"""AutoDock Vina docking utilities.

The Streamlit app uses these helpers to dock generated SMILES against a
PDBbind pocket. Meeko prepares PDBQT inputs and the Vina Python bindings run
the docking search.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class VinaResult:
    smiles: str
    vina_energy_kcal_mol: float | None
    vina_score: float | None
    status: str
    error: str = ""


def vina_available() -> tuple[bool, str]:
    """Return whether Vina + Meeko command-line tools are available."""
    missing: list[str] = []
    for exe in ("vina", "mk_prepare_ligand.py", "mk_prepare_receptor.py"):
        if shutil.which(exe) is None:
            missing.append(exe)

    if missing:
        return False, "Missing " + ", ".join(missing)
    return True, "AutoDock Vina + Meeko available"


def pdbbind_complex_dir(pdb_id: str, data_root: Path) -> Path:
    return data_root / "pdbbind" / "refined-set" / pdb_id.lower()


def dock_smiles_batch(
    smiles_list: Iterable[str],
    pocket_pdb: str,
    data_root: Path,
    exhaustiveness: int = 4,
    n_poses: int = 1,
    box_padding: float = 6.0,
) -> list[VinaResult]:
    """Dock SMILES against a PDBbind pocket and return Vina energies.

    `vina_score` is reported as `-vina_energy_kcal_mol` so it is a positive
    affinity-like proxy that can be compared visually with pKd.
    """
    ok, message = vina_available()
    if not ok:
        return [
            VinaResult(str(smi), None, None, "unavailable", message)
            for smi in smiles_list
        ]

    data_root = data_root.resolve()
    complex_dir = pdbbind_complex_dir(pocket_pdb, data_root)
    protein_pdb = complex_dir / f"{pocket_pdb.lower()}_protein.pdb"
    ref_ligand_sdf = complex_dir / f"{pocket_pdb.lower()}_ligand.sdf"
    if not protein_pdb.exists() or not ref_ligand_sdf.exists():
        return [
            VinaResult(str(smi), None, None, "missing_pdbbind_files", str(complex_dir))
            for smi in smiles_list
        ]

    center, box_size = _box_from_reference_ligand(ref_ligand_sdf, padding=box_padding)
    results: list[VinaResult] = []

    with tempfile.TemporaryDirectory(prefix="gnnbind_vina_") as td:
        tmp = Path(td)
        clean_protein_pdb = _clean_receptor_pdb(protein_pdb, tmp)
        receptor_pdbqt = _prepare_receptor(clean_protein_pdb, tmp)
        for idx, smiles in enumerate(smiles_list):
            smiles = str(smiles)
            try:
                ligand_sdf = tmp / f"ligand_{idx}.sdf"
                ligand_pdbqt = tmp / f"ligand_{idx}.pdbqt"
                _write_smiles_sdf(smiles, ligand_sdf, center)
                _prepare_ligand(ligand_sdf, ligand_pdbqt)
                energy = _run_vina(
                    receptor_pdbqt,
                    ligand_pdbqt,
                    center,
                    box_size,
                    exhaustiveness=exhaustiveness,
                    n_poses=n_poses,
                )
                results.append(
                    VinaResult(
                        smiles=smiles,
                        vina_energy_kcal_mol=energy,
                        vina_score=-energy,
                        status="ok",
                    )
                )
            except Exception as exc:
                results.append(VinaResult(smiles, None, None, "failed", str(exc)))

    return results


def _run_command(args: list[str], cwd: Path) -> None:
    proc = subprocess.run(args, cwd=str(cwd), text=True, capture_output=True)
    if proc.returncode != 0:
        stderr = proc.stderr.strip() or proc.stdout.strip()
        raise RuntimeError(stderr or f"Command failed: {' '.join(args)}")


def _prepare_receptor(protein_pdb: Path, tmp: Path) -> Path:
    out_base = tmp / "receptor"
    _run_command(
        [
            "mk_prepare_receptor.py",
            "--read_pdb",
            str(protein_pdb),
            "-o",
            str(out_base),
            "-p",
            "-a",
        ],
        cwd=tmp,
    )
    candidates = sorted(tmp.glob("receptor*.pdbqt"))
    if not candidates:
        raise RuntimeError("Meeko receptor preparation did not write a PDBQT file")
    return candidates[0]


def _clean_receptor_pdb(protein_pdb: Path, tmp: Path) -> Path:
    """Write a protein-only receptor PDB for Meeko preparation."""
    cleaned = tmp / "receptor_clean.pdb"
    keep_prefixes = ("ATOM  ", "TER", "END")
    with protein_pdb.open() as src, cleaned.open("w") as dst:
        for line in src:
            if line.startswith(keep_prefixes):
                if line.startswith("ATOM  ") and line[17:20] == "HIS":
                    line = f"{line[:17]}HIE{line[20:]}"
                dst.write(line)
        dst.write("END\n")
    return cleaned


def _prepare_ligand(ligand_sdf: Path, ligand_pdbqt: Path) -> None:
    _run_command(
        ["mk_prepare_ligand.py", "-i", str(ligand_sdf), "-o", str(ligand_pdbqt)],
        cwd=ligand_sdf.parent,
    )
    if not ligand_pdbqt.exists():
        raise RuntimeError("Meeko ligand preparation did not write a PDBQT file")


def _write_smiles_sdf(smiles: str, out_sdf: Path, center: np.ndarray) -> None:
    from rdkit import Chem
    from rdkit.Chem import AllChem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("RDKit could not parse SMILES")

    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = 0xC0FFEE
    if AllChem.EmbedMolecule(mol, params) != 0:
        raise ValueError("RDKit conformer generation failed")

    try:
        AllChem.MMFFOptimizeMolecule(mol, maxIters=200)
    except Exception:
        AllChem.UFFOptimizeMolecule(mol, maxIters=200)

    conf = mol.GetConformer()
    coords = np.array(
        [
            [
                conf.GetAtomPosition(i).x,
                conf.GetAtomPosition(i).y,
                conf.GetAtomPosition(i).z,
            ]
            for i in range(mol.GetNumAtoms())
        ],
        dtype=float,
    )
    shift = center - coords.mean(axis=0)
    for i in range(mol.GetNumAtoms()):
        pos = conf.GetAtomPosition(i)
        conf.SetAtomPosition(i, (pos.x + shift[0], pos.y + shift[1], pos.z + shift[2]))

    writer = Chem.SDWriter(str(out_sdf))
    writer.write(mol)
    writer.close()


def _box_from_reference_ligand(ligand_sdf: Path, padding: float) -> tuple[np.ndarray, list[float]]:
    from rdkit import Chem

    supplier = Chem.SDMolSupplier(str(ligand_sdf), removeHs=False)
    mol = next((m for m in supplier if m is not None), None)
    if mol is None or mol.GetNumConformers() == 0:
        raise ValueError(f"Could not read reference ligand coordinates: {ligand_sdf}")

    conf = mol.GetConformer()
    coords = np.array(
        [
            [
                conf.GetAtomPosition(i).x,
                conf.GetAtomPosition(i).y,
                conf.GetAtomPosition(i).z,
            ]
            for i in range(mol.GetNumAtoms())
        ],
        dtype=float,
    )
    center = coords.mean(axis=0)
    span = coords.max(axis=0) - coords.min(axis=0)
    box_size = np.maximum(span + padding * 2, 18.0).round(3).tolist()
    return center, box_size


def _run_vina(
    receptor_pdbqt: Path,
    ligand_pdbqt: Path,
    center: np.ndarray,
    box_size: list[float],
    exhaustiveness: int,
    n_poses: int,
) -> float:
    out_pdbqt = ligand_pdbqt.with_name(f"{ligand_pdbqt.stem}_out.pdbqt")
    args = [
        "vina",
        "--receptor",
        str(receptor_pdbqt),
        "--ligand",
        str(ligand_pdbqt),
        "--center_x",
        f"{center[0]:.3f}",
        "--center_y",
        f"{center[1]:.3f}",
        "--center_z",
        f"{center[2]:.3f}",
        "--size_x",
        f"{box_size[0]:.3f}",
        "--size_y",
        f"{box_size[1]:.3f}",
        "--size_z",
        f"{box_size[2]:.3f}",
        "--exhaustiveness",
        str(int(exhaustiveness)),
        "--num_modes",
        str(int(n_poses)),
        "--out",
        str(out_pdbqt),
    ]
    proc = subprocess.run(args, cwd=str(ligand_pdbqt.parent), text=True, capture_output=True)
    if proc.returncode != 0:
        stderr = proc.stderr.strip() or proc.stdout.strip()
        raise RuntimeError(stderr or "vina command failed")

    for line in proc.stdout.splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[0] == "1":
            return float(parts[1])

    raise RuntimeError("Could not parse Vina affinity from command output")
