"""
Heterogeneous protein-ligand graph construction for GNNBindOptimizer.
Node types: ligand (atom-level), residue (pocket residue-level)
Edge types: ligand-bond, residue-contact, residue-interacts-ligand (bidirectional)
"""
import requests
from pathlib import Path
from typing import List, Optional

import numpy as np
from scipy.spatial.distance import cdist

import torch
from torch_geometric.data import HeteroData

from rdkit import Chem
from rdkit.Chem import AllChem, rdchem
import Bio.PDB as PDB
from Bio.PDB import PDBParser

POCKET_CUTOFF    = 6.0
PROT_EDGE_CUTOFF = 8.0
PL_EDGE_CUTOFF   = 5.0
LIGAND_NODE_DIM  = 28
PROTEIN_NODE_DIM = 25

ATOM_TYPES  = ["H","C","N","O","F","P","S","Cl","Br","I"]
DEGREES     = [0, 1, 2, 3, 4, 5]
HYBRIDIZE   = [rdchem.HybridizationType.SP, rdchem.HybridizationType.SP2,
               rdchem.HybridizationType.SP3, rdchem.HybridizationType.SP3D,
               rdchem.HybridizationType.SP3D2]
BOND_TYPES  = [rdchem.BondType.SINGLE, rdchem.BondType.DOUBLE,
               rdchem.BondType.TRIPLE, rdchem.BondType.AROMATIC]
AA_TYPES    = ["ALA","ARG","ASN","ASP","CYS","GLN","GLU","GLY","HIS","ILE",
               "LEU","LYS","MET","PHE","PRO","SER","THR","TRP","TYR","VAL"]
KD_SCALE    = {"ILE":4.5,"VAL":4.2,"LEU":3.8,"PHE":2.8,"CYS":2.5,"MET":1.9,
               "ALA":1.8,"GLY":-0.4,"THR":-0.7,"SER":-0.8,"TRP":-0.9,"TYR":-1.3,
               "PRO":-1.6,"HIS":-3.2,"GLU":-3.5,"GLN":-3.5,"ASP":-3.5,"ASN":-3.5,
               "LYS":-3.9,"ARG":-4.5}
HBOND_DONORS    = {"N","O"}
HBOND_ACCEPTORS = {"N","O","F"}


def one_hot(val, choices, allow_unknown=True):
    vec = [int(val == c) for c in choices]
    if allow_unknown:
        vec.append(int(val not in choices))
    return vec


def featurize_ligand_atom(atom) -> List[float]:
    sym = atom.GetSymbol()
    return (one_hot(sym, ATOM_TYPES) + one_hot(atom.GetDegree(), DEGREES)
            + one_hot(atom.GetHybridization(), HYBRIDIZE)
            + [atom.GetFormalCharge(), int(atom.IsInRing()),
               int(atom.GetIsAromatic()), atom.GetTotalNumHs()])


def featurize_ligand_bond(bond) -> List[float]:
    bt = bond.GetBondType()
    return (one_hot(bt, BOND_TYPES, allow_unknown=False)
            + [int(bond.IsInRing()), int(bond.GetIsAromatic())])


def featurize_residue(res) -> List[float]:
    rn = res.get_resname().strip()
    atom_names = [a.get_name() for a in res.get_atoms()]
    return (one_hot(rn, AA_TYPES)
            + [int("N" in atom_names), int("CA" in atom_names),
               int("C" in atom_names), int("O" in atom_names),
               KD_SCALE.get(rn, 0.0) / 4.5])


def get_ca_coords(residues):
    coords = []
    for res in residues:
        if "CA" in res:
            coords.append(res["CA"].get_vector().get_array())
        else:
            atoms = list(res.get_atoms())
            coords.append(np.mean([a.get_vector().get_array() for a in atoms], axis=0)
                          if atoms else np.zeros(3))
    return np.array(coords, dtype=np.float32)


def get_ligand_mol_from_pdb(pdb_path: Path) -> Optional[Chem.Mol]:
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("X", str(pdb_path))
    model = next(structure.get_models())
    hetatms = [r for r in model.get_residues()
               if r.id[0].startswith("H_") and r.get_resname().strip() != "HOH"]
    if not hetatms:
        return None
    res_name = hetatms[0].get_resname().strip()
    try:
        url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{res_name}/property/IsomericSMILES/JSON"
        r = requests.get(url, timeout=10)
        if r.status_code == 200:
            smiles = r.json()["PropertyTable"]["Properties"][0]["IsomericSMILES"]
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
                return mol
    except Exception:
        pass
    atoms = list(hetatms[0].get_atoms())
    em = Chem.RWMol()
    for atom in atoms:
        sym = atom.element.strip() if atom.element.strip() else "C"
        em.AddAtom(Chem.Atom(sym))
    mol = em.GetMol()
    conf = Chem.Conformer(mol.GetNumAtoms())
    for i, atom in enumerate(atoms):
        c = atom.get_vector().get_array()
        conf.SetAtomPosition(i, (float(c[0]), float(c[1]), float(c[2])))
    mol = Chem.RWMol(mol)
    mol.AddConformer(conf, assignId=True)
    return mol.GetMol()


def pdb_to_hetero_graph(pdb_path, mol, label_pkd,
                         label_pose=0.0, label_select=0.0):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("X", str(pdb_path))
    model = next(structure.get_models())
    mol = Chem.RemoveHs(Chem.AddHs(mol))
    if mol.GetNumConformers() == 0:
        AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
    if mol.GetNumConformers() == 0:
        return None
    conf = mol.GetConformer()
    lig_coords = np.array([conf.GetAtomPosition(i) for i in range(mol.GetNumAtoms())], dtype=np.float32)
    all_res = [r for r in model.get_residues() if r.id[0]==" " and r.get_resname() in AA_TYPES]
    pocket_residues = [r for r in all_res
                       if cdist(np.array([a.get_vector().get_array() for a in r.get_atoms()]),
                                lig_coords).min() <= POCKET_CUTOFF]
    if len(pocket_residues) < 3:
        return None
    x_lig  = torch.tensor([featurize_ligand_atom(a) for a in mol.GetAtoms()], dtype=torch.float)
    x_prot = torch.tensor([featurize_residue(r)     for r in pocket_residues], dtype=torch.float)
    lig_src, lig_dst, lig_eatts = [], [], []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        ef = featurize_ligand_bond(bond)
        dist = float(np.linalg.norm(lig_coords[i] - lig_coords[j]))
        lig_src += [i,j]; lig_dst += [j,i]; lig_eatts += [ef+[dist], ef+[dist]]
    ca_coords = get_ca_coords(pocket_residues)
    D_prot = cdist(ca_coords, ca_coords)
    pi, pj = np.where((D_prot>0) & (D_prot<=PROT_EDGE_CUTOFF))
    prot_src = pi.tolist(); prot_dst = pj.tolist()
    prot_eatts = [[float(D_prot[i,j])] for i,j in zip(pi,pj)]
    D_pl = cdist(ca_coords, lig_coords)
    pi2, li2 = np.where(D_pl<=PL_EDGE_CUTOFF)
    pl_src_p=[]; pl_dst_l=[]; pl_eatts=[]
    lp_src_l=[]; lp_dst_p=[]; lp_eatts=[]
    for p_idx, l_idx in zip(pi2, li2):
        dist = float(D_pl[p_idx, l_idx])
        la = mol.GetAtomWithIdx(int(l_idx))
        ef = [dist, int(la.GetSymbol() in HBOND_DONORS), int(la.GetSymbol() in HBOND_ACCEPTORS)]
        pl_src_p.append(int(p_idx)); pl_dst_l.append(int(l_idx)); pl_eatts.append(ef)
        lp_src_l.append(int(l_idx)); lp_dst_p.append(int(p_idx)); lp_eatts.append(ef)
    data = HeteroData()
    data["ligand"].x   = x_lig
    data["ligand"].pos = torch.tensor(lig_coords, dtype=torch.float)
    data["residue"].x  = x_prot
    data["residue"].pos = torch.tensor(ca_coords, dtype=torch.float)
    data["ligand","bond","ligand"].edge_index      = torch.tensor([lig_src, lig_dst], dtype=torch.long)
    data["ligand","bond","ligand"].edge_attr       = torch.tensor(lig_eatts, dtype=torch.float) if lig_eatts else torch.zeros((0,7))
    data["residue","contact","residue"].edge_index = torch.tensor([prot_src, prot_dst], dtype=torch.long)
    data["residue","contact","residue"].edge_attr  = torch.tensor(prot_eatts, dtype=torch.float) if prot_eatts else torch.zeros((0,1))
    data["residue","interacts","ligand"].edge_index = (torch.tensor([pl_src_p, pl_dst_l], dtype=torch.long) if pl_src_p else torch.zeros((2,0),dtype=torch.long))
    data["residue","interacts","ligand"].edge_attr  = torch.tensor(pl_eatts, dtype=torch.float) if pl_eatts else torch.zeros((0,3))
    data["ligand","interacts","residue"].edge_index = (torch.tensor([lp_src_l, lp_dst_p], dtype=torch.long) if lp_src_l else torch.zeros((2,0),dtype=torch.long))
    data["ligand","interacts","residue"].edge_attr  = torch.tensor(lp_eatts, dtype=torch.float) if lp_eatts else torch.zeros((0,3))
    data.y_affinity = torch.tensor([label_pkd],    dtype=torch.float)
    data.y_pose     = torch.tensor([label_pose],   dtype=torch.float)
    data.y_select   = torch.tensor([label_select], dtype=torch.float)
    data.pdb_id     = Path(pdb_path).stem
    return data
