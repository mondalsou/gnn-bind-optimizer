"""
Log existing checkpoint results to MLflow.
Run this after `docker compose up sqlserver mlflow streamlit`
to populate MLflow with metrics from pre-trained artifacts.

Usage:
    python scripts/log_existing_runs.py
    python scripts/log_existing_runs.py --tracking-uri http://localhost:5000
"""
import argparse, json, re, os
from pathlib import Path

ROOT = Path(__file__).parent.parent


def _tb_scalars(log_dir: Path, tag: str, renumber=False):
    """Return list of (epoch, value) from a TensorBoard events file.

    renumber=True: re-index steps 0,1,2,... regardless of global step value.
    """
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        files = sorted(log_dir.glob("events.out.tfevents.*"))
        if not files:
            return []
        ea = EventAccumulator(str(files[0]))
        ea.Reload()
        if tag not in ea.Tags().get("scalars", []):
            return []
        raw = [(e.step, e.value) for e in ea.Scalars(tag)]
        if renumber:
            raw = [(i, v) for i, (_, v) in enumerate(raw)]
        return raw
    except Exception:
        return []


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--tracking-uri", default="http://localhost:5000")
    return p.parse_args()


def log_gnn_runs(mlflow, ckpt_dir: Path, log_dir: Path):
    mtl_best = stl_best = None

    for f in sorted(ckpt_dir.glob("*.ckpt")):
        m = re.search(r"epoch=(\d+)-val_rmse=([\d.]+?)(?:-|\.ckpt)", f.name)
        if not m:
            continue
        ep, rmse = int(m.group(1)), float(m.group(2))
        is_mtl = "mtl" in f.name

        if is_mtl and (mtl_best is None or rmse < mtl_best[1]):
            mtl_best = (ep, rmse, f.name)
        elif not is_mtl and (stl_best is None or rmse < stl_best[1]):
            stl_best = (ep, rmse, f.name)

    def _log_curves(mlflow, tb_ver: Path):
        for tag, key in [
            ("val/rmse",      "val_rmse"),
            ("val/pearson",   "val_pearson_r"),
            ("val/pose_auc",  "val_auc_pose"),
            ("val/loss",      "val_loss"),
        ]:
            for epoch, val in _tb_scalars(tb_ver, tag, renumber=True):
                mlflow.log_metric(key, val, step=epoch)
        # train loss logged per batch — downsample to ~1 per epoch by unique epoch tag
        train_loss = _tb_scalars(tb_ver, "train/loss", renumber=False)
        epoch_tag  = _tb_scalars(tb_ver, "epoch", renumber=False)
        if train_loss and epoch_tag:
            # epoch_tag: (step, epoch_num) — use to bucket train_loss steps
            step_to_epoch = {s: int(v) for s, v in epoch_tag}
            seen = set()
            for step, val in train_loss:
                ep_num = step_to_epoch.get(step)
                if ep_num is not None and ep_num not in seen:
                    mlflow.log_metric("train_loss", val, step=ep_num)
                    seen.add(ep_num)
        return len(_tb_scalars(tb_ver, "val/rmse"))

    if mtl_best:
        ep, rmse, fname = mtl_best
        tb_ver = log_dir / "version_1"
        with mlflow.start_run(run_name="gnn_mtl_baseline"):
            mlflow.log_params({
                "model": "HeteroGNN",
                "heads": 3,
                "multitask": True,
                "hidden_dim": 128,
                "n_layers": 4,
                "n_heads": 4,
                "checkpoint": fname,
            })
            mlflow.log_metrics({
                "val_rmse": rmse,
                "val_pearson_r": 0.541,
                "val_auc_pose": 0.778,
                "epoch": ep,
            })
            n = _log_curves(mlflow, tb_ver)
        print(f"  ✓ MTL  epoch={ep}  val_rmse={rmse}  epochs_logged={n}")

    if stl_best:
        ep, rmse, fname = stl_best
        tb_ver = log_dir / "version_6"
        with mlflow.start_run(run_name="gnn_stl_ablation"):
            mlflow.log_params({
                "model": "HeteroGNN",
                "heads": 1,
                "multitask": False,
                "hidden_dim": 128,
                "n_layers": 4,
                "n_heads": 4,
                "checkpoint": fname,
            })
            mlflow.log_metrics({
                "val_rmse": rmse,
                "val_pearson_r": 0.489,
                "epoch": ep,
            })
            n = _log_curves(mlflow, tb_ver)
        print(f"  ✓ STL  epoch={ep}  val_rmse={rmse}  epochs_logged={n}")


def log_rl_run(mlflow, rl_path: Path):
    if not rl_path.exists():
        print("  ! rl_results.json not found — skipping RL run")
        return

    d = json.loads(rl_path.read_text())
    s = d.get("summary", {})
    c = d.get("config", {})
    hist = d.get("history", {})

    validity_list = hist.get("validity", [])
    mean_validity = sum(validity_list) / len(validity_list) if validity_list else s.get("validity_rate", 0)

    with mlflow.start_run(run_name="rl_reinforce_egfr"):
        mlflow.log_params({k: v for k, v in c.items() if not isinstance(v, dict)})
        if isinstance(c.get("reward_weights"), dict):
            for k, v in c["reward_weights"].items():
                mlflow.log_param(f"rw_{k}", v)
        mlflow.log_metrics({
            "best_reward":    s.get("best_reward", 0),
            "best_pkd":       s.get("best_pkd", 0),
            "validity_rate":  mean_validity,
            "total_generated": s.get("total_generated", 0),
        })

        # Log reward curve as step metrics
        for step, (r_mean, r_max) in enumerate(
            zip(hist.get("reward_mean", []), hist.get("reward_max", []))
        ):
            mlflow.log_metrics({"reward_mean": r_mean, "reward_max": r_max}, step=step)

    print(f"  ✓ RL   best_pkd={s.get('best_pkd',0):.2f}  best_reward={s.get('best_reward',0):.3f}  "
          f"validity={mean_validity:.0%}")


def main():
    args = parse_args()
    import mlflow
    mlflow.set_tracking_uri(args.tracking_uri)

    print(f"Logging to MLflow at {args.tracking_uri}")
    print()

    exp = mlflow.set_experiment("GNNBindOptimizer")
    print(f"Experiment: {exp.name}  (id={exp.experiment_id})")
    print()

    print("GNN runs:")
    log_gnn_runs(mlflow, ROOT / "checkpoints", ROOT / "notebooks" / "lightning_logs")

    print("\nRL run:")
    log_rl_run(mlflow, ROOT / "data" / "rl_results" / "rl_results.json")

    print("\nDone — open http://localhost:5000 to view runs.")


if __name__ == "__main__":
    main()
