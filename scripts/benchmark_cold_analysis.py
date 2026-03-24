#!/usr/bin/env python3
import argparse
import json
import shutil
import sys
import time
from pathlib import Path


def _load_stage_timings(target_dir: Path) -> dict:
    timings_path = target_dir / "analysis_stage_timings.json"
    return json.loads(timings_path.read_text(encoding="utf-8"))


def _run_once(
    *,
    repo_root: Path,
    data_path: Path,
    iteration: int,
    target_dir: Path,
    customized_drug: Path | None,
    customized_pathway: Path | None,
    cmap_dir: Path | None,
    random_background_sampling_times: int,
    perturb_change: float,
    centroid: bool,
    direction_mode: str,
    ignore_hcmarkers: bool,
    ignore_dynamic_markers: bool,
    ignore_pathway_perturabtion: bool,
    ignore_drug_perturabtion: bool,
    dynamic_marker_strategy: str,
    dynamic_marker_cpu_workers: int | None,
    dynamic_marker_force_serial: bool,
    hierarchical_marker_cpu_workers: int | None,
):
    sys.path.insert(0, str(repo_root))
    from UNAGI.UNAGI_tool import UNAGI

    unagi = UNAGI()
    started_at = time.time()
    unagi.analyse_UNAGI(
        data_path=str(data_path),
        iteration=int(iteration),
        random_background_sampling_times=int(random_background_sampling_times),
        run_perturbation=True,
        customized_pathway=str(customized_pathway) if customized_pathway is not None else None,
        target_dir=str(target_dir),
        customized_drug=str(customized_drug) if customized_drug is not None else None,
        cmap_dir=str(cmap_dir) if cmap_dir is not None else None,
        perturb_change=float(perturb_change),
        centroid=bool(centroid),
        direction_mode=str(direction_mode),
        ignore_hcmarkers=bool(ignore_hcmarkers),
        ignore_dynamic_markers=bool(ignore_dynamic_markers),
        ignore_pathway_perturabtion=bool(ignore_pathway_perturabtion),
        ignore_drug_perturabtion=bool(ignore_drug_perturabtion),
        dynamic_marker_strategy=str(dynamic_marker_strategy),
        dynamic_marker_cpu_workers=dynamic_marker_cpu_workers,
        dynamic_marker_force_serial=bool(dynamic_marker_force_serial),
        hierarchical_marker_cpu_workers=hierarchical_marker_cpu_workers,
    )
    elapsed_seconds = time.time() - started_at
    return elapsed_seconds, _load_stage_timings(target_dir)


def main():
    parser = argparse.ArgumentParser(description="Benchmark cold and warm UNAGI post-analysis runs.")
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--iteration", type=int, required=True)
    parser.add_argument("--target-dir", required=True)
    parser.add_argument("--customized-drug")
    parser.add_argument("--customized-pathway")
    parser.add_argument("--cmap-dir")
    parser.add_argument("--random-background-sampling-times", type=int, default=5)
    parser.add_argument("--perturb-change", type=float, default=0.5)
    parser.add_argument("--direction-mode", default="two_sided")
    parser.add_argument("--centroid", action="store_true")
    parser.add_argument("--ignore-hcmarkers", action="store_true")
    parser.add_argument("--ignore-dynamic-markers", action="store_true")
    parser.add_argument("--ignore-pathway-perturabtion", action="store_true")
    parser.add_argument("--ignore-drug-perturabtion", action="store_true")
    parser.add_argument("--dynamic-marker-strategy", default="optimized")
    parser.add_argument("--dynamic-marker-cpu-workers", type=int)
    parser.add_argument("--dynamic-marker-force-serial", action="store_true")
    parser.add_argument("--hierarchical-marker-cpu-workers", type=int)
    args = parser.parse_args()

    repo_root = Path(args.repo_root).expanduser().resolve()
    data_path = Path(args.data_path).expanduser().resolve()
    target_dir = Path(args.target_dir).expanduser().resolve()
    target_dir.parent.mkdir(parents=True, exist_ok=True)
    if target_dir.exists():
        shutil.rmtree(target_dir)

    run_kwargs = {
        "repo_root": repo_root,
        "data_path": data_path,
        "iteration": args.iteration,
        "target_dir": target_dir,
        "customized_drug": Path(args.customized_drug).expanduser().resolve() if args.customized_drug else None,
        "customized_pathway": Path(args.customized_pathway).expanduser().resolve() if args.customized_pathway else None,
        "cmap_dir": Path(args.cmap_dir).expanduser().resolve() if args.cmap_dir else None,
        "random_background_sampling_times": args.random_background_sampling_times,
        "perturb_change": args.perturb_change,
        "centroid": args.centroid,
        "direction_mode": args.direction_mode,
        "ignore_hcmarkers": args.ignore_hcmarkers,
        "ignore_dynamic_markers": args.ignore_dynamic_markers,
        "ignore_pathway_perturabtion": args.ignore_pathway_perturabtion,
        "ignore_drug_perturabtion": args.ignore_drug_perturabtion,
        "dynamic_marker_strategy": args.dynamic_marker_strategy,
        "dynamic_marker_cpu_workers": args.dynamic_marker_cpu_workers,
        "dynamic_marker_force_serial": args.dynamic_marker_force_serial,
        "hierarchical_marker_cpu_workers": args.hierarchical_marker_cpu_workers,
    }

    cold_elapsed, cold_timings = _run_once(**run_kwargs)
    cold_snapshot = target_dir.parent / f"{target_dir.name}_cold_stage_timings.json"
    cold_snapshot.write_text(json.dumps(cold_timings, indent=2), encoding="utf-8")

    warm_elapsed, warm_timings = _run_once(**run_kwargs)
    warm_snapshot = target_dir.parent / f"{target_dir.name}_warm_stage_timings.json"
    warm_snapshot.write_text(json.dumps(warm_timings, indent=2), encoding="utf-8")

    summary = {
        "target_dir": str(target_dir),
        "cold_elapsed_seconds": cold_elapsed,
        "warm_elapsed_seconds": warm_elapsed,
        "speedup": (cold_elapsed / warm_elapsed) if warm_elapsed > 0 else None,
        "cold_stage_timings_path": str(cold_snapshot),
        "warm_stage_timings_path": str(warm_snapshot),
        "cold_total_seconds": cold_timings.get("total_seconds"),
        "warm_total_seconds": warm_timings.get("total_seconds"),
    }
    summary_path = target_dir.parent / f"{target_dir.name}_cold_warm_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
