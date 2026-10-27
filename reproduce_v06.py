"""Render the retained V06 4+4+2 and white-reference correction."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml


HERE = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=HERE / "config.yaml")
    parser.add_argument("--input-video", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--dry-run", action="store_true", help="Validate and build transforms without writing videos.")
    return parser.parse_args()


def resolve(base: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def inspect_video(path: Path) -> dict[str, float | int]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open input video: {path}")
    result = {
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "fps": float(cap.get(cv2.CAP_PROP_FPS)),
    }
    cap.release()
    return result


def validate_video(observed: dict[str, float | int], expected: dict[str, Any]) -> None:
    for name in ("width", "height", "frame_count"):
        if int(observed[name]) != int(expected[name]):
            raise ValueError(f"Input {name}={observed[name]} but V06 requires {expected[name]}.")
    if not np.isclose(float(observed["fps"]), float(expected["fps"]), rtol=0, atol=0.02):
        raise ValueError(f"Input FPS={observed['fps']} but V06 requires {expected['fps']}.")


def run(command: list[str]) -> None:
    print(subprocess.list2cmdline(command), flush=True)
    subprocess.run(command, cwd=HERE, check=True)


def copy_file(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def base_metadata(video: Path, observed: dict[str, float | int], config: dict[str, Any]) -> dict[str, Any]:
    geometry = config["geometry"]
    runtime = config["runtime"]
    return {
        "video": str(video),
        "frame_count": int(observed["frame_count"]),
        "start_frame": 0,
        "step": 1,
        "preview_fps": float(runtime["preview_fps"]),
        "rectified_fps": float(runtime["rectified_fps"]),
        "source_fps": float(observed["fps"]),
        "tracking_mode": "manual-fixed",
        "tracking_mask_json": None,
        "calibration_json": None,
        "output_mode": "fixed-context",
        "output_size": [int(geometry["output_width"]), int(geometry["output_height"])],
        "output_width": int(geometry["output_width"]),
        "output_height": int(geometry["output_height"]),
        "rectified_square_size": int(geometry["rectified_square_size"]),
        "global_translation_h": None,
        "chamber_width_mm": float(geometry["chamber_width_mm"]),
        "rectified_px_per_mm": float(geometry["rectified_px_per_mm"]),
    }


def prepare_geometry_runs(
    video: Path,
    output_root: Path,
    observed: dict[str, float | int],
    config: dict[str, Any],
) -> tuple[Path, Path]:
    assets = HERE / "v06_assets"
    runtime = output_root / f"_runtime_inputs_{datetime.now():%Y%m%d_%H%M%S}"
    pass1, pass2 = runtime / "pass1", runtime / "pass2"
    pass1.mkdir(parents=True)
    pass2.mkdir(parents=True)
    for stage in ("pass1", "pass2"):
        copy_file(assets / stage / "manual_chamber_keyframes.json", runtime / stage / "manual_chamber_keyframes.json")
        copy_file(assets / stage / "tracked_chamber_corners.csv", runtime / stage / "tracked_chamber_corners.csv")
    copy_file(assets / "pass1" / "tracked_chamber_metadata.jsonl", pass1 / "tracked_chamber_metadata.jsonl")
    metadata = base_metadata(video, observed, config)
    # The second-pass coordinates are in the first-pass rectified coordinate system.
    # Pairing is represented by one existing path; the final renderer uses the retained
    # per-frame first-pass homographies and renders from the original video.
    metadata["outputs"] = {
        "rectified_video": str(video),
        "metadata_jsonl": str((pass1 / "tracked_chamber_metadata.jsonl").resolve()),
    }
    (pass1 / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    pass2_metadata = dict(metadata)
    pass2_metadata["video"] = str(video)
    pass2_metadata["outputs"] = {"rectified_video": str(video)}
    pass2_metadata["initial_raw_px_per_mm"] = float(config["geometry"]["rectified_px_per_mm"])
    (pass2 / "run_metadata.json").write_text(json.dumps(pass2_metadata, indent=2), encoding="utf-8")
    return pass1, pass2


def prepare_reference_run(
    fourplusfour_dir: Path,
    output_root: Path,
    observed: dict[str, float | int],
    config: dict[str, Any],
) -> Path:
    reference = output_root / "_runtime_inputs" / "reference_2pt"
    reference.mkdir(parents=True, exist_ok=True)
    keyframes = reference / "reference_points_keyframes.json"
    copy_file(HERE / "v06_assets" / "reference_2pt" / keyframes.name, keyframes)
    four_meta = json.loads((fourplusfour_dir / "run_metadata.json").read_text(encoding="utf-8"))
    four_video = Path(four_meta["outputs"]["rectified_video"]).resolve()
    settings = config["reference_2point"]
    metadata = {
        "reference_video": str(four_video),
        "point_order": ["A", "B"],
        "frame_count": int(observed["frame_count"]),
        "source_fps": float(four_meta["source_fps"]),
        "output_size": four_meta["output_size"],
        "reference_points_qc_only": True,
        "reference_points_keyframes": str(keyframes.resolve()),
        "reference_mode": settings["reference_mode"],
        "reference_manual_index": None,
        "two_point_timeline": settings["timeline"],
        "two_point_anchor_smoothing_window": int(settings["annotation_smoothing_window"]),
        "minimum_distance_px": float(settings["minimum_reference_distance"]),
    }
    (reference / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return reference


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    video = (args.input_video or resolve(config_path.parent, config["input_video"])).resolve()
    output_root = (args.output_root or resolve(config_path.parent, config["output_root"])).resolve()
    if not video.is_file():
        raise FileNotFoundError(f"Input video not found: {video}")
    output_root.mkdir(parents=True, exist_ok=True)
    for target_name in ("fourplusfour", "final_v06"):
        target = output_root / target_name
        if target.exists() and any(target.iterdir()):
            raise FileExistsError(
                f"Refusing to overwrite existing output: {target}. Select a new output directory."
            )
    observed = inspect_video(video)
    validate_video(observed, config["expected_input"])
    print(f"Validated input: {observed}", flush=True)
    pass1, pass2 = prepare_geometry_runs(video, output_root, observed, config)
    geometry = config["geometry"]
    fourplusfour = output_root / "fourplusfour"
    command = [
        sys.executable,
        str(HERE / "core" / "rerender_manual_keyframes_smooth.py"),
        "--second-pass-residual",
        "--first-pass-run",
        str(pass1),
        "--second-pass-run",
        str(pass2),
        "--out-dir",
        str(fourplusfour),
        "--timeline",
        str(geometry["timeline"]),
        "--anchor-smoothing-window",
        str(geometry["pass2_anchor_smoothing_window"]),
        "--max-residual-corner-shift-warning-px",
        str(geometry["max_residual_corner_shift_warning_px"]),
    ]
    if args.dry_run:
        command.append("--dry-run")
    run(command)
    if args.dry_run:
        print("Dry-run completed after validating the retained 4+4 transforms.", flush=True)
        return
    reference = prepare_reference_run(fourplusfour, output_root, observed, config)
    settings = config["reference_2point"]
    final = output_root / "final_v06"
    command = [
        sys.executable,
        str(HERE / "core" / "rerender_manual_keyframes_smooth.py"),
        "--apply-2point-similarity-correction",
        "--fourplusfour-run",
        str(fourplusfour),
        "--reference-2pt-run",
        str(reference),
        "--out-dir",
        str(final),
        "--timeline",
        str(settings["timeline"]),
        "--anchor-smoothing-window",
        str(settings["render_smoothing_window"]),
        "--reference-mode",
        str(settings["reference_mode"]),
        "--minimum-reference-distance",
        str(settings["minimum_reference_distance"]),
        "--max-scale-warning-percent",
        str(settings["max_scale_warning_percent"]),
        "--max-angle-warning-deg",
        str(settings["max_angle_warning_deg"]),
        "--max-translation-warning-px",
        str(settings["max_translation_warning_px"]),
    ]
    white = config["white_reference"]
    if white["enabled"]:
        calibration = resolve(config_path.parent, white["calibration_directory"])
        command.extend(["--white-reference-calibration", str(calibration), "--white-reference-id", str(white["id"])])
    run(command)
    print(f"Final quantitative video: {final / 'tracked_full_frame_rectified.mp4'}", flush=True)


if __name__ == "__main__":
    main()
