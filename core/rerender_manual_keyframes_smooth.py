#!/usr/bin/env python3
"""Re-render manual-keyframe chamber outputs with a smoother corner timeline.

This does not ask the user to mark points again. It reuses an existing manual
run folder containing:
  - run_metadata.json
  - manual_chamber_keyframes.json
  - tracked_chamber_corners.csv

Default behavior linearly interpolates between manual keyframes instead of
holding each keyframe for one second and then jumping.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

import track_chamber_rect_fixed_context as base
import track_chamber_rect_tracking_modes as modes


cv2.setUseOptimized(True)
cv2.setNumThreads(max(1, (os.cpu_count() or 2) - 1))


DEFAULT_GOOD_MANUAL_ROOT = Path(
    r"C:\Users\fifa-\Downloads\Compressed\Rewrite_MS_FLUID_Combine\Chamber Tracking\Good_manual"
)
DEFAULT_OUTPUTS_ROOT = Path(
    r"C:\Users\fifa-\Downloads\Compressed\Rewrite_MS_FLUID_Combine\Chamber Tracking"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Re-render existing manual chamber keyframes with smooth per-frame interpolation."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_GOOD_MANUAL_ROOT,
        help="A manual run folder, or a folder containing multiple manual run folders.",
    )
    parser.add_argument(
        "--outputs-root",
        type=Path,
        default=DEFAULT_OUTPUTS_ROOT,
        help="Root folder for new output run folders.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        help="Output folder override. Only valid when --input is one run folder.",
    )
    parser.add_argument(
        "--timeline",
        choices=("linear", "hold"),
        default="linear",
        help="linear interpolates between manual anchors; hold reproduces the old step behavior.",
    )
    parser.add_argument(
        "--anchor-smoothing-window",
        type=int,
        default=1,
        help="Odd moving-average window over manual keyframes before interpolation. 1 disables it.",
    )
    parser.add_argument("--metadata-jsonl", type=Path, default=Path("tracked_chamber_metadata.jsonl"))
    parser.add_argument("--preview-fps", type=float, help="Override overlay FPS. Default uses run metadata.")
    parser.add_argument("--output-mode", choices=("fixed-context", "preserve-full-frame"), help="Override output mode.")
    parser.add_argument("--output-width", type=int, help="Override fixed-context width.")
    parser.add_argument("--output-height", type=int, help="Override fixed-context height.")
    parser.add_argument("--rectified-square-size", type=int, help="Override chamber target square size.")
    parser.add_argument("--calibration-json", type=Path, help="Override calibration JSON. None means no undistortion.")
    parser.add_argument("--second-pass-residual", action="store_true")
    parser.add_argument("--first-pass-run", type=Path)
    parser.add_argument("--second-pass-run", type=Path)
    parser.add_argument("--apply-2point-similarity-correction", action="store_true")
    parser.add_argument("--fourplusfour-run", type=Path)
    parser.add_argument("--reference-2pt-run", type=Path)
    parser.add_argument(
        "--white-reference-calibration",
        type=Path,
        help="Optional white-reference calibration folder to apply during explicit homography export.",
    )
    parser.add_argument("--white-reference-id", help="White reference ID used for metadata.")
    parser.add_argument("--reference-mode", choices=("first", "median", "manual-index"), default="median")
    parser.add_argument("--reference-manual-index", type=int)
    parser.add_argument("--minimum-reference-distance", type=float, default=200.0)
    parser.add_argument("--max-scale-warning-percent", type=float, default=2.0)
    parser.add_argument("--max-angle-warning-deg", type=float, default=1.0)
    parser.add_argument("--max-translation-warning-px", type=float, default=10.0)
    parser.add_argument("--max-residual-corner-shift-warning-px", type=float, default=25.0)
    parser.add_argument("--dry-run", action="store_true", help="Print planned rerenders without writing video outputs.")
    return parser.parse_args()


def is_run_dir(path: Path) -> bool:
    return (
        (path / "run_metadata.json").exists()
        and (path / "manual_chamber_keyframes.json").exists()
        and (path / "tracked_chamber_corners.csv").exists()
    )


def discover_runs(input_path: Path) -> list[Path]:
    if is_run_dir(input_path):
        return [input_path]
    runs = sorted(p for p in input_path.iterdir() if p.is_dir() and is_run_dir(p))
    if not runs:
        raise FileNotFoundError(f"No manual run folders found under: {input_path}")
    return runs


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def require_file(path: Path, description: str) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"{description} not found: {path}")
    return path


def require_run(path: Path | None, option: str) -> Path:
    if path is None:
        raise ValueError(f"{option} is required for this mode.")
    if not path.is_dir():
        raise FileNotFoundError(f"Run folder not found for {option}: {path}")
    return path.resolve()


def load_white_reference_correction(calibration_dir: Path | None) -> dict | None:
    if calibration_dir is None:
        return None
    calibration_dir = calibration_dir.resolve()
    metadata_path = require_file(calibration_dir / "calibration_metadata.json", "White-reference calibration metadata")
    metadata = load_json(metadata_path)
    outputs = metadata.get("outputs") if isinstance(metadata.get("outputs"), dict) else {}
    required = {
        "dark_ref_rgb": outputs.get("dark_ref_rgb", "dark_ref_rgb.npy"),
        "flatfield_gain": outputs.get("flatfield_gain", "flatfield_gain.npy"),
        "wb_gain": outputs.get("wb_gain", "wb_gain.npy"),
    }
    arrays = {
        name: np.load(require_file(calibration_dir / filename, f"White-reference {name}"))
        for name, filename in required.items()
    }
    dark_ref_rgb = arrays["dark_ref_rgb"].astype(np.float32)
    gain_map = arrays["flatfield_gain"].astype(np.float32)
    wb_gain = arrays["wb_gain"].astype(np.float32)
    if dark_ref_rgb.shape != (3,):
        raise ValueError("White-reference dark_ref_rgb must have shape (3,).")
    if gain_map.ndim != 2:
        raise ValueError("White-reference flatfield_gain must be a 2D array.")
    if wb_gain.shape != (3,):
        raise ValueError("White-reference wb_gain must have shape (3,).")
    return {
        "calibration_dir": str(calibration_dir),
        "metadata": metadata,
        "dark_ref_rgb": dark_ref_rgb,
        "gain_map": gain_map,
        "wb_gain": wb_gain,
        "flatfield_strength": float(metadata.get("flatfield_strength", 0.7)),
    }


def apply_white_reference_bgr(frame_bgr: np.ndarray, correction: dict) -> np.ndarray:
    gain_map = correction["gain_map"]
    if gain_map.shape != frame_bgr.shape[:2]:
        raise ValueError(
            f"White-reference gain map shape {gain_map.shape[::-1]} does not match frame "
            f"{frame_bgr.shape[1]}x{frame_bgr.shape[0]}."
        )
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    rgb = np.clip(rgb - correction["dark_ref_rgb"].reshape(1, 1, 3), 0.0, 1.0)
    rgb = rgb * (gain_map[..., None] ** np.float32(correction["flatfield_strength"]))
    rgb = rgb * correction["wb_gain"].reshape(1, 1, 3)
    rgb_u8 = np.clip(rgb * 255.0, 0, 255).astype(np.uint8)
    return cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2BGR)


def resolved_equal(left: Path, right: Path) -> bool:
    return os.path.normcase(str(left.resolve())) == os.path.normcase(str(right.resolve()))


def metadata_output_path(run_meta: dict, key: str, run_dir: Path) -> Path:
    outputs = run_meta.get("outputs")
    if not isinstance(outputs, dict) or not outputs.get(key):
        raise RuntimeError(f"Required metadata field outputs.{key} is missing in {run_dir / 'run_metadata.json'}")
    return Path(outputs[key])


def validate_homography(value, field_name: str, frame_index: int) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (3, 3):
        raise ValueError(f"{field_name} at frame {frame_index} must be a 3x3 matrix.")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{field_name} at frame {frame_index} contains non-finite values.")
    if abs(float(np.linalg.det(matrix))) < 1e-10:
        raise ValueError(f"{field_name} at frame {frame_index} is singular.")
    if abs(float(matrix[2, 2])) > 1e-12:
        matrix = matrix / matrix[2, 2]
    return matrix


def load_jsonl(path: Path) -> list[dict]:
    require_file(path, "Metadata JSONL")
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number} of {path}: {exc}") from exc
            rows.append(row)
    if not rows:
        raise RuntimeError(f"No frame metadata found in {path}")
    return rows


def load_first_pass_homographies(first_pass_run: Path) -> tuple[dict, list[dict]]:
    run_meta = load_json(require_file(first_pass_run / "run_metadata.json", "First-pass run metadata"))
    jsonl_path = metadata_output_path(run_meta, "metadata_jsonl", first_pass_run)
    rows = load_jsonl(jsonl_path)
    result: list[dict] = []
    previous = None
    for row in rows:
        if "frame_index" not in row or "time_sec" not in row or "output_h" not in row:
            raise RuntimeError(
                f"First-pass metadata requires frame_index, time_sec, and output_h: {jsonl_path}"
            )
        frame_index = int(row["frame_index"])
        if previous is not None and frame_index <= previous:
            raise ValueError(f"First-pass frame indices are not strictly increasing in {jsonl_path}")
        previous = frame_index
        result.append(
            {
                "frame_index": frame_index,
                "time_sec": float(row["time_sec"]),
                "output_h": validate_homography(row["output_h"], "output_h", frame_index),
                "rectification_h": row.get("rectification_h"),
            }
        )
    return run_meta, result


def compute_canonical_square(output_width: int, output_height: int, rectified_square_size: int) -> np.ndarray:
    if output_width <= 0 or output_height <= 0 or rectified_square_size <= 0:
        raise ValueError("Output width, height, and rectified square size must be positive.")
    if rectified_square_size > min(output_width, output_height):
        raise ValueError("Rectified square size must fit inside the output canvas.")
    return base.centered_target_square(output_width, output_height, rectified_square_size).astype(np.float32)


def compute_residual_homography_from_4pt(corners_4pt: np.ndarray, canonical_square: np.ndarray) -> np.ndarray:
    corners = np.asarray(corners_4pt, dtype=np.float32)
    if corners.shape != (4, 2) or not np.all(np.isfinite(corners)):
        raise ValueError("Second-pass corners must contain four finite x/y points.")
    matrix = cv2.getPerspectiveTransform(corners, np.asarray(canonical_square, dtype=np.float32))
    return validate_homography(matrix, "second_pass_residual_h", -1)


def compose_4plus4_homographies(h_first: np.ndarray, h_residual: np.ndarray) -> np.ndarray:
    return validate_homography(h_residual @ h_first, "h_4plus4", -1)


def validate_pairing(reference_video: Path, expected_video: Path, label: str) -> None:
    require_file(reference_video, f"{label} reference video")
    require_file(expected_video, f"{label} expected video")
    if not resolved_equal(reference_video, expected_video):
        raise ValueError(
            f"{label} coordinate-system mismatch. Annotated video is {reference_video}, "
            f"but the required paired video is {expected_video}."
        )


def load_csv_timeline(path: Path) -> list[dict]:
    with path.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise RuntimeError(f"No frame rows found in {path}")
    return rows


def output_dir_for_run(args: argparse.Namespace, run_dir: Path, multi_run: bool) -> Path:
    if args.out_dir is not None:
        if multi_run:
            raise ValueError("--out-dir can only be used with one input run folder.")
        return args.out_dir
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = f"smooth_{args.timeline}"
    if args.anchor_smoothing_window > 1:
        suffix += f"_avg{args.anchor_smoothing_window}"
    return args.outputs_root / f"{run_dir.name}_{suffix}_{stamp}"


def normalize_window(window: int) -> int:
    if window <= 1:
        return 1
    if window % 2 == 0:
        window += 1
    return window


def smooth_anchor_corners(anchors: list[tuple[int, np.ndarray]], window: int) -> list[tuple[int, np.ndarray]]:
    window = normalize_window(window)
    if window <= 1 or len(anchors) < 3:
        return anchors

    half = window // 2
    out: list[tuple[int, np.ndarray]] = []
    for i, (processed, _corners) in enumerate(anchors):
        lo = max(0, i - half)
        hi = min(len(anchors), i + half + 1)
        stacked = np.stack([corners for _p, corners in anchors[lo:hi]], axis=0)
        out.append((processed, np.mean(stacked, axis=0).astype(np.float32)))
    return out


def load_calibration_matrices(
    calibration_json: Path,
    frame_size: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = load_json(calibration_json)
    if "camera_matrix" not in data or "dist_coeffs" not in data:
        raise ValueError("Calibration JSON must contain camera_matrix and dist_coeffs.")
    width, height = frame_size
    camera_matrix = np.asarray(data["camera_matrix"], dtype=np.float32).reshape(3, 3)
    dist_coeffs = np.asarray(data["dist_coeffs"], dtype=np.float32).reshape(-1)
    new_camera_matrix, _roi = cv2.getOptimalNewCameraMatrix(
        camera_matrix,
        dist_coeffs,
        (width, height),
        0.0,
        (width, height),
    )
    return camera_matrix, dist_coeffs, new_camera_matrix


def undistort_anchor_corners(
    anchors: list[tuple[int, np.ndarray]],
    calibration_json: Path | None,
    frame_size: tuple[int, int],
    source_run_already_calibrated: bool,
) -> list[tuple[int, np.ndarray]]:
    if calibration_json is None or source_run_already_calibrated:
        return anchors

    camera_matrix, dist_coeffs, new_camera_matrix = load_calibration_matrices(calibration_json, frame_size)
    out: list[tuple[int, np.ndarray]] = []
    for processed, corners in anchors:
        points = corners.astype(np.float32).reshape(-1, 1, 2)
        undistorted = cv2.undistortPoints(points, camera_matrix, dist_coeffs, P=new_camera_matrix).reshape(4, 2)
        out.append((processed, undistorted.astype(np.float32)))
    print(
        "Applied calibration to manual keyframe coordinates because source run was uncalibrated.",
        flush=True,
    )
    return out


def interpolate_corners(
    processed_index: int,
    anchors: list[tuple[int, np.ndarray]],
    timeline: str,
) -> np.ndarray:
    if not anchors:
        raise ValueError("At least one keyframe is required.")
    if processed_index <= anchors[0][0]:
        return anchors[0][1].copy()

    for i in range(len(anchors) - 1):
        p0, c0 = anchors[i]
        p1, c1 = anchors[i + 1]
        if p0 <= processed_index <= p1:
            if timeline == "hold" or p1 <= p0:
                return c0.copy()
            alpha = (processed_index - p0) / float(p1 - p0)
            return ((1.0 - alpha) * c0 + alpha * c1).astype(np.float32)

    return anchors[-1][1].copy()


def make_export_args(args: argparse.Namespace, run_meta: dict, out_dir: Path) -> SimpleNamespace:
    calibration_json = args.calibration_json
    if calibration_json is None and run_meta.get("calibration_json"):
        calibration_json = Path(run_meta["calibration_json"])

    return SimpleNamespace(
        start_frame=int(run_meta.get("start_frame", 0)),
        step=int(run_meta.get("step", 1)),
        metadata_jsonl=args.metadata_jsonl,
        tracking_mode="manual-keyframe-rerender",
        tracking_mask_json=None,
        calibration_json=calibration_json,
        output_mode=args.output_mode or run_meta.get("output_mode", "fixed-context"),
        output_width=int(args.output_width or run_meta.get("output_width", 1024)),
        output_height=int(args.output_height or run_meta.get("output_height", 1024)),
        rectified_square_size=int(args.rectified_square_size or run_meta.get("rectified_square_size", 512)),
        geometry_area_tol=0.65,
        geometry_aspect_tol=0.45,
        out_dir=out_dir,
    )


def build_tracks(
    frame_rows: list[dict],
    keyframes: list[dict],
    timeline: str,
    anchor_smoothing_window: int,
    export_args: SimpleNamespace,
    frame_size: tuple[int, int],
    source_run_already_calibrated: bool,
) -> tuple[list[modes.FrameTrack], list[tuple[int, np.ndarray]]]:
    anchors = sorted(
        (
            int(kf.get("processed", kf.get("frame_index", 0))),
            np.asarray(kf["corners"], dtype=np.float32),
        )
        for kf in keyframes
    )
    anchors = undistort_anchor_corners(
        anchors,
        export_args.calibration_json,
        frame_size,
        source_run_already_calibrated,
    )
    anchors = smooth_anchor_corners(anchors, anchor_smoothing_window)
    initial_area = modes.polygon_area(anchors[0][1])
    initial_aspect = modes.aspect_ratio(anchors[0][1])

    tracks: list[modes.FrameTrack] = []
    for processed, row in enumerate(frame_rows):
        corners = interpolate_corners(processed, anchors, timeline)
        geometry_valid = modes.geometry_is_valid(corners, initial_area, initial_aspect, export_args)
        tracks.append(
            modes.FrameTrack(
                frame_index=int(float(row["frame_index"])),
                time_sec=float(row.get("time_sec", processed)),
                ok=geometry_valid,
                weak=not geometry_valid,
                track_count=0,
                inlier_count=0,
                track_consistency=0.0,
                geometry_valid=geometry_valid,
                corners=corners,
                tracking_h=np.eye(3, dtype=np.float32),
            )
        )
    return tracks, anchors


def export_with_explicit_homographies(
    *,
    source_video: Path,
    output_dir: Path,
    frame_rows: list[dict],
    homographies: list[np.ndarray],
    per_frame_metadata: list[dict],
    start_frame: int,
    step: int,
    source_fps: float,
    preview_fps: float,
    rectified_fps: float,
    output_size: tuple[int, int],
    rectified_square_size: int,
    metadata_jsonl_name: Path,
    run_metadata_extra: dict,
    white_reference_correction: dict | None = None,
) -> None:
    if not (len(frame_rows) == len(homographies) == len(per_frame_metadata)):
        raise ValueError("Frame rows, explicit homographies, and per-frame metadata must have equal lengths.")
    if step <= 0:
        raise ValueError("Source step must be positive.")
    if source_fps <= 0 or preview_fps <= 0 or rectified_fps <= 0:
        raise ValueError("Source, preview, and rectified FPS metadata must be positive.")
    require_file(source_video, "Original calibrated source video")
    output_dir.mkdir(parents=True, exist_ok=True)
    overlay_path = output_dir / "tracked_chamber_overlay.mp4"
    rectified_path = output_dir / "tracked_full_frame_rectified.mp4"
    corners_csv_path = output_dir / "tracked_chamber_corners.csv"
    metadata_path = base.resolve_optional_output_path(output_dir, metadata_jsonl_name)
    if metadata_path is None:
        raise ValueError("Explicit-homography modes require --metadata-jsonl.")

    cap = cv2.VideoCapture(str(source_video))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open original calibrated source video: {source_video}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    ok, first = cap.read()
    if not ok:
        cap.release()
        raise RuntimeError(f"Could not read start frame {start_frame} from {source_video}")

    rectified_writer, rectified_path = base.make_rectified_writer(rectified_path, rectified_fps, output_size)
    overlay_writer = base.make_writer(overlay_path, preview_fps, output_size)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    target_square = compute_canonical_square(output_size[0], output_size[1], rectified_square_size)

    written = 0
    try:
        with corners_csv_path.open("w", newline="", encoding="utf-8") as csv_file, metadata_path.open(
            "w", encoding="utf-8"
        ) as metadata_file:
            writer = csv.writer(csv_file)
            writer.writerow(["frame_index", "time_sec", "output_w", "output_h"])
            for i, (row, matrix, extra) in enumerate(zip(frame_rows, homographies, per_frame_metadata)):
                if i == 0:
                    frame = first
                else:
                    frame = None
                    for _ in range(step):
                        ok, frame = cap.read()
                        if not ok:
                            break
                    if not ok or frame is None:
                        raise RuntimeError(
                            f"Original source ended after {written} frames; expected {len(frame_rows)}."
                        )
                matrix = validate_homography(matrix, "explicit_output_h", int(row["frame_index"]))
                warped = cv2.warpPerspective(frame, matrix, output_size)
                if white_reference_correction is not None:
                    warped = apply_white_reference_bgr(warped, white_reference_correction)
                rectified_writer.write(warped)
                overlay = warped.copy()
                cv2.polylines(overlay, [np.round(target_square).astype(np.int32)], True, (0, 255, 0), 2)
                cv2.putText(
                    overlay,
                    f"frame {int(row['frame_index'])} explicit composed homography",
                    (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )
                overlay_writer.write(overlay)
                writer.writerow(
                    [
                        int(row["frame_index"]),
                        f"{float(row['time_sec']):.6f}",
                        output_size[0],
                        output_size[1],
                    ]
                )
                record = {
                    "frame_index": int(row["frame_index"]),
                    "time_sec": float(row["time_sec"]),
                    "output_size": list(output_size),
                }
                record.update(extra)
                metadata_file.write(json.dumps(record) + "\n")
                written += 1
    finally:
        cap.release()
        rectified_writer.release()
        overlay_writer.release()

    run_metadata = {
        "video": str(source_video),
        "source_video": str(source_video),
        "rendered_from_original_video": True,
        "frame_count": written,
        "start_frame": start_frame,
        "step": step,
        "preview_fps": preview_fps,
        "rectified_fps": rectified_fps,
        "source_fps": source_fps,
        "calibration_json": None,
        "output_mode": "fixed-context",
        "output_size": list(output_size),
        "output_width": output_size[0],
        "output_height": output_size[1],
        "rectified_square_size": rectified_square_size,
        "rectified_px_per_mm": rectified_square_size / base.CHAMBER_WIDTH_MM,
        "outputs": {
            "overlay_video": str(overlay_path),
            "rectified_video": str(rectified_path),
            "corners_csv": str(corners_csv_path),
            "metadata_jsonl": str(metadata_path),
        },
    }
    run_metadata.update(run_metadata_extra)
    (output_dir / "run_metadata.json").write_text(json.dumps(run_metadata, indent=2), encoding="utf-8")
    print(f"Wrote: {rectified_path}")
    print(f"Wrote: {overlay_path}")
    print(f"Wrote: {metadata_path}")
    print(f"Wrote: {output_dir / 'run_metadata.json'}")


def load_second_pass_4pt_annotations(
    second_pass_run: Path,
    first_pass_meta: dict,
    args: argparse.Namespace,
) -> tuple[dict, list[dict], list[modes.FrameTrack]]:
    second_meta = load_json(require_file(second_pass_run / "run_metadata.json", "Second-pass run metadata"))
    keyframe_data = load_json(
        require_file(second_pass_run / "manual_chamber_keyframes.json", "Second-pass keyframes")
    )
    frame_rows = load_csv_timeline(
        require_file(second_pass_run / "tracked_chamber_corners.csv", "Second-pass frame timeline")
    )
    expected_video = metadata_output_path(first_pass_meta, "rectified_video", Path(first_pass_meta.get("video", ".")))
    if "video" not in second_meta:
        raise RuntimeError("Required metadata field video is missing in second-pass run_metadata.json")
    validate_pairing(Path(second_meta["video"]), expected_video, "Second-pass")
    if second_meta.get("calibration_json"):
        raise ValueError("Second-pass points are already rectified; calibration_json must be null.")
    keyframes = keyframe_data.get("keyframes")
    if not isinstance(keyframes, list) or not keyframes:
        raise RuntimeError("Second-pass manual_chamber_keyframes.json contains no keyframes.")
    export_args = make_export_args(args, second_meta, second_pass_run)
    export_args.calibration_json = None
    tracks, _anchors = build_tracks(
        frame_rows,
        keyframes,
        args.timeline,
        args.anchor_smoothing_window,
        export_args,
        (int(second_meta.get("output_width", 1024)), int(second_meta.get("output_height", 1024))),
        True,
    )
    return second_meta, frame_rows, tracks


def default_mode_output_dir(args: argparse.Namespace, stem: str) -> Path:
    if args.out_dir is not None:
        return args.out_dir
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return args.outputs_root / f"{stem}_{stamp}"


def rerender_second_pass_residual(args: argparse.Namespace) -> Path:
    first_run = require_run(args.first_pass_run, "--first-pass-run")
    second_run = require_run(args.second_pass_run, "--second-pass-run")
    first_meta, first_rows = load_first_pass_homographies(first_run)
    if first_meta.get("calibration_json"):
        raise ValueError(
            "Final 4+4 requires an original calibrated video; first-pass calibration_json must be null."
        )
    if "video" not in first_meta:
        raise RuntimeError("Required metadata field video is missing in first-pass run_metadata.json")
    source_video = require_file(Path(first_meta["video"]), "Original calibrated video")
    second_meta, second_rows, second_tracks = load_second_pass_4pt_annotations(second_run, first_meta, args)
    if len(first_rows) != len(second_rows):
        raise ValueError(
            f"Frame-count mismatch: first pass has {len(first_rows)}, second pass has {len(second_rows)}."
        )
    if int(first_meta.get("start_frame", 0)) != 0 and int(first_meta.get("start_frame", 0)) != int(
        second_meta.get("start_frame", 0)
    ):
        raise ValueError("First-pass and second-pass start_frame metadata are incompatible.")
    output_size_value = first_meta.get("output_size")
    if not isinstance(output_size_value, list) or len(output_size_value) != 2:
        raise RuntimeError("Required first-pass output_size metadata is missing or invalid.")
    output_size = (int(output_size_value[0]), int(output_size_value[1]))
    square_size = int(first_meta.get("rectified_square_size", 0))
    canonical = compute_canonical_square(output_size[0], output_size[1], square_size)

    homographies: list[np.ndarray] = []
    metadata_rows: list[dict] = []
    warnings_found: list[str] = []
    for ordinal, (first_row, second_row, track) in enumerate(zip(first_rows, second_rows, second_tracks)):
        expected_first_index = int(first_meta.get("start_frame", 0)) + ordinal * int(
            first_meta.get("step", 1)
        )
        if first_row["frame_index"] != expected_first_index:
            raise ValueError(
                f"First-pass frame alignment failed at processed index {ordinal}: "
                f"expected {expected_first_index}, found {first_row['frame_index']}."
            )
        expected_second_index = int(second_meta.get("start_frame", 0)) + ordinal * int(
            second_meta.get("step", 1)
        )
        if int(float(second_row["frame_index"])) != expected_second_index:
            raise ValueError(
                f"Second-pass frame alignment failed at processed index {ordinal}: "
                f"expected {expected_second_index}, found {second_row['frame_index']}."
            )
        residual = compute_residual_homography_from_4pt(track.corners, canonical)
        composed = compose_4plus4_homographies(first_row["output_h"], residual)
        shift = float(np.max(np.linalg.norm(track.corners - canonical, axis=1)))
        if shift > args.max_residual_corner_shift_warning_px:
            warnings_found.append(
                f"frame {first_row['frame_index']}: residual corner shift {shift:.3f}px exceeds "
                f"{args.max_residual_corner_shift_warning_px:.3f}px"
            )
        homographies.append(composed)
        metadata_rows.append(
            {
                "first_pass_output_h": first_row["output_h"].tolist(),
                "second_pass_corners": track.corners.tolist(),
                "second_pass_residual_h": residual.tolist(),
                "h_4plus4": composed.tolist(),
                "residual_max_corner_shift_px": shift,
            }
        )
    out_dir = default_mode_output_dir(args, f"{first_run.name}_final_4plus4")
    print(f"Final 4+4 | frames={len(first_rows)} | source={source_video} | output={out_dir}")
    for warning in warnings_found:
        print(f"Warning: {warning}")
    if args.dry_run:
        return out_dir
    export_with_explicit_homographies(
        source_video=source_video,
        output_dir=out_dir,
        frame_rows=first_rows,
        homographies=homographies,
        per_frame_metadata=metadata_rows,
        start_frame=int(first_meta.get("start_frame", 0)),
        step=int(first_meta.get("step", 1)),
        source_fps=float(first_meta.get("source_fps", 0)),
        preview_fps=float(args.preview_fps or first_meta.get("preview_fps", 20.0)),
        rectified_fps=float(first_meta.get("rectified_fps", first_meta.get("source_fps", 30.0))),
        output_size=output_size,
        rectified_square_size=square_size,
        metadata_jsonl_name=args.metadata_jsonl,
        run_metadata_extra={
            "second_pass_residual_rerender": True,
            "first_pass_run": str(first_run),
            "second_pass_run": str(second_run),
            "homography_composition": "H_4plus4 = H_residual_pass2 @ H_first",
            "second_pass_timeline": args.timeline,
            "second_pass_anchor_smoothing_window": normalize_window(args.anchor_smoothing_window),
            "warnings": warnings_found,
        },
    )
    return out_dir


def compute_two_point_similarity(
    current_points: np.ndarray,
    reference_points: np.ndarray,
    minimum_distance: float,
) -> tuple[np.ndarray, dict]:
    current = np.asarray(current_points, dtype=np.float64)
    target = np.asarray(reference_points, dtype=np.float64)
    if current.shape != (2, 2) or target.shape != (2, 2):
        raise ValueError("Current and target reference points must each have shape (2, 2).")
    if not np.all(np.isfinite(current)) or not np.all(np.isfinite(target)):
        raise ValueError("Reference points must contain finite values.")
    vector = current[1] - current[0]
    target_vector = target[1] - target[0]
    distance = float(np.linalg.norm(vector))
    target_distance = float(np.linalg.norm(target_vector))
    if distance < minimum_distance or target_distance < minimum_distance:
        raise ValueError(
            f"A-B distance is below the minimum {minimum_distance:.3f}px "
            f"(current={distance:.3f}, target={target_distance:.3f})."
        )
    scale = target_distance / distance
    angle = math.atan2(float(target_vector[1]), float(target_vector[0])) - math.atan2(
        float(vector[1]), float(vector[0])
    )
    cosine = math.cos(angle)
    sine = math.sin(angle)
    scaled_rotation = scale * np.asarray([[cosine, -sine], [sine, cosine]], dtype=np.float64)
    translation = target[0] - scaled_rotation @ current[0]
    matrix = np.eye(3, dtype=np.float64)
    matrix[:2, :2] = scaled_rotation
    matrix[:2, 2] = translation
    matrix = validate_homography(matrix, "s_2pt", -1)
    return matrix, {
        "distance_px": distance,
        "target_distance_px": target_distance,
        "scale_correction": scale,
        "scale_correction_percent": 100.0 * (scale - 1.0),
        "angle_correction_deg": math.degrees(
            math.atan2(math.sin(angle), math.cos(angle))
        ),
        "translation_x": float(translation[0]),
        "translation_y": float(translation[1]),
        "translation_magnitude_px": float(np.linalg.norm(translation)),
    }


def load_fourplusfour_homographies(run_dir: Path) -> tuple[dict, list[dict]]:
    run_meta = load_json(require_file(run_dir / "run_metadata.json", "Final 4+4 run metadata"))
    if run_meta.get("second_pass_residual_rerender") is not True:
        raise RuntimeError("fourplusfour-run is not marked second_pass_residual_rerender=true.")
    if run_meta.get("rendered_from_original_video") is not True:
        raise RuntimeError("fourplusfour-run is not marked rendered_from_original_video=true.")
    rows = load_jsonl(metadata_output_path(run_meta, "metadata_jsonl", run_dir))
    result = []
    for row in rows:
        if "frame_index" not in row or "time_sec" not in row or "h_4plus4" not in row:
            raise RuntimeError("Final 4+4 JSONL requires frame_index, time_sec, and h_4plus4.")
        frame_index = int(row["frame_index"])
        result.append(
            {
                "frame_index": frame_index,
                "time_sec": float(row["time_sec"]),
                "h_4plus4": validate_homography(row["h_4plus4"], "h_4plus4", frame_index),
            }
        )
    return run_meta, result


def rerender_two_point_correction(args: argparse.Namespace) -> Path:
    import track_reference_points_manual_keyframes as reference

    four_run = require_run(args.fourplusfour_run, "--fourplusfour-run")
    reference_run = require_run(args.reference_2pt_run, "--reference-2pt-run")
    four_meta, four_rows = load_fourplusfour_homographies(four_run)
    reference_meta = load_json(
        require_file(reference_run / "run_metadata.json", "2-point reference run metadata")
    )
    if "reference_video" not in reference_meta:
        raise RuntimeError("Required metadata field reference_video is missing in reference run metadata.")
    four_video = metadata_output_path(four_meta, "rectified_video", four_run)
    validate_pairing(Path(reference_meta["reference_video"]), four_video, "2-point reference")
    if "reference_points_keyframes" not in reference_meta:
        raise RuntimeError(
            "Required metadata field reference_points_keyframes is missing in reference run metadata."
        )
    _keyframe_data, anchors = reference.load_reference_point_keyframes(
        Path(reference_meta["reference_points_keyframes"])
    )
    anchors = reference.smooth_reference_point_anchors(
        anchors, normalize_window(args.anchor_smoothing_window)
    )
    target_points = reference.choose_reference_points(
        anchors, args.reference_mode, args.reference_manual_index
    )
    point_timeline = reference.build_reference_point_timeline(
        len(four_rows), anchors, args.timeline, args.minimum_reference_distance
    )
    if int(reference_meta.get("frame_count", -1)) != len(four_rows):
        raise ValueError(
            f"Reference/final frame-count mismatch: {reference_meta.get('frame_count')} vs {len(four_rows)}."
        )
    if "source_video" not in four_meta:
        raise RuntimeError("Required source_video field is missing in final 4+4 run metadata.")
    source_video = require_file(Path(four_meta["source_video"]), "Original calibrated source video")
    output_size_value = four_meta.get("output_size")
    if not isinstance(output_size_value, list) or len(output_size_value) != 2:
        raise RuntimeError("Required final 4+4 output_size metadata is missing or invalid.")
    output_size = (int(output_size_value[0]), int(output_size_value[1]))
    square_size = int(four_meta.get("rectified_square_size", 0))
    compute_canonical_square(output_size[0], output_size[1], square_size)
    white_reference_correction = load_white_reference_correction(args.white_reference_calibration)
    if white_reference_correction is not None and white_reference_correction["gain_map"].shape != (
        output_size[1],
        output_size[0],
    ):
        raise ValueError(
            "White-reference calibration shape does not match 4+4+2 output size: "
            f"{white_reference_correction['gain_map'].shape[::-1]} vs {output_size}."
        )

    homographies = []
    metadata_rows = []
    warnings_found: list[str] = []
    previous_correction: np.ndarray | None = None
    for row, points in zip(four_rows, point_timeline):
        correction, correction_metrics = compute_two_point_similarity(
            points, target_points, args.minimum_reference_distance
        )
        composed = validate_homography(
            correction @ row["h_4plus4"], "h_final_4plus4_2pt", row["frame_index"]
        )
        frame_warnings = []
        if abs(correction_metrics["scale_correction_percent"]) > args.max_scale_warning_percent:
            frame_warnings.append("scale correction exceeds warning threshold")
        if abs(correction_metrics["angle_correction_deg"]) > args.max_angle_warning_deg:
            frame_warnings.append("angle correction exceeds warning threshold")
        if correction_metrics["translation_magnitude_px"] > args.max_translation_warning_px:
            frame_warnings.append("translation correction exceeds warning threshold")
        if previous_correction is not None:
            jump = float(np.linalg.norm(correction - previous_correction))
            correction_metrics["adjacent_similarity_matrix_delta"] = jump
        previous_correction = correction
        for text in frame_warnings:
            warnings_found.append(f"frame {row['frame_index']}: {text}")
        qc_metrics = reference.compute_reference_metrics(points, target_points)
        homographies.append(composed)
        metadata_rows.append(
            {
                "h_4plus4": row["h_4plus4"].tolist(),
                "reference_points": points.tolist(),
                "reference_points_target": target_points.tolist(),
                **qc_metrics,
                **correction_metrics,
                "s_2pt": correction.tolist(),
                "h_final_4plus4_2pt": composed.tolist(),
                "warnings": frame_warnings,
            }
        )
    suffix = (
        f"{four_run.name}_final_4plus4_ref2pt_white_reference"
        if white_reference_correction is not None
        else f"{four_run.name}_final_4plus4_ref2pt_corrected"
    )
    out_dir = default_mode_output_dir(args, suffix)
    print(f"Final 4+4+2 | frames={len(four_rows)} | source={source_video} | output={out_dir}")
    if white_reference_correction is not None:
        print(f"Applying white-reference calibration: {white_reference_correction['calibration_dir']}")
    for warning in warnings_found:
        print(f"Warning: {warning}")
    if args.dry_run:
        return out_dir
    export_with_explicit_homographies(
        source_video=source_video,
        output_dir=out_dir,
        frame_rows=four_rows,
        homographies=homographies,
        per_frame_metadata=metadata_rows,
        start_frame=int(four_meta.get("start_frame", 0)),
        step=int(four_meta.get("step", 1)),
        source_fps=float(four_meta.get("source_fps", 0)),
        preview_fps=float(args.preview_fps or four_meta.get("preview_fps", 20.0)),
        rectified_fps=float(four_meta.get("rectified_fps", four_meta.get("source_fps", 30.0))),
        output_size=output_size,
        rectified_square_size=square_size,
        metadata_jsonl_name=args.metadata_jsonl,
        run_metadata_extra={
            "two_point_similarity_correction": True,
            "white_reference_correction": white_reference_correction is not None,
            "white_reference_id": args.white_reference_id,
            "white_reference_calibration": (
                None if white_reference_correction is None else white_reference_correction["calibration_dir"]
            ),
            "white_reference_marked_video": (
                None
                if white_reference_correction is None
                else white_reference_correction["metadata"].get("input_video")
            ),
            "fourplusfour_run": str(four_run),
            "reference_2pt_run": str(reference_run),
            "homography_composition": "H_final = S_2pt @ H_4plus4",
            "reference_mode": args.reference_mode,
            "reference_manual_index": args.reference_manual_index,
            "two_point_timeline": args.timeline,
            "two_point_anchor_smoothing_window": normalize_window(args.anchor_smoothing_window),
            "minimum_reference_distance": args.minimum_reference_distance,
            "warning_thresholds": {
                "max_scale_warning_percent": args.max_scale_warning_percent,
                "max_angle_warning_deg": args.max_angle_warning_deg,
                "max_translation_warning_px": args.max_translation_warning_px,
            },
            "warnings": warnings_found,
        },
        white_reference_correction=white_reference_correction,
    )
    return out_dir


def rerender_run(args: argparse.Namespace, run_dir: Path, multi_run: bool) -> Path:
    run_meta = load_json(run_dir / "run_metadata.json")
    keyframe_data = load_json(run_dir / "manual_chamber_keyframes.json")
    frame_rows = load_csv_timeline(run_dir / "tracked_chamber_corners.csv")
    keyframes = keyframe_data.get("keyframes", [])
    if not keyframes:
        raise RuntimeError(f"No keyframes found: {run_dir / 'manual_chamber_keyframes.json'}")

    video_path = Path(run_meta["video"])
    if not video_path.exists():
        raise FileNotFoundError(f"Original video not found: {video_path}")

    out_dir = output_dir_for_run(args, run_dir, multi_run)
    export_args = make_export_args(args, run_meta, out_dir)

    print(
        f"Rerender {run_dir.name} | frames={len(frame_rows)} | keyframes={len(keyframes)} | "
        f"timeline={args.timeline} | anchor_avg={normalize_window(args.anchor_smoothing_window)}",
        flush=True,
    )
    print(f"  video: {video_path}", flush=True)
    print(f"  output: {out_dir}", flush=True)

    if args.dry_run:
        return out_dir

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")
    source_fps = float(run_meta.get("source_fps") or cap.get(cv2.CAP_PROP_FPS) or 30.0)
    preview_fps = float(args.preview_fps or run_meta.get("preview_fps") or 20.0)
    rectified_fps = float(run_meta.get("rectified_fps") or (source_fps / max(export_args.step, 1)))
    cap.set(cv2.CAP_PROP_POS_FRAMES, export_args.start_frame)
    ok, first = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Could not read start frame {export_args.start_frame} from {video_path}")

    frame_size = (first.shape[1], first.shape[0])
    out_dir.mkdir(parents=True, exist_ok=True)
    tracks, anchors = build_tracks(
        frame_rows,
        keyframes,
        args.timeline,
        args.anchor_smoothing_window,
        export_args,
        frame_size,
        bool(run_meta.get("calibration_json")),
    )

    undistort_maps = base.make_undistort_maps((first.shape[1], first.shape[0]), export_args.calibration_json)
    first_frame = base.preprocess_frame(first, undistort_maps)
    initial_raw_px_per_mm = float(
        run_meta.get("initial_raw_px_per_mm")
        or base.compute_px_per_mm(np.asarray(keyframes[0]["corners"], dtype=np.float32))
    )

    shutil.copy2(run_dir / "manual_chamber_keyframes.json", out_dir / "manual_chamber_keyframes.json")
    initial_corners_path = run_dir / "initial_corners.json"
    if initial_corners_path.exists():
        shutil.copy2(initial_corners_path, out_dir / "initial_corners.json")
    else:
        (out_dir / "initial_corners.json").write_text(
            json.dumps(np.asarray(keyframes[0]["corners"], dtype=np.float32).tolist(), indent=2),
            encoding="utf-8",
        )

    modes.export_outputs(
        video_path,
        out_dir,
        tracks,
        (first_frame.shape[1], first_frame.shape[0]),
        source_fps,
        preview_fps,
        rectified_fps,
        export_args,
        undistort_maps,
        initial_raw_px_per_mm,
    )

    metadata_path = out_dir / "run_metadata.json"
    metadata = load_json(metadata_path)
    metadata.update(
        {
            "source_manual_run": str(run_dir),
            "rerender_from_manual_keyframes": True,
            "manual_corner_timeline": args.timeline,
            "anchor_smoothing_window": normalize_window(args.anchor_smoothing_window),
            "manual_keyframe_count": len(keyframes),
            "smoothed_anchor_count": len(anchors),
            "note": "No new point picking was done. Corners were reconstructed from manual_chamber_keyframes.json.",
        }
    )
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"  done: {metadata_path}", flush=True)
    return out_dir


def main() -> None:
    args = parse_args()
    for option, value in (
        ("--minimum-reference-distance", args.minimum_reference_distance),
        ("--max-scale-warning-percent", args.max_scale_warning_percent),
        ("--max-angle-warning-deg", args.max_angle_warning_deg),
        ("--max-translation-warning-px", args.max_translation_warning_px),
        ("--max-residual-corner-shift-warning-px", args.max_residual_corner_shift_warning_px),
    ):
        if value <= 0:
            raise ValueError(f"{option} must be positive.")
    selected_modes = int(args.second_pass_residual) + int(args.apply_2point_similarity_correction)
    if selected_modes > 1:
        raise ValueError(
            "--second-pass-residual and --apply-2point-similarity-correction are mutually exclusive."
        )
    if selected_modes and args.anchor_smoothing_window < 1:
        raise ValueError("--anchor-smoothing-window must be at least 1 for multi-pass modes.")
    if args.second_pass_residual:
        rerender_second_pass_residual(args)
        return
    if args.apply_2point_similarity_correction:
        rerender_two_point_correction(args)
        return
    runs = discover_runs(args.input)
    multi_run = len(runs) > 1
    print(f"Found {len(runs)} run(s).", flush=True)
    for i, run_dir in enumerate(runs, 1):
        print(f"[{i}/{len(runs)}]", flush=True)
        rerender_run(args, run_dir, multi_run)


if __name__ == "__main__":
    main()
