#!/usr/bin/env python3
"""Manual two-point reference marking and QC for final 4+4 chamber videos."""

from __future__ import annotations

import argparse
import csv
import json
import math
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np


POINT_NAMES = ("A", "B")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Mark two reference points on a final 4+4 video and generate QC metrics."
    )
    parser.add_argument("video", type=Path, nargs="?", help="Final 4+4 rectified video.")
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument("--qc-only", action="store_true", help="Rebuild QC outputs from an existing reference run.")
    parser.add_argument("--input", type=Path, help="Existing reference run for --qc-only.")
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--manual-keyframe-interval-sec", type=float, default=1.0)
    parser.add_argument("--manual-keyframe-interval-frames", type=int, default=0)
    parser.add_argument("--timeline", choices=("linear", "hold"), default="linear")
    parser.add_argument("--anchor-smoothing-window", type=int, default=3)
    parser.add_argument("--reference-frame", choices=("first", "median", "manual-index"), default="median")
    parser.add_argument("--reference-manual-index", type=int)
    parser.add_argument("--minimum-distance-px", type=float, default=200.0)
    parser.add_argument(
        "--reference-points-output", type=Path, default=Path("reference_points_keyframes.json")
    )
    parser.add_argument("--metadata-jsonl", type=Path, default=Path("reference_points_metadata.jsonl"))
    return parser.parse_args()


def normalize_window(window: int) -> int:
    if window < 1:
        raise ValueError("--anchor-smoothing-window must be at least 1.")
    if window % 2 == 0:
        raise ValueError("--anchor-smoothing-window must be odd.")
    return window


def load_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Required JSON file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def resolve_output(output_dir: Path, path: Path) -> Path:
    return path if path.is_absolute() else output_dir / path


def load_reference_point_keyframes(path: Path) -> tuple[dict, list[tuple[int, np.ndarray]]]:
    data = load_json(path)
    keyframes = data.get("keyframes")
    if not isinstance(keyframes, list) or not keyframes:
        raise RuntimeError(f"No reference keyframes found in {path}")
    anchors: list[tuple[int, np.ndarray]] = []
    previous = -1
    for index, keyframe in enumerate(keyframes):
        if "processed" not in keyframe or "points" not in keyframe:
            raise RuntimeError(f"Reference keyframe {index} requires processed and points fields.")
        processed = int(keyframe["processed"])
        points = np.asarray(keyframe["points"], dtype=np.float64)
        if processed <= previous:
            raise ValueError("Reference keyframe processed indices must be strictly increasing.")
        if points.shape != (2, 2) or not np.all(np.isfinite(points)):
            raise ValueError(f"Reference keyframe {index} must contain two finite x/y points.")
        previous = processed
        anchors.append((processed, points))
    return data, anchors


def smooth_reference_point_anchors(
    anchors: list[tuple[int, np.ndarray]], window: int
) -> list[tuple[int, np.ndarray]]:
    window = normalize_window(window)
    if window == 1 or len(anchors) < 3:
        return [(processed, points.copy()) for processed, points in anchors]
    half = window // 2
    result = []
    for index, (processed, _points) in enumerate(anchors):
        lo = max(0, index - half)
        hi = min(len(anchors), index + half + 1)
        result.append((processed, np.mean([item[1] for item in anchors[lo:hi]], axis=0)))
    return result


def interpolate_reference_points(
    processed_index: int, anchors: list[tuple[int, np.ndarray]], timeline: str
) -> np.ndarray:
    if not anchors:
        raise ValueError("At least one reference anchor is required.")
    if processed_index <= anchors[0][0]:
        return anchors[0][1].copy()
    for index in range(len(anchors) - 1):
        p0, points0 = anchors[index]
        p1, points1 = anchors[index + 1]
        if p0 <= processed_index <= p1:
            if timeline == "hold" or p1 <= p0:
                return points0.copy()
            alpha = (processed_index - p0) / float(p1 - p0)
            return (1.0 - alpha) * points0 + alpha * points1
    return anchors[-1][1].copy()


def build_reference_point_timeline(
    frame_count: int,
    anchors: list[tuple[int, np.ndarray]],
    timeline: str,
    minimum_distance_px: float,
) -> list[np.ndarray]:
    if frame_count <= 0:
        raise ValueError("Reference video frame count must be positive.")
    result = []
    for processed in range(frame_count):
        points = interpolate_reference_points(processed, anchors, timeline)
        distance = float(np.linalg.norm(points[1] - points[0]))
        if distance < minimum_distance_px:
            raise ValueError(
                f"A-B distance at processed frame {processed} is {distance:.3f}px; "
                f"minimum is {minimum_distance_px:.3f}px."
            )
        result.append(points)
    return result


def choose_reference_points(
    anchors: list[tuple[int, np.ndarray]], mode: str, manual_index: int | None
) -> np.ndarray:
    if mode == "first":
        return anchors[0][1].copy()
    if mode == "median":
        return np.median(np.stack([points for _processed, points in anchors]), axis=0)
    if manual_index is None:
        raise ValueError("--reference-manual-index is required with reference mode manual-index.")
    if manual_index < 0 or manual_index >= len(anchors):
        raise IndexError(
            f"--reference-manual-index {manual_index} is outside 0..{len(anchors) - 1}."
        )
    return anchors[manual_index][1].copy()


def wrapped_angle_delta_deg(angle: float, reference_angle: float) -> float:
    return math.degrees(math.atan2(math.sin(angle - reference_angle), math.cos(angle - reference_angle)))


def compute_reference_metrics(points: np.ndarray, reference_points: np.ndarray) -> dict:
    points = np.asarray(points, dtype=np.float64)
    reference_points = np.asarray(reference_points, dtype=np.float64)
    vector = points[1] - points[0]
    reference_vector = reference_points[1] - reference_points[0]
    distance = float(np.linalg.norm(vector))
    reference_distance = float(np.linalg.norm(reference_vector))
    if distance <= 0 or reference_distance <= 0:
        raise ValueError("Reference point distance must be greater than zero.")
    centroid = np.mean(points, axis=0)
    reference_centroid = np.mean(reference_points, axis=0)
    angle = math.atan2(float(vector[1]), float(vector[0]))
    reference_angle = math.atan2(float(reference_vector[1]), float(reference_vector[0]))
    scale_relative = distance / reference_distance
    return {
        "distance_px": distance,
        "centroid_x": float(centroid[0]),
        "centroid_y": float(centroid[1]),
        "angle_rad": angle,
        "angle_deg": math.degrees(angle),
        "scale_relative_to_reference": scale_relative,
        "distance_delta_px": distance - reference_distance,
        "distance_delta_percent": 100.0 * (scale_relative - 1.0),
        "centroid_dx": float(centroid[0] - reference_centroid[0]),
        "centroid_dy": float(centroid[1] - reference_centroid[1]),
        "angle_delta_deg": wrapped_angle_delta_deg(angle, reference_angle),
    }


def probe_video(video: Path) -> dict:
    if not video.is_file():
        raise FileNotFoundError(f"Reference video not found: {video}")
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open reference video: {video}")
    result = {
        "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "source_fps": float(cap.get(cv2.CAP_PROP_FPS) or 0.0),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    }
    cap.release()
    if result["frame_count"] <= 0 or result["source_fps"] <= 0:
        raise RuntimeError(f"Reference video has invalid frame count or FPS: {video}")
    return result


def draw_picker(frame: np.ndarray, points: list[tuple[float, float]], context: str, minimum: float) -> np.ndarray:
    shown = frame.copy()
    for index, point in enumerate(points):
        xy = tuple(np.round(point).astype(int))
        cv2.circle(shown, xy, 6, (0, 255, 0), -1)
        cv2.putText(shown, POINT_NAMES[index], (xy[0] + 8, xy[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    distance = None
    if len(points) == 2:
        cv2.line(shown, tuple(np.round(points[0]).astype(int)), tuple(np.round(points[1]).astype(int)), (0, 255, 0), 2)
        distance = float(np.linalg.norm(np.asarray(points[1]) - np.asarray(points[0])))
    lines = [
        "mode: 2-point reference | order: A -> B",
        context,
        "Left click add | Enter confirm | U/Backspace undo | C clear | Esc cancel",
        f"A-B distance: {distance:.2f}px" if distance is not None else "A-B distance: waiting",
    ]
    if distance is not None and distance < minimum:
        lines.append(f"WARNING: distance must be >= {minimum:.1f}px")
    for index, text in enumerate(lines):
        color = (0, 0, 255) if text.startswith("WARNING") else (0, 255, 0)
        cv2.putText(shown, text, (15, 30 + 28 * index), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA)
    return shown


def pick_points(frame: np.ndarray, context: str, minimum: float) -> np.ndarray:
    window = "2-point reference"
    points: list[tuple[float, float]] = []

    def mouse(event, x, y, _flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 2:
            points.append((float(x), float(y)))

    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window, mouse)
    while True:
        cv2.imshow(window, draw_picker(frame, points, context, minimum))
        key = cv2.waitKey(30) & 0xFF
        if key in (13, 10):
            if len(points) != 2:
                continue
            result = np.asarray(points, dtype=np.float64)
            if float(np.linalg.norm(result[1] - result[0])) < minimum:
                continue
            return result
        if key in (8, ord("u"), ord("U")) and points:
            points.pop()
        elif key in (ord("c"), ord("C")):
            points.clear()
        elif key == 27:
            raise KeyboardInterrupt("2-point reference marking cancelled by user.")


def collect_keyframes(video: Path, interval: int, minimum: float) -> list[dict]:
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open reference video: {video}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    keyframes = []
    processed = 0
    try:
        while processed < total:
            cap.set(cv2.CAP_PROP_POS_FRAMES, processed)
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError(f"Could not read reference frame {processed} from {video}")
            points = pick_points(
                frame,
                f"keyframe {len(keyframes) + 1} | frame {processed}/{total - 1}",
                minimum,
            )
            keyframes.append(
                {
                    "processed": processed,
                    "frame_index": processed,
                    "time_sec": processed / fps,
                    "points": points.tolist(),
                    "source": "manual",
                    "pick_mode": "2-point-reference",
                }
            )
            processed += interval
    finally:
        cap.release()
        cv2.destroyAllWindows()
    return keyframes


def write_qc_outputs(
    output_dir: Path,
    video: Path,
    keyframes_path: Path,
    timeline_mode: str,
    smoothing_window: int,
    reference_mode: str,
    reference_manual_index: int | None,
    minimum_distance: float,
    metadata_jsonl_name: Path,
) -> None:
    video_info = probe_video(video)
    keyframe_data, raw_anchors = load_reference_point_keyframes(keyframes_path)
    anchors = smooth_reference_point_anchors(raw_anchors, smoothing_window)
    reference = choose_reference_points(anchors, reference_mode, reference_manual_index)
    if float(np.linalg.norm(reference[1] - reference[0])) < minimum_distance:
        raise ValueError("Selected reference A-B distance is below --minimum-distance-px.")
    timeline = build_reference_point_timeline(
        video_info["frame_count"], anchors, timeline_mode, minimum_distance
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "reference_points.csv"
    jsonl_path = resolve_output(output_dir, metadata_jsonl_name)
    fields = [
        "frame_index", "time_sec", "a_x", "a_y", "b_x", "b_y", "distance_px",
        "centroid_x", "centroid_y", "angle_rad", "angle_deg", "scale_relative_to_reference",
        "distance_delta_px", "distance_delta_percent", "centroid_dx", "centroid_dy",
        "angle_delta_deg",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as csv_file, jsonl_path.open(
        "w", encoding="utf-8"
    ) as jsonl_file:
        writer = csv.DictWriter(csv_file, fieldnames=fields)
        writer.writeheader()
        for frame_index, points in enumerate(timeline):
            metrics = compute_reference_metrics(points, reference)
            row = {
                "frame_index": frame_index,
                "time_sec": frame_index / video_info["source_fps"],
                "a_x": float(points[0, 0]), "a_y": float(points[0, 1]),
                "b_x": float(points[1, 0]), "b_y": float(points[1, 1]),
                **metrics,
            }
            writer.writerow(row)
            jsonl_file.write(json.dumps({
                "frame_index": frame_index,
                "time_sec": row["time_sec"],
                "reference_points": points.tolist(),
                "reference_points_target": reference.tolist(),
                **metrics,
            }) + "\n")
    metadata = {
        "reference_video": str(video.resolve()),
        "point_order": list(POINT_NAMES),
        "frame_count": video_info["frame_count"],
        "source_fps": video_info["source_fps"],
        "output_size": [video_info["width"], video_info["height"]],
        "reference_points_qc_only": True,
        "reference_points_keyframes": str(keyframes_path.resolve()),
        "reference_mode": reference_mode,
        "reference_manual_index": reference_manual_index,
        "reference_points_target": reference.tolist(),
        "two_point_timeline": timeline_mode,
        "two_point_anchor_smoothing_window": smoothing_window,
        "minimum_distance_px": minimum_distance,
        "outputs": {"reference_points_csv": str(csv_path), "metadata_jsonl": str(jsonl_path)},
    }
    (output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Wrote: {csv_path}")
    print(f"Wrote: {jsonl_path}")
    print(f"Wrote: {output_dir / 'run_metadata.json'}")


def main() -> None:
    args = parse_args()
    window = normalize_window(args.anchor_smoothing_window)
    if args.minimum_distance_px <= 0:
        raise ValueError("--minimum-distance-px must be positive.")
    if args.qc_only:
        if args.input is None:
            raise ValueError("--input is required with --qc-only.")
        input_run = args.input.resolve()
        run_meta = load_json(input_run / "run_metadata.json")
        if "reference_video" not in run_meta or "reference_points_keyframes" not in run_meta:
            raise RuntimeError(
                "Reference run metadata requires reference_video and reference_points_keyframes."
            )
        output_dir = args.out_dir or input_run
        write_qc_outputs(
            output_dir, Path(run_meta["reference_video"]), Path(run_meta["reference_points_keyframes"]),
            args.timeline, window, args.reference_frame, args.reference_manual_index,
            args.minimum_distance_px, args.metadata_jsonl,
        )
        return
    if args.video is None:
        raise ValueError("A reference video is required unless --qc-only is used.")
    if not args.interactive:
        raise ValueError("New 2-point annotations require --interactive.")
    video_info = probe_video(args.video)
    interval = args.manual_keyframe_interval_frames
    if interval <= 0:
        if args.manual_keyframe_interval_sec <= 0:
            raise ValueError("--manual-keyframe-interval-sec must be positive.")
        interval = max(1, int(round(args.manual_keyframe_interval_sec * video_info["source_fps"])))
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.out_dir or args.video.parent / f"{args.video.stem}_ref2pt_{stamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    keyframes = collect_keyframes(args.video, interval, args.minimum_distance_px)
    keyframes_path = resolve_output(output_dir, args.reference_points_output)
    keyframes_path.write_text(json.dumps({
        "note": "2-point reference annotations on final 4+4 rectified video.",
        "reference_video": str(args.video.resolve()),
        "point_order": list(POINT_NAMES),
        "keyframes": keyframes,
    }, indent=2), encoding="utf-8")
    write_qc_outputs(
        output_dir, args.video, keyframes_path, args.timeline, window, args.reference_frame,
        args.reference_manual_index, args.minimum_distance_px, args.metadata_jsonl,
    )


if __name__ == "__main__":
    main()
