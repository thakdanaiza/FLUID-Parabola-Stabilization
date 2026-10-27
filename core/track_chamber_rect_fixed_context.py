"""
Track a rectangular chamber and write a fixed-context rectified video.

This is a simpler output version of track_chamber_rect.py. Tracking stays as
the existing initial 4-corner selection + LK/RANSAC homography flow. The default
rectified output is a fixed 1024x1024 canvas with the chamber centered as a
512x512 square, which avoids huge mostly-black preserve-full-frame canvases.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np


CHAMBER_WIDTH_MM = 40.0
DEFAULT_OUTPUTS_ROOT = Path(
    r"c:\Users\fifa-\Downloads\Compressed\Rewrite_MS_FLUID_Combine\Chamber Tracking"
)


@dataclass
class FrameTrack:
    frame_index: int
    time_sec: float
    ok: bool
    weak: bool
    track_count: int
    inlier_count: int
    confidence: float
    corners: np.ndarray
    tracking_h: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Track chamber corners and export fixed-context rectification."
    )
    parser.add_argument("video", type=Path, help="Input video path.")
    parser.add_argument(
        "--outputs-root",
        type=Path,
        default=DEFAULT_OUTPUTS_ROOT,
        help="Base output folder. Results are written to <outputs-root>/<video_stem>_<timestamp>.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        help="Full output folder override. If omitted, uses <outputs-root>/<video_stem>_<timestamp>.",
    )
    parser.add_argument("--start-frame", type=int, default=0, help="0-based frame index.")
    parser.add_argument("--num-frames", type=int, default=0, help="0 means process until EOF.")
    parser.add_argument("--step", type=int, default=1, help="Process every Nth frame.")
    parser.add_argument("--interactive", action="store_true", help="Click 4 corners on first frame.")
    parser.add_argument(
        "--corners-json",
        type=Path,
        help="JSON file with [[x,y], [x,y], [x,y], [x,y]] corners.",
    )
    parser.add_argument(
        "--calibration-json",
        type=Path,
        help=(
            "Optional real camera calibration JSON with camera_matrix and dist_coeffs. "
            "No undistortion is applied unless this is provided."
        ),
    )
    parser.add_argument(
        "--metadata-jsonl",
        type=Path,
        help="Optional per-frame metadata JSONL path. Relative paths are written inside the output folder.",
    )
    parser.add_argument(
        "--output-mode",
        choices=("fixed-context", "preserve-full-frame"),
        default="fixed-context",
        help="Rectified output mode. fixed-context is the default.",
    )
    parser.add_argument("--output-width", type=int, default=1024)
    parser.add_argument("--output-height", type=int, default=1024)
    parser.add_argument(
        "--rectified-square-size",
        type=int,
        default=512,
        help="Square target size for the chamber in rectified output coordinates.",
    )
    parser.add_argument("--max-corners", type=int, default=600)
    parser.add_argument("--quality", type=float, default=0.01)
    parser.add_argument("--min-distance", type=int, default=8)
    parser.add_argument("--ransac-thresh", type=float, default=3.0)
    parser.add_argument("--min-tracks", type=int, default=40)
    parser.add_argument("--preview-fps", type=float, default=20.0)
    parser.add_argument(
        "--track-region",
        choices=("ring", "chamber"),
        default="ring",
        help="Use chamber border ring features by default to avoid the moving fluid region.",
    )
    parser.add_argument(
        "--ring-margin",
        type=float,
        default=0.16,
        help="Inset fraction removed from the chamber mask when --track-region=ring.",
    )
    return parser.parse_args()


def order_points(points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32).reshape(4, 2)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).ravel()
    ordered = np.zeros((4, 2), dtype=np.float32)
    ordered[0] = pts[np.argmin(s)]
    ordered[2] = pts[np.argmax(s)]
    ordered[1] = pts[np.argmin(diff)]
    ordered[3] = pts[np.argmax(diff)]
    return ordered


def load_json_array(path: Path) -> np.ndarray:
    return np.asarray(json.loads(path.read_text(encoding="utf-8")), dtype=np.float32)


def load_corners(path: Path) -> np.ndarray:
    return order_points(load_json_array(path))


def pick_corners_interactive(frame: np.ndarray) -> np.ndarray:
    points: list[tuple[int, int]] = []
    display = frame.copy()
    window_name = "Click chamber corners: TL, TR, BR, BL. Press Enter when done."

    def on_mouse(event, x, y, _flags, _param):
        if event != cv2.EVENT_LBUTTONDOWN or len(points) >= 4:
            return
        points.append((x, y))
        cv2.circle(display, (x, y), 6, (0, 255, 255), -1)
        cv2.putText(
            display,
            str(len(points)),
            (x + 8, y - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
        if len(points) > 1:
            cv2.line(display, points[-2], points[-1], (0, 255, 255), 2)
        cv2.imshow(window_name, display)

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, on_mouse)
    cv2.imshow(window_name, display)

    while True:
        key = cv2.waitKey(20) & 0xFF
        if key in (13, 10) and len(points) == 4:
            break
        if key == 27:
            raise RuntimeError("Corner selection canceled.")

    cv2.destroyWindow(window_name)
    return order_points(np.asarray(points, dtype=np.float32))


def auto_detect_rectangle(frame: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 50, 150)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates = []
    frame_area = frame.shape[0] * frame.shape[1]
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < frame_area * 0.02:
            continue
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
        if len(approx) != 4 or not cv2.isContourConvex(approx):
            continue
        candidates.append((area, approx.reshape(4, 2)))

    if not candidates:
        raise RuntimeError(
            "Could not auto-detect a 4-corner chamber. Re-run with --interactive "
            "or provide --corners-json."
        )

    _area, pts = max(candidates, key=lambda item: item[0])
    return order_points(pts)


def make_tracking_mask(shape: tuple[int, int], corners: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    cv2.fillConvexPoly(mask, corners.astype(np.int32), 255)
    if args.track_region != "ring":
        return mask

    center = corners.mean(axis=0)
    margin = float(np.clip(args.ring_margin, 0.0, 0.45))
    inner = center + (corners - center) * (1.0 - margin)
    cv2.fillConvexPoly(mask, inner.astype(np.int32), 0)
    return mask


def detect_features(gray: np.ndarray, corners: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    mask = make_tracking_mask(gray.shape, corners, args)
    points = cv2.goodFeaturesToTrack(
        gray,
        maxCorners=args.max_corners,
        qualityLevel=args.quality,
        minDistance=args.min_distance,
        mask=mask,
        blockSize=7,
    )
    if points is None or len(points) < 4:
        if args.track_region == "ring":
            print("Warning: ring tracking mask found too few features; retrying full chamber mask.")
            fallback_args = argparse.Namespace(**vars(args))
            fallback_args.track_region = "chamber"
            return detect_features(gray, corners, fallback_args)
        raise RuntimeError("Not enough trackable features inside chamber region.")
    return points.astype(np.float32)


def compute_px_per_mm(corners: np.ndarray) -> float:
    top = np.linalg.norm(corners[1] - corners[0])
    bottom = np.linalg.norm(corners[2] - corners[3])
    return float((top + bottom) * 0.5 / CHAMBER_WIDTH_MM)


def make_writer(path: Path, fps: float, size: tuple[int, int], fourcc_name: str = "mp4v") -> cv2.VideoWriter:
    fourcc = cv2.VideoWriter_fourcc(*fourcc_name)
    writer = cv2.VideoWriter(str(path), cv2.CAP_FFMPEG, fourcc, fps, size)
    if not writer.isOpened():
        writer = cv2.VideoWriter(str(path), fourcc, fps, size)
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer: {path}")
    return writer


def make_rectified_writer(path: Path, fps: float, size: tuple[int, int]) -> tuple[cv2.VideoWriter, Path]:
    try:
        return make_writer(path, fps, size), path
    except RuntimeError:
        fallback_path = path.with_suffix(".avi")
        return make_writer(fallback_path, fps, size, "MJPG"), fallback_path


def resolve_output_dir(args: argparse.Namespace) -> Path:
    if args.out_dir is not None:
        return args.out_dir
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return args.outputs_root / f"{args.video.stem}_{timestamp}"


def resolve_optional_output_path(output_dir: Path, path: Path | None) -> Path | None:
    if path is None:
        return None
    if path.is_absolute():
        return path
    return output_dir / path


def make_undistort_maps(frame_size: tuple[int, int], calibration_json: Path | None):
    if calibration_json is None:
        return None

    data = json.loads(calibration_json.read_text(encoding="utf-8"))
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
    return cv2.initUndistortRectifyMap(
        camera_matrix,
        dist_coeffs,
        None,
        new_camera_matrix,
        (width, height),
        cv2.CV_16SC2,
    )


def preprocess_frame(frame: np.ndarray, undistort_maps) -> np.ndarray:
    if undistort_maps is None:
        return frame
    map1, map2 = undistort_maps
    return cv2.remap(frame, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)


def centered_target_square(output_width: int, output_height: int, square_size: int) -> np.ndarray:
    cx = output_width / 2.0
    cy = output_height / 2.0
    half = square_size / 2.0
    return np.array(
        [
            [cx - half, cy - half],
            [cx + half, cy - half],
            [cx + half, cy + half],
            [cx - half, cy + half],
        ],
        dtype=np.float32,
    )


def origin_target_square(square_size: int) -> np.ndarray:
    return np.array(
        [
            [0, 0],
            [square_size - 1, 0],
            [square_size - 1, square_size - 1],
            [0, square_size - 1],
        ],
        dtype=np.float32,
    )


def compute_fixed_context_h(corners: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    target = centered_target_square(args.output_width, args.output_height, args.rectified_square_size)
    return cv2.getPerspectiveTransform(corners.astype(np.float32), target)


def compute_origin_rectification_h(corners: np.ndarray, square_size: int) -> np.ndarray:
    return cv2.getPerspectiveTransform(corners.astype(np.float32), origin_target_square(square_size))


def transformed_frame_corners(frame_size: tuple[int, int], homography: np.ndarray) -> np.ndarray:
    width, height = frame_size
    frame_corners = np.array(
        [
            [0, 0],
            [width - 1, 0],
            [width - 1, height - 1],
            [0, height - 1],
        ],
        dtype=np.float32,
    )
    return cv2.perspectiveTransform(frame_corners.reshape(1, 4, 2), homography).reshape(4, 2)


def compute_preserve_full_frame_geometry(
    tracks: list[FrameTrack],
    frame_size: tuple[int, int],
    square_size: int,
) -> tuple[np.ndarray, tuple[int, int], list[np.ndarray]]:
    rectification_hs = [compute_origin_rectification_h(track.corners, square_size) for track in tracks]
    all_corners = np.vstack(
        [transformed_frame_corners(frame_size, rectification_h) for rectification_h in rectification_hs]
    )
    min_xy = all_corners.min(axis=0)
    max_xy = all_corners.max(axis=0)
    translate_x = -float(min_xy[0])
    translate_y = -float(min_xy[1])
    canvas_w = max(1, int(np.ceil(max_xy[0] - min_xy[0])))
    canvas_h = max(1, int(np.ceil(max_xy[1] - min_xy[1])))
    canvas_w += canvas_w % 2
    canvas_h += canvas_h % 2
    translation_h = np.array(
        [
            [1.0, 0.0, translate_x],
            [0.0, 1.0, translate_y],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    return translation_h, (canvas_w, canvas_h), rectification_hs


def draw_overlay(frame: np.ndarray, corners: np.ndarray, status_text: str, weak: bool) -> np.ndarray:
    out = frame.copy()
    color = (0, 165, 255) if weak else (0, 255, 255)
    cv2.polylines(out, [corners.astype(np.int32).reshape(-1, 1, 2)], True, color, 3, cv2.LINE_AA)
    for i, (x, y) in enumerate(corners.astype(int), start=1):
        cv2.circle(out, (x, y), 5, (0, 0, 255), -1)
        cv2.putText(out, str(i), (x + 8, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(out, status_text, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    return out


def track_frames(
    cap: cv2.VideoCapture,
    first_frame: np.ndarray,
    initial_corners: np.ndarray,
    source_fps: float,
    args: argparse.Namespace,
    undistort_maps,
) -> list[FrameTrack]:
    prev_frame = preprocess_frame(first_frame, undistort_maps)
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    prev_pts = detect_features(prev_gray, initial_corners, args)
    base_pts = prev_pts.reshape(-1, 2).copy()
    base_corners = initial_corners.copy()
    current_corners = initial_corners.copy()
    tracks: list[FrameTrack] = []
    frame_index = args.start_frame
    processed = 0

    while True:
        if processed > 0:
            ok = False
            frame = None
            for _ in range(args.step):
                ok, frame = cap.read()
                frame_index += 1
                if not ok:
                    break
            if not ok or frame is None:
                break
        else:
            frame = first_frame

        if args.num_frames and processed >= args.num_frames:
            break

        work_frame = preprocess_frame(frame, undistort_maps)
        gray = cv2.cvtColor(work_frame, cv2.COLOR_BGR2GRAY)
        tracking_h = np.eye(3, dtype=np.float32)
        track_ok = True
        weak = False
        track_count = len(prev_pts)
        inlier_count = track_count

        if processed > 0:
            next_pts, status, _err = cv2.calcOpticalFlowPyrLK(
                prev_gray,
                gray,
                prev_pts,
                None,
                winSize=(31, 31),
                maxLevel=3,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
            )

            if next_pts is None or status is None:
                track_ok = False
                good_base = np.empty((0, 2), dtype=np.float32)
                good_next = np.empty((0, 2), dtype=np.float32)
            else:
                status = status.reshape(-1).astype(bool)
                good_base = base_pts[status]
                good_next = next_pts.reshape(-1, 2)[status]
                track_count = len(good_next)

            if track_ok and track_count >= args.min_tracks:
                homography, inliers = cv2.findHomography(
                    good_base,
                    good_next,
                    cv2.RANSAC,
                    args.ransac_thresh,
                )
                inlier_count = 0 if inliers is None else int(inliers.sum())
                if homography is not None and inlier_count >= args.min_tracks:
                    tracking_h = homography.astype(np.float32)
                    current_corners = cv2.perspectiveTransform(
                        base_corners.reshape(1, 4, 2),
                        tracking_h,
                    ).reshape(4, 2)
                    current_corners = order_points(current_corners)
                    prev_pts = good_next.reshape(-1, 1, 2).astype(np.float32)
                    base_pts = good_base.copy()
                else:
                    track_ok = False
            else:
                inlier_count = 0
                track_ok = False

            if not track_ok:
                weak = True

        confidence = 0.0 if track_count <= 0 else float(inlier_count / track_count)
        try:
            prev_pts = detect_features(gray, current_corners, args)
            base_pts = prev_pts.reshape(-1, 2).copy()
            base_corners = current_corners.copy()
        except RuntimeError:
            if processed == 0:
                raise
            weak = True
            track_ok = False

        tracks.append(
            FrameTrack(
                frame_index=frame_index,
                time_sec=frame_index / source_fps if source_fps else float(processed),
                ok=track_ok,
                weak=weak,
                track_count=track_count,
                inlier_count=inlier_count,
                confidence=confidence,
                corners=current_corners.copy(),
                tracking_h=tracking_h.copy(),
            )
        )

        if processed % 25 == 0:
            print(
                f"Processed {processed} frames | frame_index={frame_index} | "
                f"tracks={track_count} | inliers={inlier_count} | weak={weak}",
                flush=True,
            )

        prev_gray = gray
        processed += 1

    return tracks


def build_output_geometry(
    tracks: list[FrameTrack],
    frame_size: tuple[int, int],
    args: argparse.Namespace,
) -> tuple[tuple[int, int], list[np.ndarray], list[np.ndarray], np.ndarray | None]:
    if args.output_mode == "fixed-context":
        output_size = (args.output_width, args.output_height)
        rectification_hs = [compute_fixed_context_h(track.corners, args) for track in tracks]
        output_hs = rectification_hs
        return output_size, rectification_hs, output_hs, None

    translation_h, output_size, rectification_hs = compute_preserve_full_frame_geometry(
        tracks,
        frame_size,
        args.rectified_square_size,
    )
    output_hs = [translation_h @ rectification_h for rectification_h in rectification_hs]
    return output_size, rectification_hs, output_hs, translation_h


def export_outputs(
    video_path: Path,
    output_dir: Path,
    tracks: list[FrameTrack],
    frame_size: tuple[int, int],
    source_fps: float,
    preview_fps: float,
    rectified_fps: float,
    args: argparse.Namespace,
    undistort_maps,
    initial_raw_px_per_mm: float,
) -> None:
    overlay_path = output_dir / "tracked_chamber_overlay.mp4"
    rectified_path = output_dir / "tracked_full_frame_rectified.mp4"
    corners_csv_path = output_dir / "tracked_chamber_corners.csv"
    run_metadata_path = output_dir / "run_metadata.json"
    metadata_jsonl_path = resolve_optional_output_path(output_dir, args.metadata_jsonl)
    output_size, rectification_hs, output_hs, global_translation_h = build_output_geometry(
        tracks,
        frame_size,
        args,
    )

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not re-open video: {video_path}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, args.start_frame)
    ok, first = cap.read()
    if not ok:
        raise RuntimeError(f"Could not read start frame {args.start_frame}.")

    first_frame = preprocess_frame(first, undistort_maps)
    overlay_writer = make_writer(overlay_path, preview_fps, (first_frame.shape[1], first_frame.shape[0]))
    rectified_writer, rectified_path = make_rectified_writer(rectified_path, rectified_fps, output_size)
    metadata_file = None
    if metadata_jsonl_path is not None:
        metadata_jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_file = metadata_jsonl_path.open("w", encoding="utf-8")

    try:
        with corners_csv_path.open("w", newline="", encoding="utf-8") as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(
                [
                    "frame_index",
                    "time_sec",
                    "ok",
                    "weak",
                    "track_count",
                    "inlier_count",
                    "confidence",
                    "tl_x",
                    "tl_y",
                    "tr_x",
                    "tr_y",
                    "br_x",
                    "br_y",
                    "bl_x",
                    "bl_y",
                    "output_w",
                    "output_h",
                ]
            )

            for i, track in enumerate(tracks):
                if i == 0:
                    frame = first
                else:
                    ok = False
                    frame = None
                    for _ in range(args.step):
                        ok, frame = cap.read()
                        if not ok:
                            break
                    if not ok or frame is None:
                        break

                work_frame = preprocess_frame(frame, undistort_maps)
                output_h = output_hs[i]
                warped = cv2.warpPerspective(work_frame, output_h, output_size)
                rectified_writer.write(warped)

                status_text = (
                    f"frame {track.frame_index} tracks {track.track_count} "
                    f"inliers {track.inlier_count} conf {track.confidence:.2f}"
                )
                overlay_writer.write(draw_overlay(work_frame, track.corners, status_text, track.weak))

                csv_writer.writerow(
                    [
                        track.frame_index,
                        f"{track.time_sec:.6f}",
                        int(track.ok),
                        int(track.weak),
                        track.track_count,
                        track.inlier_count,
                        f"{track.confidence:.4f}",
                        *[f"{v:.3f}" for v in track.corners.reshape(-1)],
                        output_size[0],
                        output_size[1],
                    ]
                )

                if metadata_file is not None:
                    metadata_file.write(
                        json.dumps(
                            {
                                "frame_index": track.frame_index,
                                "time_sec": track.time_sec,
                                "ok": track.ok,
                                "weak": track.weak,
                                "track_count": track.track_count,
                                "inlier_count": track.inlier_count,
                                "confidence": track.confidence,
                                "corners": track.corners.tolist(),
                                "tracking_h": track.tracking_h.tolist(),
                                "rectification_h": rectification_hs[i].tolist(),
                                "output_h": output_h.tolist(),
                            }
                        )
                        + "\n"
                    )
    finally:
        if metadata_file is not None:
            metadata_file.close()

    overlay_writer.release()
    rectified_writer.release()
    cap.release()

    run_metadata = {
        "video": str(video_path),
        "frame_count": len(tracks),
        "start_frame": args.start_frame,
        "step": args.step,
        "preview_fps": preview_fps,
        "rectified_fps": rectified_fps,
        "source_fps": source_fps,
        "calibration_json": str(args.calibration_json) if args.calibration_json else None,
        "output_mode": args.output_mode,
        "output_size": list(output_size),
        "output_width": args.output_width,
        "output_height": args.output_height,
        "rectified_square_size": args.rectified_square_size,
        "global_translation_h": global_translation_h.tolist() if global_translation_h is not None else None,
        "track_region": args.track_region,
        "ring_margin": args.ring_margin,
        "chamber_width_mm": CHAMBER_WIDTH_MM,
        "rectified_px_per_mm": args.rectified_square_size / CHAMBER_WIDTH_MM,
        "initial_raw_px_per_mm": initial_raw_px_per_mm,
        "outputs": {
            "overlay_video": str(overlay_path),
            "rectified_video": str(rectified_path),
            "corners_csv": str(corners_csv_path),
            "metadata_jsonl": str(metadata_jsonl_path) if metadata_jsonl_path else None,
        },
    }
    run_metadata_path.write_text(json.dumps(run_metadata, indent=2), encoding="utf-8")

    print(f"Wrote: {corners_csv_path}")
    if metadata_jsonl_path is not None:
        print(f"Wrote: {metadata_jsonl_path}")
    print(f"Wrote: {overlay_path}")
    print(f"Wrote: {rectified_path}")
    print(f"Wrote: {run_metadata_path}")


def main() -> None:
    args = parse_args()
    output_dir = resolve_output_dir(args)
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {args.video}")

    source_fps = cap.get(cv2.CAP_PROP_FPS) or args.preview_fps
    preview_fps = min(args.preview_fps, source_fps / max(args.step, 1))
    rectified_fps = source_fps / max(args.step, 1)
    cap.set(cv2.CAP_PROP_POS_FRAMES, args.start_frame)
    ok, first = cap.read()
    if not ok:
        raise RuntimeError(f"Could not read start frame {args.start_frame}.")

    undistort_maps = make_undistort_maps((first.shape[1], first.shape[0]), args.calibration_json)
    first_frame = preprocess_frame(first, undistort_maps)

    if args.corners_json:
        initial_corners = load_corners(args.corners_json)
    elif args.interactive:
        initial_corners = pick_corners_interactive(first_frame)
    else:
        initial_corners = auto_detect_rectangle(first_frame)

    initial_raw_px_per_mm = compute_px_per_mm(initial_corners)
    (output_dir / "initial_corners.json").write_text(
        json.dumps(initial_corners.tolist(), indent=2),
        encoding="utf-8",
    )

    tracks = track_frames(
        cap,
        first,
        initial_corners,
        source_fps,
        args,
        undistort_maps,
    )
    cap.release()

    if not tracks:
        raise RuntimeError("No frames were processed.")

    export_outputs(
        args.video,
        output_dir,
        tracks,
        (first_frame.shape[1], first_frame.shape[0]),
        source_fps,
        preview_fps,
        rectified_fps,
        args,
        undistort_maps,
        initial_raw_px_per_mm,
    )


if __name__ == "__main__":
    main()
