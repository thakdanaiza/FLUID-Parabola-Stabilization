"""
Track/rectify chamber video with selectable camera tracking modes.

This version keeps chamber corners for rectification, but avoids assuming that
features inside the chamber are valid camera-tracking features after fluid
enters. Default mode is fixed: use the initial chamber corners for every frame.
Use external-mask mode with a user-supplied rigid-region polygon mask when
camera motion must be estimated.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

import track_chamber_rect_fixed_context as base


cv2.setUseOptimized(True)
cv2.setNumThreads(max(1, (os.cpu_count() or 2) - 1))


@dataclass
class FrameTrack:
    frame_index: int
    time_sec: float
    ok: bool
    weak: bool
    track_count: int
    inlier_count: int
    track_consistency: float
    geometry_valid: bool
    corners: np.ndarray
    tracking_h: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Track chamber corners with fixed, LK, or external-mask tracking."
    )
    parser.add_argument("video", type=Path, help="Input video path.")
    parser.add_argument(
        "--outputs-root",
        type=Path,
        default=base.DEFAULT_OUTPUTS_ROOT,
        help="Base output folder. Results are written to <outputs-root>/<video_stem>_<timestamp>.",
    )
    parser.add_argument("--out-dir", type=Path, help="Full output folder override.")
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--num-frames", type=int, default=0, help="0 means process until EOF.")
    parser.add_argument("--step", type=int, default=1)
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument("--corners-json", type=Path)
    parser.add_argument("--calibration-json", type=Path)
    parser.add_argument("--metadata-jsonl", type=Path)
    parser.add_argument(
        "--tracking-mode",
        choices=("fixed", "lk", "external-mask"),
        default="fixed",
        help="fixed keeps initial chamber corners; external-mask tracks only user-selected rigid regions.",
    )
    parser.add_argument(
        "--tracking-mask-json",
        type=Path,
        help="JSON polygon or polygons defining rigid tracking regions on the reference frame.",
    )
    parser.add_argument(
        "--pick-tracking-mask",
        action="store_true",
        help="Interactively pick rigid tracking polygons on the reference frame.",
    )
    parser.add_argument(
        "--tracking-mask-output",
        type=Path,
        default=Path("rigid_tracking_mask.json"),
        help="Output JSON path for --pick-tracking-mask. Relative paths are written inside the run folder.",
    )
    parser.add_argument(
        "--output-mode",
        choices=("fixed-context", "preserve-full-frame"),
        default="fixed-context",
    )
    parser.add_argument("--output-width", type=int, default=1024)
    parser.add_argument("--output-height", type=int, default=1024)
    parser.add_argument("--rectified-square-size", type=int, default=512)
    parser.add_argument("--max-corners", type=int, default=600)
    parser.add_argument("--quality", type=float, default=0.01)
    parser.add_argument("--min-distance", type=int, default=8)
    parser.add_argument("--ransac-thresh", type=float, default=3.0)
    parser.add_argument("--min-tracks", type=int, default=40)
    parser.add_argument("--preview-fps", type=float, default=20.0)
    parser.add_argument(
        "--ring-margin",
        type=float,
        default=0.16,
        help="Only used by --tracking-mode lk. Inner chamber area is excluded.",
    )
    parser.add_argument("--geometry-area-tol", type=float, default=0.65)
    parser.add_argument("--geometry-aspect-tol", type=float, default=0.45)
    return parser.parse_args()


def load_tracking_polygons(path: Path) -> list[np.ndarray]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        data = data.get("polygons", data.get("polygon"))
    if data is None:
        raise ValueError("tracking mask JSON must contain a polygon or polygons.")

    if not isinstance(data, list) or not data:
        raise ValueError("tracking mask JSON must contain a non-empty polygon list.")

    first = data[0]
    if isinstance(first, list) and len(first) == 2 and all(isinstance(v, (int, float)) for v in first):
        polygon = np.asarray(data, dtype=np.float32)
        if polygon.ndim == 2 and polygon.shape[1] == 2:
            return [polygon]

    polygons = []
    for polygon_data in data:
        polygon = np.asarray(polygon_data, dtype=np.float32)
        if polygon.ndim != 2 or polygon.shape[1] != 2 or len(polygon) < 3:
            raise ValueError("Each tracking polygon must contain at least three [x, y] points.")
        polygons.append(polygon)
    if polygons:
        return polygons
    raise ValueError("tracking mask JSON must be [[x,y], ...] or [[[x,y], ...], ...].")


def polygons_mask(shape: tuple[int, int], polygons: list[np.ndarray]) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    for polygon in polygons:
        if len(polygon) >= 3:
            cv2.fillPoly(mask, [polygon.astype(np.int32)], 255)
    return mask


def ring_mask_without_chamber_interior(
    shape: tuple[int, int],
    corners: np.ndarray,
    ring_margin: float,
) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    cv2.fillConvexPoly(mask, corners.astype(np.int32), 255)
    center = corners.mean(axis=0)
    margin = float(np.clip(ring_margin, 0.0, 0.45))
    inner = center + (corners - center) * (1.0 - margin)
    cv2.fillConvexPoly(mask, inner.astype(np.int32), 0)
    return mask


def detect_features(gray: np.ndarray, mask: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    points = cv2.goodFeaturesToTrack(
        gray,
        maxCorners=args.max_corners,
        qualityLevel=args.quality,
        minDistance=args.min_distance,
        mask=mask,
        blockSize=7,
    )
    if points is None or len(points) < 4:
        raise RuntimeError("Not enough trackable features in the selected rigid tracking mask.")
    return points.astype(np.float32)


def count_features(gray: np.ndarray, mask: np.ndarray, args: argparse.Namespace) -> int:
    points = cv2.goodFeaturesToTrack(
        gray,
        maxCorners=args.max_corners,
        qualityLevel=args.quality,
        minDistance=args.min_distance,
        mask=mask,
        blockSize=7,
    )
    return 0 if points is None else int(len(points))


def draw_tracking_polygons(frame: np.ndarray, polygons: list[np.ndarray], current: list[tuple[int, int]]) -> np.ndarray:
    out = frame.copy()
    fill = out.copy()
    for idx, polygon in enumerate(polygons, start=1):
        pts = polygon.astype(np.int32).reshape(-1, 1, 2)
        cv2.fillPoly(fill, [pts], (0, 80, 80))
    out = cv2.addWeighted(fill, 0.35, out, 0.65, 0)

    for idx, polygon in enumerate(polygons, start=1):
        pts = polygon.astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(out, [pts], True, (0, 255, 255), 2, cv2.LINE_AA)
        x, y = polygon[0].astype(int)
        cv2.putText(out, str(idx), (x + 6, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    if current:
        pts = np.asarray(current, dtype=np.int32)
        for x, y in pts:
            cv2.circle(out, (int(x), int(y)), 4, (0, 0, 255), -1)
        if len(pts) >= 2:
            cv2.polylines(out, [pts.reshape(-1, 1, 2)], False, (0, 0, 255), 2, cv2.LINE_AA)

    help_lines = [
        "Rigid tracking mask only: color patch, chip frame, label, fixture. Avoid chamber/fluid.",
        "Left click: point | Enter: finish polygon | N: new polygon | U/Backspace: undo",
        "C: clear all | S: save | Esc: cancel",
    ]
    y = 28
    for line in help_lines:
        cv2.putText(out, line, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2, cv2.LINE_AA)
        y += 28
    return out


def finish_current_polygon(polygons: list[np.ndarray], current: list[tuple[int, int]]) -> None:
    if len(current) >= 3:
        polygons.append(np.asarray(current, dtype=np.float32))
        current.clear()


def pick_tracking_mask_interactive(
    frame: np.ndarray,
    output_json: Path,
    preview_path: Path,
    args: argparse.Namespace,
) -> Path:
    polygons: list[np.ndarray] = []
    current: list[tuple[int, int]] = []
    window_name = "Pick rigid tracking mask polygons"

    def on_mouse(event, x, y, _flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN:
            current.append((x, y))
            cv2.imshow(window_name, draw_tracking_polygons(frame, polygons, current))

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, on_mouse)
    cv2.imshow(window_name, draw_tracking_polygons(frame, polygons, current))

    while True:
        key = cv2.waitKey(20) & 0xFF
        if key in (13, 10):
            finish_current_polygon(polygons, current)
        elif key in (ord("n"), ord("N")):
            finish_current_polygon(polygons, current)
        elif key in (ord("u"), ord("U"), 8):
            if current:
                current.pop()
            elif polygons:
                polygons.pop()
        elif key in (ord("c"), ord("C")):
            polygons.clear()
            current.clear()
        elif key in (ord("s"), ord("S")):
            finish_current_polygon(polygons, current)
            if not polygons:
                print("Select at least one polygon with three or more points before saving.", flush=True)
            else:
                break
        elif key == 27:
            cv2.destroyWindow(window_name)
            raise RuntimeError("Tracking mask selection canceled.")

        cv2.imshow(window_name, draw_tracking_polygons(frame, polygons, current))

    cv2.destroyWindow(window_name)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    preview_path.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps({"polygons": [polygon.tolist() for polygon in polygons]}, indent=2),
        encoding="utf-8",
    )
    preview = draw_tracking_polygons(frame, polygons, [])
    cv2.imwrite(str(preview_path), preview)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mask = polygons_mask(gray.shape, polygons)
    feature_count = count_features(gray, mask, args)
    print(f"Saved tracking mask: {output_json}", flush=True)
    print(f"Saved tracking mask preview: {preview_path}", flush=True)
    print(f"goodFeaturesToTrack inside tracking mask: {feature_count}", flush=True)
    return output_json


def polygon_area(corners: np.ndarray) -> float:
    return float(abs(cv2.contourArea(corners.astype(np.float32))))


def aspect_ratio(corners: np.ndarray) -> float:
    width = 0.5 * (np.linalg.norm(corners[1] - corners[0]) + np.linalg.norm(corners[2] - corners[3]))
    height = 0.5 * (np.linalg.norm(corners[3] - corners[0]) + np.linalg.norm(corners[2] - corners[1]))
    return float(width / max(height, 1e-6))


def geometry_is_valid(
    corners: np.ndarray,
    initial_area: float,
    initial_aspect: float,
    args: argparse.Namespace,
) -> bool:
    area = polygon_area(corners)
    aspect = aspect_ratio(corners)
    if not np.isfinite(area) or not np.isfinite(aspect) or area <= 1:
        return False
    area_delta = abs(area / max(initial_area, 1e-6) - 1.0)
    aspect_delta = abs(aspect / max(initial_aspect, 1e-6) - 1.0)
    return area_delta <= args.geometry_area_tol and aspect_delta <= args.geometry_aspect_tol


def initial_tracking_mask(
    gray_shape: tuple[int, int],
    initial_corners: np.ndarray,
    args: argparse.Namespace,
) -> np.ndarray | None:
    if args.tracking_mode == "fixed":
        return None
    if args.tracking_mode == "external-mask":
        if args.tracking_mask_json is None:
            raise ValueError("--tracking-mode external-mask requires --tracking-mask-json.")
        return polygons_mask(gray_shape, load_tracking_polygons(args.tracking_mask_json))
    if args.tracking_mask_json is not None:
        return polygons_mask(gray_shape, load_tracking_polygons(args.tracking_mask_json))
    return ring_mask_without_chamber_interior(gray_shape, initial_corners, args.ring_margin)


def track_frames(
    cap: cv2.VideoCapture,
    first_frame: np.ndarray,
    initial_corners: np.ndarray,
    source_fps: float,
    args: argparse.Namespace,
    undistort_maps,
) -> list[FrameTrack]:
    prev_frame = base.preprocess_frame(first_frame, undistort_maps)
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    tracking_mask = initial_tracking_mask(prev_gray.shape, initial_corners, args)
    prev_pts = None if tracking_mask is None else detect_features(prev_gray, tracking_mask, args)
    base_pts = None if prev_pts is None else prev_pts.reshape(-1, 2).copy()
    base_corners = initial_corners.copy()
    current_corners = initial_corners.copy()
    initial_area = polygon_area(initial_corners)
    initial_aspect = aspect_ratio(initial_corners)
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

        work_frame = base.preprocess_frame(frame, undistort_maps)
        gray = cv2.cvtColor(work_frame, cv2.COLOR_BGR2GRAY)
        tracking_h = np.eye(3, dtype=np.float32)
        track_ok = True
        weak = False
        track_count = 0
        inlier_count = 0
        if processed == 0 and args.tracking_mode != "fixed" and prev_pts is not None:
            track_count = len(prev_pts)
            inlier_count = track_count

        if args.tracking_mode == "fixed":
            current_corners = initial_corners.copy()
        elif processed > 0 and prev_pts is not None and base_pts is not None:
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
                    current_corners = base.order_points(current_corners)
                    prev_pts = good_next.reshape(-1, 1, 2).astype(np.float32)
                    base_pts = good_base.copy()
                else:
                    track_ok = False
            else:
                track_ok = False
        elif args.tracking_mode != "fixed":
            track_count = 0 if prev_pts is None else len(prev_pts)
            inlier_count = track_count

        if args.tracking_mode != "fixed" and tracking_mask is not None:
            try:
                prev_pts = detect_features(gray, tracking_mask, args)
                base_pts = prev_pts.reshape(-1, 2).copy()
                base_corners = current_corners.copy()
            except RuntimeError:
                track_ok = False

        track_consistency = 0.0 if track_count <= 0 else float(inlier_count / track_count)
        geometry_valid = geometry_is_valid(current_corners, initial_area, initial_aspect, args)
        weak = (not track_ok) or (not geometry_valid)

        tracks.append(
            FrameTrack(
                frame_index=frame_index,
                time_sec=frame_index / source_fps if source_fps else float(processed),
                ok=track_ok and geometry_valid,
                weak=weak,
                track_count=track_count,
                inlier_count=inlier_count,
                track_consistency=track_consistency,
                geometry_valid=geometry_valid,
                corners=current_corners.copy(),
                tracking_h=tracking_h.copy(),
            )
        )

        if processed % 25 == 0:
            print(
                f"Processed {processed} frames | frame_index={frame_index} | "
                f"tracks={track_count} | inliers={inlier_count} | "
                f"track_consistency={track_consistency:.3f} | geometry_valid={geometry_valid} | weak={weak}",
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
        rectification_hs = [base.compute_fixed_context_h(track.corners, args) for track in tracks]
        return output_size, rectification_hs, rectification_hs, None

    translation_h, output_size, rectification_hs = base.compute_preserve_full_frame_geometry(
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
    metadata_jsonl_path = base.resolve_optional_output_path(output_dir, args.metadata_jsonl)
    output_size, rectification_hs, output_hs, global_translation_h = build_output_geometry(
        tracks,
        frame_size,
        args,
    )
    render_videos = not getattr(args, "keyframes_only", False)
    cap = None
    first = None
    overlay_writer = None
    rectified_writer = None
    if render_videos:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise FileNotFoundError(f"Could not re-open video: {video_path}")
        cap.set(cv2.CAP_PROP_POS_FRAMES, args.start_frame)
        ok, first = cap.read()
        if not ok:
            raise RuntimeError(f"Could not read start frame {args.start_frame}.")

        first_frame = base.preprocess_frame(first, undistort_maps)
        overlay_writer = base.make_writer(
            overlay_path,
            preview_fps,
            (first_frame.shape[1], first_frame.shape[0]),
        )
        rectified_writer, rectified_path = base.make_rectified_writer(
            rectified_path,
            rectified_fps,
            output_size,
        )
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
                    "track_consistency",
                    "geometry_valid",
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
                output_h = output_hs[i]
                if render_videos:
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

                    work_frame = base.preprocess_frame(frame, undistort_maps)
                    warped = cv2.warpPerspective(work_frame, output_h, output_size)
                    rectified_writer.write(warped)

                    status_text = (
                        f"frame {track.frame_index} tracks {track.track_count} "
                        f"inliers {track.inlier_count} tc {track.track_consistency:.2f} "
                        f"geom {int(track.geometry_valid)}"
                    )
                    overlay_writer.write(
                        base.draw_overlay(work_frame, track.corners, status_text, track.weak)
                    )

                row = [
                    track.frame_index,
                    f"{track.time_sec:.6f}",
                    int(track.ok),
                    int(track.weak),
                    track.track_count,
                    track.inlier_count,
                    f"{track.track_consistency:.4f}",
                    int(track.geometry_valid),
                    *[f"{v:.3f}" for v in track.corners.reshape(-1)],
                    output_size[0],
                    output_size[1],
                ]
                csv_writer.writerow(row)

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
                                "track_consistency": track.track_consistency,
                                "geometry_valid": track.geometry_valid,
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

    if overlay_writer is not None:
        overlay_writer.release()
    if rectified_writer is not None:
        rectified_writer.release()
    if cap is not None:
        cap.release()

    run_metadata = {
        "video": str(video_path),
        "frame_count": len(tracks),
        "start_frame": args.start_frame,
        "step": args.step,
        "preview_fps": preview_fps,
        "rectified_fps": rectified_fps,
        "source_fps": source_fps,
        "tracking_mode": args.tracking_mode,
        "tracking_mask_json": str(args.tracking_mask_json) if args.tracking_mask_json else None,
        "calibration_json": str(args.calibration_json) if args.calibration_json else None,
        "output_mode": args.output_mode,
        "output_size": list(output_size),
        "output_width": args.output_width,
        "output_height": args.output_height,
        "rectified_square_size": args.rectified_square_size,
        "global_translation_h": global_translation_h.tolist() if global_translation_h is not None else None,
        "chamber_width_mm": base.CHAMBER_WIDTH_MM,
        "rectified_px_per_mm": args.rectified_square_size / base.CHAMBER_WIDTH_MM,
        "initial_raw_px_per_mm": initial_raw_px_per_mm,
        "opencv_threads": cv2.getNumThreads(),
        "opencv_optimized": bool(cv2.useOptimized()),
        "keyframes_only": not render_videos,
        "outputs": {
            "overlay_video": str(overlay_path) if render_videos else None,
            "rectified_video": str(rectified_path) if render_videos else None,
            "corners_csv": str(corners_csv_path),
            "metadata_jsonl": str(metadata_jsonl_path) if metadata_jsonl_path else None,
        },
    }
    run_metadata_path.write_text(json.dumps(run_metadata, indent=2), encoding="utf-8")

    print(f"Wrote: {corners_csv_path}")
    if metadata_jsonl_path is not None:
        print(f"Wrote: {metadata_jsonl_path}")
    if render_videos:
        print(f"Wrote: {overlay_path}")
        print(f"Wrote: {rectified_path}")
    else:
        print("Skipped overlay and rectified videos (--keyframes-only).")
    print(f"Wrote: {run_metadata_path}")


def main() -> None:
    args = parse_args()
    output_dir = base.resolve_output_dir(args)
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

    undistort_maps = base.make_undistort_maps((first.shape[1], first.shape[0]), args.calibration_json)
    first_frame = base.preprocess_frame(first, undistort_maps)

    if args.corners_json:
        initial_corners = base.load_corners(args.corners_json)
    elif args.interactive:
        initial_corners = base.pick_corners_interactive(first_frame)
    else:
        initial_corners = base.auto_detect_rectangle(first_frame)

    if args.pick_tracking_mask:
        mask_json = base.resolve_optional_output_path(output_dir, args.tracking_mask_output)
        if mask_json is None:
            raise RuntimeError("Could not resolve tracking mask output path.")
        preview_path = mask_json.with_name(f"{mask_json.stem}_preview.jpg")
        args.tracking_mask_json = pick_tracking_mask_interactive(
            first_frame,
            mask_json,
            preview_path,
            args,
        )
        args.tracking_mode = "external-mask"

    initial_raw_px_per_mm = base.compute_px_per_mm(initial_corners)
    (output_dir / "initial_corners.json").write_text(
        json.dumps(initial_corners.tolist(), indent=2),
        encoding="utf-8",
    )

    tracks = track_frames(cap, first, initial_corners, source_fps, args, undistort_maps)
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
