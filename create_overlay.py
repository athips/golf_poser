"""
Simple script to create overlay video from results
"""
import sys
import json
import argparse
from pathlib import Path

import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from overlay_visualization import create_overlay_video


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create overlay video from results.")
    parser.add_argument("video_path", help="Path to input video")
    parser.add_argument("--npz", help="Path to results .npz from main.py")
    parser.add_argument("--keypoints", default="output/keypoints.npy",
                        help="Path to keypoints.npy (default: output/keypoints.npy)")
    parser.add_argument("--confidences", default="output/confidences.npy",
                        help="Path to confidences.npy (default: output/confidences.npy)")
    parser.add_argument("--overlay-json", default="output/pose_overlays.json",
                        help="Path to pose_overlays.json (default: output/pose_overlays.json)")
    parser.add_argument("--output", default="output/overlay_video.mp4",
                        help="Output video path (default: output/overlay_video.mp4)")
    parser.add_argument("--no-skeleton", action="store_true",
                        help="Don't draw skeleton")
    parser.add_argument("--no-phases", action="store_true",
                        help="Don't draw swing phases")
    parser.add_argument("--no-info", action="store_true",
                        help="Don't draw info text")
    parser.add_argument("--right-handed", action="store_true",
                        help="Set RightHanded=True in overlay metadata (default: True)")
    parser.add_argument("--left-handed", action="store_true",
                        help="Set RightHanded=False in overlay metadata")
    return parser


def _ensure_parent(path_str: str) -> None:
    Path(path_str).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)


def _write_overlay_json(path_str: str, view: str, fps: float, right_handed: bool) -> None:
    overlay_dict = {
        "Metadata": {
            "View": view,
            "RightHanded": right_handed,
            "Framerate": float(fps),
        },
        "visualaid": {
            "SwingPhases": {}
        }
    }
    _ensure_parent(path_str)
    with open(path_str, "w", encoding="utf-8") as f:
        json.dump(overlay_dict, f, indent=2)


if __name__ == "__main__":
    args = _build_arg_parser().parse_args()

    keypoints_path = args.keypoints
    confidences_path = args.confidences
    overlay_json_path = args.overlay_json

    if args.npz:
        npz = np.load(args.npz, allow_pickle=True)
        if "keypoints" in npz and "confidences" in npz:
            _ensure_parent(keypoints_path)
            _ensure_parent(confidences_path)
            np.save(keypoints_path, npz["keypoints"])
            np.save(confidences_path, npz["confidences"])
        else:
            raise ValueError("NPZ must contain 'keypoints' and 'confidences'")

        # Create overlay JSON if it doesn't exist
        overlay_path = Path(overlay_json_path)
        if not overlay_path.is_file():
            view = str(npz.get("view", "DL"))
            fps = float(npz.get("fps", 30.0))
            if args.left_handed:
                right_handed = False
            elif args.right_handed:
                right_handed = True
            else:
                right_handed = True
            _write_overlay_json(overlay_json_path, view, fps, right_handed)

    print("Creating overlay video...")
    print(f"  Input video: {args.video_path}")
    print(f"  Keypoints: {keypoints_path}")
    print(f"  Confidences: {confidences_path}")
    print(f"  Overlay JSON: {overlay_json_path}")
    print(f"  Output: {args.output}")

    create_overlay_video(
        video_path=args.video_path,
        keypoints_path=keypoints_path,
        confidences_path=confidences_path,
        overlay_json_path=overlay_json_path,
        output_path=args.output,
        draw_skeleton_overlay=not args.no_skeleton,
        draw_phases=not args.no_phases,
        draw_info=not args.no_info
    )

    print(f"\nDone! Overlay video saved to: {args.output}")

