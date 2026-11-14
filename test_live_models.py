"""Live Verification Suite for Presence B2C AI CCTV Platform.
Runs automated or interactive live webcam verification for the 3 core safety use cases:
1. Fire & Smoke Detection
2. Electricity When Nobody Is There (Power When Absent)
3. Baby Crying Acoustic Classifier
"""

from test_live_cam import run_live_cam_test, run_interactive_cam
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Live Core Safety Models Verification")
    parser.add_argument("--camera", type=int, default=0, help="Webcam device index (default 0)")
    parser.add_argument("--interactive", action="store_true", help="Launch live interactive preview window")
    parser.add_argument("--output", default="live_test_output.jpg", help="Output annotated frame path")
    args = parser.parse_args()

    if args.interactive:
        run_interactive_cam(args.camera)
    else:
        run_live_cam_test(args.camera, args.output)
