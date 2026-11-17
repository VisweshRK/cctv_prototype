"""Small ROI calibration helper; deployments should save results in go2rtc config."""
from __future__ import annotations
import argparse
import cv2

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("image")
    args = parser.parse_args()
    image = cv2.imread(args.image)
    if image is None:
        raise SystemExit("could not read image")
    print({"width": image.shape[1], "height": image.shape[0], "message": "Define appliance boxes in your deployment config"})

if __name__ == "__main__":
    main()
