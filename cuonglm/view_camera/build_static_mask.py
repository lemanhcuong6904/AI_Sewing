"""Learn the fixed fabric-coloured layer of the machine bed from a folder of frames.

The camera is fixed, so a beige layer that sits in the same place in most frames is not the fabric
being moved.  The result is saved as static_mask.png next to this file and used by app.py and
view_camera.py.

Run:  python cuonglm/view_camera/build_static_mask.py <folder of frames>
The fabric is found automatically (warmer than the felt); only if a colour was picked in app.py
(settings.json) is that colour used instead.
"""
import glob
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from segment import STATIC_PATH, build_static_mask, save_static  # noqa: E402


def main():
    if len(sys.argv) < 2 or not os.path.isdir(sys.argv[1]):
        sys.exit(__doc__)
    settings = os.path.join(os.path.dirname(os.path.abspath(__file__)), "settings.json")
    try:
        with open(settings, encoding="utf-8") as f:
            fab = json.load(f).get("fabric_bgr")
    except (OSError, ValueError):
        fab = None
    files = []
    for ext in ("png", "jpg", "jpeg", "bmp"):
        files += glob.glob(os.path.join(sys.argv[1], f"*.{ext}"))
    frames = [cv2.imdecode(np.fromfile(f, np.uint8), cv2.IMREAD_COLOR) for f in sorted(files)]
    frames = [f for f in frames if f is not None]
    if len(frames) < 5:
        sys.exit(f"Only {len(frames)} frame(s): need at least 5 with the fabric in different places.")
    static = build_static_mask(frames, None if not fab else np.array(fab, np.float32))
    save_static(static)
    print(f"{len(frames)} frames -> {STATIC_PATH} ({int(static.sum())} px = {static.mean():.1%} of the image)")


if __name__ == "__main__":
    main()
