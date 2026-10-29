"""Run the corner pipeline on every frame of adjustor_data/<dir>/data and report which frames get a corner.

No labels are used.  The fabric colour is taken automatically from the brightest large blob (white
fabric); in the app you pick it with a click.  DexiNed outputs are cached in the temp folder.

Usage:
    python cuonglm/curve_matcher/check_dataset.py [square-has_light square-no_light]
Writes contact sheets (corner + fitted edges drawn on each frame) next to this file: _check_<dir>_<n>.png
"""
import glob
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..")))
from cuonglm.curve_matcher import edge_detect as E  # noqa: E402
from cuonglm.curve_matcher.config import load_machine  # noqa: E402
from cuonglm.curve_matcher.corner import find_bottom_left_corner  # noqa: E402

ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
CACHE = os.path.join(os.environ.get("TEMP", "/tmp"), "cm_probs")
os.makedirs(CACHE, exist_ok=True)
THRESH, OPEN_PX, TOL = 0.25, 21, 25          # the app's defaults


def edge_prob(key, img):
    f = os.path.join(CACHE, key + ".npy")
    if os.path.exists(f):
        return np.load(f).astype(np.float32)
    p = E.infer_edge_map(img)
    np.save(f, p.astype(np.float16))
    return p


def main(dirs):
    for ds in dirs:
        files = sorted(glob.glob(os.path.join(ROOT, "adjustor_data", ds, "data", "*.png")),
                       key=lambda f: int(os.path.splitext(os.path.basename(f))[0]))
        found, none, tiles = [], [], []
        for f in files:
            idx = int(os.path.splitext(os.path.basename(f))[0])
            img = cv2.imdecode(np.fromfile(f, np.uint8), cv2.IMREAD_COLOR)
            h, w = img.shape[:2]
            g = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (0, 0), 20)
            y, x = np.unravel_index(int(np.argmax(g)), g.shape)
            lab = E.bgr_to_lab_image(img)
            cm = E.color_match_mask(lab, E.bgr_to_lab(E.sample_color(img, x, y, radius=8)), TOL)
            c, _, _ = E.extract_fabric(edge_prob(f"{ds}_{idx}", img), thresh=THRESH, open_px=OPEN_PX,
                                       color_match=cm, lab_img=lab)
            m = load_machine((w, h))
            corner, why = (None, "no fabric region") if c is None else \
                find_bottom_left_corner(c, (w, h), m.u1, m.u2, "auto")
            v = img.copy()
            if c is not None:
                cv2.polylines(v, [c.astype(np.int32)], True, (0, 255, 255), 1)
            if corner is not None:
                p = corner.point
                cv2.line(v, tuple(p.astype(int)), tuple((p + corner.h_dir * corner.h_len).astype(int)), (0, 255, 0), 2)
                cv2.line(v, tuple(p.astype(int)), tuple((p + corner.v_dir * corner.v_len).astype(int)), (255, 128, 0), 2)
                cv2.circle(v, tuple(p.round().astype(int)), 6, (0, 0, 255), -1)
                found.append((idx, corner.theta))
            else:
                none.append((idx, why))
            cv2.putText(v, f"{idx} " + (f"theta {corner.theta:+.1f}" if corner else "NONE"), (4, 18), 0, 0.6, (0, 0, 255), 2)
            tiles.append(cv2.resize(v, (400, 300)))
        print(f"{ds}: {len(found)}/{len(files)} frames have a bottom-left corner")
        print("  found :", ", ".join(f"{i}({t:+.0f})" for i, t in found))
        print("  none  :", ", ".join(str(i) for i, _ in none))
        for s in range(0, len(tiles), 12):
            grp = tiles[s:s + 12]
            while len(grp) % 4:
                grp.append(np.zeros_like(tiles[0]))
            cv2.imwrite(os.path.join(HERE, f"_check_{ds}_{s // 12}.png"),
                        np.vstack([np.hstack(grp[i:i + 4]) for i in range(0, len(grp), 4)]))


if __name__ == "__main__":
    main(sys.argv[1:] or ["square-has_light", "square-no_light"])
