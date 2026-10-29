import sys, os; sys.path.insert(0, ".")
import cv2, numpy as np
import corner_first as CF
f = "../adjustor_data/square-has_light/data/%s.png" % sys.argv[1]
img = cv2.imdecode(np.fromfile(f, np.uint8), 1); lab = CF.bgr_to_lab_image(img, blur=2.0)
g = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (0, 0), 20); y, x = np.unravel_index(int(np.argmax(g)), g.shape)
ref = np.median(lab[y-8:y+9, x-8:x+9].reshape(-1, 3), 0).astype(np.float32)
d = CF.detect(img, ref, 25, np.array([1., 0.]), np.array([0., -1.]), lab=lab, l_weight=1.0)
print("kept", [(int(a), int(b)) for a, b in d.kept]); print(d.why[-90:])
print("label", open("../adjustor_data/square-has_light/label/%s.txt" % sys.argv[1]).read().strip())
