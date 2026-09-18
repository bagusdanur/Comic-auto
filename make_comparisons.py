import cv2
import numpy as np
from PIL import Image
from pathlib import Path

art_dir = Path(r"C:\Users\kanim\.gemini\antigravity-ide\brain\d02d2196-ce6c-4d74-9afc-8fd95df40b52")
crop_dir = Path("test_debug/test_page001_out")

# 1. Bubble 003
b03_before = cv2.imread(str(art_dir / ".user_uploaded/media_1788514564966.png"))
b03_after = cv2.imread(str(crop_dir / "bubble-003_final_typeset.png"))
# crop after to match the card
# Resize to match height for side by side
h = 240
b03_before_r = cv2.resize(b03_before, (int(b03_before.shape[1] * (h / b03_before.shape[0])), h))
b03_after_r = cv2.resize(b03_after, (int(b03_after.shape[1] * (h / b03_after.shape[0])), h))

# Put label bars
def make_labeled_card(img, label, color=(0, 0, 180)):
    canvas = np.full((img.shape[0] + 40, img.shape[1], 3), 30, dtype=np.uint8)
    canvas[40:, :] = img
    cv2.putText(canvas, label, (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
    return canvas

c1_before = make_labeled_card(b03_before_r, "BEFORE (Ada sisa raw T dan D)", (0, 0, 200))
c1_after = make_labeled_card(b03_after_r, "AFTER FIX (Bersih & Rapi)", (0, 180, 0))
comp1 = np.hstack([c1_before, c1_after])
cv2.imwrite(str(art_dir / "fix_comparison_1.png"), comp1)

# 2. Bubble 004
b04_before = cv2.imread(str(art_dir / ".user_uploaded/media_1788514570648.png"))
b04_after = cv2.imread(str(crop_dir / "bubble-004_final_typeset.png"))
h = 240
b04_before_r = cv2.resize(b04_before, (int(b04_before.shape[1] * (h / b04_before.shape[0])), h))
b04_after_r = cv2.resize(b04_after, (int(b04_after.shape[1] * (h / b04_after.shape[0])), h))
c2_before = make_labeled_card(b04_before_r, "BEFORE (Ada sisa raw SL dan quote)", (0, 0, 200))
c2_after = make_labeled_card(b04_after_r, "AFTER FIX (Bersih & Rapi)", (0, 180, 0))
comp2 = np.hstack([c2_before, c2_after])
cv2.imwrite(str(art_dir / "fix_comparison_2.png"), comp2)

# 3. Bubble 005
b05_before = cv2.imread(str(art_dir / ".user_uploaded/media_1788514572639.png"))
b05_after = cv2.imread(str(crop_dir / "bubble-005_final_typeset.png"))
h = 240
b05_before_r = cv2.resize(b05_before, (int(b05_before.shape[1] * (h / b05_before.shape[0])), h))
b05_after_r = cv2.resize(b05_after, (int(b05_after.shape[1] * (h / b05_after.shape[0])), h))
c3_before = make_labeled_card(b05_before_r, "BEFORE (Ada sisa raw C dan kurung)", (0, 0, 200))
c3_after = make_labeled_card(b05_after_r, "AFTER FIX (Bersih & Rapi)", (0, 180, 0))
comp3 = np.hstack([c3_before, c3_after])
cv2.imwrite(str(art_dir / "fix_comparison_3.png"), comp3)

print("Comparison images generated successfully!")
