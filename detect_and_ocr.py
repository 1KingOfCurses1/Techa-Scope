 
import os
import cv2
import numpy as np
import pytesseract

# If tesseract isn't on PATH, set it explicitly:
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# --- Files expected next to this script ---
CFG = "yolov4-tiny.cfg"
WEIGHTS = "yolov4-tiny.weights"
NAMES = "coco.names"
IMAGE = "test.jpg"  # change if needed

# --- Sanity checks ---
for f in (CFG, WEIGHTS, NAMES):
    if not os.path.exists(f):
        raise SystemExit(f"[ERROR] Missing {f}. Download model files first.")

img = cv2.imread(IMAGE)
if img is None:
    raise SystemExit(f"[ERROR] Could not read {IMAGE}")

# --- YOLOv4-tiny setup ---
net = cv2.dnn.readNetFromDarknet(CFG, WEIGHTS)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

with open(NAMES, "r", encoding="utf-8") as f:
    CLASSES = [c.strip() for c in f if c.strip()]

try:
    out_layers = net.getUnconnectedOutLayersNames()
except AttributeError:
    ln = net.getLayerNames()
    out_layers = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

CONF_T = 0.5
NMS_T  = 0.4
INPUT_SIZE = (416, 416)

# --- Object detection ---
H, W = img.shape[:2]
blob = cv2.dnn.blobFromImage(img, 1/255.0, INPUT_SIZE, swapRB=True, crop=False)
net.setInput(blob)
outs = net.forward(out_layers)

boxes, confidences, class_ids = [], [], []
for out in outs:
    for det in out:
        scores = det[5:]
        cls = int(np.argmax(scores))
        conf = float(scores[cls]) * float(det[4])
        if conf >= CONF_T:
            cx, cy, w, h = det[0]*W, det[1]*H, det[2]*W, det[3]*H
            x, y = int(cx - w/2), int(cy - h/2)
            boxes.append([x, y, int(w), int(h)])
            confidences.append(conf)
            class_ids.append(cls)

idxs = cv2.dnn.NMSBoxes(boxes, confidences, CONF_T, NMS_T)

# Draw detected objects (green)
for i in (idxs.flatten() if len(idxs) else []):
    x, y, w, h = boxes[i]
    label = f"{CLASSES[class_ids[i]]} {confidences[i]:.2f}"
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 200, 0), 2)
    cv2.putText(img, label, (x, max(12, y-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,200,0), 2)

# --- OCR helpers ---
def preprocess_for_ocr(bgr):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    big  = cv2.resize(gray, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_LINEAR)
    _, th = cv2.threshold(big, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    return th

# --- OCR on whole image ---
pre = preprocess_for_ocr(img)
full_text = pytesseract.image_to_string(pre, config="--oem 3 --psm 6", lang="eng")

# Word-level boxes (red)
data = pytesseract.image_to_data(pre, output_type=pytesseract.Output.DICT, config="--oem 3 --psm 6", lang="eng")
n = len(data["text"])
for i in range(n):
    txt = data["text"][i].strip()
    conf_str = data["conf"][i]
    conf = int(conf_str) if conf_str.isdigit() else -1
    if txt and conf > 50:
        x, y, w, h = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
        cv2.rectangle(img, (x, y), (x+w, y+h), (200, 0, 0), 2)
        cv2.putText(img, txt, (x, max(12, y-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,0,0), 2)

# Print OCR text to terminal
print("\n===== OCR TEXT =====\n")
print(full_text.strip())
print("\n====================\n")

# Save + show
out_path = "output_detect_ocr.jpg"
cv2.imwrite(out_path, img)
print(f"[INFO] Saved annotated image to {out_path}")

cv2.imshow("Objects (green) + OCR words (red)", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
