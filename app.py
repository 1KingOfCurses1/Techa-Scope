import os, base64
from io import BytesIO

from flask import Flask, request, jsonify, render_template
import numpy as np
import cv2
import pytesseract
import re

# ---- Model files expected next to app.py ----
CFG = "yolov4-tiny.cfg"
WEIGHTS = "yolov4-tiny.weights"
NAMES = "coco.names"

CONF_T = 0.5
NMS_T  = 0.4
INPUT_SIZE = (416, 416)

# If Tesseract isn't on PATH, set it explicitly:
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# ---- Sanity checks ----
for f in (CFG, WEIGHTS, NAMES):
    if not os.path.exists(f):
        raise SystemExit(f"[ERROR] Missing {f}. Put it next to app.py")

# ---- Load YOLO once (fast) ----
net = cv2.dnn.readNetFromDarknet(CFG, WEIGHTS)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

with open(NAMES, "r", encoding="utf-8") as f:
    CLASSES = [c.strip() for c in f if c.strip()]

try:
    OUT_LAYERS = net.getUnconnectedOutLayersNames()
except AttributeError:
    ln = net.getLayerNames()
    OUT_LAYERS = [ln[i[0]-1] for i in net.getUnconnectedOutLayers()]

# ---- Helpers ----
def detect_objects(bgr, conf_thresh=CONF_T):
    H, W = bgr.shape[:2]
    blob = cv2.dnn.blobFromImage(bgr, 1/255.0, INPUT_SIZE, swapRB=True, crop=False)
    net.setInput(blob)
    outs = net.forward(OUT_LAYERS)

    boxes, confidences, class_ids = [], [], []
    for out in outs:
        for det in out:
            scores = det[5:]
            cls = int(np.argmax(scores))
            conf = float(scores[cls]) * float(det[4])  # class score * objectness
            if conf >= conf_thresh:
                cx, cy, w, h = det[0]*W, det[1]*H, det[2]*W, det[3]*H
                x, y = int(cx - w/2), int(cy - h/2)
                boxes.append([x, y, int(w), int(h)])
                confidences.append(conf)
                class_ids.append(cls)

    idxs = cv2.dnn.NMSBoxes(boxes, confidences, conf_thresh, NMS_T)
    dets = []
    if len(idxs):
        for i in idxs.flatten():
            x, y, w, h = boxes[i]
            dets.append({
                "label": CLASSES[class_ids[i]],
                "conf": float(confidences[i]),
                "box": [int(x), int(y), int(w), int(h)]
            })
    return dets

def preprocess_for_ocr(bgr):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    big  = cv2.resize(gray, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_LINEAR)
    _, th = cv2.threshold(big, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    return th

def run_ocr(bgr):
    pre = preprocess_for_ocr(bgr)
    text = pytesseract.image_to_string(pre, config="--oem 3 --psm 6", lang="eng")
    data = pytesseract.image_to_data(pre, output_type=pytesseract.Output.DICT,
                                     config="--oem 3 --psm 6", lang="eng")
    words = []
    for i in range(len(data["text"])):
        t = data["text"][i].strip()
        conf_s = str(data["conf"][i])
        conf = int(conf_s) if conf_s.isdigit() else -1
        if t and conf > 50:
            words.append({"text": t,
                          "box": [int(data["left"][i]), int(data["top"][i]),
                                  int(data["width"][i]), int(data["height"][i])],
                          "conf": conf})
    return text, words

def annotate(bgr, dets, words):
    img = bgr.copy()
    # objects (green)
    for d in dets:
        x, y, w, h = d["box"]
        cv2.rectangle(img, (x, y), (x+w, y+h), (0,200,0), 2)
        cv2.putText(img, f'{d["label"]} {d["conf"]:.2f}', (x, max(10, y-6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,200,0), 2)
    # words (red)
    for w in words:
        x, y, ww, hh = w["box"]
        cv2.rectangle(img, (x, y), (x+ww, y+hh), (200,0,0), 2)
        cv2.putText(img, w["text"], (x, max(10, y-6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,0,0), 2)
    return img

def bgr_to_base64_jpeg(bgr, quality=90):
    ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise RuntimeError("JPEG encode failed")
    return base64.b64encode(buf).decode("ascii")

def check_real_word(val_list):
    cleaned = [re.sub(r"[^a-zA-Z0-9]+", "", w) for w in val_list]
    return [w for w in cleaned if w and check_word(w.lower())]

def check_word(word):
    return word in word_set

def check_empty(lst):
    if not lst:
        lst.append("")

word_set = set()
if os.path.exists("clean_oxford_10000_words.txt"):
    with open("clean_oxford_10000_words.txt", "r", encoding="utf-8") as f:
        for line in f:
            w = line.strip()
            if w:
                word_set.add(w)

# ---- Flask app ----
app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload():
    conf = float(request.form.get("conf", CONF_T))

    if "file" not in request.files:
        return jsonify({"message": "No 'file' in form-data"}), 400

    file = request.files["file"]
    data = np.frombuffer(file.read(), np.uint8)
    bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if bgr is None:
        return jsonify({"message": "Could not decode image"}), 400

    # Run detection
    dets = detect_objects(bgr, conf_thresh=conf)

    # Pick one "primary" detected object
    detected_object = ""
    if dets:
        top = max(dets, key=lambda d: d["conf"])
        detected_object = top["label"]

    # OCR + annotate
    ocr_text, ocr_words = run_ocr(bgr)
    annotated = annotate(bgr, dets, ocr_words)

    # Filter OCR words using Oxford list
    val = (ocr_text or "").split()
    check1 = check_real_word(val)
    check_empty(check1)
    final_words = set(check1)
    item_name = " ".join(final_words)

    # Speech text, same style as Image_Recognition.py
    speech_text = f"Detected Object: {detected_object}\nItem name: {item_name}"

    return jsonify({
        "message": "ok",
        "objects": dets,
        "detected_object": detected_object,
        "item_name": item_name,
        "ocr_text": (ocr_text or "").strip(),
        "speech_text": speech_text.strip(),
        "annotated_image": bgr_to_base64_jpeg(annotated)
    })

if __name__ == "__main__":
    app.run(debug=True)
 
