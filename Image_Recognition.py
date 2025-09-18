# Image_Recognition.py
# Shan Truong
# September 18, 2025
# Backend code for image recognition using OpenCV for Kimberly.
# Detect objects (YOLO) + OCR + TTS

import cv2
import numpy as np
import pytesseract
import re
import pyttsx3
import os
import sys

# ----------------- helpers for your dictionary checks -----------------
def check_real_word(val):
    # strip non-alnum
    val = [re.sub(r"[^a-zA-Z0-9]+", "", w) for w in val]
    # keep only real words from your set
    return [w for w in val if check_word(w.lower())]

def check_word(word):
    return word in word_set

def check_empty(lst):
    if not lst:
        lst.append("")

# ----------------- word list (hash set) -----------------
word_set = set()
with open("clean_oxford_10000_words.txt", "r", encoding="utf-8") as file:
    for line in file:
        word_set.add(line.strip())

try:
    # ----------------- load image -----------------
    image_path = "bottle.jpg"  # <-- was "images.png"; use a file that exists
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Can not find image file: {image_path}")

    # normalize working size
    HEIGHT, WIDTH = 600, 1200
    image_resize = cv2.resize(image, (WIDTH, HEIGHT))
    gray_image = cv2.cvtColor(image_resize, cv2.COLOR_BGR2GRAY)

    # ----------------- YOLOv4-tiny model (cfg FIRST, weights SECOND) -----------------
    CFG = "yolov4-tiny.cfg"
    WEIGHTS = "yolov4-tiny.weights"
    NAMES = "coco.names"

    for f in (CFG, WEIGHTS, NAMES):
        if not os.path.exists(f):
            raise FileNotFoundError(f"Missing required file: {f}")

    net = cv2.dnn.readNetFromDarknet(CFG, WEIGHTS)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    with open(NAMES, "r", encoding="utf-8") as f:
        classes = [line.strip() for line in f if line.strip()]

    # Output layers (version-safe)
    try:
        output_layers = net.getUnconnectedOutLayersNames()
    except AttributeError:
        layer_names = net.getLayerNames()
        unconnected = net.getUnconnectedOutLayers()  # 1-based indices
        output_layers = [layer_names[i[0] - 1] for i in unconnected]

    # ----------------- YOLO forward -----------------
    blob = cv2.dnn.blobFromImage(image_resize, scalefactor=1/255.0, size=(416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    outputs = net.forward(output_layers)

    conf_threshold = 0.5
    nms_threshold = 0.4
    boxes, confidences, class_ids = [], [], []

    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = int(np.argmax(scores))
            # multiply class score by objectness (detection[4]) for better confidence
            confidence = float(scores[class_id]) * float(detection[4])
            if confidence > conf_threshold:
                cx = int(detection[0] * WIDTH)
                cy = int(detection[1] * HEIGHT)
                w  = int(detection[2] * WIDTH)
                h  = int(detection[3] * HEIGHT)
                x  = cx - w // 2
                y  = cy - h // 2
                boxes.append([x, y, w, h])
                confidences.append(confidence)
                class_ids.append(class_id)

    # Non-max suppression
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

    # Keep one label (your code used a single string)
    detected_objects = ""
    if len(idxs):
        # take the highest confidence detection as "primary"
        best_i = int(idxs.flatten()[np.argmax([confidences[i] for i in idxs.flatten()])])
        detected_objects = classes[class_ids[best_i]]

    # ----------------- OCR (Tesseract) -----------------
    # If tesseract isn't on PATH, set this:
    # pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

    # Pipeline 1: OTSU -> scale up
    gray = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    custom_config = r'--oem 3 --psm 6'
    detected_text1 = pytesseract.image_to_string(gray, config=custom_config)

    # Pipeline 2: denoise + adaptive threshold + scale
    gray2 = cv2.cvtColor(image_resize, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.GaussianBlur(gray2, (5, 5), 0)
    gray2 = cv2.adaptiveThreshold(gray2, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                  cv2.THRESH_BINARY, 11, 2)
    gray2 = cv2.resize(gray2, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    custom_config2 = r'--oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789'
    detected_text2 = pytesseract.image_to_string(gray2, config=custom_config2)

    print("Detected Text1:", detected_text1)
    print("Detected Text2:", detected_text2)

    # ----------------- postprocess words -----------------
    val1 = detected_text1.split()
    val2 = detected_text2.split()

    check1 = check_real_word(val1)
    check2 = check_real_word(val2)

    check_empty(check1)
    check_empty(check2)

    final_words = set(check1).union(set(check2))
    speech_text = f"Detected Object: {detected_objects}\nItem name: {' '.join(final_words)}"
    print(speech_text)

    # ----------------- TTS -----------------
    engine = pyttsx3.init()
    engine.say(speech_text)
    engine.runAndWait()

except FileNotFoundError as e:
    print("Error:", e)
except cv2.error as e:
    print("OpenCV Error:", e)
except Exception as e:
    print("Unexpected Error:", e)
