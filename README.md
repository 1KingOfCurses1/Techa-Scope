# TechaScope

TechaScope is a vision assistant that uses **computer vision and OCR** to help recognize objects and read out any text it finds.  
It combines **YOLOv4-tiny object detection**, **Tesseract OCR**, and **Text-to-Speech (TTS)** in a simple web app.

---

## How it works
- The **camera feed** is accessed directly in the browser.  
- When you press **CAPTURE** (or wait for auto-capture), a snapshot is sent to the backend.  
- The **backend (Flask + OpenCV)** runs:
  - Object detection with YOLOv4-tiny
  - Text detection with Tesseract OCR
  - Builds a combined message (object + detected text)  
- The browser then **displays the results** and can **speak them aloud** using built-in text-to-speech.

---

## Features
- 🎥 **Live camera input** with capture button  
- 🔁 **Auto-capture mode** every few seconds (can be toggled in code)  
- 🔊 **Text-to-Speech (TTS)** to read detected objects and text aloud  
- ✋ **Manual control** — you can choose when to trigger speech  
- 🖼 **Annotated image preview** showing bounding boxes and recognized text  
- 📖 OCR words are filtered with a dictionary to reduce noise  

---

## Running the program
1. Start the backend server:
   ```bash
   py app.py
   ```
   The server runs on `http://127.0.0.1:5000`.

2. Open the frontend:
   - In your browser, go to `http://127.0.0.1:5000`
   - Allow camera access
   - Press **CAPTURE** or let it auto-capture

3. See results on the page:
   - Detected object name
   - Item name from OCR text
   - Full speech text (also spoken aloud if enabled)

---

## Notes
- Auto-read can be turned **on/off** in the frontend JavaScript.  
- Confidence thresholds and capture intervals can be adjusted in the code.  
- Model weights (`yolov4-tiny.weights`) are required locally but ignored in GitHub.  

---

## Authors
Shan Truong, Nick Sorbara, Mia Gassojne, Manu

