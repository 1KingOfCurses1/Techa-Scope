import cv2
import pytesseract

img = cv2.imread("images.png")
if img is None:
    raise SystemExit("Could not read test.jpg")

text = pytesseract.image_to_string(img, lang="eng")
print("=== OCR TEXT ===")
print(text)
