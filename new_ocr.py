import cv2
import numpy as np
import pyttsx3
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

img = cv2.imread("book.jpg")
# 2. Resize the image
img = cv2.resize(img, None, fx=0.5, fy=0.5)
# 3. Convert image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# 4. Convert image to black and white (using adaptive threshold)
adaptive_threshold = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 85, 11)

config = "--psm 3"
text = pytesseract.image_to_string(adaptive_threshold, config=config)
print(text)

engine = pyttsx3.init()

engine.say(text)
engine.runAndWait()