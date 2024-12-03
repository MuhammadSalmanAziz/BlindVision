import cv2
import pytesseract
import pyttsx3
import time
import threading

# Set the tesseract executable path for Windows (adjust the path if needed)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Initialize text-to-speech engine
tts_engine = pyttsx3.init()
tts_engine.setProperty('rate', 150)  # Set speaking rate

def perform_ocr(image_np):
    try:
        gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

        custom_config = r'--oem 3 --psm 3'
        d = pytesseract.image_to_data(thresh, output_type=pytesseract.Output.DICT, config=custom_config)
        
        return d
    except Exception as e:
        print(f"Error during OCR processing: {e}")
        return None

def draw_boxes_and_text(image_np, ocr_data):
    n_boxes = len(ocr_data['level'])
    for i in range(n_boxes):
        (x, y, w, h) = (ocr_data['left'][i], ocr_data['top'][i], ocr_data['width'][i], ocr_data['height'][i])
        text = ocr_data['text'][i]
        if int(ocr_data['conf'][i]) > 60:  # Confidence threshold
            cv2.rectangle(image_np, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(image_np, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return image_np

def speak(text):
    tts_engine.say(text)
    tts_engine.runAndWait()

def main():
    cap = cv2.VideoCapture(0)  # Capture video from the laptop's webcam

    if not cap.isOpened():
        print("Error: Could not open video capture")
        speak("Error: Could not open video capture")
        return

    print("Press 'q' to quit")
    speak("Press q to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            speak("Failed to grab frame")
            break

        # Perform OCR on the current frame
        try:
            ocr_data = perform_ocr(frame)
            if ocr_data is not None:
                text = " ".join([ocr_data['text'][i] for i in range(len(ocr_data['text'])) if int(ocr_data['conf'][i]) > 60])
                if text.strip():  # If OCR result is not empty
                    print(f"OCR Result: {text}")

                    # Run the TTS engine in a separate thread
                    tts_thread = threading.Thread(target=speak, args=(text,))
                    tts_thread.start()

                frame = draw_boxes_and_text(frame, ocr_data)
            else:
                speak("Error during OCR processing")
        except Exception as e:
            print(f"Error during OCR: {e}")
            speak(f"Error during OCR: {e}")

        cv2.imshow('Live OCR', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
