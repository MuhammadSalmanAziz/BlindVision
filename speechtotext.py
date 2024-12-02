import speech_recognition as sr
import subprocess
import threading

# Function to recognize speech
def recognize_speech():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Please say 'object detection', 'OCR', or 'close':")
        audio = recognizer.listen(source)

    try:
        command = recognizer.recognize_google(audio).lower()
        print(f"You said: {command}")
        return command
    except sr.UnknownValueError:
        print("Sorry, I did not understand that.")
        return None
    except sr.RequestError:
        print("Sorry, my speech service is down.")
        return None

# Main function
def main():
    current_process = None

    while True:
        command = recognize_speech()
        if command:
            if 'object detection' in command:
                if current_process:
                    current_process.terminate()
                    print("Previous process terminated.")
                print("Running object detection...")
                current_process = subprocess.Popen(['python', 'detect4cls.py'])
            elif 'ocr' in command:
                if current_process:
                    current_process.terminate()
                    print("Previous process terminated.")
                print("Running OCR...")
                current_process = subprocess.Popen(['python', 'ocr.py'])
            elif 'close' in command:
                if current_process:
                    current_process.terminate()
                    print("Process terminated. Please provide a new command.")
                    current_process = None
            else:
                print("Command not recognized. Please say 'object detection', 'OCR', or 'close'.")

if __name__ == "__main__":
    main()
