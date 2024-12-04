from ultralytics import YOLO
import cv2
import pyttsx3
import time

# Initialize the YOLO model
model = YOLO("D:\Blindvision\yolov8n.pt")

# Initialize text-to-speech engine
tts_engine = pyttsx3.init()
tts_engine.setProperty('rate', 150)  # Set speaking rate
tts_engine.setProperty('volume', 0.9)  # Set volume leqvel

# Define the classes we want to detect
desired_classes = ['person', 'cell phone', 'mouse', 'book']

def describe_scene(detected_class_names):
    if not detected_class_names:
        return "No objects detected"
    
    unique_objects = list(set(detected_class_names))
    object_counts = {obj: detected_class_names.count(obj) for obj in unique_objects}
    
    description = "In your surroundings, I see "
    descriptions = []
    
    for obj, count in object_counts.items():
        if count == 1:
            descriptions.append(f"a {obj}")
        else:
            descriptions.append(f"{count} {obj}s")
    
    description += ", ".join(descriptions) + "."
    return description

def main():
    cap = cv2.VideoCapture(0)  # Capture video from the laptop's webcam

    if not cap.isOpened():
        print("Error: Could not open video capture")
        tts_engine.say("Error: Could not open video capture")
        tts_engine.runAndWait()
        return

    print("Press 'q' to quit")
    tts_engine.say("Press q to quit")
    tts_engine.runAndWait()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            tts_engine.say("Failed to grab frame")
            tts_engine.runAndWait()
            break

        # Perform prediction on the current frame
        results = model.predict(source=frame, show=True)

        # Extract detected classes
        detected_classes = []
        for result in results:
            for box in result.boxes:
                class_id = int(box.cls)
                class_name = model.names[class_id]
                if class_name in desired_classes:
                    detected_classes.append(class_name)

        # Construct the scene description
        scene_description = describe_scene(detected_classes)
        print(scene_description)
        tts_engine.say(scene_description)

        # Run and wait for the TTS engine to finish
        tts_engine.runAndWait()

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
