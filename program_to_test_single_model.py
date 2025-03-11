from ultralytics import YOLO
import cv2
import cvzone
import math

# Initialize webcam
cap = cv2.VideoCapture(0)  # For webcam
cap.set(3, 640)  # Set width
cap.set(4, 480)  # Set height

# Load the trained YOLO models
model1 = YOLO('Completely_new_datasets/runs/detect/train4/weights/best.pt')

# Define class names for each model
classNames_model1 = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']

# Set confidence threshold
conf_threshold = 0.25  # Adjust this value as needed

while True:
    success, img = cap.read()
    if not success:
        break

    # Perform detection using both models
    results1 = model1(img, stream=True)

    highest_conf = 0
    highest_box = None
    highest_cls = None

    # Process results from model1
    for r in results1:
        boxes = r.boxes
        for box in boxes:
            conf = box.conf[0].item()  # Get confidence score

            if conf >= conf_threshold and conf > highest_conf:  # Check if confidence is above threshold and highest so far
                highest_conf = conf
                highest_box = box
                highest_cls = int(box.cls[0])

    # Draw the bounding box for the detection with the highest confidence
    if highest_box is not None:
        x1, y1, x2, y2 = map(int, highest_box.xyxy[0])
        # Draw rectangle for model1 detections (Blue)
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 3)

        highest_conf = round(highest_conf * 100, 2)  # Confidence
        if highest_cls < len(classNames_model1):
            label = f'{classNames_model1[highest_cls]} {highest_conf}%'
            cvzone.putTextRect(img, label, (max(0, x1), max(35, y1)), scale=1, thickness=1)

    # Display the frame
    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()


