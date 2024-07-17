import cv2
import supervision as sv
from ultralytics import YOLOv10
import os

# Use raw string for the file path to avoid unicode escape issues
model = YOLOv10(r'C:\Users\user\Desktop\languages\ааа\computervision\shet\best.pt')

bounding_box_annotator = sv.BoundingBoxAnnotator()
label_annotator = sv.LabelAnnotator()

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Failed to open webcam")

output_dir = 'output_images'
os.makedirs(output_dir, exist_ok=True)
img_counter = 0

while True:
    ret, frame = cap.read()
    
    if not ret:
        break

    results = model(frame)[0]
    detections = sv.Detections.from_ultralytics(results)

    annotated_image = bounding_box_annotator.annotate(
        scene=frame, detections=detections)
    annotated_image = label_annotator.annotate(
        scene=annotated_image, detections=detections)

    cv2.imshow('Webcam', annotated_image)

    k = cv2.waitKey(1)

    if k % 256 == 27:
        print("Exiting...")
        break

cap.release()
cv2.destroyAllWindows()
