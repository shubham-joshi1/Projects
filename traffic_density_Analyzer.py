from ultralytics import YOLO
import cv2

# Load the pre-trained YOLOv8 model
model = YOLO('yolov8n.pt')  # 'yolov8n.pt' is the smallest version of yolov8 models.

# Open the video file or webcam
video_path = r"C:\Users\Shubham\Downloads\2103099-uhd_3840_2160_30fps.mp4"  
cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Resize frame for faster processing
    frame = cv2.resize(frame, (800, 600))

    # Perform inference with YOLOv8
    results = model(frame)

    # Extract detected objects
    detections = results[0].boxes.data.cpu().numpy()  # YOLOv8 outputs detections in this format

    vehicle_count = 0
    for detection in detections:
        x1, y1, x2, y2, confidence, class_id = detection
        class_id = int(class_id)

        # Only count vehicles (class IDs vary depending on the YOLO model)
        # Common vehicle classes in COCO dataset: 'car', 'truck', 'bus', 'motorcycle' 
        if class_id in [2, 3, 5, 7]:  # Adjust based on your model's class mappings
            vehicle_count += 1
            # Draw bounding boxes on the frame
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(frame, f"Vehicle {vehicle_count}", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the vehicle count
    cv2.putText(frame, f"Vehicles Detected: {vehicle_count}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Show the frame
    cv2.imshow("Traffic Density Analyzer", frame)

    # Break loop on 'q' key
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
