# Project
Traffic Density Analyzer

Brief explanation
YOLOv8 Model:

The script uses the yolov8n.pt model (YOLOv8 Nano) for real-time performance. Replace it with yolov8m.pt or yolov8l.pt for higher accuracy if needed.
Detection Logic:

Each detection includes bounding box coordinates, confidence score, and class ID.
Class IDs are mapped to COCO dataset classes (e.g., 2 for cars, 3 for motorcycles, etc.).
Counting Vehicles:

Only specific class IDs corresponding to vehicles (e.g., cars, trucks, buses) are considered.
Bounding Boxes:

Bounding boxes and labels are drawn on the video feed for visual feedback.
