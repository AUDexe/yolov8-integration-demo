# Import the YOLO class from the ultralytics library
from ultralytics import YOLO

# Load a pretrained model (recommended for training)
model = YOLO("yolov8m.pt")

# Export the initialized YOLO model to the ONNX (Open Neural Network Exchange) format to run the model on JavaScript
model.export(format="onnx")