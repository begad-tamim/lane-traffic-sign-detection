import torch
from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO('best.pt')

# Load dataset
data_path = 'lane-traffic-sign-detection\data\combined'
train_data = torch.utils.data.DataLoader(data_path, batch_size=16, shuffle=True)

# Training configuration
epochs = 50
learning_rate = 0.001

# Train the model
model.train(data=train_data, epochs=epochs, lr=learning_rate)

# Save the trained model
model.save('yolov8_lane_traffic_sign_detection.pt')