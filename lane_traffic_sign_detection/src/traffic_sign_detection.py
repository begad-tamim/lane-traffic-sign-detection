import cv2
import numpy as np
from ultralytics import YOLO
import os

class TrafficSignDetector:
    def __init__(self, model_path, class_names):
        self.model = YOLO(model_path)
        self.classes = class_names

    def detect_signs(self, image):
        results = self.model(image)
        detected_signs = []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = box.conf[0]
                cls = int(box.cls[0])
                if 0 <= cls < len(self.classes):
                    label = f'{self.classes[cls]} {conf:.2f}'
                    detected_signs.append((label, (x1, y1, x2 - x1, y2 - y1)))
                    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                else:
                    print(f"Detected class index {cls} is out of range for class names list.")
        return image, detected_signs

def main():
    model_path = r'C:\Users\begad\OneDrive\Term 7\Computer Vision\FInal Project\lane-traffic-sign-detection\models\best.pt'  # Ensure this path is correct
    class_names = [
        # 'stop sign', 'yield sign', 'speed limit', 'no entry', 'traffic light', 'pedestrian crossing', 'school zone',
        # 'construction', 'railroad crossing', 'one way', 'do not enter', 'no parking', 'no u-turn', 'no left turn',
        # 'no right turn', 'roundabout', 'speed bump', 'slippery road', 'animal crossing', 'bicycle lane'
    ]  # COCO dataset classes + additional traffic signs
    detector = TrafficSignDetector(model_path, class_names)

    # Directory containing test images
    test_images_dir = r'C:\Users\begad\OneDrive\Term 7\Computer Vision\FInal Project\lane-traffic-sign-detection\data\valid\images'

    # Process each image in the test images directory
    for image_name in os.listdir(test_images_dir):
        image_path = os.path.join(test_images_dir, image_name)
        image = cv2.imread(image_path)
        if image is None:
            print(f"Failed to load image {image_path}")
            continue

        result_image, _ = detector.detect_signs(image)
        cv2.imshow('Traffic Sign Detection', result_image)
        cv2.waitKey(0)  # Wait for a key press to move to the next image

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()