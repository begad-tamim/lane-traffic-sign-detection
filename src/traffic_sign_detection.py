import cv2
import numpy as np
from ultralytics import YOLO
import os

class TrafficSignDetector:
    def __init__(self, model_path, class_names):
        """
        Initializes the TrafficSignDetection class.

        Args:
            model_path (str): Path to the pre-trained YOLO model.
            class_names (list): List of class names for traffic sign detection.
        """
        self.model = YOLO(model_path)
        self.classes = class_names

    def detect_signs(self, image):
        """
        Detects traffic signs in the given image using the pre-trained model.

        Args:
            image (numpy.ndarray): The input image in which traffic signs are to be detected.

        Returns:
            tuple: A tuple containing:
                - image (numpy.ndarray): The input image with detected signs annotated.
                - detected_signs (list): A list of tuples, each containing:
                    - label (str): The class name and confidence score of the detected sign.
                    - bbox (tuple): The bounding box coordinates (x, y, width, height) of the detected sign.
        """
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
    """
    Main function to perform traffic sign detection on test images.
    This function initializes the TrafficSignDetector with the specified model path and class names.
    It then processes each image in the test images directory, performing traffic sign detection
    and displaying the results.
    Steps:
    1. Initialize the TrafficSignDetector with the model path and class names.
    2. Iterate through each image in the test images directory.
    3. Load the image and perform traffic sign detection.
    4. Display the detection results.
    5. Wait for a key press to move to the next image.
    6. Close all OpenCV windows after processing all images.
    Note:
    - Ensure the model path and test images directory are correct.
    - The class names list is currently commented out and should be updated with the actual class names if needed.
    """
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