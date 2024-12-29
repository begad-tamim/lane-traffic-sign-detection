import cv2  
import numpy as np  
from ultralytics import YOLO 
import os  

class TrafficSignDetector:
    def __init__(self, model_path, class_names):
        
        # Initialize the TrafficSignDetection class with model path and class names
        self.model = YOLO(model_path) 
        self.classes = class_names 

    def detect_signs(self, image):
        
        # Detect traffic signs in the given image
        results = self.model(image)
        
        # Initialize list to store detected signs
        detected_signs = []
        for result in results:
            for box in result.boxes:
                
                # Get bounding box coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0]) 
                
                # Get confidence score
                conf = box.conf[0]  
                
                # Get class index
                cls = int(box.cls[0])  
                if 0 <= cls < len(self.classes):
                    
                    # Create label with class name and confidence
                    label = f'{self.classes[cls]} {conf:.2f}'  
                    
                    # Append detected sign to list
                    detected_signs.append((label, (x1, y1, x2 - x1, y2 - y1)))  
                    
                    # Draw bounding box on image
                    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2) 
                    
                    # Put label text on image
                    cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)  
                else:
                    print(f"Detected class index {cls} is out of range for class names list.") 
        return image, detected_signs

def main():
    
    # Main function to perform traffic sign detection on test images
    # Path to the pre-trained YOLO model
    model_path = r'C:\Users\begad\OneDrive\Term 7\Computer Vision\FInal Project\lane-traffic-sign-detection\models\best.pt' 
    class_names = [
        # List of class names for traffic sign detection
        # 'stop sign', 'yield sign', 'speed limit', 'no entry', 'traffic light', 'pedestrian crossing', 'school zone',
        # 'construction', 'railroad crossing', 'one way', 'do not enter', 'no parking', 'no u-turn', 'no left turn',
        # 'no right turn', 'roundabout', 'speed bump', 'slippery road', 'animal crossing', 'bicycle lane'
    ]
    
    # Initialize the TrafficSignDetector
    detector = TrafficSignDetector(model_path, class_names)  

    test_images_dir = r'C:\Users\begad\OneDrive\Term 7\Computer Vision\FInal Project\lane-traffic-sign-detection\data\valid\images'

    # Iterate through each image in the test images directory
    for image_name in os.listdir(test_images_dir):  
        
        # Get the full path of the image
        image_path = os.path.join(test_images_dir, image_name)  
        
        # Load the image
        image = cv2.imread(image_path)  
        if image is None:
            print(f"Failed to load image {image_path}")  # Print error if image loading fails
            continue

        # Perform traffic sign detection
        result_image, _ = detector.detect_signs(image)  
        
        # Display the detection results
        cv2.imshow('Traffic Sign Detection', result_image)  
        
        # Wait for a key press to move to the next image
        cv2.waitKey(0)  

    cv2.destroyAllWindows() 

if __name__ == "__main__":
    main()