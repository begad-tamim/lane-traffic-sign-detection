import cv2
import numpy as np
import os

def detect_lanes(image):
    
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise and improve edge detection
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Use Canny edge detection
    edges = cv2.Canny(blurred, 50, 150)
    
    # Define a region of interest (ROI) to focus on the lane area
    height, width = edges.shape
    
    # Triangle ROI for simplicity
    roi_vertices = np.array([[(0, height), (width / 2, height / 2), (width, height)]], dtype=np.int32)  
    
    # Create a blank mask
    mask = np.zeros_like(edges) 
    
    # Fill the ROI with white color (255)
    cv2.fillPoly(mask, roi_vertices, 255) 
    
    # Apply the mask to the edges image
    masked_edges = cv2.bitwise_and(edges, mask) 
    
    # Use Hough Transform to detect lines in the masked edge image
    lines = cv2.HoughLinesP(masked_edges, 1, np.pi / 180, threshold=50, minLineLength=100, maxLineGap=50) 
    
    # Create a copy of the original image to draw the detected lanes
    lane_image = np.copy(image)
    
    lane_detected = False
    
    # Draw the detected lines on the lane image
    if lines is not None:
        lane_detected = True
        for line in lines:
            
            # Extract line coordinates
            x1, y1, x2, y2 = line[0] 
            
            # Draw a green line
            cv2.line(lane_image, (x1, y1), (x2, y2), (0, 255, 0), 3) 
    
    return lane_image, lane_detected

def classify_lane_or_curve(lane_detected, image):
    
    # Classify the image as "Lane" if lanes are detected, otherwise as "Curve"
    if lane_detected:
        
        return "Lane"
    else:
        return "Curve"

def main():
    
    # Directory containing test images
    test_images_dir = r'C:\Users\begad\OneDrive\Term 7\Computer Vision\FInal Project\lane-traffic-sign-detection\data\test\images'  # Ensure this path is correct
    
    # Process each image in the test images directory
    for image_name in os.listdir(test_images_dir):
        image_path = os.path.join(test_images_dir, image_name)
        image = cv2.imread(image_path)
        
        if image is None:
            print(f"Failed to load image {image_path}")
            continue
        
        # Detect lanes in the current image
        lane_image, lane_detected = detect_lanes(image)
        
        # Classify the image as "lane" or "curve"
        lane_or_curve = classify_lane_or_curve(lane_detected, lane_image)
        print(f"Image '{image_name}' classified as: {lane_or_curve}")
        
        # Display the result
        cv2.imshow(f'{image_name} - {lane_or_curve}', lane_image)
        
        # Wait for a key press to move to the next image
        cv2.waitKey(0)  
    
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()