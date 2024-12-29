import cv2
import numpy as np
import os

def detect_lanes(image):
    """
    Detects lanes in the given image using edge detection and Hough Transform.
    Args:
        image (numpy.ndarray): The input image in which lanes are to be detected.
    Returns:
        tuple: A tuple containing:
            - lane_image (numpy.ndarray): The original image with detected lanes drawn on it.
            - lane_detected (bool): A boolean indicating whether any lanes were detected.
    """
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise and improve edge detection
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Use Canny edge detection
    edges = cv2.Canny(blurred, 50, 150)
    
    # Define a region of interest (ROI) to focus on the lane area
    height, width = edges.shape
    roi_vertices = np.array([[(0, height), (width / 2, height / 2), (width, height)]], dtype=np.int32) # Triangle ROI for simplicity 
    mask = np.zeros_like(edges) # Create a blank mask
    cv2.fillPoly(mask, roi_vertices, 255) # Fill the ROI with white color (255)
    masked_edges = cv2.bitwise_and(edges, mask) # Apply the mask to the edges image
    
    # Use Hough Transform to detect lines in the masked edge image
    lines = cv2.HoughLinesP(masked_edges, 1, np.pi / 180, threshold=50, minLineLength=100, maxLineGap=50) 
    
    # Create a copy of the original image to draw the detected lanes
    lane_image = np.copy(image)
    
    lane_detected = False
    # Draw the detected lines on the lane image
    if lines is not None:
        lane_detected = True
        for line in lines:
            x1, y1, x2, y2 = line[0] # Extract line coordinates
            cv2.line(lane_image, (x1, y1), (x2, y2), (0, 255, 0), 3) # Draw a green line
    
    return lane_image, lane_detected

def classify_lane_or_curve(lane_detected, image):
    if lane_detected:
        # Here you can add additional logic to detect if the lane is a curve.
        # For now, we assume if lanes are detected, it's a "lane", else "curve".
        return "Lane"
    else:
        return "Curve"

def main():
    """
    Main function to process test images for lane detection and classification.
    This function reads images from a specified directory, detects lanes in each image,
    classifies the image as either "lane" or "curve", and displays the result.
    Steps:
    1. Reads images from the test images directory.
    2. For each image:
       - Loads the image.
       - Detects lanes in the image.
       - Classifies the image as "lane" or "curve".
       - Displays the classified image with the detected lanes.
    Note:
    - Ensure the test images directory path is correct.
    - Press any key to move to the next image when the result is displayed.
    Raises:
    - Prints an error message if an image fails to load.
    """
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
        cv2.waitKey(0)  # Wait for a key press to move to the next image
    
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
