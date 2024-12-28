def load_image(image_path):
    """Load an image from the specified path."""
    import cv2
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found at {image_path}")
    return image

def display_image(image, window_name='Image'):
    """Display an image in a window."""
    import cv2
    cv2.imshow(window_name, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def save_image(image, save_path):
    """Save an image to the specified path."""
    import cv2
    cv2.imwrite(save_path, image)

def resize_image(image, width, height):
    """Resize an image to the specified width and height."""
    import cv2
    return cv2.resize(image, (width, height))

def get_image_files(directory):
    """Get a list of image files in the specified directory."""
    import os
    return [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(('.png', '.jpg', '.jpeg'))]