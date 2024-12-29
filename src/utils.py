def load_image(image_path):
    """
    Load an image from the specified path.

    Args:
        image_path (str): The path to the image file.

    Returns:
        numpy.ndarray: The loaded image.

    Raises:
        FileNotFoundError: If the image is not found at the specified path.
    """
    """Load an image from the specified path."""
    import cv2
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found at {image_path}")
    return image

def display_image(image, window_name='Image'):
    """
    Display an image in a window.

    Parameters:
        image (numpy.ndarray): The image to be displayed.
        window_name (str, optional): The name of the window in which the image will be displayed. Defaults to 'Image'.

    Returns:
        None
    """
    """Display an image in a window."""
    import cv2
    cv2.imshow(window_name, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def save_image(image, save_path):
    """
    Save an image to the specified path.

    Parameters:
    image (numpy.ndarray): The image to be saved.
    save_path (str): The path where the image will be saved.

    Returns:
    bool: True if the image is successfully saved, False otherwise.
    """
    """Save an image to the specified path."""
    import cv2
    cv2.imwrite(save_path, image)

def resize_image(image, width, height):
    """
    Resize an image to the specified width and height.

    Parameters:
        image (numpy.ndarray): The input image to be resized.
        width (int): The desired width of the resized image.
        height (int): The desired height of the resized image.

    Returns:
        numpy.ndarray: The resized image.
    """
    """Resize an image to the specified width and height."""
    import cv2
    return cv2.resize(image, (width, height))

def get_image_files(directory):
    """
    Get a list of image files in the specified directory.

    Args:
        directory (str): The path to the directory to search for image files.

    Returns:
        list: A list of file paths to image files with extensions .png, .jpg, or .jpeg.
    """
    """Get a list of image files in the specified directory."""
    import os
    return [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(('.png', '.jpg', '.jpeg'))]