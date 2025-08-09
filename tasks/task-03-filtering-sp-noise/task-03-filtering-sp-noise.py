import cv2
import numpy as np

def remove_salt_and_pepper_noise(image: np.ndarray) -> np.ndarray:
    """
    Removes salt and pepper noise from a grayscale image.

    Parameters:
        image (np.ndarray): Noisy input image (grayscale).

    Returns:
        np.ndarray: Denoised image.
    """
    # TODO: Implement noise removal here (e.g., median filtering)

    image = cv2.medianBlur(image, 3)

    return image  # Replace this with your filtering implementation

if __name__ == "__main__":
    noisy_image = cv2.imread("../../img/head.png", cv2.IMREAD_GRAYSCALE)
    
    if noisy_image is None:
        raise FileNotFoundError("Could not find or load head.png")
        
    # Apply noise removal
    denoised_image = remove_salt_and_pepper_noise(noisy_image)
    
    # Save the result
    cv2.imwrite("head_filtered.png", denoised_image)
    
    cv2.imshow('Noisy Image', noisy_image)
    cv2.imshow('Denoised Image', denoised_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
