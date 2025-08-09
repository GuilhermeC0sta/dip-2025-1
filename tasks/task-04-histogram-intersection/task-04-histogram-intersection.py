import numpy as np
import cv2

def compute_histogram_intersection(img1: np.ndarray, img2: np.ndarray) -> float:
    """
    Compute the histogram intersection similarity score between two grayscale images.

    This function calculates the similarity between the grayscale intensity 
    distributions of two images by computing the intersection of their 
    normalized 256-bin histograms.

    The histogram intersection is defined as the sum of the minimum values 
    in each corresponding bin of the two normalized histograms. The result 
    ranges from 0.0 (no overlap) to 1.0 (identical histograms).

    Parameters:
        img1 (np.ndarray): First input image as a 2D NumPy array (grayscale).
        img2 (np.ndarray): Second input image as a 2D NumPy array (grayscale).

    Returns:
        float: Histogram intersection score in the range [0.0, 1.0].

    Raises:
        ValueError: If either input is not a 2D array (i.e., not grayscale).
    """    
    if img1.ndim != 2 or img2.ndim != 2:
        raise ValueError("Both input images must be 2D grayscale arrays.")

    ### START CODE HERE ###
    
    hist1, _ = np.histogram(img1, bins=256, range=(0, 256), density=True)
    hist2, _ = np.histogram(img2, bins=256, range=(0, 256), density=True)
    
    
    intersection = np.sum(np.minimum(hist1, hist2))
    ### END CODE HERE ###

    return float(intersection)


if __name__ == "__main__":
    # Caminhos das imagens
    img1_path = "../../img/head.png"
    img2_path = "../../img/head_filtered.png"

    # Carregar imagens em escala de cinza
    img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)

    # Verificação
    if img1 is None or img2 is None:
        raise FileNotFoundError(f"Não foi possível carregar {img1_path} ou {img2_path}")

    # Calcula interseção de histograma
    score = compute_histogram_intersection(img1, img2)
    print(f"Pontuação de interseção do histograma: {score:.4f}")