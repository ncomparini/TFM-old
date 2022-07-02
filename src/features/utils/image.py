import numpy as np


def get_2_dimensional_matrix(matrix: np.ndarray):
    num_dim = len(matrix.shape)
    if num_dim == 4:
        return matrix[0][..., -1]
    elif num_dim == 3:
        return matrix[..., -1]
    elif num_dim == 2:
        return matrix
    else:
        print(f"ERROR. Invalid matrix. Matrix shape: {matrix.shape}")
        return None


def normalize(img: np.ndarray):
    return img * 0.5 + 0.5


def extract_points(img: np.ndarray, threshold: float = 0.85):
    gray_image = get_2_dimensional_matrix(img)
    img_norm = normalize(gray_image)
    points = np.argwhere(img_norm > threshold)

    return points
