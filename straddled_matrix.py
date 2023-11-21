import math
import numpy as np


# Test constant init in MLP layers
def straddled_matrix(shape1: int, shape2: int) -> np.ndarray:
    small_matrix = np.identity(shape2)
    matrix = small_matrix
    for i in range(math.ceil(shape1 / shape2)):
        matrix = np.concatenate((matrix, small_matrix), axis=0)
    return matrix[:shape1, :]
