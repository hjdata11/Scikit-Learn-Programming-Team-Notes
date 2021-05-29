# [희소 행렬 sparse matrix 저장 (0을 많이 포함한 2차원 배열 저장)]

import numpy as np
from scipy import sparse

eye = np.eye(4)
print("Numpy 배열:\n", eye)

# Numpy 배열을 CSR 포맷의 Scipy 희박 행렬로 변환
sparse_matrix = sparse.csr_matrix(eye)
print("\nScipy의 CSR행렬:\n", sparse_matrix)

data = np.ones(4)
row_indices = np.arange(4)
col_indices = np.arange(4)
eye_coo = sparse.coo_matrix((data, (row_indices, col_indices)))
print("COO 표현:\n", eye_coo)