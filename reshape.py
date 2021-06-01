import numpy as np

array1 = np.arange(8)
array3d = array1.reshape((2, 2, 2))
print('array3d:\n', array3d.tolist())

# 3차원 -> 2차원
array5 = array3d.reshape(-1, 1)
print('array5:\n', array5.tolist())

# 1차원 -> 2차원
array6 = array1.reshape(-1, 1)
print('array6:\n', array6.tolist())

# 인덱싱
array1d = np.arange(start=1, stop=10)
array3 = array1d[array1d>5]
indexs = array1d[array3d]
array4 = array1d[indexs]
print(array4)