import numpy as np

# 인덱싱
array1d = np.arange(start=1, stop=10)
array3 = array1d[array1d>5]
indexs = array1d[array3d]
array4 = array1d[indexs]
print(array4)

# argsort(인덱싱 배열)
name_array = np.array(['John', 'Mike', 'Sarah', 'Kate', 'Samuel'])
score_array = np.array([78, 95, 84, 98, 88])

sort_indices_asc = np.argsort(score_array)
print(name_array[sort_indices_asc])