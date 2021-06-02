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

# ix 연산 유형
# data_df.ix[0, 0], data_df.ix['one', 0], data_df.ix[3, 'Name'], data_df.ix[0:2, [0, 1]], data_df[0:2, [0:3]], data_df.ix[0:3, ['Name', 'Year']], data_df.ix[:], data_df.ix[:, :], data_df[data_df.Year >= 2014]

# column 이름 변환
data_df = data_df.rename(columns={'index':'old_index'})