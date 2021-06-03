# 데이터 인코딩

# 레이블 인코딩
from sklearn.preprocessing import LabelEncoder

items = ['TV', '냉장고', '전자레인지', '컴퓨터', '선풍기', '선풍기', '믹서', '믹서']

encoder = LabelEncoder()
encoder.fit(items)
labels = encoder.transform(items)
# 숫자의 크고 작음이 특성에 반영되기 때문에 회귀와 같은 ML 알고리즘에는 부적합
print(labels)
print(encoder.inverse_transform(labels))

# 원핫 인코딩
from sklearn.preprocessing import OneHotEncoder
import numpy as np

labels = labels.reshape(-1, 1)

oh_encoder = OneHotEncoder()
oh_encoder.fit(labels)
oh_labels = oh_encoder.transform(labels)
print(oh_labels.toarray())

import pandas as pd

df = pd.DataFrame({'item':items})
labels = pd.get_dummies(df)
print(labels)