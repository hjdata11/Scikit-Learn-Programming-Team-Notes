# 피처 스케일링과 정규화

from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris
import pandas as pd

# 가능하다면 전체 데이터의 스케일링 변환을 적용한 뒤 학습과 테스트 데이터로 분리
# 여의치 않다면 테스트 데이터 변환 시에는 fit()이나 fit_transfrom()을 적용하지 않고 학습데이터로 이미 fit()된 Scaler 객체를 이용해 transform()으로 변환

iris = load_iris()
iris_data = iris.data
iris_df = pd.DataFrame(data=iris_data, columns=iris.feature_names)

scaler = StandardScaler()
scaler.fit(iris_df)
iris_scaled = scaler.transform(iris_df)
iris_df_scaled = pd.DataFrame(data=iris_scaled, columns=iris.feature_names)
print(iris_df_scaled.mean())
print(iris_df_scaled.var())