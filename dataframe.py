import pandas as pd

titanic_df = pd.read_csv('train.csv')

# 총 데이터 건수와 데이터 타입, NULL 건수 확인
titanic_df.info()
# 개략적 분포도 확인
titanic_df.describe()
# 데이터 분포도 확인
titanic_df['Pclass'].value_counts()

# DataFrame <-> array
col_name = ['col1', 'col2', 'col3']
list2 = [[1, 2, 3], [11, 12, 13]]
dict2 = {'col1':[1, 11], 'col2':[2, 22], 'col3':[3, 33]}

array2 = np.array(list2)
df_list2 = pd.DataFrame(list2, columns=col_name)
print(df_list2)
df_array2 = pd.DataFrame(array2, columns=col_name)
print(df_array2)
df_dict = pd.DataFrame(dict2)
print(df_dict)

# ndarray
array3 = df_dict.values
print(array3)
# list
list3 = df_dict.values.tolist()
print(list3)
# dict
dict3 = df_dict.to_dict('list')
print(dict3)

# Dataframe 생성 수정
titanic_df['Age_by_10'] = titanic_df['Age']*10
titanic_df['Family_No'] = titanic_df['SibSp'] + titanic_df['Parch'] + 1
titanic_df.head(3)
# Dataframe 삭제
drop_result = titanic_df.drop(['Age_by_10', 'Family_No'], axis = 1, inplace=True)
titanic_df.head(3)

# 결손 데이터 확인
titanic_df.isna().sum()
# 결손 데이터 대체
titanic_df['Cabin'] = titanic_df['Cabin'].fillna('C000')

# 데이터 가공
titanic_df['Name_len'] = titanic_df['Name'].apply(lambda x : len(x))
titanic_df['Child_Adult'] = titanic_df['Age'].apply(lambda x : 'Child' if x<=15 else ('Adult' if x<=60 else 'Elderly'))