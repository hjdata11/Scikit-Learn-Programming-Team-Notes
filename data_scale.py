# 데이터 변환

def get_scaled_data(method="None", p_degree=None, input_data=None):
    if method == 'Standard':
        scaled_data = StandardScaler().fit_tranform(input_data)
    elif method == "MinMax":
        scaled_data = MinMaxScaler().fit_transform(input_data)
    elif method == "log":
        scaled_data = np.log1p(input_data)
    else:
        scaled_data = input_data
        
    return scaled_data