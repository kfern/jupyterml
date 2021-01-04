from sklearn.model_selection import train_test_split

def custom_split_data(df_data, X_columns, y_colum, train_ratio, validation_ratio, test_ratio, random_state=None, shuffle=True, stratify=None):
    dataX = df_data[X_columns]
    dataY = df_data[y_colum]
    x_train, x_test, y_train, y_test = train_test_split(dataX, dataY, test_size=1 - train_ratio, random_state=random_state, shuffle=shuffle, stratify=stratify)
    x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size=test_ratio/(test_ratio + validation_ratio), random_state=random_state, shuffle=shuffle, stratify=stratify)

    return x_train, x_test, x_val, y_train, y_test, y_val

