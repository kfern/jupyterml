import pandas as pd
from statsmodels.tsa import stattools

class MissingValuesTransformer():
    def __init__(self, index="datetime", column=None):
        self.index = index
        self.column = column

    def transform(self, X, **transform_params):
        # initialization
        df_cpy = X.copy()
        df_dp = pd.DataFrame(pd.date_range(start=df_cpy[self.index].min(), end=df_cpy[self.index].max(), freq='H').values, columns=['datetime'])
        df_dp[self.column] = pd.np.nan
        df_dp.set_index('datetime', inplace=True)

        # Add X to result
        df_cpy.set_index(self.index, inplace=True)
        df_dp.loc[df_cpy.index.values] = df_cpy

        # Missing values
        df_dp.interpolate(type='time', limit_area='inside', inplace=True)
        df_dp.dropna(inplace=True)

        # Reset index
        df_dp.reset_index(inplace=True)

        # Return result
        return df_dp

    def fit(self, X, y=None, **fit_params):
        return self


class AutoCorrelationFeaturesTransformer():
    def __init__(self, acf_threshold=.9, index="datetime", column=None):
        self.acf_threshold = acf_threshold
        self.index = index
        self.column = column
        self.new_features = 0

    def transform(self, X, **transform_params):
        df_cpy = X.copy()
        for nfeat in range(1, self.new_features + 1, 1):
            df_cpy[self.column + '-h-' + str(nfeat)] = df_cpy[self.column].shift(nfeat)

        # Drop NaN
        df_cpy.dropna(inplace=True)

        # Return result
        return df_cpy

    def fit(self, X, y=None, **fit_params):
        # Finding Regressor Inputs
        acf, _, _, _ = stattools.acf(X.set_index(self.index), nlags=20, qstat=True, fft=True, alpha = 0.05)
        acf_result = pd.Series(acf)
        self.new_features = acf_result[acf_result > self.acf_threshold].index.values.max() + 1

        return self