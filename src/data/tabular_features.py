import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import numpy as np

def load_tabular(csv_path: str):
    df = pd.read_csv(csv_path)
    X_num = df[['amount','time_gap']].values
    enc = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    X_cat = enc.fit_transform(df[['merchant_type']])
    X = np.hstack([X_num, X_cat])
    y = df['fraud'].values.astype(int)
    return X, y, enc
