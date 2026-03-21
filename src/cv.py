
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np


def train_valid_split(df):
    print("Splitting data...")
        # 時系列順に並べる
    df = df.sort_values("gameday")
    train_df,valid_df=train_test_split(df,test_size=0.2,shuffle=False)

    print("train shape",train_df.shape)
    print("valid shape",valid_df.shape)
    print(train_df["gameday"].min(), train_df["gameday"].max())
    print(valid_df["gameday"].min(),valid_df["gameday"].max())
    return train_df,valid_df



def evaluate_model(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    print("RMSE:", rmse)
    return rmse