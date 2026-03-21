import pandas as pd
import os

def load_data():
    print("Loading data...")

    data_path = "data"

    condition = pd.read_csv(os.path.join(data_path, "condition.csv"))
    condition_add = pd.read_csv(os.path.join(data_path, "condition_add.csv"))
    condition_all = pd.concat([condition, condition_add], axis=0)

    stadium =pd.read_csv(os.path.join(data_path,"stadium.csv"))
    stadium=stadium.rename(columns={"name":"stadium"})

    train=pd.read_csv(os.path.join(data_path,"train.csv"))
    train_add=pd.read_csv(os.path.join(data_path,"train_add.csv"))
    train_all = pd.concat([ train, train_add], axis=0)

    #train_alにcondition,stadium情報を集約
    train_all=pd.merge(train_all,condition_all,on="id",how="left")
    train_all=pd.merge(train_all,stadium,on="stadium",how="left")


    print("Data loaded and concatenated.")
    return train_all, None