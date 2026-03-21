from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor


"""
#モデル作成
def create_model():
    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )

    return model
#2 学習
def train_model(X_train, y_train):
    print("Training model...")
    model = create_model()
    model.fit(X_train, y_train)
    return model

#予測
def predict(model, X):
    preds = model.predict(X)
    return preds

"""

def create_model():

    model = XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )

    return model


# 学習
def train_model(X_train, y_train):
    print("Training model...")
    model = create_model()
    model.fit(X_train, y_train)
    return model


# 予測
def predict(model, X):
    preds = model.predict(X)
    return preds