import pandas as pd
import os
from src.load_data import load_data
from src.features import create_features
from src.model import create_model, train_model, predict
from src.cv import train_valid_split, evaluate_model
from config import FEATURES

def main():
    data_path = "data"
    print("Main started")

    # 1. データ読み込み
    train, test = load_data()

    # 2. 分割
    train, valid = train_valid_split(train)

    # 3. 特徴量
    train = create_features(train)
    valid = create_features(valid, reference_df=train)

    if test is not None:
        test = create_features(test, reference_df=train)

    # ============================
    # 🔥 ここが変更ポイント
    # ============================

    # X
    X_train = train[FEATURES]
    X_valid = valid[FEATURES]

    # y（fill_rate）
    y_train = train["y"] / train["capa"]
    y_valid = valid["y"] / valid["capa"]

    # ============================

    # 4. 学習
    model = train_model(X_train, y_train)

    # 5. 評価（train）
    train_pred = predict(model, X_train)

    # 🔥 元スケールに戻す
    train_pred_actual = train_pred * train["capa"]

    print("Train RMSE")
    evaluate_model(train["y"], train_pred_actual)

    # 6. 評価（valid）
    valid_pred = predict(model, X_valid)

    # 🔥 元スケールに戻す
    valid_pred_actual = valid_pred * valid["capa"]

    print("Valid RMSE")
    evaluate_model(valid["y"], valid_pred_actual)

    print("Training finished.")

    # 7. 保存
    valid_result = valid[["gameday", "match_pair"]].copy()
    valid_result["actual"] = valid["y"].values
    valid_result["pred"] = valid_pred_actual
    valid_result.to_csv(os.path.join(data_path, "valid_result.csv"), index=False)


if __name__ == "__main__":
    main()