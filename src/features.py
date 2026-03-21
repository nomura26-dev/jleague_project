import pandas as pd
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)
pd.set_option("display.max_colwidth", 50)

def create_features(df, reference_df=None):

    df = df.copy()

    #------------------------------------------
    # 日付処理（変更なし）
    #------------------------------------------
    df["gameday"] = df["year"].astype(str) + "/" + df["gameday"].str.replace(r"\(.*?\)", "", regex=True)
    df["gameday"] = pd.to_datetime(df["gameday"], errors="coerce")
    df["month"] = df["gameday"].dt.month
    df["weekday_num"] = df["gameday"].dt.weekday
    df["is_weekend"] = (df["weekday_num"] >= 5).astype(int)

    #------------------------------------------
    # 天気（変更なし）
    #------------------------------------------
    df["is_rain"] = df["weather"].str.contains("雨", na=False).astype(int)
    df["is_sunny"] = df["weather"].str.contains("晴", na=False).astype(int)

    #==========================================
    # ★ここから分岐（重要）
    #==========================================

    #-------------------------
    # train用（今までと同じ）
    #-------------------------
    if reference_df is None:

        df = df.sort_values(["home", "gameday"])
        df["home_mean_rolling5"] = (
            df.groupby("home")["y"]
            .transform(lambda x: x.shift().rolling(5, min_periods=1).mean())
        )

        df = df.sort_values(["away", "gameday"])
        df["away_mean_rolling5"] = (
            df.groupby("away")["y"]
            .transform(lambda x: x.shift().rolling(5, min_periods=1).mean())
        )

        df["match_pair"] = df["home"] + "_" + df["away"]
        df = df.sort_values(["match_pair", "gameday"])
        df["pair_mean_rolling3"] = (
            df.groupby("match_pair")["y"]
            .transform(lambda x: x.shift().rolling(3, min_periods=1).mean())
        )

        df["fill_rate"] = df["y"] / df["capa"]

    #-------------------------
    # valid/test用（🔥ここが違う）
    #-------------------------

    else:

        ref = reference_df.copy()

        combined = pd.concat([ref, df])

        # --- home ---
        combined = combined.sort_values(["home", "gameday"])
        combined["home_mean_rolling5"] = (
            combined.groupby("home")["y"]
            .transform(lambda x: x.shift().rolling(5, min_periods=1).mean())
        )

        df["home_mean_rolling5"] = combined.loc[df.index, "home_mean_rolling5"]

        # --- away ---
        combined = combined.sort_values(["away", "gameday"])
        combined["away_mean_rolling5"] = (
            combined.groupby("away")["y"]
            .transform(lambda x: x.shift().rolling(5, min_periods=1).mean())
        )

        df["away_mean_rolling5"] = combined.loc[df.index, "away_mean_rolling5"]

        # --- pair ---
        combined["match_pair"] = combined["home"] + "_" + combined["away"]

        combined = combined.sort_values(["match_pair", "gameday"])
        combined["pair_mean_rolling3"] = (
            combined.groupby("match_pair")["y"]
            .transform(lambda x: x.shift().rolling(3, min_periods=1).mean())
        )

        df["match_pair"] = df["home"] + "_" + df["away"]
        df["pair_mean_rolling3"] = combined.loc[df.index, "pair_mean_rolling3"]


        # --- fill_rate ---
        fill_rate_mean = (ref["y"] / ref["capa"]).mean()
        df["fill_rate"] = fill_rate_mean
    return df