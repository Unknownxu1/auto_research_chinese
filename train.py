# EVOLVE-BLOCK-START
def your_function(train_df, test_df):
    import pandas as pd
    import numpy as np
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.preprocessing import OrdinalEncoder

    train_df = train_df.copy()
    test_df = test_df.copy()

    cat_cols = ["Driver", "Compound", "Race"]
    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    train_df[cat_cols] = enc.fit_transform(train_df[cat_cols].astype(str))
    test_df[cat_cols] = enc.transform(test_df[cat_cols].astype(str))

    train_df["PitStop"] = train_df["PitStop"].astype(int)
    test_df["PitStop"] = test_df["PitStop"].astype(int)

    feature_cols = [
        "Driver", "Compound", "Race", "Year", "PitStop",
        "LapNumber", "Stint", "TyreLife", "Position",
        "LapTime (s)", "LapTime_Delta", "Cumulative_Degradation",
        "RaceProgress", "Position_Change",
    ]

    X_train = train_df[feature_cols]
    y_train = train_df["PitNextLap"].astype(int)
    X_test = test_df[feature_cols]

    model = GradientBoostingClassifier(
        n_estimators=200,
        max_depth=5,
        min_samples_leaf=20,
        learning_rate=0.1,
        random_state=42,
    )

    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    submission_df = pd.DataFrame({
        "id": test_df["id"].values,
        "PitNextLap": preds.astype(int)
    })

    return submission_df
# EVOLVE-BLOCK-END


def _run():
    import pandas as pd
    train_df = pd.read_csv("train.csv")
    test_df = pd.read_csv("test.csv")
    submission_df = your_function(train_df, test_df)
    submission_df.to_csv("submission.csv", index=False)
    print("预测完成，结果已保存为 submission.csv")
    print(submission_df.head())
    print(submission_df["PitNextLap"].value_counts())


def main():
    import multiprocessing as mp
    p = mp.Process(target=_run)
    p.start()
    p.join(timeout=300)
    if p.is_alive():
        p.terminate()
        print("超时：运行超过5分钟，已终止")


if __name__ == "__main__":
    main()
