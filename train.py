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
    preds_proba = model.predict_proba(X_test)[:, 1]

    submission_df = pd.DataFrame({
        "id": test_df["id"].values,
        "PitNextLap": preds.astype(int),
        "PitNextLap_Prob": preds_proba,
    })

    return submission_df
# EVOLVE-BLOCK-END


def main():
    import os
    import sys
    import time
    import pandas as pd
    import numpy as np
    import threading
    from pathlib import Path
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_auc_score

    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)

    def _timeout_exit():
        print("Training hit the 5-minute limit; exiting.", flush=True)
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(124)

    timer = threading.Timer(300, _timeout_exit)
    timer.daemon = True
    timer.start()

    t_total_start = time.time()

    try:
        data_dir = Path("data")
        if not data_dir.exists():
            data_dir = Path("data")

        train_df = pd.read_csv(data_dir / "train.csv")
        test_df = pd.read_csv(data_dir / "test.csv")

        train_sub, val_df = train_test_split(
            train_df, test_size=0.2, random_state=42, stratify=train_df["PitNextLap"]
        )

        t_train_start = time.time()
        preds_df = your_function(train_sub, val_df)
        t_train_end = time.time()

        auc = roc_auc_score(val_df["PitNextLap"].values, preds_df["PitNextLap_Prob"].values)

        peak_vram = 0.0
        try:
            import subprocess
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5
            )
            lines = [l.strip() for l in result.stdout.strip().split("\n") if l.strip()]
            if lines:
                peak_vram = max(float(l) for l in lines)
        except:
            pass

        training_seconds = t_train_end - t_train_start
        total_seconds = time.time() - t_total_start

        print("---")
        print(f"AUC:          {auc:.6f}")
        print(f"training_seconds: {training_seconds:.1f}")
        print(f"total_seconds:    {total_seconds:.1f}")
        print(f"peak_vram_mb:     {peak_vram:.1f}")

    finally:
        timer.cancel()


if __name__ == "__main__":
    main()