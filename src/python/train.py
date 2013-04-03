from sklearn.ensemble import RandomForestClassifier
import pandas as pd



features = pd.read_csv("./data/train/train.csv")

classifier = RandomForestClassifier(
    n_estimator = 100,
    max_features = None,
    verbose = 2,
    compute_importance = True,
    )

