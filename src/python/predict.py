from sklearn.externals import joblib
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import numpy as np
from loss import rmsle


print("loading data")
## load test data
test_features = pd.read_csv("./data/test/features-test.csv")
test_features = test_features.set_index("review_id")

print("loading model")
## load model
filename = "./models/rf_regressor.joblib.pkl"
model = joblib.load(filename)

print("predicting...")
## make prediction
prediction = model.predict(test_features.drop([ "date", "city"], axis = 1))

test_features["prediction"] = np.exp(prediction) -1

print("writing results to disk")
test_features = test_features["prediction"]
test_features.to_csv("./submissions/2013-05-12-text-features.csv")
