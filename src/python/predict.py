from sklearn.externals import joblib
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import numpy as np
from loss import rmsle


print("loading data")
## load test data
test_features = pd.read_csv("./data/test/features-test.csv")
test_features = test_features.set_index("review_id")
test_features = test_features.drop([ "date", "city"], axis = 1)
print("loading model")
## load model
model_path  = "./models/"
model1 = joblib.load(model_path + "model1.joblib.pkl")
model2 = joblib.load(model_path + "model2.joblib.pkl")
model3 = joblib.load(model_path + "model3.joblib.pkl")

user_full_features = np.array(['average_stars','review_count_user',  'votes_cool', 'votes_funny', 'votes_useful_user', 'log_review_count', 'votes_useful_ave', 'votes_useful_ave_log', 'user_biz_stars_diff', 'user_rev_stars_diff'])
user_partial_features = np.array(['average_stars', 'review_count_user', 'log_review_count', 'user_biz_stars_diff', 'user_rev_stars_diff'])
user_partial_drop = np.setdiff1d(user_full_features, user_partial_features)

print("predicting...")
## make prediction
test_features1 = test_features[test_features["private_profile"] == 0]
test_features1["prediction"] = np.exp(model1.predict(test_features1)) -1

test_features2 = test_features[test_features["private_profile"] == 1].drop(user_partial_drop, axis = 1)
test_features2["prediction"] = np.exp(model2.predict(test_features2)) -1

test_features3 = test_features[test_features["private_profile"] == 2].drop(user_full_features, axis = 1)
test_features3["prediction"] = np.exp(model3.predict(test_features3)) -1

test_features = test_features1.append(test_features2).append(test_features3)

print("writing results to disk")
test_features = test_features["prediction"]
test_features.to_csv("./submissions/2013-05-25.csv")
