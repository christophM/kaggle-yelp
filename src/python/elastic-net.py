import numpy as np
from sklearn.linear_model import ElasticNetCV
import pandas as pd
from loss import rmsle

            
print("reading data")
data = pd.read_csv("./data/train/features-train.csv")
features = data.drop(["votes_useful", "city", "date"], axis = 1)
features = features.set_index("review_id")
target = data.votes_useful.map(lambda x: np.log(x + 1))
np.random.seed(42)
model = ElasticNetCV(l1_ratio = 1.0, verbose = 1, n_jobs = 2)
print("fitting model")
model.fit(features, target)
prediction = model.predict(test_features.drop([ "date", "city"], axis = 1))
print(rmsle(target, model.predict(features)))

test_features = test_features["prediction"]
test_features.to_csv("./submissions/2013-04-06-text-features.csv")