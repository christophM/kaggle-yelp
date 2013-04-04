from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



data = pd.read_csv("./data/train/features-train.csv")
features = data.drop(["votes_useful", "city", "date"], axis = 1)
features = features.set_index("review_id")
target = data.votes_useful
print(features.describe())
## TODO predict log of target
model = RandomForestRegressor(compute_importances = True, oob_score = True, verbose = 3, n_jobs = -1, n_estimators = 60)

## TODO set seed

model.fit(features, target)

importances = model.feature_importances_
indices = np.argsort(importances)[::-1]

print("feature ranking:")

for f in xrange(10):
    print "%d. feature %s (%f)" % (f + 1, features.columns[f], importances[indices[f]])

#plt.scatter(data.votes_useful, data.stars_rev)
#plt.show()
# TODO save random forest and put prediction in extra file

## predict


test_features = pd.read_csv("./data/test/features-test.csv")
test_features = test_features.set_index("review_id")
prediction = model.predict(test_features.drop([ "date", "city"], axis = 1))

test_features["prediction"] = prediction

test_features = test_features["prediction"]
test_features.to_csv("./submissions/2013-04-02-versuch2.csv")
