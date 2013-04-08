from sklearn.ensemble import RandomForestRegressor
from sklearn.externals import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectPercentile, f_regression


print("reading data")
data = pd.read_csv("./data/train/features-train.csv")
features = data.drop(["votes_useful", "city", "date"], axis = 1)
features = features.set_index("review_id")
target = data.votes_useful.map(lambda x: np.log(x + 1))

selector = SelectPercentile(f_regression, percentile=30)
selector.fit(features, target)
reduced_features = selector.transform(features)


np.random.seed(42)
model = RandomForestRegressor(compute_importances = True, oob_score = True, verbose = 1, n_jobs = 2, n_estimators = 100, max_features = "sqrt")
print("fitting model")
model.fit(reduced_features, target)

importances = model.feature_importances_
indices = np.argsort(importances)[::-1]
print("-------------------------------------------------------------------------")
print("Score on training data: " + str(model.score(reduced_features, target)))
print("Feature ranking:")
#for f in xrange(10):
#    print "%d. feature %s (%f)" % (f + 1, reduced_features.columns[f], importances[indices[f]])
print("-------------------------------------------------------------------------")
print("writing model to disk")
# save random forest    
filename = "./models/2013-04-08-rf_regressor.joblib.pkl"
_ = joblib.dump(model, filename, compress = 0)
selectorfile =  "./models/2013-04-08-selector.joblib.pkl"
_ = joblib.dump(selector,selectorfile, compress = 0)
