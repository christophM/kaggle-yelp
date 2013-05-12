from sklearn.ensemble import RandomForestRegressor
from sklearn.externals import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from loss import rmsle

def readReviews(filename):
    data = pd.read_csv(filename)
    features = data.drop(["votes_useful", "city", "date"], axis = 1).set_index("review_id")
    target = data.votes_useful.map(lambda x: np.log(x + 1))
    return target, features

     
print("reading data")
target, features = readReviews("./data/train/features-train.csv")
target_inTrain, features_inTrain = readReviews("./data/train/features-inTrain.csv")
target_inTest, features_inTest = readReviews("./data/train/features-inTest.csv")

np.random.seed(42)
model = RandomForestRegressor(compute_importances = True, oob_score = True, verbose = 2, n_jobs = 2, n_estimators = 50, max_features = "sqrt")
print("fitting model")
model.fit(features_inTrain, target_inTrain)
predicted = np.exp(model.predict(features_inTest)) - 1
actual = target_inTest
print("Rmsle score: " + str(rmsle(actual, predicted)))

## Print test set error
## Input the RandomForestRegressor, test set feature and test set known values
def rfErrCurve(rf_model,test_X,test_y):
    p = []
    for i,tree in enumerate(rf_model.estimators_):
                p.insert(i,tree.predict(test_X))
                print rmsle(np.mean(p,axis=0),test_y)
print("error curve: ")
print(rfErrCurve(model, features_inTest, actual))
print("-------------------------------------------------------------------------")
print("Refitting model with whole training data")
model.fit(features, target)
print("writing model to disk")
# save random forest
filename = "./models/rf_regressor.joblib.pkl"
_ = joblib.dump(model, filename, compress = 0)

importances = model.feature_importances_
indices = np.argsort(importances)[::-1]
print("-------------------------------------------------------------------------")
print("Score on training data: " + str(model.score(features, target)))
print("Feature ranking:")
for f in xrange(importances.size - 1):
    print "%d. feature %s (%f)" % (f + 1, features.columns[f], importances[indices[f]])