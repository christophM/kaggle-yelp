from sklearn.ensemble import GradientBoostingRegressor
from sklearn.externals import joblib
import pandas as pd
import numpy as np
import pylab as pl
import matplotlib.pyplot as plt
from loss import rmsle

def readReviews(filename):
    data = pd.read_csv(filename)
    features = data.drop(["votes_useful", "city", "date"], axis = 1).set_index("review_id")
    target = data.votes_useful.map(lambda x: np.log(x + 1))
    return target, features

################################################################################
print("reading data")
target, features = readReviews("./data/train/features-train.csv")
target_inTrain, features_inTrain = readReviews("./data/train/features-inTrain.csv")
target_inTest, features_inTest = readReviews("./data/train/features-inTest.csv")

################################################################################
np.random.seed(42)
params = {'n_estimators': 100, 'max_depth': 5, 'min_samples_split': 2, 'loss': 'ls', 'verbose': 2}
model = GradientBoostingRegressor(**params)
print("fitting model on " + str(len(features.columns)) + " features")
model.fit(features_inTrain, target_inTrain)
predicted = np.exp(model.predict(features_inTest)) - 1
actual = target_inTest
print("Rmsle score: " + str(rmsle(actual, predicted)))

################################################################################

# Plot training deviance

# compute test set deviance
test_score = np.zeros((params['n_estimators'],), dtype=np.float64)

for i, y_pred in enumerate(model.staged_decision_function(features_inTest)):
    test_score[i] = model.loss_(target_inTest, y_pred)

pl.figure(figsize=(12, 6))
pl.subplot(1, 2, 1)
pl.title('Deviance')
pl.plot(np.arange(params['n_estimators']) + 1, model.train_score_, 'b-',
        label='Training Set Deviance')
pl.plot(np.arange(params['n_estimators']) + 1, test_score, 'r-',
        label='Test Set Deviance')
pl.legend(loc='upper right')
pl.xlabel('Boosting Iterations')
pl.ylabel('Deviance')
###############################################################################
# Plot feature importance
feature_importance = model.feature_importances_
# make importances relative to max importance
feature_importance = 100.0 * (feature_importance / feature_importance.max())
sorted_idx = np.argsort(feature_importance)
pos = np.arange(sorted_idx.shape[0]) + .5
pl.subplot(1, 2, 2)
pl.barh(pos, feature_importance[sorted_idx], align='center')
pl.yticks(pos, features_inTest.columns[sorted_idx])
pl.xlabel('Relative Importance')
pl.title('Variable Importance')
pl.show()
################################################################################
print("-------------------------------------------------------------------------")
print("Refitting model with whole training data")
model.fit(features, target)
print("writing model to disk")
#save random forest
filename = "./models/gbm_regressor.joblib.pkl"
_ = joblib.dump(model, filename, compress = 0)


