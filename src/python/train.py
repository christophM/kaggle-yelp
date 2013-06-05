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
    target = pd.Series(np.log(data.votes_useful + 1).values, index = features.index)
    return target, features

def eval_model(model, train, train_target, test, test_target, plot = True):
    np.random.seed(42)
    print("fitting model on " + str(len(train.columns)) + " features")
    model.fit(train, train_target)
    predicted = np.exp(model.predict(test)) - 1
    actual = test_target
    print("Rmsle score: " + str(rmsle(actual, predicted)))
    # Plot training deviance
    # compute test set deviance
    if plot:
        test_score = np.zeros((params['n_estimators'],), dtype=np.float64)
        for i, y_pred in enumerate(model.staged_decision_function(test)):
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

def plot_feature_importance(model, train):
        # Plot feature importance
        feature_importance = model.feature_importances_
        # make importances relative to max importance
        feature_importance = 100.0 * (feature_importance / feature_importance.max())
        sorted_idx = np.argsort(feature_importance)
        pos = np.arange(sorted_idx.shape[0]) + .5
        pl.subplot(1, 2, 2)
        pl.barh(pos, feature_importance[sorted_idx], align='center')
        pl.yticks(pos, train.columns[sorted_idx])
        pl.xlabel('Relative Importance')
        pl.title('Variable Importance')
        pl.show()

def fit_and_save(model, featurez, target, filename, plot = True):
    print "fitting " + filename
    model.fit(featurez, target)
    if (plot):
        plot_feature_importance(model, featurez)
    model_path = "./models/"
    print "writing " + filename + " to disk"
    _ = joblib.dump(model, model_path + filename, compress = 0)
 
################################################################################
print("reading data")
target, features = readReviews("./data/train/features-train.csv")
target_inTrain, features_inTrain = readReviews("./data/train/features-inTrain.csv")
target_inTest, features_inTest = readReviews("./data/train/features-inTest.csv")

################################################################################
user_full_features = np.array(['average_stars','review_count_user',  'votes_cool', 'votes_funny', 'votes_useful_user', 'log_review_count', 'votes_useful_ave', 'votes_useful_ave_log', 'user_biz_stars_diff', 'user_rev_stars_diff'])
user_partial_features = np.array(['average_stars', 'review_count_user', 'log_review_count', 'user_biz_stars_diff', 'user_rev_stars_diff'])
user_partial_drop = np.setdiff1d(user_full_features, user_partial_features)

params = {'n_estimators': 50, 'max_depth': 5, 'min_samples_split': 2, 'loss': 'ls', 'verbose': 2, 'subsample': 0.9}
model = GradientBoostingRegressor(**params)

## model1: data with all user information
## subset: user_privacy = 1
## model2: data with partial user information
## subset: user_privacy = 1 + throw away some features
## model3: data with no user information
## subset: user_privacy = 3 + throw away all user features
#data_drop = [user_partial_drop, np.array([]), user_full_features]
#for drop_features in data_drop:
#    all_info_index = features_inTrain["private_profile"] == 2
#    all_info_index2 = features_inTest["private_profile"] == 2
#    train_features = features_inTrain[all_info_index].drop(drop_features, axis = 1).dropna(axis = 0)
#    test_features = features_inTest[all_info_index2].drop(drop_features, axis = 1).dropna(axis = 0)
#    eval_model(model, train_features, target_inTrain[all_info_index], test_features, target_inTest[all_info_index2], False)


################################################################################
print("-------------------------------------------------------------------------")
print("Refitting model with whole training data")
all_info_index = features["private_profile"] == 0
fit_and_save(model, features[all_info_index] , target[all_info_index], "model1.joblib.pkl")
fit_and_save(model, features[all_info_index].drop(user_partial_drop, axis = 1), target[all_info_index], "model2.joblib.pkl")
fit_and_save(model, features.drop(user_full_features, axis =   1), target, "model3.joblib.pkl")
             


