import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.feature_extraction import DictVectorizer
import math
from datetime import datetime

def readBusiness(filename):
    return  pd.read_csv(filename, header = 0, index_col = "business_id")

def readCheckin(filename):
    return readBusiness(filename)

def readReview(filename):
    return pd.read_csv(filename, header = 0, index_col = "review_id")

def readUser(filename):
     return pd.read_csv(filename, header = 0, index_col = "user_id")

def readTextFeatures(filename):
    return pd.read_csv(filename, header = 0, index_col = "review_id")

def combineTestTrain(train, test):
    data = train.combine_first(test)
    return data.drop_duplicates()

def processCheckin(checkin, index):
    checkin = checkin.fillna(0)
    checkin = checkin.drop("type", axis = 1)
    # new feature: number of checkins 
    checkin['nCheckins'] = checkin.apply(sum, axis = 1)
    checkin = checkin[["nCheckins"]]
    checkin = checkin.drop_duplicates()
    checkin = checkin.reindex(index)
    checkin = checkin.fillna(0)
    return checkin

def getBusinessFeatures(business, checkin):
    # combine business and checkin information
    business = business.combine_first(checkin)
    ## find cutoff take biggest 10 or 20 categories
    cat_list = business.categories.fillna("").map(lambda x: str.split(x, ",")).values
    freq_cats = pd.Series([category for categories in cat_list for category in categories]).value_counts().ix[1:40].index
    dic_vec = DictVectorizer()
    def cats_to_dict(cats):
        result_dict = {}
        for x in cats:
            if x in freq_cats:
                result_dict.setdefault(x, 1)  
            return result_dict
    category_features = dic_vec.fit_transform(business.categories.fillna("").map(lambda x: cats_to_dict(str.split(x, ","))).values).toarray()
    cat_panda = pd.DataFrame(category_features, index = business.index, columns = dic_vec.feature_names_)
    business = business.combine_first(cat_panda)
    ## write a Feature Encoder
    business = business.drop(["categories", "full_address", "neighborhoods", "name", "state", "type"], axis = 1)
    city_freq = dict(business.city.value_counts())
    frequent_cities = dict( (k, v) for k, v in city_freq.items() if v > 100).keys()
    business["city"] = business.city.map(lambda x: x if x in frequent_cities else "Other")
    return business

def getUserFeatures(user):
    ## rename votes_useful
    user["votes_useful_user"] = user.votes_useful
    user = user.drop("votes_useful", axis = 1)
    ## add feature
    user["log_review_count"] = user.review_count.map(math.log)
    user["votes_useful_ave"] = user.votes_useful_user / (user.review_count + 1)
    user["votes_useful_ave_log"] = np.log(user.votes_useful_ave + 1)
    ## drop unused
    user = user.drop(["name", "type"], axis = 1)
    return user 
    

def imputeMedian(column):
    return column.fillna(column.median())

def processReviews(reviews, business, user, text_features, cutoff_date):
    ## some date features
    reviews["date"] = reviews.date.map(pd.to_datetime)
    reviews = reviews.ix[reviews.date < cutoff_date].copy()
    reviews['month'] = reviews.date.map(lambda x: x.month)
    reviews['year'] = reviews.date.map(lambda x: x.year)
    reviews['weekday'] = reviews.date.map(lambda x: x.weekday())
    reviews['time_offset'] = reviews.date.map(lambda x: (cutoff_date - x).days)
    reviews['log_time_offset'] = np.log(reviews.time_offset)
    ## drop the text
    reviews = reviews.drop(["text", "type"], axis = 1)
    ## merge with user and business
    reviews = reviews.reset_index()
    res = pd.merge(reviews, business.reset_index(), on = "business_id", how = "left", suffixes = ["_rev", "_biz"])
    res = pd.merge(res, user.reset_index(), on = "user_id", how = "left", suffixes = ["_rev", "_user"])
    res = res.drop(["user_id", "business_id"], axis = 1)
    res = res.set_index("review_id")
    ## users with private profile
    res["private_profile"] = res.private_profile.fillna(2)
    ## add some features
    res["user_rev_stars_diff"] = res.average_stars - res.stars_rev
    res["user_biz_stars_diff"] =  res.average_stars - res.stars_biz 
    res["rev_biz_stars_diff"] = res.stars_rev - res.stars_biz
    res = res.combine_first(text_features)
    return(res)

def main():
    data_path = "./data/"
    test_path = data_path + "test/"
    train_path = data_path + "train/"
    print("Reading files: ")
    print("  business")
    businesses_train = readBusiness(train_path + "yelp_academic_dataset_business.csv")
    businesses_test = readBusiness(test_path + "yelp_test_set_business.csv")
    business_raw = combineTestTrain(businesses_train, businesses_test)
    print("  checkins")
    checkins_train = readCheckin(train_path + "yelp_academic_dataset_checkin.csv")
    checkins_test = readCheckin(test_path + "yelp_test_set_checkin.csv")
    checkin = combineTestTrain(checkins_train, checkins_test)
    checkin = processCheckin(checkin, business_raw.index)
    # TODO handle categories
    # TODO handle cities
    business = getBusinessFeatures(business_raw, checkin)
    print("  reviews")
    print("      train")
    reviews_train = readReview(train_path + "yelp_academic_dataset_review.csv")
    reviews_train = reviews_train.drop(["votes_cool", "votes_funny"], axis = 1)
    print("      test")
    reviews_test = readReview(test_path + "yelp_test_set_review.csv")
    print("  users")
    users_train = readUser(train_path + "yelp_academic_dataset_user.csv")
    users_train["private_profile"] = pd.Series(np.zeros(users_train.shape[0]), index = users_train.index)
    users_test = readUser(test_path + "yelp_test_set_user.csv")
    users_test["private_profile"] = pd.Series(np.ones(users_test.shape[0]), index = users_test.index)
    ## merge users
    user = combineTestTrain(users_train, users_test)
    user = getUserFeatures(user)
    print("getting text features")
    textFeaturesTrain  = readTextFeatures("./data/train/features-text-train.csv")
    textFeaturesTest  = readTextFeatures("./data/test/features-text-test.csv")
    print("handling reviews")
    print("   test")
    featuresTest = processReviews(reviews_test, business, user, textFeaturesTest, datetime(2013, 3, 12))
    assert(featuresTest.index.size == 22956)
    print("   train")
    features = processReviews(reviews_train, business, user, textFeaturesTrain, datetime(2013, 1, 19))
    assert(features.index.size == 229907)
    inTrain = processReviews(reviews_train, business, user, textFeaturesTrain, datetime(2012, 5, 1))
    inTest_indices = features.index.diff(inTrain.index)
    inTest = features.copy().ix[inTest_indices]
    inTest = inTest.reset_index().rename(columns = {"index": "review_id"})
    assert(inTrain.index.size + inTest.index.size == features.index.size)
    print("writing files")
    features.to_csv("./data/train/features-train.csv")
    featuresTest.to_csv("./data/test/features-test.csv")
    inTrain.to_csv("./data/train/features-inTrain.csv")
    inTest.to_csv("./data/train/features-inTest.csv", index = False)


if __name__ == "__main__":
    main()
