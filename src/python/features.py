import pandas as pd
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
    ## drop unused
    user = user.drop(["name", "type"], axis = 1)
    user = user.drop_duplicates()
    return user 
    

def imputeMedian(column):
    return column.fillna(column.median())

def processReviews(reviews, business, user):
    ## some date features
    reviews.date = reviews.date.map(pd.to_datetime)
    reviews['month'] = reviews.date.map(lambda x: x.month)
    reviews['year'] = reviews.date.map(lambda x: x.year)
    reviews['weekday'] = reviews.date.map(lambda x: x.weekday())
    reviews['date_numeric_days'] = reviews.date.map(lambda x: (x - datetime(1970, 1, 1)).days)
    ## drop the text
    reviews = reviews.drop(["text", "type"], axis = 1)
    ## merge with user and business
    reviews = reviews.reset_index()
    res = pd.merge(reviews, business.reset_index(), on = "business_id", how = "left", suffixes = ["_rev", "_biz"])
    res = pd.merge(res, user.reset_index(), on = "user_id", how = "left", suffixes = ["_rev", "_user"])
    res = res.drop(["user_id", "business_id"], axis = 1)
    res = res.set_index("review_id")
    ## impute missing values with median
    nulls = ["average_stars", "review_count_user", "votes_cool",
             "votes_funny", "votes_useful_user", "log_review_count",
             "votes_useful_ave"]
    res[nulls] = res[nulls].apply(imputeMedian)
    ## add some features
    res["user_rev_stars_diff"] = res.average_stars - res.stars_rev
    res["user_biz_stars_diff"] =  res.average_stars - res.stars_biz 
    res["rev_biz_stars_diff"] = res.stars_rev - res.stars_biz
    return(res)

def main():
    data_path = "./data/"
    test_path = data_path + "test/"
    train_path = data_path + "train/"
    print("Reading files: ")
    print("  business")
    businesses_train = readBusiness(train_path + "yelp_academic_dataset_business.csv")
    businesses_test = readBusiness(test_path + "yelp_test_set_business.csv")
    business = combineTestTrain(businesses_train, businesses_test)
    
    print("  checkins")
    checkins_train = readCheckin(train_path + "yelp_academic_dataset_checkin.csv")
    checkins_test = readCheckin(test_path + "yelp_test_set_checkin.csv")
    checkin = combineTestTrain(checkins_train, checkins_test)
    checkin = processCheckin(checkin, business.index)

    # TODO handle categories
    # TODO handle cities
    business = getBusinessFeatures(business, checkin)

    print("  reviews")
    print("      train")
    reviews_train = readReview(train_path + "yelp_academic_dataset_review.csv")
    reviews_train = reviews_train.drop(["votes_cool", "votes_funny"], axis = 1)
    print("      test")
    reviews_test = readReview(test_path + "yelp_test_set_review.csv")
    print("  users")
    users_train = readUser(train_path + "yelp_academic_dataset_user.csv")
    users_test = readUser(test_path + "yelp_test_set_user.csv")
    ## merge users
    user = combineTestTrain(users_train, users_test)
    user = getUserFeatures(user)

    print("handling reviews")
    print("   test")
    featuresTest = processReviews(reviews_test, business, user)
    print("   train")
    features = processReviews(reviews_train, business, user)
    features
    
    print("writing files")
    features.to_csv("./data/train/features-train.csv")
    featuresTest.to_csv("./data/test/features-test.csv")


if __name__ == "__main__":
    main()
