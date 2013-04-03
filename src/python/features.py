import pandas as pd
import pandas.io.sql as pd_sql
import math
from datetime import datetime
import sqlite3 as db
import os


data_path = "./data/"
test_path = data_path + "test/"
train_path = data_path + "train/"

print("Reading files: ")
print("  business")
businesses_train = pd.read_csv(train_path + "yelp_academic_dataset_business.csv", header = 0)
businesses_test = pd.read_csv(test_path + "yelp_test_set_business.csv", header = 0)

print("  checkins")
checkins_train = pd.read_csv(train_path + "yelp_academic_dataset_checkin.csv", header = 0)
checkins_test = pd.read_csv(test_path + "yelp_test_set_checkin.csv", header = 0)

print("  reviews")
print("      train")
reviews_train = pd.read_csv(train_path + "yelp_academic_dataset_review.csv", header = 0)
print("      test")
reviews_test = pd.read_csv(test_path + "yelp_test_set_review.csv", header = 0)

print("  users")
users_train = pd.read_csv(train_path + "yelp_academic_dataset_user.csv", header = 0)
users_test = pd.read_csv(test_path + "yelp_test_set_user.csv", header = 0)



# businesses_test =

def processBusinessess(business, checkin):
    ## delete unused columns
    delete_cols = ["categories", "full_address", "neighborhoods", "name", "state", "type"]
    business = business.drop(delete_cols, axis = 1)
    ## aggregate cities
    city_freq = dict(business.city.value_counts())
    frequent_cities = dict( (k, v) for k, v in city_freq.items() if v > 100).keys()
    business.city = business.city.map(lambda x: x if x in frequent_cities else "Other")
    ## aggregate checkins
    # drop type variable
    #checkin = checkin.drop("type", axis = 1)
    # replace NaN with 0 as reported in forum
    #checkin = checkin.fillna(0)
    #checkin['nCheckins'] = checkin.apply(lambda row: sum(row[2:]), axis = 1)
    ## drop checkin info
    #checkin = checkin[['business_id', 'nCheckins']]
    ## merge business and checkin data
    #return (pd.merge(business, checkin, how = "left", on = "business_id"))
    return(business)



def processReviews(reviews):
    ## some date features
    reviews.date = reviews.date.map(pd.to_datetime)
    reviews['month'] = reviews.date.map(lambda x: x.month)
    reviews['year'] = reviews.date.map(lambda x: x.year)
    reviews['weekday'] = reviews.date.map(lambda x: x.weekday())
    ## drop the text
    reviews = reviews.drop("text", axis = 1)
    return(reviews)

def processUsers(users):
    ## test users have no votes
    # users["user_ave_votes_cool"] = users.apply(lambda user: user.votes_cool / user.review_count, axis = 1)
    # users["user_ave_votes_funny"] = users.apply(lambda user: user.votes_funny / user.review_count, axis = 1)
    # cusers["user_ave_votes_usful"] = users.apply(lambda user: user.votes_useful / user.review_count, axis = 1)
    users = users.drop(["name", "type"], axis = 1)
    users["log_review_count"] = users.review_count.map(math.log)
    return(users)


print("handling businesses")
busyFeatTrain = processBusinessess(businesses_train, checkins_train)
busyFeatTest = processBusinessess(businesses_test, checkins_test)

print("handling reviews")
revFeatTest = processReviews(reviews_test)
revFeatTrain = processReviews(reviews_train)

print("handling users")
userFeatTest = processUsers(users_test)
userFeatTrain = processUsers(users_train)


print("creating databases")
os.remove("./data/yelp.db")
con = db.connect("./data/yelp.db")
pd_sql.write_frame(busyFeatTest, "businessTest", con)
pd_sql.write_frame(busyFeatTrain, "businessTrain", con)
pd_sql.write_frame(userFeatTrain, "userTrain", con)
pd_sql.write_frame(userFeatTest, "userTest", con)
pd_sql.write_frame(revFeatTest, "reviewTest", con)
pd_sql.write_frame(revFeatTrain, "reviewTrain", con)
print("merging")
os.system("sqlite3 -batch -init ./src/sql/mergeAll.sql ./data/yelp.db")

print("finished")

