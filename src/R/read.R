################################################################################
##
## Read and clean data
##
################################################################################

## read

train_path <- "./data/train/"
businessesTrain <- read.csv(paste(train_path, "yelp_academic_dataset_business.csv", sep = ""))
reviewsTrain <- read.csv(paste(train_path, "yelp_academic_dataset_review.csv", sep = ""))
usersTrain <- read.csv(paste(train_path, "yelp_academic_dataset_user.csv", sep =""))
checkinsTrain <- read.csv(paste(train_path, "yelp_academic_dataset_checkin.csv", sep =""))


 
test_path <- "./data/test/"
businessesTest <- read.csv(paste(test_path, "yelp_test_set_business.csv", sep = ""))
reviewsTest <- read.csv(paste(test_path, "yelp_test_set_review.csv", sep = ""))
usersTest <- read.csv(paste(test_path, "yelp_test_set_user.csv", sep =""))

## remove duplicates
usersTest <- usersTest[-which(duplicated(usersTest)), ]


checkinsTest <- read.csv(paste(test_path, "yelp_test_set_checkin.csv", sep =""))


usersTest$votes_useful <- usersTest$votes_funny <- usersTest$votes_cool <- NA
reviewsTest$votes_useful <- reviewsTest$votes_funny <- reviewsTest$votes_cool <- NA

businesses <- rbind(businessesTrain, businessesTest)
reviews <- rbind(reviewsTrain, reviewsTest)
users <- rbind(usersTrain, usersTest)
checkins <- rbind(checkinsTrain, checkinsTest)

businesses$type <- reviews$type <- users$type <- checkins$type <- NULL


save(businesses, reviews, users, checkins, file = "./data/rdata/raw.RData")

