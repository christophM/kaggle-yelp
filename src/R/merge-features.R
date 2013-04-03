################################################################################
##
## Merge all data sets and add some features
##
################################################################################
load("./data/rdata/review-features.RData")
load("./data/rdata/users-features.RData")
load("./data/rdata/business-features.RData")


#dat <- merge(reviews, businesses, by = "business_id", all.x = TRUE, suffixes = c(".rev", ".busy"))
#rm(reviews, businesses)
#gc()
dat <- reviews
dat <- merge(dat, users, by = "user_id", all.x = TRUE, suffixes = c(".rev", ".user"))

#dat$user_busy_diff <- dat$average_stars -
#  names(dat)
  

save(dat, file = "./data/rdata/full-featured.RData")
