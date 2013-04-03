################################################################################
##
## Build review related features
##
################################################################################

load("./data/rdata/raw.RData")

reviews$date <- as.Date(reviews$date)




reviews$month <- format(reviews$date, "%m")
reviews$year <- format(reviews$date, "%Y")


reviews$text <- NULL

save(reviews, file = "./data/review-features.RData")
