################################################################################
##
## Build review related features
##
################################################################################

load("./data/rdata/raw.RData")

reviews$date <- as.Date(reviews$date)




reviews$month <- as.numeric(format(reviews$date, "%m"))
reviews$year <- as.numeric(format(reviews$date, "%Y"))


reviews$text <- NULL

save(reviews, file = "./data/rdata/review-features.RData")
