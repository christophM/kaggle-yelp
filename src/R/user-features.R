################################################################################
##
## Build some user specific features
##
################################################################################
load("./data/rdata/raw.RData")
  

# not using right now
users$name <- NULL

## average of votes
users$user_ave_votes_cool <- users$votes_cool / users$review_count
users$user_ave_votes_funny <- users$votes_funny / users$review_count
users$user_ave_votes_useful <- users$votes_useful / users$review_count




users$log_review_count <- log(users$review_count)


save(users, file = "./data/rdata/users-features.RData")

