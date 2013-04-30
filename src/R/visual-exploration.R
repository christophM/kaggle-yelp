################################################################################
##
## Just plotting around
##
################################################################################
library("ggplot2")
dat <- read.csv("./data/train/features-train.csv")

dat$log_y <- log(dat$votes_useful)

p <- ggplot(dat, aes(y = log_y))

## exclamation marks are bad
p + geom_point(aes(x = count_exmarks))

## interesting: 
p + geom_point(aes(x = average_stars), position = position_jitter())

## location:
p + geom_point(aes(x = longitude, y = latitude, alpha = log_y))
