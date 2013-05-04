################################################################################
##
## Just plotting around
##
################################################################################
library("ggplot2")
dat <- read.csv("./data/train/features-train.csv")
names(dat)
dat$log_y <- log(dat$votes_useful + 1)

p <- ggplot(dat, aes(y = log_y))

## exclamation marks are bad
p + geom_point(aes(x = count_exmarks))

## interesting: 
p + geom_point(aes(x = average_stars), position = position_jitter())

## location:
p + geom_point(aes(x = longitude, y = latitude, colour = log_y), size = 1.8)

## influence of time
p + geom_point(aes(x = log(time_offset)))
cor(dat$log_y, log(dat$time_offset))
