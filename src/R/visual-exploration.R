################################################################################
##
## Just plotting around
##
################################################################################
library("ggplot2")
dat <- read.csv("./data/train/features-train.csv")
names(dat)
dat$log_y <- log(dat$votes_useful + 1)
dat$review_id <- NULL
dat$date <- NULL

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

## ARI Readability index
p + geom_point(aes(x = textlength))
lm(log_y ~ log(ARI_Readability), data = dat)

## votes_useful_ave
p + geom_point(aes(x = log(votes_useful_ave), colour = textlength))
summary(lm(log_y ~ log(votes_useful_ave +1) + textlength , data = dat))

dat$votes_useful_ave_log <- log(dat$votes_useful_ave + 1)
## a tree 
library("party")
sub = dat[sample(seq(from=1, to=nrow(dat)), size = 10000), ]
sub = sub[-which(names(sub) == "votes_useful")]
tree = ctree(log_y ~ . , data = sub, control = ctree_control(mincriterion = 0.99999999))
plot(tree)
