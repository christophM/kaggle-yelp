################################################################################
##
##  Explore the data
##
################################################################################
library("ggplot2")

load("../../data/raw.RData")

## no dependent on the rating itself:
ggplot(reviews) + geom_boxplot(aes(x = stars, y = log(votes_useful), group = stars)) 
