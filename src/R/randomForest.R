################################################################################
##
## Try a simple random Forest
##
################################################################################
load("./data/rdata/full-featured.RData")
library("randomForest")


## impute missing values
datImp <- apply(dat[)], 2, function(x) {
  if(is.numeric(x)) {
    m <- median(x)
    x[is.na(x)] <- m
  }
  x
})



test <- datImp[which(is.na(datImp$votes_useful.rev)), ]
train <- datImp[- which(is.na(datImp$votes_useful.rev)), ]

names(dat)

train <- na.omit(train)[1:10000, ]
y = train$votes_useful.rev
x = train[-which(names(train) %in% c("business_id", "user_id", "review_id", "votes_funny.rev", "votes_cool.rev", "votes_useful.rev"))]
rf <- randomForest(y = log(y +1), x = x, do.trace = TRUE, ntree = 100)
plot(rf)
varImpPlot(rf)



ggplot(train) + geom_point(aes(y = votes_useful.rev, x = date))


predict(rf, test)
