################################################################################
##
## Try a simple random Forest
##
################################################################################

library("randomForest")
train  <- read.csv("./data/train/features-train.csv")
test <- read.csv("./data/test/features-test.csv")
names(train)


y = train$votes_useful
x = train[-which(names(train) %in% c("votes_useful", "review_id", "date"))]
rf <- randomForest(y = log(y +1), x = x, do.trace = TRUE, ntree = 500)
print(x$mse)
plot(rf)
varImpPlot(rf)


predict(rf, test)
