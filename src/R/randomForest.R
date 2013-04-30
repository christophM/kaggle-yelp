################################################################################
##
## Try a simple random Forest
##
################################################################################

library("randomForest")
train  <- read.csv("./data/train/features-train.csv")
test <- read.csv("./data/test/features-test.csv")
names(train)


y = log(train$votes_useful + 1)
x = train[-which(names(train) %in% c("votes_useful", "review_id", "date"))]
set.seed(42)
rf <- randomForest(y = log(y +1), x = x, do.trace = TRUE, ntree = 500)
save(rf, file = "./models/randomForestR-1-5-2013.RData")
print(x$mse)
plot(rf)
varImpPlot(rf)


predict(rf, test)

## prediction: dont forget to remove log
