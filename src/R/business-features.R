################################################################################
##
## Build business related features
##
################################################################################

load("./data/rdata/raw.RData")

businesses$categories <- NULL
businesses$full_address <- NULL
businesses$neighbourhoods <- NULL
businesses$name <- NULL
businesses$state <- NULL


# aggregate cities
tab <- table(businesses$city)
businesses$city <- as.character(businesses$city)
businesses$city[tab[businesses$city] < 100] <- "Other"
businesses$city <- as.factor(businesses$city)
summary(businesses$city)

checkins$n_checkins <- apply(checkins[-1], 1, function(x) sum(x, na.rm = TRUE))

checkins <- checkins[c("business_id", "n_checkins")]

cr = checkins["business_id"]
br = businesses["business_id"]
x <- merge(br, cr, all.x = TRUE, all.y = FALSE)
head(checkins)
head(businesses)
length(intersect(businesses$business_id, checkins$business_id)) / length(unique(checkins$business_id))
businesses2 <- merge(businesses, checkins, all.x = TRUE, by = "business_id")
nrow(businesses2) / nrow(businesses)


businesses <- businesses2
businesses$n_checkins[is.na(businesses$n_checkins)] <- 0

save(businesses, file = "./data/rdata/business-features.RData")
