.mode csv
.header on
.out ./data/test/test.csv


SELECT * 
FROM  (reviewTest AS rev
LEFT JOIN userTest AS use
ON rev.user_id == use.user_id)
LEFT JOIN  businessTest  AS bus
ON rev.business_id == bus.business_id;



.out ./data/train/train.csv



SELECT * 
FROM  (reviewTrain AS rev
LEFT JOIN userTrain AS use
ON rev.user_id == use.user_id)
LEFT JOIN  businessTrain  AS bus
ON rev.business_id == bus.business_id;


.exit