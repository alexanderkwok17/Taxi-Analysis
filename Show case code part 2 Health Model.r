# 1. Get Package -------------------------------------------------------------
library('magrittr')
library('partykit')
library('sqldf')
library('gbm')
library('ggraptR')
library('randomForest')
library('missForest')
library('glmnet')
library('ggplot2')
library('xgboost')
library('GGally')
library('plyr')
library('dplyr')
library('tidyr')
library('randomForest')
library('rpart')
library('caret')
library("ROCR")
library('lattice')
library('ISLR')
library('tree')
library('MASS')
library('mice')
library('dummies')
library('VIM')
library('RJDBC')
library('pROC')
library("ggraptR")
library('e1071')
library('xgboost')
library('GGally')
library("r2pmml")
library("pmml")
library("randomForest")
library("XML")
library('Ckmeans.1d.dp')
library('DiagrammeR')
getwd()
options(java.parameters = "-Xmx61g")

# 2. Link To Database ------------------
#special_file_path = "C:/USERDATA/nakk1/My Documents/special/secure"
special_file_path = "C:/Users/nmhgg/Downloads/secure"
getwd()
## UserID & PW for different system
special_v = readLines(special_file_path)
redshift = special_v[grepl("^REDSHIFT", special_v)]
redshift_user = sub("^.*=", "\\1", redshift[grepl("USER", redshift)])
redshift_pw =sub("^.*=", "\\1", redshift[grepl("PASSWORD", redshift)])

rm(special_v); rm(redshift)

driver <- JDBC("com.amazon.redshift.jdbc41.Driver", "RedshiftJDBC41-1.1.10.1010.jar", identifier.quote="`")
## Preprod
url <- paste0("jdbc:redshift://data-cap-preprod-cluster.wowmp.com.au:443/datacap?user=",redshift_user,"&password=",redshift_pw ,"?ssl=true?sslfactory=org.postgresql.ssl.NonValidatingFactory")
## Prod
#url <- paste0("jdbc:redshift://data-cap-prod-cluster.wowmp.com.au:443/datacap?user=",redshift_user,"&password=",redshift_pw ,"?ssl=true?sslfactory=org.postgresql.ssl.NonValidatingFactory")

conn <- dbConnect(driver, url) 

# 3. Source Data Base Import
# Health_model_part1_nodist_noprod ----
Health_model_part1_nodist_noprod <- dbGetQuery(conn,"SELECT * FROM  loyalty_modeling.noproduce_1")
Health_model_part2_nodist_noprod <- dbGetQuery(conn,"SELECT * FROM  loyalty_modeling.noproduce_2")
Health_model_part3_nodist_noprod <- dbGetQuery(conn,"SELECT * FROM  loyalty_modeling.noproduce_3")
Health_model_part4_nodist_noprod <- dbGetQuery(conn,"SELECT * FROM  loyalty_modeling.noproduce_4")
Health_model_part5_nodist_noprod <- dbGetQuery(conn,"SELECT * FROM  loyalty_modeling.noproduce_5")


# Store dataset ----
getwd()

# Consolidate all dataset to one -----
model_dataset <- Health_model_part1_nodist_noprod
Health_model_part2_nodist_noprod
model_dataset <- merge(model_dataset, Health_model_part2_nodist_noprod, by = 'crn')
model_dataset <- subset(model_dataset, select = -c(hsr_segment_noproduce.x,hsr_segment_noproduce.y) )
#head(names(model_dataset$hsr_segment_noproduce.))
model_dataset <- merge(model_dataset, Health_model_part3_nodist_noprod, by = 'crn')
model_dataset <- merge(model_dataset, Health_model_part4_nodist_noprod, by = 'crn')
model_dataset <- merge(model_dataset, Health_model_part5_nodist_noprod, by = 'crn')
rm(Health_model_part3_nodist_noprod,Health_model_part2_nodist_noprod)
# model on target segment.x -----
dim(model_dataset)
# further remove variables that are affecting the results ------
# names(model_dataset)[3940:3966]
# model_dataset <- model_dataset[,c(1:3939,3967:5689)]
# names(model_dataset)[1380:2870]
# names(model_dataset)[151:542]
# model_dataset <- model_dataset[,c(1:150,543:1379,2871:5662)]
# model_dataset <- subset(model_dataset, select=-c(fix_dsct_othr_wowd_rec))
# model_dataset <- subset(model_dataset, select=-c(fix_dsct_wowd_alt_earn_rt ))
# model_dataset <- subset(model_dataset, select=-c(fix_dsct_wowd_alt_s_amt ))


# model fitting -----

table(model_dataset$hsr_segment_noproduce)
model_dataset$hsr_segment_noproduce[model_dataset$hsr_segment_noproduce == 3] <- 1
model_dataset$hsr_segment_noproduce[model_dataset$hsr_segment_noproduce == 2] <- 0
# remove cat and sub cat spend ----
remove <- grep("ctgry", names(model_dataset), value=TRUE)
model_dataset <- model_dataset[, !(colnames(model_dataset) %in% remove)]

which( colnames(model_dataset)=="hsr_segment_noproduce" )

idcols <- c("crn", "hsr_segment_noproduce"); 
cols <- c(idcols, names(model_dataset)[-which(names(model_dataset) %in% idcols)]); 
model_dataset <- model_dataset[cols] 

h <- sample(nrow(model_dataset), dim(model_dataset)*0.7)
inTrain <- data.frame(model_dataset[h,-1])
inTest <- data.frame(model_dataset[-h,-1])
names(inTrain)[1:5]

feature.names <- names(model_dataset)[4:ncol(model_dataset)-1]
feature.names[1:10]

dtrain <- xgb.DMatrix(data.matrix(inTrain[,feature.names]), label = inTrain$hsr_segment_noproduce, missing = NA)

dval <- xgb.DMatrix(data.matrix(inTest[,feature.names]), label=inTest$hsr_segment_noproduce, missing = NA)

watchlist <- list(eval = dval, train = dtrain)

# QUick model to pick out a subset of varables ----

param <- list(  objective           = "binary:logitraw", 
                booster =  "gbtree",#"gblinear"
                eta                 = 0.001,
                max_depth           = 6,  # changed from default of 6
                subsample           = 0.6,
                colsample_bytree    = 0.6,
                eval_metric         = "auc",
 alpha = 0.0001, 
 lambda = 1
)

xgb_train <- xgb.train(   params              = param, 
                          data                = dtrain, 
                          nrounds             = 300, # changed from 300
                          verbose             = 2, 
                          #early.stop.round    = 10,
                          watchlist           = watchlist,
                          maximize            = TRUE)

# Compute feature importance matrix
importance_matrix <- xgb.importance(feature.names, model = xgb_train)
#importance_matrix <- xgb.importance(feature.names, model = xgb_train, filename_dump = "D:/Alexander/Alexander/Spend model/xgbdump.txt")
# Nice graph
important_mat<-data.frame(importance_matrix)
importance_matrix <- importance_matrix[order(Weight, decreasing = TRUE ),]
importance_matrix[1:100]
xgb.plot.importance(importance_matrix[1:100])
variables_top100 <- importance_matrix$Feature[1:100]

# Xgboost with 100 var----
importance_matrix$Feature[1:100]
inTrain_100 <-  subset(inTrain, select=c("segment.x",importance_matrix$Feature[1:100]))
inTest_100 <-  subset(inTest, select=c("segment.x",importance_matrix$Feature[1:100]))
feature.names <- importance_matrix$Feature[1:100]
dtrain <- xgb.DMatrix(data.matrix(inTrain_100[,feature.names]), label = inTrain_100$segment.x, missing = NA)

dval <- xgb.DMatrix(data.matrix(inTest_100[,feature.names]), label=inTest_100$segment.x, missing = NA)

#save(inTest,file = 'testing_set.rDATA')
#save(inTrain,file = 'training_set.rDATA')
#save(model_dataset,file = 'full_set.rDATA')
#load("D:/Alexander/Alexander/Health model/Modeling Code/10000 Sample/full_set.rDATA")
#load("D:/Alexander/Alexander/Health model/Modeling Code/10000 Sample/training_set.rDATA")
#load("D:/Alexander/Alexander/Health model/Modeling Code/10000 Sample/testing_set.rDATA")

# MODEL WITH JUST 100 VARIABLES ---- 
watchlist <- list(eval = dval, train = dtrain)
param <- list(  objective           = "binary:logitraw", 
                booster =  "gbtree",#"gblinear"
                eta                 = 0.001,
                max_depth           = 6,  # changed from default of 6
                subsample           = 0.6,
                colsample_bytree    = 0.6,
                eval_metric         = "auc",
                alpha = 0.0001, 
                lambda = 1
)

xgb_train_100 <- xgb.train(   params              = param, 
                          data                = dtrain, 
                          nrounds             = 300, # changed from 300
                          verbose             = 2, 
                          #early.stop.round    = 10,
                          watchlist           = watchlist,
                          maximize            = TRUE)

# Compute feature importance matrix
importance_matrix <- xgb.importance(feature.names, model = xgb_train_100)
#importance_matrix <- xgb.importance(feature.names, model = xgb_train, filename_dump = "D:/Alexander/Alexander/Spend model/xgbdump.txt")
# Nice graph
important_mat<-data.frame(importance_matrix)
importance_matrix <- importance_matrix[order(Weight, decreasing = TRUE ),]
importance_matrix[1:20]
xgb.plot.importance(importance_matrix[1:20])
# 3063 : PRODUCE - VEG / FRESHCUTS / HARD PRODUCE
# 0576 : CONFECTIONERY
# 0544 : CARBONATED SOFT DRINKS

# Look at raw score from the model on the whole set ----
test_score <- predict(xgb_train_100,dval)
train_score <- predict(xgb_train_100,dtrain)

roc(inTest$segment.x,test_score )
roc(inTrain$segment.x,train_score )


################################
################################
################################
### Plot model Variable ########
################################
################################
################################
png("plot-%d.png")
ggplot(model_dataset, aes(x=fix_dsct_wowd_alt_s_amt, color = factor(hsr_segment_noproduce), fill =  factor(hsr_segment_noproduce) )) + 
  geom_density(alpha = 0.3) +
  stat_density(geom = "path",position = "identity", kernel = "gaussian") +
  #  facet_wrap(~modelg5, ncol=2 ) + #, scales="free") 
  labs(title ="WOW dollars customer earned through ALT stretch in the last 8 weeks" ,x = "$")+ 
  scale_fill_discrete(name="Healthy Indicator") + scale_colour_discrete(name="Healthy Indicator")

ggplot(model_dataset, aes(x=fix_dsct_rt_ea, color = factor(hsr_segment_noproduce), fill =  factor(hsr_segment_noproduce) )) + 
  geom_density(alpha = 0.3) +
  stat_density(geom = "path",position = "identity", kernel = "gaussian") +
  #  facet_wrap(~modelg5, ncol=2 ) + #, scales="free") 
  labs(title ="Discount rate based on $ amount of discount divide by total spend of original price (8wks)" ,x = "rate")+ 
  scale_fill_discrete(name="Healthy Indicator") + scale_colour_discrete(name="Healthy Indicator")

ggplot(model_dataset, aes(x=fix_dsct_othr_wowd_amt, color = factor(hsr_segment_noproduce), fill =  factor(hsr_segment_noproduce) )) + 
  geom_density(alpha = 0.3) +
  stat_density(geom = "path",position = "identity", kernel = "gaussian") +
  #  facet_wrap(~modelg5, ncol=2 ) + #, scales="free") 
  labs(title ="WOW dollars customer earned exclude orange ticket in the last 8 weeks" ,x = "dollar")+ 
  scale_fill_discrete(name="Healthy Indicator") + scale_colour_discrete(name="Healthy Indicator")

ggplot(model_dataset, aes(x=fix_dsct_wowd_all_amt, color = factor(hsr_segment_noproduce), fill =  factor(hsr_segment_noproduce) )) + 
  geom_density(alpha = 0.3) +
  stat_density(geom = "path",position = "identity", kernel = "gaussian") +
  #  facet_wrap(~modelg5, ncol=2 ) + #, scales="free") 
  labs(title ="Total WOW dollars customer earned in the last 8 weeks" ,x = "dollar")+ 
  scale_fill_discrete(name="Healthy Indicator") + scale_colour_discrete(name="Healthy Indicator")

ggplot(model_dataset, aes(x=fix_dsct_othr_wowd_rec, color = factor(hsr_segment_noproduce), fill =  factor(hsr_segment_noproduce) )) + 
  geom_density(alpha = 0.3) +
  stat_density(geom = "path",position = "identity", kernel = "gaussian") +
  #  facet_wrap(~modelg5, ncol=2 ) + #, scales="free") 
  labs(title ="Number of products customer puchased which earn WOW dollar exclude orange ticket in the last 8 weeks" ,x = "No. items")+ 
  scale_fill_discrete(name="Healthy Indicator") + scale_colour_discrete(name="Healthy Indicator")
# 6 ----
ggplot(model_dataset, aes(x=fix_earn_rate_8w, color = factor(hsr_segment_noproduce), fill =  factor(hsr_segment_noproduce) )) + 
  geom_density(alpha = 0.3) +
  stat_density(geom = "path",position = "identity", kernel = "gaussian") +
  #  facet_wrap(~modelg5, ncol=2 ) + #, scales="free") 
  labs(title ="8 weeks earn rate" ,x = "rate")+ 
  scale_fill_discrete(name="Healthy Indicator") + scale_colour_discrete(name="Healthy Indicator")

ggplot(model_dataset, aes(x=fix_dsct_arcl_rec_rt_ea, color = factor(hsr_segment_noproduce), fill =  factor(hsr_segment_noproduce) )) + 
  geom_density(alpha = 0.3) +
  stat_density(geom = "path",position = "identity", kernel = "gaussian") +
  #  facet_wrap(~modelg5, ncol=2 ) + #, scales="free") 
  labs(title ="Discount rate based on number of item-based discount products (EA type) in the last 8 weeks" ,x = "rate")+ 
  scale_fill_discrete(name="Healthy Indicator") + scale_colour_discrete(name="Healthy Indicator")

ggplot(model_dataset, aes(x=fix_liq_wine_spd_p_52w, color = factor(hsr_segment_noproduce), fill =  factor(hsr_segment_noproduce) )) + 
  geom_density(alpha = 0.3) +
  stat_density(geom = "path",position = "identity", kernel = "gaussian") +
  #  facet_wrap(~modelg5, ncol=2 ) + #, scales="free") 
  labs(title ="Wine spend percentage in the last 52 weeks" ,x = "%")+ 
  scale_fill_discrete(name="Healthy Indicator") + scale_colour_discrete(name="Healthy Indicator")

ggplot(model_dataset, aes(x=fix_none_swipe_amt_8w, color = factor(hsr_segment_noproduce), fill =  factor(hsr_segment_noproduce) )) + 
  geom_density(alpha = 0.3) +
  stat_density(geom = "path",position = "identity", kernel = "gaussian") +
  #  facet_wrap(~modelg5, ncol=2 ) + #, scales="free") 
  labs(title ="None Swipe amount in the last 8 weeks" ,x = "amount")+ 
  scale_fill_discrete(name="Healthy Indicator") + scale_colour_discrete(name="Healthy Indicator")

ggplot(model_dataset, aes(x=fix_dsct_atcl_amt_ea, color = factor(hsr_segment_noproduce), fill =  factor(hsr_segment_noproduce) )) + 
  geom_density(alpha = 0.3) +
  stat_density(geom = "path",position = "identity", kernel = "gaussian") +
  #  facet_wrap(~modelg5, ncol=2 ) + #, scales="free") 
  labs(title ="Total discount $ amount of item-based discount Products (EA type) purchased in the last 8 weeks " ,x = "$")+ 
  scale_fill_discrete(name="Healthy Indicator") + scale_colour_discrete(name="Healthy Indicator")
# 11 ----
ggplot(model_dataset, aes(x=fix_dsct_othr_wowd_earn_rt, color = factor(hsr_segment_noproduce), fill =  factor(hsr_segment_noproduce) )) + 
  geom_density(alpha = 0.3) +
  stat_density(geom = "path",position = "identity", kernel = "gaussian") +
  #  facet_wrap(~modelg5, ncol=2 ) + #, scales="free") 
  labs(title ="Total discount $ amount of item-based discount Products (EA type) purchased in the last 8 weeks " ,x = "$")+ 
  scale_fill_discrete(name="Healthy Indicator") + scale_colour_discrete(name="Healthy Indicator")

ggplot(model_dataset, aes(x=fix_dsct_comb_rec_rt_ea, color = factor(hsr_segment_noproduce), fill =  factor(hsr_segment_noproduce) )) + 
  geom_density(alpha = 0.3) +
  stat_density(geom = "path",position = "identity", kernel = "gaussian") +
  #  facet_wrap(~modelg5, ncol=2 ) + #, scales="free") 
  labs(title ="Combined discount rate based on number of discount products  in the last 8 weeks " ,x = "rate")+ 
  scale_fill_discrete(name="Healthy Indicator") + scale_colour_discrete(name="Healthy Indicator")

ggplot(model_dataset, aes(x=fix_dsct_wowd_all_earn_rt, color = factor(hsr_segment_noproduce), fill =  factor(hsr_segment_noproduce) )) + 
  geom_density(alpha = 0.3) +
  stat_density(geom = "path",position = "identity", kernel = "gaussian") +
  #  facet_wrap(~modelg5, ncol=2 ) + #, scales="free") 
  labs(title ="Total WOW dollar earn rate in the last 8 weeks" ,x = "rate")+ 
  scale_fill_discrete(name="Healthy Indicator") + scale_colour_discrete(name="Healthy Indicator")

ggplot(model_dataset, aes(x=p_tot_professionals, color = factor(hsr_segment_noproduce), fill =  factor(hsr_segment_noproduce) )) + 
  geom_density(alpha = 0.3) +
  stat_density(geom = "path",position = "identity", kernel = "gaussian") +
  #  facet_wrap(~modelg5, ncol=2 ) + #, scales="free") 
  labs(title ="percentage of professionals with the area" ,x = "%")+ 
  scale_fill_discrete(name="Healthy Indicator") + scale_colour_discrete(name="Healthy Indicator")

ggplot(model_dataset, aes(x=m_tot_professionals, color = factor(hsr_segment_noproduce), fill =  factor(hsr_segment_noproduce) )) + 
  geom_density(alpha = 0.3) +
  stat_density(geom = "path",position = "identity", kernel = "gaussian") +
  #  facet_wrap(~modelg5, ncol=2 ) + #, scales="free") 
  labs(title ="percentage of male professionals with the area" ,x = "%")+ 
  scale_fill_discrete(name="Healthy Indicator") + scale_colour_discrete(name="Healthy Indicator")

ggplot(model_dataset, aes(x=p_tot_labourers, color = factor(hsr_segment_noproduce), fill =  factor(hsr_segment_noproduce) )) + 
  geom_density(alpha = 0.3) +
  stat_density(geom = "path",position = "identity", kernel = "gaussian") +
  #  facet_wrap(~modelg5, ncol=2 ) + #, scales="free") 
  labs(title ="percentage of labourers with the area" ,x = "%")+ 
  scale_fill_discrete(name="Healthy Indicator") + scale_colour_discrete(name="Healthy Indicator")

ggplot(model_dataset, aes(x=m_4000_over_tot, color = factor(hsr_segment_noproduce), fill =  factor(hsr_segment_noproduce) )) + 
  geom_density(alpha = 0.3) +
  stat_density(geom = "path",position = "identity", kernel = "gaussian") +
  #  facet_wrap(~modelg5, ncol=2 ) + #, scales="free") 
  labs(title ="percentage of male earning 4000 or above within the area" ,x = "%")+ 
  scale_fill_discrete(name="Healthy Indicator") + scale_colour_discrete(name="Healthy Indicator")

ggplot(model_dataset, aes(x=fix_dsct_wowd_alt_earn_rt, color = factor(hsr_segment_noproduce), fill =  factor(hsr_segment_noproduce) )) + 
  geom_density(alpha = 0.3) +
  stat_density(geom = "path",position = "identity", kernel = "gaussian") +
  #  facet_wrap(~modelg5, ncol=2 ) + #, scales="free") 
  labs(title ="ATL stretch WOW dollar earn rate in the last 8 weeks." ,x = "rate")+ 
  scale_fill_discrete(name="Healthy Indicator") + scale_colour_discrete(name="Healthy Indicator")

ggplot(model_dataset, aes(x=p_tot_mach_oper_drivers, color = factor(hsr_segment_noproduce), fill =  factor(hsr_segment_noproduce) )) + 
  geom_density(alpha = 0.3) +
  stat_density(geom = "path",position = "identity", kernel = "gaussian") +
  #  facet_wrap(~modelg5, ncol=2 ) + #, scales="free") 
  labs(title ="total percentage operation drivers (within census area)" ,x = "%")+ 
  scale_fill_discrete(name="Healthy Indicator") + scale_colour_discrete(name="Healthy Indicator")

ggplot(model_dataset, aes(x=fix_trn_tot_spend_a_8w, color = factor(hsr_segment_noproduce), fill =  factor(hsr_segment_noproduce) )) + 
  geom_density(alpha = 0.3) +
  stat_density(geom = "path",position = "identity", kernel = "gaussian") +
  #  facet_wrap(~modelg5, ncol=2 ) + #, scales="free") 
  labs(title ="total spend amount last 8 weeks" ,x = "$")+ 
  scale_fill_discrete(name="Healthy Indicator") + scale_colour_discrete(name="Healthy Indicator")
dev.off()