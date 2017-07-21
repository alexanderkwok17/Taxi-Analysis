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
# 3. Pull DataSets from redshift

healthData <- dbGetQuery(conn,"SELECT * FROM  loyalty_modeling.wow_tgi_rating_clean_distinct")

table(healthData$hsr)

frq <- hist(healthData$hsr[healthData$hsr!= 88 & healthData$hsr != 99])

healthData[healthData$hsr == 5,1:5]
# Clustering
library(cluster)
library(HSAUR)

CRN_healthData <- dbGetQuery(conn,"select * from loyalty_modeling.wow_tgi_crn_rating_behaviour ")
Health_Model_G_I <- read.csv("~/Health_Model_G_I.csv", colClasses = c("character", "integer"))
head(CRN_healthData)
names(CRN_healthData)
dim(CRN_healthData)

#cluster_data_sample <- CRN_healthData[sample(nrow(CRN_healthData), 10000),2:24]
cluster_data_sample <- CRN_healthData #[sample(nrow(CRN_healthData),nrow(CRN_healthData)*1 ),2:24]
cluster_data_sample <- merge(cluster_data_sample, Health_Model_G_I,by = "crn")
names(cluster_data_sample)

table(cluster_data_sample$modelI4)
cluster_data_sample$modelI4[cluster_data_sample$modelI4 == 2] <- 5
table(cluster_data_sample$modelI4)
cluster_data_sample$modelI4[cluster_data_sample$modelI4 == 4] <- 2
table(cluster_data_sample$modelI4)
cluster_data_sample$modelI4[cluster_data_sample$modelI4 == 5] <- 4
table(cluster_data_sample$modelI4)
summary(cluster_data_sample$tot_sales_all)
cluster_data_sample$tot_sales_all <- replace(cluster_data_sample$tot_sales_all, cluster_data_sample$tot_sales_all > quantile(cluster_data_sample$tot_sales_all, 0.95), quantile(cluster_data_sample$tot_sales_all,0.95))

rm(percent)
cluster_data_sample[is.na(cluster_data_sample)] <- 0

names(cluster_data_sample)

head(cluster_data_sample)
dim(cluster_data_sample)
# Testing ----
###################################
km    <- kmeans(cluster_data_sample,5)
dissE <- daisy(cluster_data_sample) 
dE2   <- dissE^2
sk2   <- silhouette(km$cl, dE2)
plot(sk2)
##########################################
library(cluster)
library("fpc")
# run PCA first
pc <- princomp(cluster_data_sample, cor=TRUE, scores=TRUE)
summary(pc)
plot(pc,type="lines")
biplot(pc)

# Kmeans clustre analysis
clus <- kmeans(cluster_data_sample, centers=4)
# Fig 01
plotcluster(cluster_data_sample, clus$cluster)
# More complex
clusplot(cluster_data_sample, clus$cluster, color=TRUE, shade=TRUE, 
         labels=2, lines=0)
# Fig 03
with(cluster_data_sample[,1:11], pairs(cluster_data_sample[,1:11], col=c(1:3)[clus$cluster])) 
# cluster dendrogram
di <- dist(cluster_data_sample, method="euclidean")
tree <- hclust(di, method="ward")
cluster_data_sample$hcluster <- as.factor((cutree(tree, k=4)-2) %% 4 +1)
# that modulo business just makes the coming table look nicer
plot(tree, xlab="")
rect.hclust(tree, k=4, border="red")
########################################

cluster_data_sample$kcluster <- as.factor( clus$cluster)

head(cluster_data_sample)
write.csv(cluster_data_sample, file = "clustering_result.csv")
# Group mean each cluster
names(cluster_data_sample)
r2 <- aggregate(cluster_data_sample[,c(97) ], list(cluster_data_sample$modelI4), mean)


# rUN THE SAME FOR DOLLAR SPEND ----
is.nan.data.frame <- function(x)
  do.call(cbind, lapply(x, is.nan))
cluster_data_sample[is.nan(cluster_data_sample)]<- 0

names(cluster_data_sample[,8:18]) # qty
names(cluster_data_sample[,19:29]) # measured qty
names(cluster_data_sample[,30:40]) # dollar sales all
names(cluster_data_sample[,41:51]) # dollar sales article
names(cluster_data_sample[,52:62]) # dollar sales measured 
names(cluster_data_sample[,63:73]) # discounted items 
names(cluster_data_sample[,76:86]) # distinct prod count 

cluster_data_sample$hsr_0_to_1_5_qty <- rowSums(cluster_data_sample[,8:11]) # group hsr 0 to 1.5 together
cluster_data_sample$hsr_2_to_2_5_qty <- rowSums(cluster_data_sample[,12:13]) # group hsr 2 to 2.5 together
cluster_data_sample$hsr_3_to_3_5_qty <- rowSums(cluster_data_sample[,14:15]) # group hsr 3 to 3.5 together
cluster_data_sample$hsr_4_5_to_5_qty <- rowSums(cluster_data_sample[,17:18]) # group hsr 4.5 to 5 together

cluster_data_sample$hsr_0_to_1_5_dollar <- rowSums(cluster_data_sample[,30:33]) # group hsr 0 to 1.5 together
cluster_data_sample$hsr_2_to_2_5_dollar <- rowSums(cluster_data_sample[,34:35]) # group hsr 2 to 2.5 together
cluster_data_sample$hsr_3_to_3_5_dollar <- rowSums(cluster_data_sample[,36:37]) # group hsr 3 to 3.5 together
cluster_data_sample$hsr_4_5_to_5_dollar <- rowSums(cluster_data_sample[,39:40]) # group hsr 4.5 to 5 together



# total capture
cluster_data_sample$tot_qty <- rowSums(cluster_data_sample[,8:18],na.rm = TRUE,dims = 1)
cluster_data_sample$tot_mea_qty <- rowSums(cluster_data_sample[,19:29],na.rm = TRUE,dims = 1)
cluster_data_sample$tot_sales_all <- rowSums(cluster_data_sample[,30:40],na.rm = TRUE,dims = 1)
cluster_data_sample$tot_sales_art <- rowSums(cluster_data_sample[,41:51],na.rm = TRUE,dims = 1)
cluster_data_sample$tot_sales_mea <- rowSums(cluster_data_sample[,52:62],na.rm = TRUE,dims = 1)
cluster_data_sample$tot_dscnt <- rowSums(cluster_data_sample[,63:73],na.rm = TRUE,dims = 1)
cluster_data_sample$tot_prod_count <- rowSums(cluster_data_sample[,76:86],na.rm = TRUE,dims = 1)

head(cluster_data_sample[,8:18] / cluster_data_sample$tot_qty) 
# Discounted percentage overall 
cluster_data_sample$dscnt_pct <- cluster_data_sample$tot_dscnt / (cluster_data_sample$tot_qty + cluster_data_sample$tot_mea_qty)

########################################
# % qty sales
percent <- cluster_data_sample[,8:18] / cluster_data_sample$tot_qty
names(percent) <- paste(names(percent), '_percent', sep="")
cluster_data_sample <- cbind(cluster_data_sample, percent)
head(cluster_data_sample)
# % qty measure sales
percent <- cluster_data_sample[,19:29] / cluster_data_sample$tot_mea_qty
names(percent) <- paste(names(percent), '_percent', sep="")
cluster_data_sample <- cbind(cluster_data_sample, percent)
head(cluster_data_sample)
# error on qty_mea_4_5
cluster_data_sample$hsr_4_5_mea_qty_percent <- (replace(cluster_data_sample$hsr_4_5_mea_qty_percent,cluster_data_sample$hsr_4_5_mea_qty_percent > 1 , 1))
summary(cluster_data_sample$hsr_4_5_mea_qty_percent)

# % dollar sales all
percent <- cluster_data_sample[,30:40] / cluster_data_sample$tot_sales_all
names(percent) <- paste(names(percent), '_percent', sep="")
cluster_data_sample <- cbind(cluster_data_sample, percent)
head(cluster_data_sample)
# % dollar sales article based
percent <- cluster_data_sample[,41:51] / cluster_data_sample$tot_sales_art
names(percent) <- paste(names(percent), '_percent', sep="")
cluster_data_sample <- cbind(cluster_data_sample, percent)
head(cluster_data_sample)
# % dollar sales weighted based
percent <- cluster_data_sample[,52:62] / cluster_data_sample$tot_sales_mea
names(percent) <- paste(names(percent), '_percent', sep="")
cluster_data_sample <- cbind(cluster_data_sample, percent)
head(cluster_data_sample)
# % qty dsct
percent <- cluster_data_sample[,63:73] / cluster_data_sample$tot_dscnt
names(percent) <- paste(names(percent), '_percent', sep="")
cluster_data_sample <- cbind(cluster_data_sample, percent)
head(cluster_data_sample)
# % distinct prod count
percent <- cluster_data_sample[,76:86] / cluster_data_sample$tot_prod_count
names(percent) <- paste(names(percent), '_percent', sep="")
cluster_data_sample <- cbind(cluster_data_sample, percent)
head(cluster_data_sample)

# new set of % on groupped HSR groups qty
percent <- cluster_data_sample[,c(87:90,16)] / cluster_data_sample$tot_qty
names(percent) <- paste(names(percent), '_percent_Groupped', sep="")
cluster_data_sample <- cbind(cluster_data_sample, percent)

# new set of % on groupped HSR groups dollar
percent <- cluster_data_sample[,c(38,91:94) ] / cluster_data_sample$tot_sales_all
names(percent) <- paste(names(percent), '_percent_Grouped', sep="")
cluster_data_sample <- cbind(cluster_data_sample, percent)
head(cluster_data_sample)

names(cluster_data_sample)
# percentage spend on $ qty spend over all spend
cluster_data_sample$tot_dollar_qty_pct <- cluster_data_sample$tot_sales_art / (cluster_data_sample$tot_sales_mea + cluster_data_sample$tot_sales_art)
# Normalised total dollar spend 
# Standardisation function ----
standardised <- function(x){
  a <- max(x)
  b <- min(x)
  return( (x- b) / (a-b))
}
# Run function -------
cluster_data_sample$tot_sales_all_standardised <- standardised(cluster_data_sample$tot_sales_all)
cluster_data_sample$tot_sales_art_standardised <- standardised(cluster_data_sample$tot_sales_art)
cluster_data_sample$tot_sales_mea_standardised <- standardised(cluster_data_sample$tot_sales_mea)
# check the range of defined columns : validated the standardised column is between 0 to 1
summary(cluster_data_sample[,191:193])
head(cluster_data_sample$tot_sales_all[order(cluster_data_sample$tot_sales_all,decreasing = TRUE)])
hist(cluster_data_sample$tot_sales_all_standardised)
# too many extreme value, need to cap the max value to a more reasonable value (95% of the original distribution )
replace_sales <- replace(cluster_data_sample$tot_sales_all, cluster_data_sample$tot_sales_all >  quantile(cluster_data_sample$tot_sales_all,probs = 0.95),  quantile(cluster_data_sample$tot_sales_all,probs = 0.95))
cluster_data_sample$tot_sales_all_standardised <- standardised(replace_sales)
summary(cluster_data_sample$tot_sales_all_standardised)
# repeat for article sales
replace_sales <- replace(cluster_data_sample$tot_sales_art, cluster_data_sample$tot_sales_art >  quantile(cluster_data_sample$tot_sales_art,probs = 0.95),  quantile(cluster_data_sample$tot_sales_art,probs = 0.95))
cluster_data_sample$tot_sales_art_standardised <- standardised(replace_sales)
summary(cluster_data_sample$tot_sales_art_standardised)
# repeat for weighted sales
replace_sales <- replace(cluster_data_sample$tot_sales_mea, cluster_data_sample$tot_sales_mea >  quantile(cluster_data_sample$tot_sales_mea,probs = 0.95),  quantile(cluster_data_sample$tot_sales_mea,probs = 0.95))
cluster_data_sample$tot_sales_mea_standardised <- standardised(replace_sales)
summary(cluster_data_sample$tot_sales_mea_standardised)


# percentage of measured sales over total sales : 21.66%
sum(cluster_data_sample$tot_sales_mea) / sum(cluster_data_sample$tot_sales_art + cluster_data_sample$tot_sales_mea)

# produce summary of all created variables
summary_list <- summary(cluster_data_sample)

write.csv(summary_list , file = "summary_list.csv")

# histogram view -----
h<-hist(cluster_data_sample$avg_hsr, breaks=50, col="red", xlab="HSR Rating", 
     main="Histogram with Normal Curve")
x <- cluster_data_sample$avg_hsr
xfit<-seq(min(x),max(x),length=40) 
yfit<-dnorm(xfit,mean=mean(x),sd=sd(x)) 
yfit <- yfit*diff(h$mids[1:2])*length(x) 
lines(xfit, yfit, col="blue", lwd=2)

head(cluster_data_sample[1:10,])
cluster_data_sample <- data.frame(cluster_data_sample)
cluster_data_sample[is.nan(cluster_data_sample)] <- 0
# kmeans ----
head(cluster_data_sample)
dim(cluster_data_sample)


names(cluster_data_sample)
names(cluster_data_sample[,c(170:193)])
# Segmentation Trail 1 candidate : Use qty % weight % unique prod count % total spend standardised and qty spending %
names(cluster_data_sample)
names(cluster_data_sample[,c(103:113,114:124,169:179,190,191)])
# Segmentation Trail 2 candidate : Use qty % weight % total spend standardised and qty spending %
names(cluster_data_sample[,c(103:113, 114:124,190,191)])
# Segmentation Trial 3 Candidate : Use qty % total spend standardised 
names(cluster_data_sample[,c(103:113,190:191)])
# Segmentation Model A: Use $ % and total spend standardised
names(cluster_data_sample[,c(125:135, 191)])
# Segmentation Model B: Use $ article % and total spend standardised
names(cluster_data_sample[,c(136:146, 191)])
# Segmentation Model C: Use qty % and total spend standardised
names(cluster_data_sample[,c(103:113, 191)])
# Segmentation Model D: Use qty mea % and total Spend Standardised
names(cluster_data_sample[,c(114:124, 191)])
# Segmentation Model E: Use prod freq % and total Spend Standardised
names(cluster_data_sample[,c(169:179, 191)])
# Segmentation Model F : Use groupped prod qty % and total Spend Standardised
names(cluster_data_sample[,c(180:184,191)])
# Sementation Model G  : Use groupped prod dollar % and total Spend Standardised
names(cluster_data_sample[,c(185:189,191)])
# Segmentation Model H : Use groupped prod qty % 
names(cluster_data_sample[,c(180:184)])
# Sementation Model I  : Use groupped prod dollar % 
names(cluster_data_sample[,c(185:189)])

# Remove all customer with 0 spend in any HSR rating 
cluster_data_sample <- cluster_data_sample[cluster_data_sample$tot_sales_all != 0 ,]

names(cluster_data_sample[,c(185:189,191)])

Cluster_set <- cluster_data_sample[,c(185:189,191)]
## FULL (1:23,26:49)
Cluster_set[is.nan(Cluster_set)] <- 0
cluster_data_sample[is.nan(cluster_data_sample)] <- 0
cluster_data_sample[is.na(cluster_data_sample)] <- 0

summary(Cluster_set)

pc <- princomp(Cluster_set, cor=TRUE, scores=TRUE)
pc <- prcomp(Cluster_set, cor=TRUE, scores=TRUE)
which(abs(cov(Cluster_set)) < .001, arr.ind=TRUE)
summary(pc)
plot(pc,type="lines")

biplot(pc)


# Kmeans clustre analysis
clus <- kmeans(Cluster_set, centers=5)
table(clus$cluster) / sum(table(clus$cluster))
table(clus$cluster)
# Fig 01
plotcluster(Cluster_set, clus$cluster)
# More complex
clusplot(Cluster_set, clus$cluster, color=TRUE, shade=TRUE, 
         labels=2, lines=0)
cluster_data_sample$modelG5_2 <-   clus$cluster
#cluster_data_sample$kcluster_qtyANDmea_6<-   clus$cluster
table(cluster_data_sample$kcluster_qty)/sum(table(cluster_data_sample$kcluster_qty))
table(cluster_data_sample$modelG5, cluster_data_sample$modelG5_2)
# cluster dendrogram
di <- dist(Cluster_set, method="euclidean")
#tree <- hclust(di, method="ward")
#cluster_data_sample$hcluster <- as.factor((cutree(tree, k=4)-2) %% 4 +1)
# that modulo business just makes the coming table look nicer
#plot(tree, xlab="")
#rect.hclust(tree, k=4, border="red")

summary(cluster_data_sample$hsr_5_dollar_sales_percent, cluster_data_sample$kcluster)
tapply(cluster_data_sample$hsr_5_dollar_sales_percent, cluster_data_sample$kcluster, summary)
cbind(cluster_data_sample$crn, cluster_data_sample$kcluster)[1,]

names(cluster_data_sample[,c(1,204,206)])

write.csv(cluster_data_sample[,c(1,204,206)], file = "Health_Model_G_I.csv",row.names= FALSE)

write.csv(cluster_data_sample, file = "clustering_result_50pct.csv")

# Density plot of Average score
ggplot(CRN_healthData, aes(avg_hsr)) +
  geom_density(adjust = 3)
cluster_data_sample$kcluster
head(cluster_data_sample[,c(95:105,107:116,161:171,172,175)])
cluster_data_sample$kcluster_qty_mea_5 <- factor(cluster_data_sample$modelA)

d <- density(cluster_data_sample$hsr_4_dscnt_qty_percent)
plot(d)

# Inspect group 4 and 2 see if there can be further disect into healthier group ----
names(cluster_data_sample)

table(cluster_data_sample$modelI4)
ggplot(cluster_data_sample, aes(x=tot_sales_all/8#,group = modelI4
                                , color = factor(modelI4), fill =  factor(modelI4) )) + 
  geom_density(alpha = 0.3) +
  stat_density(geom = "path",position = "identity", kernel = "gaussian") +
 # facet_wrap(~modelI4, ncol=2 , scales="free") +
  labs(title ="Distribution of average weekly sales over 8 weeks ($)" ,x = "Dollar $")+ 
  scale_fill_discrete(name="Model I4") + scale_colour_discrete(name="Model I4")

# Total Sales Distribution ----
ggplot(cluster_data_sample, aes(x=tot_sales_all)) +
  coord_cartesian(xlim= c(0,1000)) +
  geom_histogram(binwidth =  1000) +
#  facet_wrap(~modelI4, ncol=2 , scales="free") +
  labs(title ="Distribution of Total Sales" ,x = "Dollar")
plot_data <- merge(cluster_data_sample, Health_Model_G_I)
summary(plot_data$tot_sales_all)
plot_data$tot_sales_all <- replace(plot_data$tot_sales_all,plot_data$tot_sales_all > quantile(plot_data$tot_sales_all, 0.95),quantile(plot_data$tot_sales_all, 0.95))
plot_data$modelI4[plot_data$modelI4 ==5] <- 4
table(plot_data$modelI4)


ggplot(plot_data, aes(x=tot_sales_all, fill = factor(modelI4), color = factor(modelI4) 
)) + 
  geom_density(alpha = 0.3, adjust = 3) +
  #stat_density(geom = "path",position = "identity", kernel = "gaussian") +
  labs(title ="Distribution of  Total Sales" ,x = "Dollar") +
  scale_fill_discrete(name="Health Cluster")  + scale_colour_discrete(name="Health Cluster")

hist(cluster_data_sample$tot_sales_all[cluster_data_sample$modelG5 == 1],breaks = 500)
hist(cluster_data_sample$tot_sales_all[cluster_data_sample$modelG5 == 2],breaks = 500)
hist(cluster_data_sample$tot_sales_all[cluster_data_sample$modelG5 == 3],breaks = 500)
hist(cluster_data_sample$tot_sales_all[cluster_data_sample$modelG5 == 4],breaks = 500)
hist(cluster_data_sample$tot_sales_all[cluster_data_sample$modelG5 == 5],breaks = 500)

cluster_data_sample$modelI4 <- factor(cluster_data_sample$modelI4)
cluster_data_sample$modelG5 <- factor(cluster_data_sample$modelG5)
summary(cluster_data_sample[cluster_data_sample$tot_sales_all== 0,])
summary(cluster_data_sample$modelG5   )
# Mean statistic within each cluster-----
# using dollar cluster
summary(cluster_data_sample)
head(cluster_data_sample)
cluster_data_sample <- data.frame(cluster_data_sample)

names(cluster_data_sample)[c(95:105,107:116,161:171,172,173)]

r2 <- aggregate(cluster_data_sample[,c(95:105,107:116,161:171,172,173)], list(cluster_data_sample$modelH4), mean)
r3 <- aggregate(cluster_data_sample[,c(189:193,173)], list(cluster_data_sample$modelH4), mean)

group <- cluster_data_sample$kcluster_qtyANDmea
table(group)
table(group) / sum(table(group))

table(cluster_data_sample$kcluster_qty_mea)

write.csv(r2, file = "groupMean_qty_mea_cluster_H4.csv")
write.csv(r3, file = "groupMean_new_cluster_H4.csv")

x <- density(cluster_data_sample$hsr_5_dscnt_qty_percent)
plot(x)

# Compare the two cluster ----
table (cluster_data_sample$kcluster_dollar, cluster_data_sample$kcluster_qty)
table (cluster_data_sample$kcluster_dollar)
head(cluster_data_sample[,c(1,96)])
write.csv(cluster_data_sample[,c(1,96)], file ="cluster_result.csv",row.names= FALSE)



ggraptR()


# Z scores Transformation ----
zscores <- dbGetQuery(conn,"
SELECT crn, hsr, avg(Z_scores) as average_score
FROM loyalty_modeling.wow_tgi_rating_crn_6wk_sd_scores
GROUP BY 1,2 ")
head(zscores)
dim(zscores)
length(unique(zscores$crn))
library('reshape')
zscoresTF <- cast(zscores, crn ~ hsr)

zscoresTF <- zscoresTF[zscoresTF$crn != 0 , ]
rm(zscores)
head(zscoresTF)
save(zscoresTF, file = 'Transformed.rdata') 

# remove outliers ----
# First replace null with Minimum score 
zscoresTF$`5`[is.na(zscoresTF$`5`)] <- min(zscoresTF$`5`, na.rm = TRUE)
outlierKD(zscoresTF, `5`)
summary(zscoresTF$`5`)
quantile(zscoresTF$`5`, c(0.05,0.95))
# procedure to remove outliers at 5% tile on both end
zscoresTF$`0`[is.na(zscoresTF$`0`)] <- min(zscoresTF$`0`, na.rm = TRUE)
subject <- zscoresTF$`0`
lowlim <- quantile(subject, c(0.25))
uplim <- quantile(subject, c(0.95))
subject[subject< lowlim] <- lowlim
subject[subject> uplim] <- uplim

hist(zscoresTF$`0`)
hist(subject)

zscoresTF$`0` <- subject


outlierKD(zscoresTF, `5`)
outlierKD(zscoresTF, `4`)
outlierKD(zscoresTF, `3`)
outlierKD(zscoresTF, `2`)
outlierKD(zscoresTF, `0`)
outlierKD(zscoresTF, `4.5`)
outlierKD(zscoresTF, `3.5`)
outlierKD(zscoresTF, `2.5`)
outlierKD(zscoresTF, `0.5`)
outlierKD(zscoresTF, `0.5`)


head(zscoresTF)
summary(zscoresTF$`4`)

head(zscoresTF)

head(zscoresTF[,-c(1:2,13:15)])
zscoresTF[is.na(zscoresTF)] <- 0

final <- zscoresTF[complete.cases(zscoresTF[,-c(1:2,13:15)]),-c(1:2,13:15)]

clusterSet <- final #zscoresTF[,-c(1:2,13:15)]
head(clusterSet)
pc <- princomp(clusterSet, cor=TRUE, scores=TRUE)
summary(pc)
plot(pc,type="lines")

biplot(pc)

# Kmeans clustre analysis
clus <- kmeans(clusterSet, centers=4)

zscoresTF$cluster <- clus$cluster

table(clus$cluster) / sum(table(clus$cluster))

CRN_healthData$cluster <- clus$cluster

CRN_healthData$crn
zscoresTF$crn
result <- merge(CRN_healthData, zscoresTF, by = "crn")

head(result)
result[is.na(result)]<- 0

r2 <- aggregate(result[,c(2:12)], list(result$cluster), mean)
r2

outlierKD(zscoresTF, zscoresTF$`5`)



# outliers script ----
outlierKD <- function(dt, var) {
  var_name <- eval(substitute(var),eval(dt))
  na1 <- sum(is.na(var_name))
  m1 <- mean(var_name, na.rm = T)
  par(mfrow=c(2, 2), oma=c(0,0,3,0))
  boxplot(var_name, main="With outliers")
  hist(var_name, main="With outliers", xlab=NA, ylab=NA)
  outlier <- boxplot.stats(var_name)$out
  mo <- mean(outlier)
  var_name <- ifelse(var_name %in% outlier, NA, var_name)
  boxplot(var_name, main="Without outliers")
  hist(var_name, main="Without outliers", xlab=NA, ylab=NA)
  title("Outlier Check", outer=TRUE)
  na2 <- sum(is.na(var_name))
  cat("Outliers identified:", na2 - na1, "n")
  cat("Propotion (%) of outliers:", round((na2 - na1) / sum(!is.na(var_name))*100, 1), "n")
  cat("Mean of the outliers:", round(mo, 2), "n")
  m2 <- mean(var_name, na.rm = T)
  cat("Mean without removing outliers:", round(m1, 2), "n")
  cat("Mean if we remove outliers:", round(m2, 2), "n")
  response <- readline(prompt="Do you want to remove outliers and to replace with NA? [yes/no]: ")
  if(response == "y" | response == "yes"){
    dt[as.character(substitute(var))] <- invisible(var_name)
    assign(as.character(as.list(match.call())$dt), dt, envir = .GlobalEnv)
    cat("Outliers successfully removed", "n")
    return(invisible(dt))
  } else{
    cat("Nothing changed", "n")
    return(invisible(var_name))
  }
}
