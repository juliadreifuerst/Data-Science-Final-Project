###########################################################################

######################## Final Project Assignment #########################

###########################################################################

library(tidyverse)
library(ggplot2)
library(dplyr)
library(caret)

# load the data
worldpopulation <- read.csv("WorldPopulation2022.csv")

###########################################################################

# DATA EXPLORATION 

###########################################################################

# explore the dataset
view(worldpopulation)
head(worldpopulation)
summary(worldpopulation)
names(worldpopulation)
str(worldpopulation)

###########################################################################

# DATA CLEANING 

###########################################################################

# change the column names to be easier to understand
colnames(worldpopulation)[1]="ID"
colnames(worldpopulation)[2]="country"
colnames(worldpopulation)[3]="population_2022"
colnames(worldpopulation)[4]="yearly_change"
colnames(worldpopulation)[5]="net_change"
colnames(worldpopulation)[6]="density"
colnames(worldpopulation)[7]="land_area"
colnames(worldpopulation)[8]="migrants"
colnames(worldpopulation)[9]="fertility_rate"
colnames(worldpopulation)[10]="median_age"
colnames(worldpopulation)[11]="urban_population"
colnames(worldpopulation)[12]="world_share_pop"

view(worldpopulation)

# handle missing values
sum(is.na(worldpopulation)) # there are 102 missing values
sum(is.na(worldpopulation$population_2022)) # 0 missing values
sum(is.na(worldpopulation$yearly_change)) # 0 missing values
sum(is.na(worldpopulation$net_change)) # 0 missing values
sum(is.na(worldpopulation$density)) # 0 missing values
sum(is.na(worldpopulation$land_area)) # 0 missing values
sum(is.na(worldpopulation$migrants)) # 34 missing values
sum(is.na(worldpopulation$fertility_rate)) # 34 missing values
sum(is.na(worldpopulation$median_age)) # 34 missing values
sum(is.na(worldpopulation$urban_population)) # 0 missing values
sum(is.na(worldpopulation$world_share_pop)) # 0 missing values
    # migrants, fertility_rate, and median_age have missing values

# replace missing values for variables with the mean of the column
worldpopulation$migrants <- worldpopulation$migrants %>% 
    replace_na(mean(worldpopulation$migrants, na.rm=TRUE))
sum(is.na(worldpopulation$migrants))

worldpopulation$fertility_rate <- worldpopulation$fertility_rate %>% 
  replace_na(mean(worldpopulation$fertility_rate, na.rm=TRUE))
sum(is.na(worldpopulation$fertility_rate))

worldpopulation$median_age <- worldpopulation$median_age %>% 
  replace_na(mean(worldpopulation$median_age, na.rm=TRUE))
sum(is.na(worldpopulation$median_age))

# check for outliers
ggplot(worldpopulation, aes(yearly_change)) + geom_boxplot() # only one outlier
ggplot(worldpopulation, aes(population_2022)) + geom_boxplot() # lots of outliers, two major outliers 
ggplot(worldpopulation, aes(net_change)) + geom_boxplot() # lots of outliers, 1 major outlier
ggplot(worldpopulation, aes(density)) + geom_boxplot() # lots of outliers, two major
ggplot(worldpopulation, aes(land_area)) + geom_boxplot() # lots of outliers, a few major
ggplot(worldpopulation, aes(migrants)) + geom_boxplot() # lots of outliers
ggplot(worldpopulation, aes(fertility_rate)) + geom_boxplot() # a few outliers
ggplot(worldpopulation, aes(median_age)) + geom_boxplot() # no outliers
ggplot(worldpopulation, aes(urban_population)) + geom_boxplot() # no outliers
ggplot(worldpopulation, aes(world_share_pop)) + geom_boxplot() # lots of outliers

# use min max scaling to suppress the effect of outliers
preproc <- preProcess(worldpopulation, method=c("range"))
worldpop_norm <- predict(preproc, worldpopulation)
summary(worldpop_norm)

###########################################################################

# DATA PREPROCESSING 

###########################################################################

# normalize the data
preproc1 <- preProcess(worldpop_norm, method=c("center","scale"))
worldpop_norm <- predict(preproc1, worldpop_norm)
summary(worldpop_norm)

###########################################################################

# CLUSTERING 

###########################################################################

library(stats)
library(factoextra)
# remove class labels
predictors <- worldpop_norm %>% select(-c(country, ID, population_2022)) #pop_2022's scale has a substantially higher range
head(predictors)

set.seed(123)

# k-means 
# find the knee
fviz_nbclust(predictors, kmeans, method="wss")
fviz_nbclust(predictors, kmeans, method="silhouette")
  # k=2
# fit the data
fit <- kmeans(predictors, centers=2, nstart=25)
fit
# display the cluster plot
fviz_cluster(fit, data=predictors)

# calculate PCA
pca = prcomp(predictors)
rotated_data = as.data.frame(pca$x)
rotated_data$Color <- worldpop_norm$ID
ggplot(data=rotated_data, aes(x=PC1, y=PC2, col=Color)) + geom_point(alpha=0.3) 

###########################################################################

# CLASSIFICATION 

###########################################################################

set.seed(123)

classification_data = worldpop_norm%>%select(-c(ID, country, population_2022))
classification_data$cluster_label = fit$cluster
classification_data$cluster_label <- as.factor(classification_data$cluster_label)

index=createDataPartition(y=classification_data$cluster_label, p=0.7, list=FALSE)
train_world = classification_data[index,]
test_world= classification_data[-index,]

# Model 1 - Decision Tree
train_control <- trainControl(method='cv', number=10)
tree1 <- train(cluster_label ~., data=train_world, method = "rpart", trControl=train_control) 
tree1
pred_tree <- predict(tree1, test_world)
confusionMatrix(test_world$cluster_label, pred_tree)
# 0.9429 Accuracy

# Model 2 - KNN
ctrl <- trainControl(method='cv', number=10)
knnFit <- train(cluster_label~., data=train_world,
                method='knn',
                trControl = ctrl,
                preProcess = c("center", "scale"))

plot(knnFit)

pred_knn <- predict(knnFit, test_world)
cm <- confusionMatrix(test_world$cluster_label, pred_knn)
cm
# 0.9571 Accuracy

###########################################################################

# EVALUATION 

###########################################################################

# calculate precision and recall
metrics <- as.data.frame(cm$byClass)
metrics
#precision
metrics %>% select(c(Precision))
# recall
metrics %>% select(C(Recall))

#ROC Curve - KNN
library(pROC)
# Get class probabilities for KNN
pred_prob <- predict(knnFit, test_world, type = "prob")
head(pred_prob)

# ROC plot
roc_obj <- roc((test_world$cluster_label), pred_prob[,1])

plot(roc_obj, print.auc=TRUE)
