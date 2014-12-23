#R script

#Set the working directory
setwd("C:\\Users\\Desktop\\Practical Machine Learning\\WriteUp")

library(graphics)
library(caret)
library(plyr)

#------------------------------------------
#0A. read in training and testing data
indata <- read.table("./pml-training.csv", header = T, sep = ",", na.strings=c("NA", "#DIV/0!")) #19622 obs, 160 vars
outdata <- read.table("./pml-testing.csv", header = T, sep = ",",na.strings=c("NA", "#DIV/0!"))   #20 obs, 160 vars

#------------------------------------------
#0B. First glance of the data:
#0B.1 When new_windows == "yes"
new.window <- subset(indata, new_window == "yes") #406 observation
new.window.obs <- nrow(new.window)

dq <- lapply(new.window[,8:159] , function(x) rbind( 
                  mean = mean(x, na.rm = T) ,
                  sd = sd(x, na.rm = T) ,
                  NA.Rate = sum(is.na(x)/new.window.obs))
            )

dq <- data.frame(dq)
na.rate <- dq[row.names(dq)=="NA.Rate",]
ncol(dq[, which(numcolwise(sum)(na.rate) > 0.95)]) #number of high-NA features: 6

#0B.2 When new_windows == "no"
new.window <- subset(indata, new_window == "no") #19216 observation
new.window.obs <- nrow(new.window)

dq <- lapply(new.window[,8:159] , function(x) rbind( 
      mean = mean(x, na.rm = T) ,
      sd = sd(x, na.rm = T) ,
      NA.Rate = sum(is.na(x)/new.window.obs))
)

dq <- data.frame(dq)
na.rate <- dq[row.names(dq)=="NA.Rate",]
ncol(dq[, which(numcolwise(sum)(na.rate) > 0.95)]) #number of high-NA features: 100 out of 152
names(dq[, which(numcolwise(sum)(na.rate) > 0.95)])
#------------------------------------------
#1. Clean up the data first
indata.obs <- nrow(indata)
summary <- lapply(indata[,8:159] , function(x) rbind( 
                                         mean = mean(x, na.rm = T) ,
                                         sd = sd(x, na.rm = T) ,
                                         NA.Rate = sum(is.na(x)/indata.obs) 
                                         )
                   )
summary.data<-data.frame(summary)
na.rate <- summary.data[row.names(summary.data)=="NA.Rate",]
a1 <- summary.data[, -which(numcolwise(sum)(na.rate) > 0.95)] # remove high missing variable
sd.0 <- summary.data[row.names(summary.data)=="sd",]
a2 <- a1[, -which(numcolwise(sum)(sd.0) <= 0)] # remove variable with constant number all the way (no variation)
candidates <- names(a2) 
length(candidates)
#only 51 raw-form features left. 
#Notes: If i knew how the derived features were calculated, 
#I'd have recaculated them and re-merge them back to the point-of-time level
#to avoid high missing derived features.

clean.indata <- cbind(classe = indata$classe, indata[, names(a2)])
#------------------------------------------
#2. Cross-validation 70-30 split
set.seed(221)
inTrain <- createDataPartition(y = clean.indata$classe, p = 0.7, list = FALSE)
Train <- clean.indata[inTrain,]
Test <- clean.indata[-inTrain,]
#------------------------------------------
#3. EDA
library(car)
class(Train)

#3.1 sampling 10% of Train data for easier chartting purpose.
random_sample = function (dt, sample_size) {
      # determine the number of records in the data frame
      length_dt = length (dt[,1])
      # extract the sample
      dt [sample (1:length_dt, size = sample_size),]
}

get_stratified_sample = function (dt, strat_by, sample_intensity=0.1) {
      # dt contains the data
      # strat_by contains the name of the variable according to which the sample is to be stratified
      # sample intensity is proportion of data to be sampled, default = 10%
      # deterine the number of levels of factor 'strat_by' and store the number of rows relevant to the factor in table tmp
      f = factor (dt[,strat_by])
      tmp = aggregate (dt[,strat_by], by=list(f), FUN=length)
      #
      # determine the number of rows to be included within the sample for each level of the factor
      tmp$sz = round(tmp[,2] * sample_intensity, 0)
      #
      f.number = length (tmp [,1])      # determine the number of levels of the factor
      #
      # obtain the sample for the first level and store in strat_sample
      strat_sample = random_sample (subset (dt, f == tmp[1,1]), tmp[1,3])
      # obtain subsample for the remaining levels of the factor.
      if (f.number > 1) {
            for (i in 2:f.number) { 
                  rs = random_sample (subset (dt, f == tmp[i,1]), tmp[i,3])
                  strat_sample = rbind (strat_sample, rs)
            }
      }
      strat_sample    # returned to calling statement.
}

sample4plot <- get_stratified_sample(Train, "classe", 0.1)

panel.cor <- function(x, y, digits=2, prefix="", cex.cor, ...)
{
      usr <- par("usr"); on.exit(par(usr))
      par(usr = c(0, 1, 0, 1))
      r <- abs(cor(x, y))
      txt <- format(c(r, 0.123456789), digits=digits)[1]
      txt <- paste(prefix, txt, sep="")
      if(missing(cex.cor)) cex.cor <- 0.8/strwidth(txt)
      text(0.5, 0.5, txt, cex = cex.cor * r)
}
png(filename = "plot1_scatterplot.png",
    width = 480, height = 480, units = "px",
    bg = "white")
pairs(sample4plot[,2:11],lower.panel=panel.smooth, upper.panel=panel.cor)
dev.off()

summary(Train)
#------------------------------------------
#4. Feature Selection

# 4.1 Remove near zero variance
#nzv <- nearZeroVar(Train, freqCut = 99/1, uniqueCut = 1)
nzv <- nearZeroVar(Train)
length(nzv) #0 variables

#4.2 Remove high correlation
corr_mat <- cor(Train[, -1])
a <- data.frame(corr_mat)
too_high <- findCorrelation(corr_mat, cutoff = .85, verbose = T)
length(too_high+1) 
Train1 <- Train[, - (too_high+1)] #51-8 = 43 features left
Test1 <- Test[,- (too_high+1) ]


#4.3 Recursive Feature Elimination
x.train <- Train1[,-1]
y.train <- Train1[,1]

normalization <- preProcess(x.train)
x.train <- predict(normalization, x.train) # normalization
x.train <- as.data.frame(x.train)

subsets <- c(1:5, 10, 15, 20, 25) #fit models with subset sizes of 25, 20, 15, 10, 5, 4, 3, 2, 1;

set.seed(1)
newRF <- rfFuncs
ctrl <- rfeControl(functions = newRF, 
                   method = "repeatedcv",
                   repeats = 5,
                   verbose = T)

#many many hours to run!
rfRFE <- rfe(x.train, 
             y.train, 
             sizes = subsets, 
             metric = "ROC",
             rfeControl = ctrl)

# decide number of features to select
rfRFE

png(filename = "plot2_NumberFeatures.png",
    width = 480, height = 480, units = "px",
    bg = "white")

trellis.par.set(caretTheme()) #suggest 10 to 15 features
plot(rfRFE, type = c("g", "o")) #10 accuracy ~ .9801 while 15 = 0.9858

dev.off()

#save the importance
Imp1<-varImp(rfRFE$fit)
Imp1$Variable <- rownames(Imp1)
dim(Imp1)
colnames(Imp1) <- c("Importance","Variable")

sort.Imp1 <- Imp1[order(-Imp1$Importance),]
#save the feature importance
write.table(sort.Imp1, file = "./rfRFE_selectedVar.txt", col.names = TRUE, sep = "\t")

#4.4 selected top 15 features:
Top15.features <- sort.Imp1$Variable[1:15]

Train2 <- cbind(classe = y.train, x.train[, Top15.features])

#Fit random forest with the top 15 selected features (don't use formula format - slow)
#modFit.rf <- train(classe ~ .,data = Train2, method = "rf", prox = T, verbose = T)

set.seed(2)
modFit.rf <- randomForest(y=Train2[,1], x=Train2[,-1], 
                          mtry=10, ntree=2000, type="classification", prox=T)
modFit.rf #OOB estimate of error rate = 1.3% (trianing)


# (don't run) class centers
Top15.features #yaw_belt, magnet_dumbbell_z
P <- classCenter(Train2[,c(2,3)], Train2$classe, modFit.rf$proximity)
P <- as.data.frame(P)
P$classe <- rownames(P)
g <- qplot(yaw_belt, magnet_dumbbell_z, col = classe, data = Train2)
g+geom_point(aes(x = yaw_belt, y = magnet_dumbbell_z, col = classe), size = 7, shape = 6, data = P)


#4.5 fit Test data
x.test <- predict(normalization, Test1[,-1]) # normalization use training mean and sd
x.test <- as.data.frame(x.test)

pred <- predict(modFit.rf, x.test)
confusionMatrix(pred, Test1[,1]) #accuray = 0.985 sample error rate ~ 1.5%

#------------------------------------------
#5. Apply the formula to the outdata (20 obs)
x.outdata <- predict(normalization, outdata[,names(x.train)])
pred <- predict(modFit.rf, x.outdata)
pred

#------------------------------------------
#6. output answers for submissions
pml_write_files = function(x){
      n = length(x)
      for(i in 1:n){
            filename = paste0("problem_id_",i,".txt")
            write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
      }
}

pml_write_files(pred)


