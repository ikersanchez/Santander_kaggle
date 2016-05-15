#######################################
### Santander Customer Satisfaction ###
#######################################


#XGB
#Just trying bayesian optimization

	library(Matrix)
	library(xgboost)


	setwd("C:/Users/Iker/Desktop/Santander")
	train <- read.csv("train.csv",stringsAsFactors = F)
	test <- read.csv("test.csv",stringsAsFactors = F)

	test$TARGET <- rep(-1,nrow(test))

	full <- rbind(train,test)
	Idx <- which(full$TARGET == -1)

#Vars

	y_true <- train$TARGET
	full$TARGET <- NULL


#Sum of zeros

	full$SumZeros <- apply(full, 1, function(x) sum(x == 0))

# Removing constant features

	for (f in names(full)) {
			if (length(unique(full[[f]])) == 1) {
					cat(f, "is constant in train. We delete it.\n")
					full[[f]] <- NULL
				
			}
	}


#Remove duplicate columns

	DuplicateColumns <- which(duplicated(t(full))== TRUE)
	ToRemove <- names(full[,DuplicateColumns]) 
	vars <- setdiff(names(full),ToRemove)
	full <- full[,vars]


#Matrix

	fullmat <- sparse.model.matrix(ID ~ .-1,data = full)
	trainmat <- fullmat[-Idx,]
	testmat <- fullmat[Idx,]



#Parameters from bayesian opt. script
	param <- list(  objective           = "binary:logistic", 
					booster             = "gbtree",
					eval_metric         = "auc",
					eta                 = 0.01,
					max_depth           = 5,
					subsample           = 0.7,
					colsample_bytree    = 0.8,
					min_child_weight    = 2
					#scale_pos_weight = 1.30 this param improves my private leaderboard score (~163th) -> deals with imbalanced dataset
	)

#Cross validation

	set.seed(1234)
	cv <- xgb.cv(params = param,data = trainmat,label = y_true,verbose = T,predictions = T,nfold = 5,nround = 1000)
	min.error <- which.max(cv[,test.auc.mean])
	cv[min.error,]

#Model

	set.seed(1234)
	model1 <- xgboost(param = param,data = trainmat,label = y_true,nrounds = min.error)

#Predictions
	pred <- predict(model1,testmat)

#Submision

	sub <- data.frame(ID = test$ID, TARGET= pred)
	write.csv(sub,"xgb1.csv",row.names = FALSE)

#Importance

	#Variable importance plot
	gc()
	model <- xgb.dump(model1,with.stats = TRUE)
	names <- dimnames(trainmat)[[2]]
	importance <- xgb.importance(names,model = model1)
	data.frame(importance)
	p <- xgb.plot.importance(importance[1:15])
	print(p)