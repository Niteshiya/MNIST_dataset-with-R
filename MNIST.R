#MNIST
library(keras)
mnist <- dataset_mnist()
str(mnist)
#Train and test data
train <- mnist$train$x
train_lab <- mnist$train$y
test <- mnist$test$x
test_lab <- mnist$test$y
#plot images
par(mfrow=c(3,3))
for(i in 1:9){
  plot(as.raster(train[i,,],max=255))
}
par(mfrow=c(1,1))
hist(train[1,,],col=rainbow(10))
#reshape and rescale
train <- array_reshape(train,c(nrow(train),28*28))
str(train)
test <- array_reshape(test,c(nrow(test),28*28))
train <- train/255
test <- test /255
#One hot encoding
train_lab <- to_categorical(train_lab,10)
test_lab <- to_categorical(test_lab,10)

#model
model <- keras_model_sequential()

model %>% 
  layer_dense(units = 50,activation = "relu",input_shape=c(784)) %>%
  layer_dense(units=10,activation = "softmax")
model
#compile model
model %>% 
  compile(loss="categorical_crossentropy",
          optimizer=optimizer_rmsprop(),
          metrics="accuracy")
#fit model
history <- model %>%
  fit(train,train_lab,epochs=55,batch_size=32,validation_split=0.22)
plot(history)

#Evaluation and prediction
model %>%
  evaluate(test,test_lab)
#predict_proba() for probability
pred <- model %>% predict_classes(train)
table(predicted = pred,actual=mnist$train$y)
pred1 <- model %>% predict_classes(test)
table(predicted=pred1,actual=mnist$test$y)
