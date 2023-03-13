# APS MODELAGEM PREDITIVA VANÇADA - redes neurais
#Lila Hadba e Michel Wachslicht

library(keras)

# setwd("~/Michel/Insper/AULAS 5 SEMESTRE/Modelagem Preditiva Avançada/APS rede neural")

churn <- read.csv("churn_training.csv")

churn$churned <- ifelse(churn$churned == "no", 0, 1)

#length(churn)
#length(churn$churned)
#sum(is.na(churn$churned))

idx <- sample(1:nrow(churn), size = round(0.7 * nrow(churn)))
training <- churn[idx, ]
test <- churn[-idx, ]

x_trn <- as.matrix(training[, 1:15])
y_trn <- to_categorical(training$churned, 2)

x_tst <- as.matrix(test[, 1:15])
y_tst <- test$churned

x_trn <- scale(x_trn)
x_tst <- scale(x_tst, center = attr(x_trn, "scaled:center"),
               scale = attr (x_trn, "scaled:scale"))


rede <- keras_model_sequential()
rede <- layer_dense(rede, units = 16, activation = "relu", input_shape = ncol(x_trn)) 
rede <- layer_dropout(rede, rate = 0.2)
rede <- layer_dense(rede, units = 64, activation = "relu")
rede <- layer_dense(rede, units = 32, activation = "relu")
rede <- layer_dropout(rede, rate = 0.2)
rede <- layer_dense(rede, units = 16, activation = "relu")
rede <- layer_dense(rede, units = 2, activation = "softmax")

summary(rede)

rede <- compile(rede, loss = "categorical_crossentropy",
                optimizer = "adam",
                metrics = "accuracy")

history <- fit(rede, x_trn, y_trn, btch_size = 128, epochs = 50,
               validation_split = 0.2)

y_hat_net <- predict(rede, x_tst) %>% k_argmax() %>% as.numeric()

mean(y_hat_net != y_tst)



# loss: 0.2068 - accuracy: 0.9271 - val_loss: 0.2041 - val_accuracy: 0.9343

# save_model_hdf5(rede, "modelo_final.hdf5")


library(ranger)

rf <- ranger(churned ~ ., data = training, probability = TRUE)
prob_rf <- predict(rf, data = test)$predictions[, 2] # "data" instead of "newdata"
y_hat_rf <- ifelse(prob_rf >= 0.5, 1, 0)
mean(y_hat_rf != y_tst)
mean(y_hat_net != y_hat_rf)
