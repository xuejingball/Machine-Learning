library("ggplot2")
library("reshape2")
data1 = read.csv("J48-confidence factor.csv")
df1 = as.data.frame(data1)
ggplot(df1, aes(x=confidence.factor, y=error, colour=mode)) + 
  geom_line() + geom_point() + 
  scale_color_manual(labels = c("crossValidation","train"), values = c("red", "black")) +
  ggtitle("Confidence factor Tuning for J48 model")+
  xlab("confidence factor")+
  ylab("error rates")
ggsave("J48-confidence factor.png")

data2 = read.csv("J48-nodes.csv")
df2 = as.data.frame(data2)
ggplot(df2, aes(x=nodes, y=error, colour=mode)) + 
  geom_line() + geom_point() + 
  scale_color_manual(labels = c("crossValidation","train"), values = c("red", "black")) +
  ggtitle("Minimal number of objects Tuning for J48 model") +
  xlab("minimal number of objects")+
  ylab("error rates")
ggsave("J48-nodes.png")

dataJ48 = read.csv("J48-learning curve.csv")
dfJ48 = as.data.frame(dataJ48)
ggplot(dfJ48, aes(x=percentage, y=error, colour=mode)) + 
  geom_line() + geom_point() + 
  scale_color_manual(labels = c("crossValidation","train"), values = c("orange", "blue")) +
  ggtitle("J48 model learning curve") +
  xlab("sample(%)")+
  ylab("error rates")
ggsave("J48-learning curve.png")

#----------------------------------------------------------

data3 = read.csv("KNN-K.csv")
df3 = as.data.frame(data3)
ggplot(df3, aes(x=K, y=error, colour=mode)) + 
  geom_line() + geom_point() + 
  scale_color_manual(labels = c("crossValidation","train"), values = c("red", "black")) +
  ggtitle("K value Tuning for IBk model") +
  xlab("K value")+
  ylab("error rates")
ggsave("KNN-K.png")


dataKNN = read.csv("KNN-learning curve.csv")
dfKNN = as.data.frame(dataKNN)
ggplot(dfKNN, aes(x=percentage, y=error, colour=mode)) + 
  geom_line() + geom_point() + 
  scale_color_manual(labels = c("crossValidation","train"), values = c("orange", "blue")) +
  ggtitle("IBk model learning curve") +
  xlab("sample(%)")+
  ylab("error rates")
ggsave("KNN-learning curve.png")

#-----------------------------------------------------------

data4 = read.csv("NN-hidden layers.csv")
df4 = as.data.frame(data4)
ggplot(df4, aes(x=NumOfNodes, y=error, colour=mode)) + 
  geom_line() + geom_point() + 
  scale_color_manual(labels = c("crossValidation","train"), values = c("red", "black")) +
  ggtitle("Number of nodes in the hidden layer tuning for Multilayerperceptron model") +
  xlab("Number of nodes in the hidden layer")+
  ylab("error rates")
ggsave("NN-hiddenlayer.png")

data5 = read.csv("NN-learning rate.csv")
df5 = as.data.frame(data5)
ggplot(df5, aes(x=learning.rate, y=error, colour=mode)) + 
  geom_line() + geom_point() + 
  scale_color_manual(labels = c("crossValidation","train"), values = c("red", "black")) +
  ggtitle("Learning Rate Tuning for Multilayerperceptron model") +
  xlab("learning rate")+
  ylab("error rates")
ggsave("NN-learning rate.png")

data6 = read.csv("NN-train times.csv")
df6 = as.data.frame(data6)
ggplot(df6, aes(x=train.times, y=error, colour=mode)) + 
  geom_line() + geom_point() + 
  scale_color_manual(labels = c("crossValidation","train"), values = c("red", "black")) +
  ggtitle("train times Tuning for Multilayerperceptron model") +
  xlab("train times")+
  ylab("error rates")
ggsave("NN-train times.png")

dataNN = read.csv("NN-learning curve.csv")
dfNN = as.data.frame(dataNN)
ggplot(dfNN, aes(x=percentage, y=error, colour=mode)) + 
  geom_line() + geom_point() + 
  scale_color_manual(labels = c("crossValidation","train"), values = c("orange", "blue")) +
  ggtitle("NN model learning curve")+
  xlab("sample(%)")+
  ylab("error rates")
ggsave("NN-learning curve.png")

#--------------------------------------------------------------------------------------
data7 = read.csv("Boosting-train times.csv")
df7 = as.data.frame(data7)
ggplot(df7, aes(x=train.times, y=error, colour=mode)) + 
  geom_line() + geom_point() + 
  scale_color_manual(labels = c("crossValidation","train"), values = c("red", "black")) +
  ggtitle("train times Tuning for AdaBoostM1 model") +
  xlab("train times")+
  ylab("error rates")
ggsave("Boosting-train times.png")

dataBoost = read.csv("Boosting-learning curve.csv")
dfBoost = as.data.frame(dataBoost)
ggplot(dfBoost, aes(x=percentage, y=error, colour=mode)) + 
  geom_line() + geom_point() + 
  scale_color_manual(labels = c("crossValidation","train"), values = c("orange", "blue")) +
  ggtitle("Boosting model learning curve") +
  xlab("sample(%)")+
  ylab("error rates")
ggsave("Boosting-learning curve.png")

#--------------------------------------------------------------------------------------
data8 = read.csv("SMO-poly-complexity.csv")
df8 = as.data.frame(data8)
ggplot(df8, aes(x=complexity, y=error, colour=mode)) + 
  geom_line() + geom_point() + 
  scale_color_manual(labels = c("crossValidation","train"), values = c("red", "black")) +
  ggtitle("Complexity parameter tuning for SMO model with Polynomial kernel") +
  xlab("c")+
  ylab("error rates")
ggsave("SMO-poly-complexity.png")

data9 = read.csv("SMO-RBF-complexity.csv")
df9 = as.data.frame(data9)
ggplot(df9, aes(x=complexity, y=error, colour=mode)) + 
  geom_line() + geom_point() + 
  scale_color_manual(labels = c("crossValidation","train"), values = c("red", "black")) +
  ggtitle("Complexity parameter tuning for SMO model with Gaussian kernel") +
  xlab("c")+
  ylab("error rates")
ggsave("SMO-RBF-complexity.png")


data10 = read.csv("SMO-RBF-gamma.csv")
df10 = as.data.frame(data10)
ggplot(df10, aes(x=gamma, y=error, colour=mode)) + 
  geom_line() + geom_point() + 
  scale_color_manual(labels = c("crossValidation","train"), values = c("red", "black")) +
  ggtitle("Gamma tuning for SMO model with Gaussian kernel") +
  xlab("gamma")+
  ylab("error rates")
ggsave("SMO-RBF-gamma.png")

dataSMO = read.csv("SMO-learning curve.csv")
dfSMO = as.data.frame(dataSMO)
ggplot(dfSMO, aes(x=percentage, y=error, colour=mode)) + 
  geom_line() + geom_point() + 
  scale_color_manual(labels = c("PolynomialCV","PolynomialTrain", "GaussianCV", "GaussianTrain"), 
                     values = c("red", "black", "orange", "blue")) +
  xlab("samle(%)")+
  ylab("error rates")
  ggtitle("SMO model learning curve")
ggsave("SMO-learning curve.png")


dataSMOP = read.csv("SMO-poly-learning curve.csv")
dfSMOP = as.data.frame(dataSMOP)
ggplot(dfSMOP, aes(x=percentage, y=error, colour=mode)) + 
  geom_line() + geom_point() + 
  scale_color_manual(labels = c("crossValidation","train"), values = c("orange", "blue")) +
  ggtitle("SMO PolyKernel model learning curve")
ggsave("SMO PolyKernel-learning curve.png")



dataSMORBF = read.csv("SMO_RBF-learning curve.csv")
dfSMORBF = as.data.frame(dataSMORBF)
ggplot(dfSMORBF, aes(x=percentage, y=error, colour=mode)) + 
  geom_line() + geom_point() + 
  scale_color_manual(labels = c("crossValidation","train"), values = c("orange", "blue")) +
  ggtitle("SMO RBF model learning curve")
ggsave("SMO RBF-learning curve.png")