# 1. Stochastic Gradient Descent Algorithm
dat = read.csv("Data/MLR.csv", header = F)
true_beta = as.matrix(read.csv("True_Beta.csv", header = F),nrow = 30,ncol =1)

X = as.matrix(dat[,1:30])
Y = as.matrix(dat[,31])

#Step-size:
H_hat = t(X) %*% X 
eig_vals = eigen(H_hat) 
alpha = 1/sort(eig_vals$values, decreasing = T)[1]

alpha_sgd = alpha/nrow(X)
beta = matrix(0,nrow=30,ncol=1)
flag =1
j = 0
cost = rep(0,nrow(X))
avg_cost = mean(cost)
cost_plot = c(avg_cost)

avg_cost = 1/(2*nrow(X))*norm(((X%*%beta)-Y),"2")^2
cost_plot = c(avg_cost)
while(flag){
  indices = sample(nrow(X), size = nrow(X), replace = FALSE)
  X_new = X[indices,]
  Y_new = as.matrix(Y[indices,],nrow =1000,ncol=1)
  for(i in indices){
    delta = as.matrix(X_new[i,])%*%((t(X_new[i,])%*%beta)-Y_new[i])
    beta_new = beta - alpha_sgd*delta
    cost[i] = 1/2*norm(((t(X_new[i,])%*%beta_new)-Y_new[i,]),"2")^2
    beta = beta_new
  }
  if(abs(mean(cost)-avg_cost)<0.00001){
    flag = 0
  }
  avg_cost = mean(cost)
  cost_plot = c(cost_plot,avg_cost)
}

norm((beta-true_beta),"2")^2/30

plot(cost_plot, type = "l", 
     main = "Plot of cost against number of iterations (Stochastic Gradient Descent)", 
     xlab = "Number of iterations", ylab = "Cost", col = "black")