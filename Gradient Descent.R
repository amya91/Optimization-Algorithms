# 1. Gradient Descent Algorithm

dat = read.csv("MLR.csv", header = F)
true_beta = as.matrix(read.csv("True_Beta.csv", header = F),nrow = 30,ncol =1)

X = as.matrix(dat[,1:30])
Y = as.matrix(dat[,31])

#Part B: Step-size:
H_hat = t(X) %*% X 
eig_vals = eigen(H_hat) 
alpha = 1/sort(eig_vals$values, decreasing = T)[1]

#Gradient descent Algorithm
beta = matrix(0,nrow=30,ncol=1)
flag =1
j = 0

cost_old = 1/(2*nrow(X))*norm(((X%*%beta)-Y),"2")^2
cost_vals = c(cost_old)
while(flag){
  delta = t(X)%*%((X%*%beta)-Y)
  beta_new = beta - alpha/nrow(X)*delta
  cost = 1/(2*nrow(X))*norm(((X%*%beta_new)-Y),"2")^2
  beta = beta_new
  if(abs(cost-cost_old)<0.00001){
    flag = 0
  }
  cost_old = cost
  cost_vals = c(cost_vals,cost_old)
}

norm((beta-true_beta),"2")^2/30

plot(cost_vals, type = "l", 
     main = "Plot of cost against number of iterations (Gradient Descent)", 
     xlab = "Number of iterations", ylab = "Cost", col = "black")