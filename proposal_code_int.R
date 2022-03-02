#main code
library(plotly)
library(akima)
library(rgl)
library(h2o)
library(devtools)
library(data.table)
library(tidyr)
library(pracma)
library(pgam)
h2o.init()

#set.seed(3)
# generate data set
#x1 = runif(600, -2, 2) #unsorted; mean zero
#x2 = runif(600, -2, 2) #unsorted
x1 = seq(from=-2, to=2, length.out=600)
x2 = seq(from=-2, to=2, length.out=600)
x1 = sample(x1, replace=FALSE)
x2 = sample(x2, replace=FALSE)

n = length(x1)
g1 = function(x1){
  return(sin(2*x1))
}
g2 = function(x2){
  return((x2)^2 - mean( (x2)^2 ) ) # (X2)**2 - np.mean((X2)**2)
}
#interaction term
int = function(x1, x2){
  return( pmax(x1, 0)*(pmax(x2, 0)) ) 
}

m = g1(x1) + g2(x2) + int(x1, x2)
y = m  + 0.1*rnorm(length(m))

model_nn = function(y, data){
  df=cbind(y, data)
  df=as.h2o(df)
  # Build and train the model:
  dl <- h2o.deeplearning(x = 2:ncol(df),
                         y = 1,
                         hidden = c(64, 64),
                         epochs = 1000,
                         activation = "Rectifier",
                         training_frame = df,
                         loss = "Quadratic",
                         stopping_rounds=0,
                         regression_stop=-1,
                         nfolds=10,
                         fold_assignment = "Random",
                         mini_batch_size = 10)
}

library(xgboost)
model_xgb = function(y, data){
  xgb_train = xgb.DMatrix(data = data, label = y)
  # Build and train the model:
  # Build and train the model:
  xgbc = xgboost(data = xgb_train, max.depth = 100, nrounds = 500)
  return(xgbc)
}



#pred_y = predict(xgbc, xgb_train)





#sorting over x2 and then over x1
{
  df_dat=cbind(x1, x2) # 
  df_dat_grid=tidyr::expand_grid(df_dat[, 1], df_dat[, 2]) #n^2 x 2 
  mat = as.matrix(df_dat_grid)
  arr = array(t(mat), dim= c(ncol(df_dat), n, n))
  ordered = array(0, c(n,2,n))
  #sort first by second column (x2)
  for (a in 1:n){
    mat1 = t(arr[, , a])
    ordered[,,a] =mat1[ order(mat1[1:n, 2]), ]
  }
  # array to matrix (there is an easier way)
  ordered_matrix = matrix(0, ncol=2, nrow=n*n)
  for (b in 1:n){
    ordered_matrix[(n*b-(n-1)):(n*b), 1:2] = ordered[,,b] 
  }
  #Now sort by x1
  fill = matrix(0, ncol=2, nrow=n*n)
  for (c in 1:n){
    
    fill[(n*c-(n-1)):(n*c), 1:2 ] =  ordered_matrix[which(ordered_matrix[, 1]==unique(sort(ordered_matrix[,1]))[c]), 1:2]
    
  }
  
  #h2o object for sample
  df_dat_trans=as.h2o(fill) #n^2 x 2 
  names(df_dat_trans)=c("x1", "x2")
  
}






#shapley on population
c = 0
C_1 = 0.5
C_2 =  0.5

m_x1x2 = g1(matrix(df_dat_trans[, 1])) + g2(matrix(df_dat_trans[, 2]) ) + int( matrix(df_dat_trans[, 1]) , matrix(df_dat_trans[, 2]) ) 
m_x1 = g1(matrix(df_dat_trans[, 1]))
m_x2 = g2(matrix(df_dat_trans[, 2]))
#surface1 population
surface1_temp= as.matrix( C_1*(  m_x1x2 - m_x2 ) + C_2*( m_x1 - c) ) # on population: mean(y) = 0
surface1=t(pracma::Reshape(surface1_temp, length(x1), length(x2))) #correct; columns are varied over x2 and rows over x

#surface2 population
surface2_temp = as.matrix( C_1*(  m_x1x2 - m_x1 ) + C_2*( m_x2 - c) ) # on population: mean(y) = 0
surface2 = t(pracma::Reshape(surface2_temp, length(x1), length(x2))) #correct; columns are varied over x2 and rows over x


#shapley on sample
#train model using xgb
#model12 = model_xgb(y, cbind(x1, x2))
#model2 = model_xgb(y, cbind(x2))
#model1 = model_xgb(y, cbind(x1))

#predict using xgb
#m_x1x2_est = as.h2o(predict(model12, as.matrix(df_dat_trans)))
#m_x1_est = as.h2o(predict(model1, as.matrix(df_dat_trans[, 1])))
#m_x2_est = as.h2o(predict(model2, as.matrix(df_dat_trans[, 2] )))

#train using NN
model12_nn = model_nn(y, cbind(x1,x2))
model2_nn = model_nn(y, cbind(x2))
model1_nn = model_nn(y, cbind(x1))
#predict using NN
m_x1x2_est_nn = h2o.predict(model12_nn, as.h2o(df_dat_trans)) 
m_x1_est_nn = h2o.predict(model1_nn, as.h2o(df_dat_trans[, 1]))
m_x2_est_nn = h2o.predict(model2_nn, as.h2o(df_dat_trans[, 2]) )

#surface1 sample
#surface1_temp_est = as.matrix( C_1*(  m_x1x2_est - m_x2_est ) + C_2*( m_x1_est - mean(y)) ) # on sample: c=mean(y) 
#surface1_est=t(pracma::Reshape(surface1_temp_est, length(x1), length(x2))) #correct; columns are varied over x2 and rows over x1

#nn
surface1_temp_est_nn = as.matrix( C_1*(  m_x1x2_est_nn - m_x2_est_nn ) + C_2*( m_x1_est_nn - mean(y)) ) # on sample: c=mean(y) 
surface1_est_nn=t(pracma::Reshape(surface1_temp_est_nn, length(x1), length(x2))) #correct; columns are varied over x2 and rows over x1


#nn surf2
surface2_temp_est_nn = as.matrix( C_1*(  m_x1x2_est_nn - m_x1_est_nn ) + C_2*( m_x2_est_nn - mean(y)) ) # on sample: c=mean(y) 
surface2_est_nn=t(pracma::Reshape(surface2_temp_est_nn, length(x1), length(x2))) #correct; columns are varied over x2 and rows over x1




##################################POPULATION PLOTS#######################

par(mar = c(5.1, 1.5, 0, 0.8))
# assume: z is (x times y)
fig = plot_ly(x=sort(x1[1:n]), y=sort(x2[1:n]), z=surface1[1:n,1:n], type="surface") %>% hide_colorbar()
fig <- fig %>% layout(scene=list(xaxis=list(title='x2'),yaxis=list(title='x1'),zaxis=list(title='shap1')))
#fig

###comparison to g1
#partial1 = g1(x1)
#data_full = cbind(partial1, x1)
#ord=data_full[order(data_full[ , 2]),]
#plot(sort(x1), ord[, 1] ,type="l")
###sanity check
#lines(sort(x1), surface1[,1], col="blue")

###comparison to g2
#partial2 = g2(x2)
#data_full2 = cbind(partial2, x2)
#ord2=data_full2[order(data_full2[ , 2]),]
#plot(sort(x2), ord2[, 1] ,type="l")
####sanity check
#lines(sort(x2), surface2[1,], col="blue")
###############################SAMPLEPLOTS######################

fig_est = plot_ly(x=sort(x1[1:n]), y=sort(x2[1:n]), z=surface1_est[1:n,1:n], type="surface") %>% hide_colorbar()
fig_est <- fig_est %>% layout(scene=list(xaxis=list(title='x2'),yaxis=list(title='x1'),zaxis=list(title='shap1')))
#fig_est


#both plots
plot_ly(showscale = FALSE) %>%
  add_surface( x=~sort(x1[1:n]), y=~sort(x2[1:n]), z = ~surface1[1:n,1:n] , opacity = 1, colorscale = list(c(0,1),c("rgb(0,3,140)","rgb(0,3,140)"))) %>%
  #add_surface(x=~sort(x1[1:n]), y=~sort(x2[1:n]),z = ~surface1_est[1:n,1:n], opacity = 0.3, colorscale = list(c(0,1),c("rgb(255,107,184)","rgb(128,0,64)"))) %>%
  add_surface(x=~sort(x1[1:n]), y=~sort(x2[1:n]),z = ~surface1_est_nn[1:n,1:n], opacity = 0.3, colorscale = list(c(0,1),c("rgb(255,107,184)","rgb(128,0,64)"))) %>%
  layout(scene=list(xaxis=list(title='x2'), yaxis=list(title='x1'), zaxis=list(title='shap1')))


plot_ly(showscale = FALSE) %>%
  add_surface( x=~sort(x1[1:n]), y=~sort(x2[1:n]), z = ~surface2[1:n,1:n] , opacity = 1, colorscale = list(c(0,1),c("rgb(0,3,140)","rgb(0,3,140)"))) %>%
  #add_surface(x=~sort(x1[1:n]), y=~sort(x2[1:n]),z = ~surface1_est[1:n,1:n], opacity = 0.3, colorscale = list(c(0,1),c("rgb(255,107,184)","rgb(128,0,64)"))) %>%
  add_surface(x=~sort(x1[1:n]), y=~sort(x2[1:n]),z = ~surface2_est_nn[1:n,1:n], opacity = 0.3, colorscale = list(c(0,1),c("rgb(255,107,184)","rgb(128,0,64)"))) %>%
  layout(scene=list(xaxis=list(title='x2'), yaxis=list(title='x1'), zaxis=list(title='shap2')))





#plot_ly(showscale = FALSE) %>%
#  add_surface( x=~sort(x1[1:n]), y=~sort(x2[1:n]), z = ~surface2[1:n,1:n] , opacity = 1, colorscale = list(c(0,1),c("rgb(0,3,140)","rgb(0,3,140)"))) %>%
#  add_surface(x=~sort(x1[1:n]), y=~sort(x2[1:n]),z = ~surface2_est[1:n,1:n], opacity = 0.3, colorscale = list(c(0,1),c("rgb(255,107,184)","rgb(128,0,64)"))) %>%
#  layout(scene=list(xaxis=list(title='x2'), yaxis=list(title='x1'), zaxis=list(title='shap2')))

###########
#1) surface1 average in x2 direction
shap_mean = rowMeans(surface1_est_nn)

#2) backfitting; regress shap_mean on x1, note that you need to sort x1, since surface corresponds to sorted x1 
#library(gam)
#back1 = gam::gam(shap_mean ~ sort(x1), family=gaussian(link = "identity"))


library(mgcv)

shapley1_obs = numeric(0)
for (k in 1:n){
  shapley1_obs[k] = surface1_temp_est_nn[(k-1)*n + k]
}




back1_new = mgcv::gam(shapley1_obs ~ s(sort(x1)))

#2a) Sanity check
plot(shapley1_obs, col="blue", type="l")
lines(back1_new$fitted.values)

#one dim fitted residuals, instead of surface
res=(back1_new$fitted.values - shapley1_obs)^2



#3) plot estimated partial dep fct against estimated shap1
g1_est = back1_new$fitted.values
g1_est_sur = matrix(rep(g1_est, n), ncol=n)

plot_ly(showscale = FALSE) %>%
  add_surface( x=~sort(x1), y=~sort(x2), z = ~ g1_est_sur, opacity = 1, colorscale = list(c(0,1),c("rgb(0,3,140)","rgb(0,3,140)"))) %>%
  add_surface(x=~sort(x1), y=~sort(x2),z = ~surface1_est_nn, opacity = 0.3, colorscale = list(c(0,1),c("rgb(255,107,184)","rgb(128,0,64)"))) %>%
  layout(scene=list(xaxis=list(title='x2'), yaxis=list(title='x1'), zaxis=list(title='shap1')))


#4) Residuals
res_surface = (g1_est_sur -  surface1_est_nn)^2

plot_ly(showscale = FALSE) %>%
  add_surface( x=~sort(x1), y=~sort(x2), z = ~ res_surface, opacity = 1, colorscale = list(c(0,1),c("rgb(255,107,184)","rgb(128,0,64)"))) %>%
  layout(scene=list(xaxis=list(title='x2'), yaxis=list(title='x1'), zaxis=list(title='residuals1')))




#save and load into python: 1) true shap: x1, sort(x2), surface1
#2) est shap: x1, sort(x2), surface1_est_nn
x1_sort = sort(x1)
x2_sort = sort(x2)
setwd("/Users/ratmir/rdata")
save(x1_sort, x2_sort, surface1, surface2, surface1_est_nn, surface2_est_nn, g1_est_sur, res_surface, file = "data_int.RData")



