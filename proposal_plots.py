# so far, this code only uses linear relationships, we could go for non-linear
import numpy as np
import pandas as pd
import pyreadr
import matplotlib.pyplot as plt


data1 = pyreadr.read_r('/Users/ratmir/rdata/data_noint.RData')



x1 = data1["x1_sort"]
x2 = data1["x2_sort"]
X, Y = np.meshgrid(x1,x2)

#pop surface and es surface
surface1 = data1["surface1"] #plot
surface1_est = data1["surface1_est_nn"] #plot

#par dep surface
surface1_partial = data1["g1_est_sur"] #plot

#residual surface
surface1_residual = data1["res_surface"] #plot


#plot1: pop sur vs est sur
fig = plt.figure()
ax = plt.axes(projection='3d')
#colors
#c("rgb(0,3,140)","rgb(0,3,140)") true; opacity=1
#c("rgb(255,107,184)","rgb(128,0,64)") est; opacity=0.3
ax.plot_surface(Y, X, surface1, color=[0, 0.12, 0.55, 0.75]) # 	0, 1.2, 54.9
ax.plot_surface(Y, X, surface1_est, color=[0.9, 0, 0.251, 0.4]) # 50.2, 0, 25.1
ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')
ax.set_zlabel('$\phi_1$, $\widehat{\phi}_1$')
ax.view_init(30, -80)
plt.show()
plt.savefig('sur_noint.png', transparent=True, bbox_inches='tight')

#plot2: est sur vs partial dep
fig = plt.figure()
ax = plt.axes(projection='3d')
#colors
#c("rgb(0,3,140)","rgb(0,3,140)") true; opacity=1
#c("rgb(255,107,184)","rgb(128,0,64)") est; opacity=0.3
ax.plot_surface(Y, X, surface1_partial, color=[0, 0.12, 0.55, 0.75]) # 	0, 1.2, 54.9
ax.plot_surface(Y, X, surface1_est, color=[0.9, 0, 0.251, 0.4]) # 50.2, 0, 25.1
ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')
ax.set_zlabel('$\widehat{\phi}_1$, $\widehat{\operatorname{g}}_1$')
ax.view_init(30, -71)
plt.show()
plt.savefig('partial_int.png', transparent=True, bbox_inches='tight')


#plot3: residuals
fig = plt.figure()
ax = plt.axes(projection='3d')
#colors
#c("rgb(0,3,140)","rgb(0,3,140)") true; opacity=1
#c("rgb(255,107,184)","rgb(128,0,64)") est; opacity=0.3
ax.plot_surface(Y, X, surface1_residual, color=[0, 0.12, 0.55, 0.9]) # 	0, 1.2, 54.9
ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')
ax.set_zlabel('$\widehat{\operatorname{r}}_1^2$')
ax.view_init(30, 144)
plt.show()
plt.savefig('res_noint.png', transparent=True, bbox_inches='tight')








