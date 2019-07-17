#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 19 11:31:12 2019

@author: jps99
"""

#Create images under the GSM

import numpy as np
from numpy.linalg import pinv
import matplotlib.pyplot as plt
from SSN_mod import create_A2
from scipy.stats import norm, multivariate_normal
#plt.style.use(['ggplot','desktop_screen'])

#%% Create everything to begin with
N_E = 50

#Create covariance matrix 'C' of the covariation between gabor filters for generating a feature vector 'y'
theta = np.linspace(0,np.pi,51)
theta = np.concatenate((theta[:-1],theta[:-1]))
def create_cov_C():
    C = np.zeros((N_E,N_E))
    for i in range(N_E-1):
        for j in range(i+1,N_E):
            C[i,j] = 2.5*np.exp((np.cos(2*(theta[i] - theta[j])) - 1)/((np.pi/6)**2))
    return C+C.T + 2.5*np.diag(np.ones(N_E)) + np.diag(np.ones(N_E))*(1e-9) #Addition of small diagonal to aid inversion

C = create_cov_C()
plt.imshow(C); plt.colorbar()

y_mean = 5*np.exp((np.cos(2*(theta[:N_E] - np.pi/2)) - 1)/((np.pi/5)**2))
plt.plot(y_mean)

#Create my matrix of vectorised gabor filters
A = create_A2(N_E,16)
A = A[:,:N_E]

#%%
sig_x = 10

#np.savez('GSM_tgt_arrays4',C=C,y_mean=y_mean,sig_x=sig_x)

#Save the mean and covariance
c = 0
mu_array_tgt = np.zeros((N_E,5))
cov_array_tgt = np.zeros((N_E,N_E,5))
for z in [0,0.12,0.25,0.5,1]:
    x = z*A@y_mean
    temp = pinv(C) + ((z**2)/(sig_x**2))*A.T@A
    sig = pinv(temp)
    mu_array_tgt[:,c] = (np.reshape((z/sig_x**2)*sig@A.T@x,(N_E))).T
    cov_array_tgt[:,:,c] = sig
    c=c+1
np.savez('mu_cov_tgt_5_images4',mu_array_tgt=mu_array_tgt,cov_array_tgt=cov_array_tgt)

#Generate and save training 5 images
x1 = 1*A@y_mean #+ np.random.normal(0,sig_x,(16**2,1))
x2 = 0.5*A@y_mean #+ np.random.normal(0,sig_x,(16**2,1))
x3 = 0.25*A@y_mean #+ np.random.normal(0,sig_x,(16**2,1))
x4 = 0.12*A@y_mean #+ np.random.normal(0,sig_x,(16**2,1))
x5 = 0.0*A@y_mean #np.random.normal(0,sig_x,(16**2,1))

#plt.imshow(np.reshape(x1,(16,16))); plt.colorbar()
np.savez('5_tgt_images4',x1=x1,x2=x2,x3=x3,x4=x4,x5=x5)

#%% Now load and plot everything
#data = np.load('GSM_tgt_arrays4.npz')
#C=data['C']
#y_mean=data['y_mean']
#A = create_A2(N_E,16)
#A = A[:,:N_E]
#
##Plot non transformed moments
#mu_array = np.zeros((N_E,5))
#sd_array = np.zeros((N_E,5))
#c=0
#for z in [0,0.12,0.25,0.5,1]:
#    x = z*A@y_mean
#    temp = pinv(C) + ((z**2)/(sig_x**2))*A.T@A
#    sig = pinv(temp)
#    mu_array[:,c] = (np.reshape((z/sig_x**2)*sig@A.T@x,(N_E))).T
#    sd_array[:,c] = np.reshape(np.sqrt(np.diag(sig)),(N_E))
#    c=c+1
#
#f, axarr = plt.subplots(1,2, figsize = (10,5))
#ax= axarr[0].plot(mu_array)
#ax = axarr[1].plot(sd_array)
#
#data = np.load('5_tgt_images4.npz')
#x1=data['x1']
#x2=data['x2']
#x3=data['x3']
#x4=data['x4']
#x5=data['x5']
#plt.imshow(np.reshape(x1,(16,16))); plt.colorbar()
#plt.imshow(np.reshape(x2,(16,16))); plt.colorbar()
#plt.imshow(np.reshape(x3,(16,16))); plt.colorbar()
#plt.imshow(np.reshape(x4,(16,16))); plt.colorbar()
#plt.imshow(np.reshape(x5,(16,16))); plt.colorbar()

#%% NL transformation of y
nl_scale = 3#2.4
nl_power = 0.6
nl_baseline = (3.5/nl_scale)**(1.0/nl_power)
N_pat = 5

def np_th(u):
    return np.maximum(u,0)
  
def nl_fun(u,nl_scale,nl_baseline,nl_power):
    return nl_scale*(np_th(u+nl_baseline)**nl_power)

def get_mean_var(points_1d,mu,std,nl_scale,nl_baseline,nl_power):
    new_mean = np.empty([N_E])
    new_var = np.empty([N_E])
    for i in range(N_E):
        mu_i = mu[i]
        std_i = std[i]    
        points_resc = std_i * points_1d + mu_i
        values_dist = norm.pdf(points_resc, loc = mu_i, scale = std_i)
        norm_f = np.sum(values_dist)
        values_fun = nl_fun(points_resc, nl_scale,nl_baseline,nl_power)
        new_mean[i] = np.sum(values_dist*values_fun)/norm_f
        new_var[i] = np.sum(values_dist*(values_fun-new_mean[i])**2.0)/norm_f    
    return (new_mean, new_var)

def get_cov(points_1d,mu,std,Cov,new_mean,new_var,nl_scale,nl_baseline,nl_power):
    new_Cov = np.empty([N_E,N_E])
    n_points = len(points_1d)
    for i in range(N_E):
        new_Cov[i,i] = new_var[i]
        for j in range(i+1,N_E):
            mu_i = mu[i]
            std_i = std[i]
            mu_j = mu[j]
            std_j = std[j]
            mean_red = np.array([mu_i,mu_j])    
            Cov_red = np.array([[Cov[i,i],Cov[i,j]],[Cov[j,i],Cov[j,j]]])
            points_resc_i = std_i * points_1d + mu_i
            points_resc_j = std_j * points_1d + mu_j
            X,Y = np.meshgrid(points_resc_i, points_resc_j)
            points_2d = np.vstack((X.flatten(), Y.flatten())).T
            values_dist = multivariate_normal.pdf(points_2d, 
                                    mean=mean_red,cov=Cov_red).reshape(n_points,n_points)
            
            norm_f = np.sum(values_dist)
            values_fun_2 = np.outer(nl_fun(points_resc_i, nl_scale,nl_baseline,nl_power)-new_mean[i],
                                  nl_fun(points_resc_j, nl_scale,nl_baseline,nl_power)-new_mean[j])
            new_Cov[i,j] = np.sum(values_dist*values_fun_2)/norm_f
    
    for i in range(N_E):
        for j in range(0,i):
            new_Cov[i,j] = new_Cov[j,i]
                    
    return new_Cov

points_1d = np.linspace(-4.0,4.0, num = 201)
mu_array_tgt_transformed = np.zeros((N_E,5))
cov_array_tgt_transformed = np.zeros((N_E,N_E,5))
for alpha in range(N_pat):
        # Loading moments
        data = np.load('mu_cov_tgt_5_images4.npz')
        mu_array_tgt = data['mu_array_tgt']
        cov_array_tgt = data['cov_array_tgt']
        
        mu_current = mu_array_tgt[:,alpha]
        Sigma_current = cov_array_tgt[:,:,alpha]
        var = np.diag(Sigma_current)
        std = np.sqrt(var)        
        
        mu_new, var_new = get_mean_var(points_1d,mu_current,std,nl_scale,nl_baseline,nl_power)
        Sigma_new = get_cov(points_1d,mu_current,std,Sigma_current,mu_new,var_new,nl_scale,nl_baseline,nl_power)
        
        mu_array_tgt_transformed[:,alpha] = mu_new
        cov_array_tgt_transformed[:,:,alpha] = Sigma_new
 
np.savez('mu_cov_tgt_5_images_transformed4',mu_array_tgt_transformed=mu_array_tgt_transformed,
         cov_array_tgt_transformed=cov_array_tgt_transformed)

#%% plot the transformed moments
data = np.load('mu_cov_tgt_5_images_transformed4.npz')
mu_array_tgt_transformed = data['mu_array_tgt_transformed']
cov_array_tgt_transformed = data['cov_array_tgt_transformed']

#Mean and std
f, axarr = plt.subplots(1,2, figsize = (10,5))
ax= axarr[0].plot(mu_array_tgt_transformed)
sd_array = np.zeros((N_E,5))
for i in range(5):
    sd_array[:,i] = np.reshape(np.sqrt(np.diag(cov_array_tgt_transformed[:,:,i])),(N_E))
ax = axarr[1].plot(sd_array)

#Covariances
f, axarr = plt.subplots(5, figsize = (5,12))
for image in range(5):    
    ax1 = axarr[image].imshow(cov_array_tgt_transformed[:,:,image],vmin=-1,vmax=1, cmap = 'bwr')
    
f.subplots_adjust(right=0.8)
cbar_ax = f.add_axes([0.85, 0.12, 0.05, 0.29])
f.colorbar(ax1,cax=cbar_ax)

