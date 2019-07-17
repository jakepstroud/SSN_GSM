#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 13:10:58 2019

@author: jps99
"""
#Evaluate tensors after training

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from SSN_mod import create_A2
#from SSN_mod import create_A
#from tensorflow.contrib.opt.python.training.external_optimizer import ScipyOptimizerInterface
plt.style.use(['ggplot','desktop_screen'])


#Params
N = 100
N_E = 50
N_I = 50
N_pat = 5   #######################################################################
w_ee_np = np.pi/5
w_ei_np = np.pi/5
w_ie_np = np.pi/5
w_ii_np = np.pi/5
h_ee_np = 1.25/18
h_ei_np = 0.65/18
h_ie_np = 1.2/18
h_ii_np = 0.5/18

tau_noise = 50

dt = 1.0
t_final = 500
num_steps = tf.constant(int(t_final/dt),dtype=tf.int32)
over_tau_temp = (1/20)*np.ones(N)
over_tau_temp[N_E:] *= 2
over_tau = tf.stack([tf.diag(np.float32(over_tau_temp))]*N_pat)
noise_step = tf.sqrt(np.float32(dt*2.0*(tau_noise)))

#Load learend variables
data = np.load('Learned_vars_lr_0002_tgt4.npz')
W_f = data['W_f']
Sigma_eta_root_f = data['Sigma_eta_root_f']
h_in_f = data['h_in_f']
alpha_h_f = data['alpha_h_f']
beta_h_f = data['beta_h_f']
power_f = data['power_f']
cost_over_time = data['cost_over_time']
learning_rate = data['learning_rate']

alpha_h = tf.constant(alpha_h_f, dtype=tf.float32)
beta_h = tf.constant(beta_h_f, dtype=tf.float32)
power = tf.constant(power_f, dtype=tf.float32)

theta_array = tf.cast(2*np.pi, tf.float32) * tf.range(N_E, dtype=tf.float32)/tf.cast(N_E, tf.float32)
cos_theta_array = tf.cos(theta_array)
sin_theta_array = tf.sin(theta_array)
template = (tf.expand_dims(cos_theta_array,axis=1) @ tf.expand_dims(cos_theta_array,axis=0) +
            tf.expand_dims(sin_theta_array,axis=1) @ tf.expand_dims(sin_theta_array,axis=0) -
            tf.ones([N_E, N_E], dtype=np.float32))            

#Form W initial
W_EE = h_ee_np * tf.exp(template/tf.square(w_ee_np))  
W_EI = - h_ei_np * tf.exp(template/tf.square(w_ei_np))
W_IE = h_ie_np * tf.exp(template/tf.square(w_ie_np))  
W_II = - h_ii_np * tf.exp(template/tf.square(w_ii_np))

W_init = tf.concat([tf.concat([W_EE,W_EI], axis=1),
                tf.concat([W_IE,W_II], axis=1)], axis =0, name = "W")

#Form W
W = tf.constant(W_f, dtype=tf.float32)
W_stack = tf.stack([W]*N_pat)


Sigma_eta_root = tf.constant(Sigma_eta_root_f,dtype=tf.float32)
Sigma_eta_root_stack = tf.stack([Sigma_eta_root]*N_pat)

Sigma_eta_root_init_np = np.load('Sigma_eta_root_init_np4.npy')

#I-O function
k = 0.3
n = 2
def get_r_tf(x):
    return k*tf.pow(tf.nn.relu(x),n)

#Set up inputs
A = create_A2(N_E,16)
data = np.load('5_tgt_images4.npz')
x1=data['x1']#Contrast = 1
x2=data['x2']
x3=data['x3']
x4=data['x4']
x5=data['x5']
x_all = [x5,x4,x3,x2,x1]
h_in = tf.constant(0.001*A.T@x_all,dtype=tf.float32)        ##############################################################
#0.0005
h = alpha_h*tf.pow(tf.nn.relu(beta_h+h_in),power)
N_trials = 500
h_stack = tf.transpose(tf.squeeze(tf.stack([h]*N_trials)),[1,2,0])


#Define cost

#Load the target moments
data = np.load('mu_cov_tgt_5_images_transformed4.npz')
mu_array_tgt_transformed = data['mu_array_tgt_transformed'] # First entry is 0 contrast
cov_array_tgt_transformed = data['cov_array_tgt_transformed'] # First entry is 0 contrast
mu_tgt = tf.constant(mu_array_tgt_transformed.T,dtype=tf.float32)                             #######################################
cov_tgt = tf.constant(np.transpose(cov_array_tgt_transformed,[2,0,1]),dtype=tf.float32) ##########################################
var_tgt = tf.matrix_diag_part(cov_tgt)
cov_tgt -= tf.matrix_diag(var_tgt)

# Standard Frobenius norm    
def frob_norm(x):
    return tf.sqrt(tf.nn.l2_loss(x))

def tf_cov(x):
    mean_x = tf.reduce_mean(x, axis=2)
    x_shifted = x - tf.expand_dims(mean_x,axis=2)    
    cov_xx = x_shifted@tf.transpose(x_shifted,[0,2,1])/tf.cast(N_trials, tf.float32)
    return cov_xx

def get_cost(net_mu,net_cov):
    net_var = tf.matrix_diag_part(net_cov)
    net_cov_no_sig = net_cov - tf.matrix_diag(net_var)
    mu_cost = frob_norm(net_mu-mu_tgt)
    cov_cost = frob_norm(net_cov_no_sig-cov_tgt)
    var_cost = frob_norm(net_var-var_tgt)
    return (1/(t_final*N_E))*mu_cost + (2/(t_final*N_E))*var_cost + (1/(t_final*N_E**2))*cov_cost


x_list = tf.zeros((N_pat,N,N_trials,1))
x = tf.random_normal((N_pat,N,N_trials),2,1)
ran = tf.random_normal((N_pat,N,N_trials,num_steps),0,1)
eta = tf.constant(np.zeros((N_pat,N,N_trials)),dtype=tf.float32)
cost = tf.constant(0.0,dtype=tf.float32)
cost_time = tf.constant(0,dtype=tf.int32)

#While_loop part
t = tf.constant(0,dtype=tf.int32)
cond = lambda x,eta,t,cost,x_list: tf.less(t,num_steps)

def body(x,eta,t,cost,x_list):
    eta = eta + (1/tau_noise)*(noise_step*Sigma_eta_root_stack@ran[:,:,:,t] - dt*eta)
    x = x + over_tau@(dt*(-x + h_stack + W_stack@(get_r_tf(x)) + eta))# + np.sqrt(dt*2*tau_noise*sigma)@ran[:,t])
    x_list = tf.concat([x_list,tf.expand_dims(x,axis=3)],axis=3)
    cost = tf.cond(t>cost_time, lambda: cost + get_cost(tf.reduce_mean(x[:,:N_E,:], axis=2),tf_cov(x[:,:N_E,:])), lambda: cost)
    return x,eta,t+1,cost,x_list

x,eta,t,cost,x_list = tf.while_loop(cond,body,(x,eta,t,cost,x_list),
                             shape_invariants = (x.get_shape(),eta.get_shape(),t.get_shape(),cost.get_shape(),tf.TensorShape((N_pat,N,N_trials,None))))

final_x_list = tf.stack(x_list)


sess = tf.Session()
x_np, r_np = sess.run([final_x_list,get_r_tf(final_x_list)]); x_np=x_np[:,:,:,1:]; r_np=r_np[:,:,:,1:]
W_init_np = sess.run(W_init)

#%%Plotting
plt.plot(cost_over_time); plt.axis([None,None,0.4,1.9])

#Plot firing rates for each image
image_choice = 0; trial = 1
plt.imshow((r_np[image_choice,:N_E,trial,:]),aspect='auto');plt.colorbar()
plt.imshow((x_np[image_choice,:N_E,trial,:]),aspect='auto');plt.colorbar()

#Plot W before and after
f, axarr = plt.subplots(1,2, figsize = (10,5))
m1 = np.minimum(np.min(W_f),np.min(W_init_np))
m2 = np.maximum(np.max(W_f),np.max(W_init_np))
ax1 = axarr[0].imshow(W_init_np,vmin=m1,vmax=m2)
ax2 = axarr[1].imshow(W_f)
axx = plt.colorbar(ax1,ax=axarr[0])
axx = plt.colorbar(ax2,ax=axarr[1])


#%% Plot mean and stds

#Load rates before training
data = np.load('x_before_training4.npz')
x_np_initial = data['x_np_initial']
r_np_initial = data['r_np_initial']

#Before
sd_array = np.zeros((N_E,N_pat))
f, axarr = plt.subplots(1,2, figsize = (10,5))
ax = axarr[0].plot(np.mean(np.mean(x_np_initial[:,:N_E,:,:],axis=3),axis=2).T)
axarr[0].set(xlabel='Preferred orientation')  
axarr[0].set(ylabel='Mean mem. pot.')  
for i in range(N_pat):
    sd_array[:,i] = np.reshape(np.sqrt(np.diag(np.cov(np.mean(x_np_initial[i,:N_E,:,50:],axis=2)))),(N_E))
ax = axarr[1].plot(sd_array)
axarr[1].set(xlabel='Preferred orientation')
axarr[1].set(ylabel='Std. mem. pot.')     
axarr[0].axis([0,50,2,10])
axarr[1].axis([0,50,0,3]) #[0,50,0.03,0.2]

#After
sd_array = np.zeros((N_E,N_pat))
f, axarr = plt.subplots(1,2, figsize = (10,5))
plt.axis([None,None,1,11])
ax= axarr[0].plot(np.mean(np.mean(x_np[:,:N_E,:,:],axis=3),axis=2).T)
axarr[0].set(xlabel='Preferred orientation')  
axarr[0].set(ylabel='Mean mem. pot.')  
for i in range(N_pat):
    sd_array[:,i] = np.reshape(np.sqrt(np.diag(np.cov(np.mean(x_np[i,:N_E,:,50:],axis=2)))),(N_E))
ax = axarr[1].plot(sd_array)
axarr[1].set(xlabel='Preferred orientation')
axarr[1].set(ylabel='Std. mem. pot.')     
axarr[0].axis([0,50,2,10])
axarr[1].axis([0,50,0,3])

#Targets
f, axarr = plt.subplots(1,2, figsize = (10,5))
mu_array_tgt_transformed_plot = mu_array_tgt_transformed
cov_array_tgt_transformed_plot = cov_array_tgt_transformed
ax= axarr[0].plot(mu_array_tgt_transformed_plot)
axarr[0].set(xlabel='Preferred orientation')
axarr[0].set(ylabel='Mean mem. pot.')  
sd_array = np.zeros((N_E,N_pat))
for i in range(N_pat):
    sd_array[:,i] = np.reshape(np.sqrt(np.diag(cov_array_tgt_transformed_plot[:,:,i])),(N_E))
ax = axarr[1].plot(sd_array)
axarr[1].set(xlabel='Preferred orientation')
axarr[1].set(ylabel='Std. mem. pot.')    
axarr[0].axis([0,50,2,10])
axarr[1].axis([0,50,0,3])

#%% Plot correlation matrices

#Grab correlation for target
corel_tgt = np.zeros((N_E,N_E,N_pat))
for i in range(N_E):
    for j in range(N_E):
        corel_tgt[i,j,:] = cov_array_tgt_transformed[i,j,:]/(np.sqrt(
                cov_array_tgt_transformed[i,i,:])*np.sqrt(cov_array_tgt_transformed[j,j,:]))


f, axarr = plt.subplots(5,2, figsize = (5,12))
for image in range(5):
    net_cor = np.corrcoef(np.mean(x_np[image,:N_E,:,:],axis=2))
    ax1 = axarr[image,0].imshow(corel_tgt[:,:,image],vmin=-1,vmax=1, cmap = 'bwr')
#    axx = plt.colorbar(ax1,ax=axarr[image,0])
    ax2 = axarr[image,1].imshow(net_cor,vmin=-1,vmax=1, cmap = 'bwr')
#    axx = plt.colorbar(ax2,ax=axarr[image,1])
f.subplots_adjust(right=0.8)
cbar_ax = f.add_axes([0.85, 0.12, 0.05, 0.29])
f.colorbar(ax2, cax=cbar_ax)

#%% Plot Sigma_eta after training

plt.imshow(Sigma_eta_root_f@Sigma_eta_root_f.T); plt.colorbar()
plt.imshow(Sigma_eta_root_init_np@Sigma_eta_root_init_np.T); plt.colorbar()

#Look at two neuron's activities plotted against each other
image=4
i=20;j=25
plt.plot(x_np[image,i,0,50:],x_np[image,j,0,50:])


#%% plot Autocorrelation
def autocorr(x,lag):
    result = np.corrcoef(x[:-lag], x[lag:])
    return result[0,1]

def create_auto_cor(image,x_data):
    auto_corr_stack = np.zeros((N_E,end_lag-1,50))
    for i in range(N_E):
        print(i)
        for lag in range(1,end_lag):
            for trial in range(50):
                auto_corr_stack[i,(lag-1),trial] = autocorr(x_data[image,i,trial,:],lag)    
    return auto_corr_stack

end_lag = 250
image = 0
auto_corr_stack_im0 = create_auto_cor(image,x_np)

plt.plot(np.mean(np.mean(auto_corr_stack_initial,axis=2),axis=0),label='Before training contrast 1')
plt.plot(np.mean(np.mean(auto_corr_stack,axis=2),axis=0),label='After training contrast 1')
plt.plot(np.mean(np.mean(auto_corr_stack_im0,axis=2),axis=0),label='After training contrast 0')
plt.legend()
    
    
    
    