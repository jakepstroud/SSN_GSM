#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 25 09:54:29 2019

"""

#TF methods Jake

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from SSN_mod import create_A2
import time
#plt.style.use(['ggplot','desktop_screen'])

#%% Setup params
N = 100
N_E = 50
N_I = 50
N_pat = 5   #Number of input images
N_trials = 250 #Number of independent noisy trials
angle = np.pi/2 #Dominant orientation of input images

#Weight matrix params
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

#Setup up time constants for neurons and for the noise process
over_tau_temp = (1/20)*np.ones(N)   #tau_E = 20ms
over_tau_temp[N_E:] *= 2            #tau_I = 10ms
over_tau = tf.stack([tf.diag(np.float32(over_tau_temp))]*N_pat)
noise_step = tf.sqrt(np.float32(dt*2.0*(tau_noise)))

#%% Create TF variables
h_ee = tf.Variable(h_ee_np, dtype=tf.float32)
h_ei = tf.Variable(h_ei_np, dtype=tf.float32)
h_ie = tf.Variable(h_ie_np, dtype=tf.float32)
h_ii = tf.Variable(h_ii_np, dtype=tf.float32)

w_ee = tf.Variable(w_ee_np, dtype=tf.float32)
w_ei = tf.Variable(w_ei_np, dtype=tf.float32)
w_ie = tf.Variable(w_ie_np, dtype=tf.float32)
w_ii = tf.Variable(w_ii_np, dtype=tf.float32)

#Variables for noise covariance matrix
d_sigma = tf.Variable(np.pi/2, dtype=tf.float32)
rho = tf.Variable(1, dtype=tf.float32)
var_e = tf.Variable(np.square(0.23), dtype=tf.float32)
var_i = tf.Variable(np.square(0.19), dtype=tf.float32)

#Variables for feedforward input nonlinaerity
alpha_h = tf.Variable(9, dtype=tf.float32)
beta_h = tf.Variable(0.7, dtype=tf.float32)
power = tf.Variable(2, dtype=tf.float32)

#Function to keep certain params always positive
def pos_fun(uu):
    return tf.log(1+tf.exp(uu-0.6))

#%% Create weight matrix W
def form_W(h_eef,h_eif,h_ief,h_iif,w_eef,w_eif,w_ief,w_iif):
    theta_array = tf.cast(2*np.pi, tf.float32) * tf.range(N_E, dtype=tf.float32)/tf.cast(N_E, tf.float32)
    cos_theta_array = tf.cos(theta_array)
    sin_theta_array = tf.sin(theta_array)
    template = (tf.expand_dims(cos_theta_array,axis=1) @ tf.expand_dims(cos_theta_array,axis=0) +
            tf.expand_dims(sin_theta_array,axis=1) @ tf.expand_dims(sin_theta_array,axis=0) -
            tf.ones([N_E, N_E], dtype=np.float32))            

    #Form W
    W_EE = pos_fun(h_ee) * tf.exp(template/tf.square(w_ee))  
    W_EI = - pos_fun(h_ei) * tf.exp(template/tf.square(w_ei))
    W_IE = pos_fun(h_ie) * tf.exp(template/tf.square(w_ie))  
    W_II = - pos_fun(h_ii) * tf.exp(template/tf.square(w_ii))
    
    W = tf.concat([tf.concat([W_EE,W_EI], axis=1),
                    tf.concat([W_IE,W_II], axis=1)], axis =0, name = "W")
    
    W_stack = tf.stack([W]*N_pat)
    return W, W_stack

W, W_stack = form_W(h_ee,h_ei,h_ie,h_ii,w_ee,w_ei,w_ie,w_ii)

#%% Create noise covariance matrix Sigma
def form_sigma(d_sigma,var_e,rho,var_i):
    theta_array = tf.cast(2*np.pi, tf.float32) * tf.range(N_E, dtype=tf.float32)/tf.cast(N_E, tf.float32)
    cos_theta_array = tf.cos(theta_array)
    sin_theta_array = tf.sin(theta_array)
    template = (tf.expand_dims(cos_theta_array,axis=1) @ tf.expand_dims(cos_theta_array,axis=0) +
            tf.expand_dims(sin_theta_array,axis=1) @ tf.expand_dims(sin_theta_array,axis=0) -
            tf.ones([N_E, N_E], dtype=np.float32))
    
    sqr_width = tf.square(d_sigma)
    S_EE = pos_fun(var_e) * tf.exp(template/sqr_width)  
    S_EI = tf.tanh(rho) * tf.sqrt(pos_fun(var_e)*pos_fun(var_i)) * tf.exp(template/sqr_width)
    S_IE = tf.transpose(S_EI)
    S_II = pos_fun(var_i) * tf.exp(template/sqr_width)
    Sigma_eta = tf.concat([tf.concat([S_EE,S_EI], axis=1),
                tf.concat([S_IE,S_II], axis=1)], axis = 0, name = "Sigma_eta")
    Sigma_eta_root = tf.linalg.cholesky(Sigma_eta + 1e-4*tf.eye(N))
    Sigma_eta_root_stack = tf.stack([Sigma_eta_root]*N_pat)
    return Sigma_eta_root, Sigma_eta_root_stack

Sigma_eta_root, Sigma_eta_root_stack = form_sigma(d_sigma,var_e,rho,var_i)

#%% Neural I--O function
k = 0.3
n = 2
def get_r_tf(x):
    return k*tf.pow(tf.nn.relu(x),n)

#%% Set up image inputs
A = create_A2(N_E,16)
def create_input(angle):
    data = np.load('5_tgt_images4.npz')
    x1=data['x1']#Contrast = 1
    x2=data['x2']
    x3=data['x3']
    x4=data['x4']
    x5=data['x5']
    x_all = [x5,x4,x3,x2,x1]
    h_in = tf.constant(0.001*A.T@x_all,dtype=tf.float32)
    h = alpha_h*tf.pow(tf.nn.relu(beta_h+h_in),power) #Transform input images through nonlinearity
    
    h_stack = tf.transpose(tf.squeeze(tf.stack([h]*N_trials)),[1,2,0])
    return h_stack

h_stack = create_input(angle)

#%% Define cost

#Load the target moments
def load_targets():
    data = np.load('mu_cov_tgt_5_images_transformed4.npz')
    mu_array_tgt_transformed = data['mu_array_tgt_transformed'] # First entry is 0 contrast
    cov_array_tgt_transformed = data['cov_array_tgt_transformed'] # First entry is 0 contrast
    mu_tgt = tf.constant(mu_array_tgt_transformed.T,dtype=tf.float32)                             #######################################
    cov_tgt = tf.constant(np.transpose(cov_array_tgt_transformed,[2,0,1]),dtype=tf.float32) ##########################################
    var_tgt = tf.matrix_diag_part(cov_tgt)
    cov_tgt -= tf.matrix_diag(var_tgt)
    return cov_tgt, var_tgt, mu_tgt
    
cov_tgt, var_tgt, mu_tgt = load_targets()
    
# Standard Frobenius norm    
def frob_norm(x_in):
    return tf.sqrt(tf.nn.l2_loss(x_in))

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
    return (1/(t_final*N_E))*mu_cost + (2/(t_final*N_E))*var_cost + (1/(t_final*(N_E**2)))*cov_cost

#%% Initialise variables
x_list = tf.zeros((N_pat,N,N_trials,1))
x = tf.random_normal((N_pat,N,N_trials),2,1)
ran = tf.random_normal((N_pat,N,N_trials,num_steps),0,1)
eta = tf.constant(np.zeros((N_pat,N,N_trials)),dtype=tf.float32)
cost = tf.constant(0.0,dtype=tf.float32)
cost_time = tf.constant(0,dtype=tf.int32)

#%% Main while_loop part
t = tf.constant(0,dtype=tf.int32)
cond = lambda x,eta,t,cost,x_list: tf.less(t,num_steps)

def body(x,eta,t,cost,x_list):
    eta = eta + (1/tau_noise)*(noise_step*Sigma_eta_root_stack@ran[:,:,:,t] - dt*eta) #Noise process
    
    x = x + over_tau@(dt*(-x + h_stack + W_stack@(get_r_tf(x)) + eta)) #Neural activity
    x_list = tf.concat([x_list,tf.expand_dims(x,axis=3)],axis=3) #Append neural activity each time
    
    #Evaluate cost at each time point
    cost = tf.cond(t>cost_time, lambda: cost + get_cost(tf.reduce_mean(x[:,:N_E,:], axis=2),tf_cov(x[:,:N_E,:])), lambda: cost)
    
    return x,eta,t+1,cost,x_list

#Execute the while loop
x,eta,t,cost,x_list = tf.while_loop(cond,body,(x,eta,t,cost,x_list),
                             shape_invariants = (x.get_shape(),eta.get_shape(),t.get_shape(),cost.get_shape(),
                                                 tf.TensorShape((N_pat,N,N_trials,None))))

#Grab neural activity after one full loop
final_x_list = tf.stack(x_list) #Here, you can evaluate your cost on this tensor of neural activity over all time

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

learning_rate = 0.0001
train_step = tf.train.AdamOptimizer(learning_rate).minimize(cost)

sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())  

n_train_steps = 20000 # Number of training steps
cost_over_time = np.zeros(n_train_steps+1)
cost_over_time[0] = sess.run(cost)
m = cost_over_time[0]
print("initial cost", cost_over_time[0])

x_np_initial = sess.run(final_x_list) #Grab initial neural activity if you want

tic = time.time()
for i in range(n_train_steps):
    
    (_, cost_over_time[i+1]) = sess.run([train_step,cost]) #Execute one optimisation step and grab the cost
    
    if cost_over_time[i+1] < m: #If the new error is smaller than the previous smallest error, grab all variables
        m = cost_over_time[i+1]
        x_np,W_f,Sigma_eta_root_f,alpha_h_f,beta_h_f,power_f = sess.run([final_x_list,W,Sigma_eta_root,alpha_h,beta_h,power])
        
    if i%10==0:
        if i>49:        
            print('Iter', i, ' , Sliding window cost', np.round(np.mean(cost_over_time[i-50:i+1]),3), 
                  ' , Current cost', np.round(cost_over_time[i+1],3), ' , Sec per step', np.round((time.time()-tic)/10,3)) #Print Iteration number and cost
        else:        
            print('Iter',i, ' , cost', cost_over_time[i+1])      
        tic = time.time()

#%%
#####SAVE############
np.savez('Learned_vars',W_f=W_f,Sigma_eta_root_f=Sigma_eta_root_f,
         alpha_h_f=alpha_h_f,beta_h_f=beta_h_f,power_f=power_f,
         cost_over_time=cost_over_time,learning_rate=learning_rate)
#####################

#%% Plot results of training
data = np.load('Learned_vars.npz')
data = np.load('Learned_vars_lr_0002_tgt4.npz')
W_f = data['W_f']
Sigma_eta_root_f = data['Sigma_eta_root_f']
alpha_h_f = data['alpha_h_f']
beta_h_f = data['beta_h_f']
power_f = data['power_f']
cost_over_time = data['cost_over_time']

plt.plot(cost_over_time)

#Plot noise covariance matrix
plt.imshow(Sigma_eta_root_f@Sigma_eta_root_f.T); plt.colorbar()

#Plot weight matrix
plt.imshow(W_f); plt.colorbar()

#Plot neural activity before and after training
contrast = 4 #Index 4 is full contrast (0 is 0 contrast)
plt.imshow((x_np_initial[contrast,:N_E,0,:]),aspect='auto');plt.colorbar()
plt.imshow((x_np[contrast,:N_E,0,:]),aspect='auto');plt.colorbar()


## Plot network and target moments
data = np.load('mu_cov_tgt_5_images_transformed4.npz')
mu_array_tgt_transformed = data['mu_array_tgt_transformed'] # First entry is 0 contrast
cov_array_tgt_transformed = data['cov_array_tgt_transformed'] # First entry is 0 contrast
    
#Plot network moments before training
sd_array = np.zeros((N_E,N_pat))
f, axarr = plt.subplots(1,2, figsize = (10,5))
ax= axarr[0].plot(np.mean(np.mean(x_np_initial[:,:N_E,:,:],axis=3),axis=2).T)
for i in range(5):
    sd_array[:,i] = np.reshape(np.sqrt(np.diag(np.cov(np.mean(x_np_initial[i,:N_E,:,50:],axis=2)))),(N_E))
ax = axarr[1].plot(sd_array)

#After training
sd_array = np.zeros((N_E,N_pat))
f, axarr = plt.subplots(1,2, figsize = (10,5))
ax= axarr[0].plot(np.mean(np.mean(x_np[:,:N_E,:,:],axis=3),axis=2).T)
for i in range(5):
    sd_array[:,i] = np.reshape(np.sqrt(np.diag(np.cov(np.mean(x_np[i,:N_E,:,50:],axis=2)))),(N_E))
ax = axarr[1].plot(sd_array)

# Plot target moments
f, axarr = plt.subplots(1,2, figsize = (10,5))
mu_array_tgt_transformed_plot = mu_array_tgt_transformed
cov_array_tgt_transformed_plot = cov_array_tgt_transformed
ax = axarr[0].plot(mu_array_tgt_transformed_plot)
sd_array = np.zeros((N_E,N_pat))
for i in range(N_pat):
    sd_array[:,i] = np.reshape(np.sqrt(np.diag(cov_array_tgt_transformed_plot[:,:,i])),(N_E))
ax = axarr[1].plot(sd_array)
