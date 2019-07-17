#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 18 11:01:00 2019

@author: jps99
"""
#Module for SSN
import numpy as np

def gengabor(sz,lam,theta,psi,sig,gam):
    radius = (int(sz[0]/2.0), int(sz[1]/2.0))
    [x, y] = np.meshgrid(range(-radius[0], radius[0]), range(-radius[1], radius[1]))
    
    x1 = x * np.cos(theta) + y * np.sin(theta)
    y1 = -x * np.sin(theta) + y * np.cos(theta)
    return np.exp(- (x1**2 + (gam**2)*y1**2)/(2*sig**2)) * np.cos(2*np.pi*x1/lam + psi)

def create_A(n_E_neurons,size):
    A = np.empty((size**2,n_E_neurons))
    for i in range(n_E_neurons):
        th = i*np.pi/(n_E_neurons)
        g = gengabor((16,16), 5, th, 0,np.pi/1.5,0.8)
        A[:,i] = np.reshape(g,(16**2))
    A = np.append(A,A,axis=1)
    return A

def create_A2(n_E_neurons,size):
    A = np.empty((size**2,n_E_neurons))
    for i in range(n_E_neurons):
        th = i*np.pi/(n_E_neurons)
        g = gengabor((16,16), 6, th, 0,np.pi/1.5,0.6)
        A[:,i] = np.reshape(g,(16**2))
    A = np.append(A,A,axis=1)
    return A