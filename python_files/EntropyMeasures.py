#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 16:24:00 2019

@author: cwinkl
"""

import numpy as np


def test_feasibility(p): 
    
    if not np.isclose(np.sum(p), 1): 
        print("probability distribution must sum to 1")
        return False
    
def Entropy(p):
    with np.errstate(divide='ignore', invalid='ignore'):
        ent_arr = np.multiply(p, np.log2(p))
    if np.isnan(ent_arr).any(): 
        if (np.array(p)[np.where(np.isnan(ent_arr))] == 0).all(): 
             np.nan_to_num(ent_arr)
    return -np.nansum(ent_arr)


def JointEntropy(p):
    #p is the x-y-joint probability matrix
    with np.errstate(divide='ignore', invalid='ignore'):
        return -np.nansum(np.multiply(p, np.log2(p)))
    
    
def ConditionalEntropy(p_joint,  p_condition):
    return JointEntropy(p_joint)-Entropy(p_condition)

def MutualInformation(p_joint, px, py):
    return Entropy(px) + Entropy(py) - JointEntropy(p_joint)

def ConditionalMutualInformation(p_joint, pyz,  pxz, pz):
    #I(X;Y|Z) = H(X|Z) - H(X|Y,Z)
    
    CE_XYZ = ConditionalEntropy(p_joint, pyz)
    CE_XZ = ConditionalEntropy(pxz, pz)
    return CE_XZ-CE_XYZ

def KLDivergence(p, q):
    
    if np.isclose(q, 0).any():
        if (p[np.where(np.isclose(q, 0))[0]]>0).any(): 
            return np.inf
    else: 
        return -np.nansum(np.multiply(p, np.log2(np.divide(p, q))))
