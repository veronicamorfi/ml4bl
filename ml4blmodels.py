#!/usr/bin/env python
# coding: utf-8

# In[1]:


import keras
import tensorflow as tf


# In[2]:


import librosa
import numpy as np
import pandas
import pickle
import os
from os import listdir
from os.path import isfile, join
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import h5py
import math
import random
import re
import csv
from sklearn.manifold import TSNE, MDS

from ml4blmodels import *

# In[3]:


from keras.models import Model, load_model
from keras.layers import Dropout, concatenate, Concatenate, Activation, Input, Dense, Conv2D, GRU, MaxPooling2D, MaxPooling1D, Flatten, Reshape, LeakyReLU, PReLU, BatchNormalization, Bidirectional, TimeDistributed, Lambda, GlobalMaxPool1D, GlobalMaxPool2D, GlobalAveragePooling2D, Multiply, GlobalAveragePooling2D
from keras.optimizers import adam_v2
import keras.backend as K
from keras import regularizers
from keras.initializers import random_normal, glorot_uniform, glorot_normal
import tensorflow as tf
from keras.callbacks import TensorBoard, ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau, Callback, EarlyStopping


# # Network

# In[10]:


def _convBNpr(a, dilation, num_filters, kernel):
    c1 = Conv2D(filters=num_filters, kernel_size=kernel, strides=(1, 1), dilation_rate=dilation, padding='same', use_bias=False, kernel_initializer=glorot_uniform(seed=123), kernel_regularizer=regularizers.l2(1e-4))(a)
    c1 = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(c1)
    c1 = LeakyReLU(alpha=0.3)(c1)
    return c1

def createBaseModel(emb_size, input_shape):
    "Creates our DL embedding model for one item"

    a = Input(shape=(input_shape)) 
    
    c = _convBNpr(a, 1, 12, (3,3))
    c = _convBNpr(c, 2, 32, (3,3))
    c = _convBNpr(c, 4, 64, (3,3))
    c = _convBNpr(c, 8, 128, (3,3))
    # attention with sigmoid
    a1 = Conv2D(filters=128, kernel_size=(1,1), strides=(1, 1), padding='same', activation = 'sigmoid',  use_bias=True, kernel_regularizer=regularizers.l2(1e-4),kernel_initializer=glorot_uniform(seed=123))(c)

    
    # sum of sum of attention
    s = Lambda(lambda x: K.sum(K.sum(x,axis=1, keepdims=True), axis=2, keepdims=True))(a1)
    s = Lambda(lambda x: K.repeat_elements(x, 170, axis=1))(s)
    s = Lambda(lambda x: K.repeat_elements(x, 150, axis=2))(s)
    
    # probability matrix of attention
    p = Lambda(lambda x: x[0]/x[1])([a1,s])
    
    # inner product of attention and projection matrices
    m = Multiply()([c, p])
    
    # output
    out_sum = Lambda(lambda x: K.sum(K.sum(x, axis=1), axis=1))(m)
    
    # attention side
    d3 = Dense(emb_size*100, kernel_initializer=glorot_normal(seed=222), kernel_regularizer=regularizers.l2(1e-4),activation='relu')(out_sum)
    d3 = Dropout(.2, seed=222)(d3)
    d4 = Dense(emb_size*10, kernel_initializer=glorot_normal(seed=333), kernel_regularizer=regularizers.l2(1e-4),activation='relu')(d3)
    d4 = Dropout(.2, seed=333)(d4)
    d5 = Dense(emb_size, kernel_initializer=glorot_normal(seed=132), kernel_regularizer=regularizers.l2(1e-4))(d4)
    d5 = Dropout(.2, seed = 132)(d5)
    
    # maxpool side
    x = _convBNpr(c, 1, 64, (3,3)) 
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = _convBNpr(x, 1, 32, (3,3))
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = _convBNpr(x, 1, 12, (3,3))

    f = Flatten()(x)
    df1 = Dense(emb_size*100, kernel_initializer=glorot_normal(seed=456), kernel_regularizer=regularizers.l2(1e-4), activation='relu')(f)#
    df1 = Dropout(.2, seed=456)(df1)
    df2 = Dense(emb_size*10, kernel_initializer=glorot_normal(seed=654), kernel_regularizer=regularizers.l2(1e-4), activation='relu')(df1)#
    df2 = Dropout(.2, seed=654)(df2)
    df3 = Dense(emb_size, kernel_initializer=glorot_normal(seed=546), kernel_regularizer=regularizers.l2(1e-4))(df2)#
    df3 = Dropout(.2, seed=546)(df3)

    concat = Concatenate(axis=-1)([d5, df3])
    dd = Dense(emb_size, kernel_initializer=glorot_normal(seed=999), kernel_regularizer=regularizers.l2(1e-4))(concat)

    sph = Lambda(lambda  x: K.l2_normalize(x,axis=1))(dd)
    
    # base model creation
    base_model = Model(a,sph) 
    return base_model


def createModelMatrix(emb_size, input_shape=(170, 150, 1)):
    "Creates our DL embedding model for a triplet of 3 items"

    base_model = createBaseModel(emb_size, input_shape)
    
    # triplet framework
    input_anchor = Input(shape=(input_shape))
    input_positive = Input(shape=(input_shape))
    input_negative = Input(shape=(input_shape)) 
    
    net_anchor = base_model(input_anchor)
    net_positive = base_model(input_positive)
    net_negative = base_model(input_negative)
    
    #base_model.summary()
    
    merged_vector = concatenate([net_anchor, net_positive, net_negative], axis=-1)
    
    model = Model([input_anchor, input_positive, input_negative], outputs=merged_vector)

    return model, base_model


#############################################################################################
# # Functions

# In[11]:


def masked_weighted_triplet_loss(margin, emb_size, m = 0 , w = 0, lh = 1):
    def lossFunction(y_true,y_pred):
        
        weight = y_true[:, 0] # acc
        cons = y_true[:, 1] # consistency
        trials = y_true[:, 2] # number of trials
        
        anchor = y_pred[:, 0:emb_size]
        positive = y_pred[:, emb_size:emb_size*2]
        negative = y_pred[:, emb_size*2:emb_size*3]

        # distance between the anchor and the positive
        pos_dist = K.sqrt(K.sum(K.square(anchor - positive), axis=1)) # l2 distance
        #pos_dist = K.sum(K.abs(anchor-positive), axis=1) # l1 distance

        # distance between the anchor and the negative
        neg_dist = K.sqrt(K.sum(K.square(anchor - negative), axis=1)) # l2 distance
        #neg_dist = K.sum(K.abs(anchor-negative), axis=1) # l1 distance

        loss_h = 0
        loss_l = 0
        
        if lh == 1:
            # DOES NOT WORK WITH MASKED LOSS
            # low-high margin loss
            p_c = K.square(neg_dist) - K.square(pos_dist) - margin  
            p_i = K.square(neg_dist) - K.square(pos_dist)
            
            loss_1 = cons*(1-K.exp(p_c)) + (1-cons)*(1-K.exp(-K.abs(p_i)))
            
        if m != 0:
            # masked loss
            basic_loss = pos_dist - neg_dist + margin
            
            threshold = K.max(basic_loss) * m
            mask = 2 + margin - K.maximum(basic_loss, threshold) 

            loss_1 = basic_loss * mask
        
        if w == 1:
            # weighted based on acc
            weighted_loss = weight*loss_1
        else:
            # non-weighted
            weighted_loss = loss_1
            
        loss = K.maximum(weighted_loss, 0.0)

        return loss
    return lossFunction


# In[12]:


def discard_some_low(triplet_list, cons, acc):
    low_margin = []
    high_margin = []
    
    for i in range(len(triplet_list)):
        if float(triplet_list[i][-1]) < cons: # low margin
            if float(triplet_list[i][-2]) >= acc: # ACC 
                low_margin.append(triplet_list[i])
        else: # high margin
            high_margin.append(triplet_list[i])
    
    random.seed(123)
    random.shuffle(low_margin)
    random.shuffle(high_margin)
    
    low_margin.extend(high_margin)
        
    return low_margin


# In[13]:


def balance_input(triplet_list, cons, hi_balance = 6, lo_balance = 6):
    batchsize = hi_balance + lo_balance
    low_margin = []
    high_margin = []
    
    for i in range(len(triplet_list)):
        if float(triplet_list[i][-1]) < cons: # low margin
            low_margin.append(triplet_list[i])
        else: # high margin
            high_margin.append(triplet_list[i])
    
    random.seed(123)
    random.shuffle(low_margin)
    random.shuffle(high_margin)
    
    new_triplet_list = []
    maxlen = np.maximum(len(low_margin), len(high_margin))
    
    hi_start = 0
    lo_start = 0
    for i in range(0,int(maxlen/hi_balance)*batchsize,batchsize):
        for j in range(hi_start,hi_start+hi_balance,1):
            new_triplet_list.append(high_margin[np.mod(j,len(high_margin))])
        hi_start+=hi_balance
        for j in range(lo_start, lo_start+lo_balance,1):
            new_triplet_list.append(low_margin[np.mod(j,len(low_margin))])
        lo_start+=lo_balance
            
    return low_margin, high_margin, new_triplet_list


#########################################################################################
# # Generators

# In[14]:


def train_generator_mixed(triplet_list, M, S, luscinia_triplets, M_l, S_l, batchsize, lo, hi, lu, emb_size, path_mel):
    
    acc_gt = np.zeros((batchsize, emb_size))
   
    random.seed(123)
    random.shuffle(luscinia_triplets)
    
    while 1:
    
        anchors_input = np.empty((batchsize, 170, 150, 1))
        positives_input = np.empty((batchsize, 170, 150, 1))
        negatives_input = np.empty((batchsize, 170, 150, 1))
        
        imax = int(len(triplet_list)/(lo+hi))
        
        list_cnt = 0
        luscinia_cnt = 0
        
        for i in range(imax):        
            for j in range(batchsize):
                
                if j < (lo+hi):
                    triplet = triplet_list[list_cnt]
                    list_cnt += 1
                    
                    tr_anc = triplet[3][:-4]+'.pckl'
                    tr_pos = triplet[1][:-4]+'.pckl'
                    tr_neg = triplet[2][:-4]+'.pckl'
                    acc_gt[j][0] = float(triplet[-2]) # acc
                    acc_gt[j][1] = 1 if float(triplet[-1])>=0.7 else 0 # cons
                    acc_gt[j][2] = int(triplet[-3]) # number of trials

                else:
                    triplet = luscinia_triplets[luscinia_cnt]
                    luscinia_cnt += 1
                    
                    tr_anc = triplet[2][:-4]+'.pckl'
                    tr_pos = triplet[0][:-4]+'.pckl'
                    tr_neg = triplet[1][:-4]+'.pckl'
                    acc_gt[j][0] = 1 # acc
                    acc_gt[j][1] = 1  # cons
                    acc_gt[j][2] = 1 # number of trials
                
                f = open(path_mel+tr_anc, 'rb')
                anc = pickle.load(f).T
                f.close()
                anc = (anc - M)/S
                anc = np.expand_dims(anc, axis=-1)
                
                f = open(path_mel+tr_pos, 'rb')
                pos = pickle.load(f).T
                f.close()
                pos = (pos - M)/S
                pos = np.expand_dims(pos, axis=-1)
                
                f = open(path_mel+tr_neg, 'rb')
                neg = pickle.load(f).T
                f.close()
                neg = (neg - M)/S
                neg = np.expand_dims(neg, axis=-1)
                
                anchors_input[j] = anc
                positives_input[j] = pos
                negatives_input[j] = neg
                
            yield [anchors_input, positives_input, negatives_input], acc_gt


# In[15]:


def train_generator_luscinia(triplet_list, M, S, batchsize, emb_size, path_mel, ordered = True):
    
    acc_gt = np.zeros((batchsize, emb_size))
    
    random.seed(123)
    random.shuffle(triplet_list) 
    
    while 1:
    
        anchors_input = np.empty((batchsize, 170, 150, 1))
        positives_input = np.empty((batchsize, 170, 150, 1))
        negatives_input = np.empty((batchsize, 170, 150, 1))
        
        imax = int(len(triplet_list)/batchsize)
                
        for i in range(imax):        
            for j in range(batchsize):
                triplet = triplet_list[i*batchsize+j]
                
                tr_anc = triplet[2][:-4]+'.pckl'
                tr_pos = triplet[0][:-4]+'.pckl'
                tr_neg = triplet[1][:-4]+'.pckl'
                acc_gt[j][0] = 1 # acc
                acc_gt[j][1] = 1  # cons
                acc_gt[j][2] = 1 # number of trials
                 
                f = open(path_mel+tr_anc, 'rb')
                anc = pickle.load(f).T
                f.close()
                anc = (anc - M)/S
                anc = np.expand_dims(anc, axis=-1)
                
                f = open(path_mel+tr_pos, 'rb')
                pos = pickle.load(f).T
                f.close()
                pos = (pos - M)/S
                pos = np.expand_dims(pos, axis=-1)
                
                f = open(path_mel+tr_neg, 'rb')
                neg = pickle.load(f).T
                f.close()
                neg = (neg - M)/S
                neg = np.expand_dims(neg, axis=-1)
                
                anchors_input[j] = anc
                positives_input[j] = pos
                negatives_input[j] = neg
                
            yield [anchors_input, positives_input, negatives_input], acc_gt


# In[16]:


def train_generator(triplet_list, M, S, batchsize, emb_size, path_mel, ordered = True):
    
    acc_gt = np.zeros((batchsize, emb_size))
    
    random.seed(123)
    random.shuffle(triplet_list) 
    
    while 1:
    
        anchors_input = np.empty((batchsize, 170, 150, 1))
        positives_input = np.empty((batchsize, 170, 150, 1))
        negatives_input = np.empty((batchsize, 170, 150, 1))
        
        imax = int(len(triplet_list)/batchsize)
                
        for i in range(imax):        
            for j in range(batchsize):
                triplet = triplet_list[i*batchsize+j]
                
                tr_anc = triplet[3][:-4]+'.pckl'
                
                if ordered == False:
                    if triplet[-1] == '0':
                        tr_pos = triplet[1][:-4]+'.pckl'
                        tr_neg = triplet[2][:-4]+'.pckl'
                    else:
                        tr_pos = triplet[2][:-4]+'.pckl'
                        tr_neg = triplet[1][:-4]+'.pckl'
                else: 
                    tr_pos = triplet[1][:-4]+'.pckl'
                    tr_neg = triplet[2][:-4]+'.pckl'
                    acc_gt[j][0] = float(triplet[-2]) # acc
                    acc_gt[j][1] = 1 if float(triplet[-1])>=0.7 else 0 # cons
                    acc_gt[j][2] = int(triplet[-3]) # number of trials
                   
                f = open(path_mel+tr_anc, 'rb')
                anc = pickle.load(f).T
                f.close()
                anc = (anc - M)/S
                anc = np.expand_dims(anc, axis=-1)
                
                f = open(path_mel+tr_pos, 'rb')
                pos = pickle.load(f).T
                f.close()
                pos = (pos - M)/S
                pos = np.expand_dims(pos, axis=-1)
                
                f = open(path_mel+tr_neg, 'rb')
                neg = pickle.load(f).T
                f.close()
                neg = (neg - M)/S
                neg = np.expand_dims(neg, axis=-1)
                
                anchors_input[j] = anc
                positives_input[j] = pos
                negatives_input[j] = neg
                
            yield [anchors_input, positives_input, negatives_input], acc_gt


