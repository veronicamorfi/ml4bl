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


# In[8]:


path_mel = './ML4BL_ZF/melspecs/'
path_files = './ML4BL_ZF/files/'

train_triplet_file = 'train_triplets_50_70_single.pckl'
train_gt_file = 'train_gt_50_70_single.pckl'
train_cons_file = 'train_cons_50_70_single.pckl'
train_trials_file = 'train_trials_50_70_single.pckl'

test_triplet_file = 'test_triplets_50_70_single.pckl'
test_gt_file = 'test_gt_50_70_single.pckl'
test_cons_file = 'test_cons_50_70_single.pckl'
test_trials_file = 'test_trials_50_70_single.pckl'


# In[9]:


luscinia_triplets_file = 'luscinia_triplets_filtered.csv'


# In[10]:


luscinia_triplets = []
with open(path_files+luscinia_triplets_file, 'r',  newline='') as csvfile:
    csv_r = csv.reader(csvfile, delimiter=',')
    for row in csv_r:
        luscinia_triplets.append(row)


# In[26]:


luscinia_triplets = luscinia_triplets[1:]
luscinia_train_len = round(8*len(luscinia_triplets)/10)
luscinia_val_len = len(luscinia_triplets) - luscinia_train_len


# In[12]:


f = open(path_files+'mean_std_luscinia_pretraining.pckl', 'rb')
train_dict = pickle.load(f)
M_l = train_dict['mean']
S_l = train_dict['std']
f.close()


# In[13]:


f = open(path_files+'training_setup_1_ordered_acc_single_cons_50_70_trials.pckl', 'rb')
train_dict = pickle.load(f)
train_keys = train_dict['train_keys']
training_triplets = train_dict['train_triplets']
val_keys = train_dict['val_keys']
validation_triplets = train_dict['vali_triplets']
test_triplet = train_dict['test_triplets']
test_keys = train_dict['test_keys']
M = train_dict['train_mean']
S = train_dict['train_std']
f.close()


########## Here's where the model definition previously sat

# # Training

# In[45]:


emb_size=16
margin = 0.1
m = 0
lr = 1e-8
adam = adam_v2.Adam(lr = lr)

triplet_model = createModelMatrix(emb_size=emb_size, input_shape=(170, 150, 1))[0]
triplet_model.summary()
triplet_model.compile(loss=masked_weighted_triplet_loss(margin=margin, emb_size=emb_size, m=m, w = 0, lh = 1),optimizer=adam) 


# # PRE (pretraining on Luscinia triplets)

# In[18]:


lo = 6
hi = 8
lu = 10
batchsize = lo+hi+lu 

cpCallback = ModelCheckpoint('ZF_emb_'+str(emb_size)+'D_LUSCINIA_PRE_margin_loss_backup.h5', monitor='val_loss', save_best_only=True, save_weights_only=True, mode='min', save_freq='epoch')
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, verbose=1, min_lr=1e-12)
earlystop = EarlyStopping(monitor='val_loss', min_delta=0, patience=20, verbose=1, mode='auto')

history = triplet_model.fit(train_generator_luscinia(luscinia_triplets[:int(luscinia_train_len/10)], M_l, S_l, batchsize, emb_size, path_mel),
                           steps_per_epoch=int(int(luscinia_train_len/10)/batchsize), epochs=1000, verbose=1,
                           validation_data=train_generator_luscinia(luscinia_triplets[luscinia_train_len:luscinia_train_len+200], M_l, S_l, batchsize, emb_size, path_mel),
                           validation_steps=int(200/batchsize), callbacks=[cpCallback, reduce_lr, earlystop])


# # PRE trained (training on bird decisions after pretraining on Luscinia triplets)

# In[33]:


# load pretrained model
triplet_model.load_weights('ZF_emb_'+str(emb_size)+'D_LUSCINIA_PRE_margin_loss_backup.h5')


# In[34]:


lo = 6
hi = 8
lu = 10
batchsize = lo+hi+lu 

cpCallback = ModelCheckpoint('ZF_emb_'+str(emb_size)+'D_LUSCINIA_PRE_margin_loss_trained.h5', monitor='val_loss', save_best_only=True, save_weights_only=True, mode='min', save_freq='epoch')
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, verbose=1, min_lr=1e-12)
earlystop = EarlyStopping(monitor='val_loss', min_delta=0, patience=20, verbose=1, mode='auto')

dis_tr_triplets = discard_some_low(training_triplets, 0.7, 0.7)
dis_val_triplets = discard_some_low(validation_triplets, 0.7, 0.7)

history = triplet_model.fit(train_generator(dis_tr_triplets, M, S, batchsize, emb_size, path_mel),
                           steps_per_epoch=int(len(dis_tr_triplets)/batchsize), epochs=1000, verbose=1,
                           validation_data=train_generator(dis_val_triplets, M,S, batchsize, emb_size, path_mel),
                           validation_steps=int(len(dis_val_triplets)/batchsize), callbacks=[cpCallback, reduce_lr,earlystop])


# # MIXED (training on both bird decisions and Luscinia triplets - w/o pretraining)

# In[37]:


lo = 6
hi = 8
lu = 10
batchsize = lo+hi+lu #26

cpCallback = ModelCheckpoint('ZF_emb_'+str(emb_size)+'D_LUSCINIA_MIXED_margin_loss.h5', monitor='val_loss', save_best_only=True, save_weights_only=True, mode='min', save_freq='epoch')
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, verbose=1, min_lr=1e-12)
earlystop = EarlyStopping(monitor='val_loss', min_delta=0, patience=20, verbose=1, mode='auto')

dis_tr_triplets = discard_some_low(training_triplets, 0.7, 0.7)
dis_val_triplets = discard_some_low(validation_triplets, 0.7, 0.7)

low_margin, high_margin, bal_training_triplets = balance_input(dis_tr_triplets, 0.7, hi_balance = hi, lo_balance = lo)
vlow_margin, vhigh_margin, bal_val_triplets = balance_input(dis_val_triplets, 0.7, hi_balance = hi, lo_balance = lo)


history = triplet_model.fit(train_generator_mixed(bal_training_triplets, M, S, luscinia_triplets[:luscinia_train_len],M_l, S_l, batchsize, lo, hi, lu, emb_size, path_mel),
                                steps_per_epoch=int(len(bal_training_triplets)/(lo+hi)), epochs=1000, verbose=1,
                                validation_data=train_generator_mixed(bal_val_triplets, M, S, luscinia_triplets[luscinia_train_len:],M_l, S_l, batchsize, lo, hi, lu, emb_size, path_mel),
                                validation_steps=int(len(bal_val_triplets)/(lo+hi)), callbacks=[cpCallback, reduce_lr, earlystop])


# # PRE + MIXED (training on both bird decisions and Luscinia triplets - w/ pretraining on Luscinia)

# In[ ]:


# load pre-trained model
triplet_model.load_weights('ZF_emb_'+str(emb_size)+'D_LUSCINIA_PRE_margin_loss_backup.h5')


# In[ ]:


lo = 6
hi = 8
lu = 10
batchsize = lo+hi+lu 

cpCallback = ModelCheckpoint('ZF_emb_'+str(emb_size)+'D_LUSCINIA_PRE_MIXED_margin_loss.h5', monitor='val_loss', save_best_only=True, save_weights_only=True, mode='min', save_freq='epoch')
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, verbose=1, min_lr=1e-12)
earlystop = EarlyStopping(monitor='val_loss', min_delta=0, patience=20, verbose=1, mode='auto')

dis_tr_triplets = discard_some_low(training_triplets, 0.7, 0.7)
dis_val_triplets = discard_some_low(validation_triplets, 0.7, 0.7)

low_margin, high_margin, bal_training_triplets = balance_input(dis_tr_triplets, 0.7, hi_balance = hi, lo_balance = lo)
vlow_margin, vhigh_margin, bal_val_triplets = balance_input(dis_val_triplets, 0.7, hi_balance = hi, lo_balance = lo)


history = triplet_model.fit(train_generator_mixed(bal_training_triplets, M, S, luscinia_triplets[:luscinia_train_len],M_l, S_l, batchsize, lo, hi, lu, emb_size, path_mel),
                                steps_per_epoch=int(len(bal_training_triplets)/(lo+hi)), epochs=1000, verbose=1,
                                validation_data=train_generator_mixed(bal_val_triplets, M, S, luscinia_triplets[luscinia_train_len:],M_l, S_l, batchsize, lo, hi, lu, emb_size, path_mel),
                                validation_steps=int(len(bal_val_triplets)/(lo+hi)), callbacks=[cpCallback, reduce_lr, earlystop])


# # Evaluation (for both ambiguous and unambiguous triplets - based on different distance margins)

# In[ ]:


def low_high_evaluation(path_mel, vhigh_margin, vlow_margin, triplet_model, margin = 0.00000001, max_margin = 0.0000001, step = 0.00000001):
    # pos, neg, anc
    while margin < max_margin:
        acc_cnt = 0
        high_cnt = 0
        low_cnt = 0
        for triplet in vhigh_margin:
            tr_pos = triplet[1][:-4]+'.pckl'
            tr_neg = triplet[2][:-4]+'.pckl'
            tr_anc = triplet[3][:-4]+'.pckl'

            f = open(path_mel+tr_anc, 'rb')
            anc = pickle.load(f).T
            f.close()
            anc = (anc - M)/S
            anc = np.expand_dims(anc, axis=0)
            anc = np.expand_dims(anc, axis=-1)

            f = open(path_mel+tr_pos, 'rb')
            pos = pickle.load(f).T
            f.close()
            pos = (pos - M)/S
            pos = np.expand_dims(pos, axis=0)
            pos = np.expand_dims(pos, axis=-1)

            f = open(path_mel+tr_neg, 'rb')
            neg = pickle.load(f).T
            f.close()
            neg = (neg - M)/S
            neg = np.expand_dims(neg, axis=0)
            neg = np.expand_dims(neg, axis=-1)

            y_pred = triplet_model.predict([anc, pos, neg])

            anchor1 = y_pred[:, 0:emb_size]
            positive1 = y_pred[:, emb_size:emb_size*2]
            negative1 = y_pred[:, emb_size*2:emb_size*3]

            pos_dist = np.sqrt(np.sum(np.square(anchor1 - positive1), axis=1))[0]
            neg_dist = np.sqrt(np.sum(np.square(anchor1 - negative1), axis=1))[0]

            if np.square(neg_dist) > np.square(pos_dist) + margin:
                acc_cnt += 1
                high_cnt += 1
        for triplet in vlow_margin:
            tr_pos = triplet[1][:-4]+'.pckl'
            tr_neg = triplet[2][:-4]+'.pckl'
            tr_anc = triplet[3][:-4]+'.pckl'

            f = open(path_mel+tr_anc, 'rb')
            anc = pickle.load(f).T
            f.close()
            anc = (anc - M)/S
            anc = np.expand_dims(anc, axis=0)
            anc = np.expand_dims(anc, axis=-1)

            f = open(path_mel+tr_pos, 'rb')
            pos = pickle.load(f).T
            f.close()
            pos = (pos - M)/S
            pos = np.expand_dims(pos, axis=0)
            pos = np.expand_dims(pos, axis=-1)

            f = open(path_mel+tr_neg, 'rb')
            neg = pickle.load(f).T
            f.close()
            neg = (neg - M)/S
            neg = np.expand_dims(neg, axis=0)
            neg = np.expand_dims(neg, axis=-1)

            y_pred = triplet_model.predict([anc, pos, neg])

            anchor1 = y_pred[:, 0:emb_size]
            positive1 = y_pred[:, emb_size:emb_size*2]
            negative1 = y_pred[:, emb_size*2:emb_size*3]

            pos_dist = np.sqrt(np.sum(np.square(anchor1 - positive1), axis=1))[0]
            neg_dist = np.sqrt(np.sum(np.square(anchor1 - negative1), axis=1))[0]

            if np.abs(np.square(pos_dist) - np.square(neg_dist)) <= margin:
                acc_cnt+=1
                low_cnt+=1
        print('MARGIN = ', margin)
        print('Macro-average Low-High margin accuracy: ',0.5*(high_cnt/(len(vhigh_margin)) + low_cnt/(len(vlow_margin)))*100, '%')
        print('Micro-average Low-High margin accuracy: ',(acc_cnt/(len(vhigh_margin)+len(vlow_margin)))*100, '%') 
        print('High margin accuracy: ',(high_cnt/(len(vhigh_margin)))*100, '%')  
        print('Low margin accuracy: ',(low_cnt/(len(vlow_margin)))*100, '%')  
        margin += step
        
    return 


# In[30]:


# Separate sets between low-margin (ambiguous) and high-margin (unambiguous) triplets
low_margin = pickle.load( open(path_files+'train_triplets_low_50_70_ACC70.pckl', 'rb'))
vlow_margin = pickle.load(open(path_files+'val_triplets_low_50_70_ACC70.pckl', 'rb'))
high_margin = pickle.load(open(path_files+'train_triplets_high_50_70_ACC70.pckl', 'rb'))
vhigh_margin =pickle.load(open(path_files+'val_triplets_high_50_70_ACC70.pckl', 'rb'))
tlow_margin =pickle.load(open(path_files+'test_triplets_low_50_70_ACC70.pckl', 'rb'))
thigh_margin = pickle.load(open(path_files+'test_triplets_high_50_70_ACC70.pckl', 'rb'))


# In[ ]:


# run evaluation on a high margin and low margin set of the same split
low_high_evauation(path_mel, high_margin, low_margin, triplet_model, margin = 0.0, max_margin = 0.01, step = 0.005)

