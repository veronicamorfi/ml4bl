# coding: utf-8

import apply_embedding
import pickle
import os
import numpy as np

path_mel = os.path.expanduser('~/datasets/ml4bl/ML4BL_ZF/melspecs/')

nlimit = 200    # TODO this is the maximum number of triplets to check, just to limit runtime

##########################################################################################################
# Load model, and load a set of triplet definitions
ml4blmodel = apply_embedding.load_ml4bl_model()
with open('/home/dans/datasets/ml4bl/ML4BL_ZF/files/train_triplets_high_50_70_ACC70.pckl', 'rb') as infp:
    data = pickle.load(infp)
    

# proj = apply_embedding.project_melspec_frompickle(path_mel + "/" + triplet[1].replace(".wav", ".pckl"), ml4blmodel)
# proj

def project_triplet(triplet):
    return [apply_embedding.project_melspec_frompickle(path_mel + "/" + triplet[i].replace(".wav", ".pckl"), ml4blmodel) for i in [1,2,3]]

def project_triplet_wav(triplet):
    return [apply_embedding.project_melspec_fromwav(path_mel + "/../wavs/" + triplet[i], ml4blmodel) for i in [1,2,3]]

    
# projs = project_triplet(triplet)
# np.shape(projs)

##########################################################################################################
# Now make embedding-based triplet decisions, and check if they match the gt. (Order in triplet: P, N, A)
print("Projecting and judging triplets (mel-spec)...")
ncorrect = 0
ntot = 0
for triplet in data[:nlimit]:
    projs = project_triplet(triplet)
    dist_p = np.sum((projs[0]-projs[2])**2)
    dist_n = np.sum((projs[1]-projs[2])**2)
    if dist_n > dist_p:
        ncorrect += 1
    ntot += 1
    if ntot % 100 == 0:
    	print(f"      Done {ntot}")

print(f"Correct: {ncorrect}/{ntot} (mel spec version)")

print("Projecting and judging triplets (wav)...")
ncorrect = 0
ntot = 0    
for triplet in data[:nlimit]:
    projs = project_triplet_wav(triplet)
    dist_p = np.sum((projs[0]-projs[2])**2)
    dist_n = np.sum((projs[1]-projs[2])**2)
    if dist_n > dist_p:
        ncorrect += 1
    ntot += 1
    if ntot % 100 == 0:
    	print(f"      Done {ntot}")
    
print(f"Correct: {ncorrect}/{ntot} (wav version)")

