# ML4BL (Machine Learning for Birdsong Learning)
# Loading the pretrained model and applying it
# This script by Dan Stowell. CC0.

#####################################################
import keras
import librosa
import numpy as np
import pickle, os, sys, getopt

from ml4blmodels import *

#####################################################
# config

path_modelfiles = os.path.expanduser('~/datasets/ml4bl/ML4BL_ZF/files/')
path_mel = os.path.expanduser('~/datasets/ml4bl/ML4BL_ZF/melspecs/')
path_modelpretrained = os.path.expanduser('~/Documents/ml4bl_bbsrc/model_parameters/ZF_emb_64D_LUSCINIA_MIXED_margin_loss.h5')

emb_size = 64        # dimensions in embedding
ntimeframes = 170    # timeframes in input
n_mels = 150         # frequency bands in input

# further spectrogram settings:
n_fft = 2048
hop_length = 128
win_length = 512

#####################################################
# loading the model etc

def load_ml4bl_model():
	# loading the mean+stdev (normalisation values)
	with open(path_modelfiles+'training_setup_1_ordered_acc_single_cons_50_70_trials.pckl', 'rb') as infp:
		train_dict = pickle.load(infp)

	# intialise the model
	triplet_model, single_model = createModelMatrix(emb_size=emb_size, input_shape=(170, 150, 1))

	# loading the pretrained model weights
	triplet_model.load_weights('../model_parameters/ZF_emb_64D_LUSCINIA_MIXED_margin_loss.h5')

	return {
		'train_mean': train_dict['train_mean'],
		'train_std': train_dict['train_std'],
		'triplet_model': triplet_model,
		'single_model': single_model
	}

#####################################################
# functions to project data into the embedding

def project_melspec(specarray, ml4blmodel):
	#print("PROJECT: specarray is shape: " + str(specarray.shape))
	specarray = (specarray - ml4blmodel['train_mean'])/ml4blmodel['train_std']
	specarray = np.expand_dims(specarray, axis=0)
	specarray = np.expand_dims(specarray, axis=-1)
	return ml4blmodel['single_model'].predict([specarray])[0]

def project_melspec_frompickle(infpath, ml4blmodel):
	with open(infpath, 'rb') as infp:
		specarray = pickle.load(infp).T
	return project_melspec(specarray, ml4blmodel)

def project_melspec_fromwav(infpath, ml4blmodel):
	a, sr = librosa.load(infpath, sr=48000)
	a = np.pad(a, (0, win_length)) # zero-pad to ensure end included in spec
	specarray = librosa.feature.melspectrogram(y=a, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length, win_length=win_length, pad_mode='reflect')

	if specarray.shape[1] > ntimeframes:
		print(f"WARNING: truncating input to the first {ntimeframes} spectral frames")
		specarray = specarray[:, :ntimeframes]
	elif specarray.shape[1] < ntimeframes:
		# zero-padding the time axis
		specarray = np.concatenate((specarray, np.zeros((specarray.shape[0], ntimeframes-specarray.shape[1]))), axis=1)
	specarray = specarray.T
	specarray = np.log(np.maximum(specarray, 1e-12))
	return project_melspec(specarray, ml4blmodel)

#####################################################
if __name__=='__main__':
	opts, args = getopt.getopt(sys.argv[1:],"")

	ml4blmodel = load_ml4bl_model()

	if len(args)==0:
		print("Loaded pretrained ml4bl model.")
		ml4blmodel['single_model'].summary()

		infpath = path_mel+'Yellow14_r27_1.pckl'
		print(f"Input file path: {infpath}")
		y_pred1 = project_melspec_frompickle(infpath, ml4blmodel)
		print("Output projection (y_pred1):")
		print(y_pred1)

		infpath = path_mel+'../wavs/Yellow14_r27_1.wav'
		#infpath = path_mel+'../wavs/Yellow14_r25_4.wav'
		print(f"Input file path: {infpath}")
		y_pred2 = project_melspec_fromwav(infpath, ml4blmodel)
		print("Output projection (y_pred2):")
		print(y_pred2)
		
		print("Cosine sim:")
		print(np.dot(y_pred1, y_pred2) / (np.sqrt(np.sum(y_pred1*y_pred1)) * np.sqrt(np.sum(y_pred2*y_pred2)) ))
	else:
		# Given a list of pickle filepaths as commandline arguments, output a CSV of embedding locations
		for afpath in args:
			y_pred = project_melspec_frompickle(afpath, ml4blmodel)
			print("%s,%s" % (afpath,",".join(map(str, y_pred))))

