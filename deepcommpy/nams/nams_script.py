# Belief propagation using Pytorch

import time
from time import sleep
import numpy as np
import numpy.matlib as mnp
import scipy.io as sio
import pickle 
import mat73
import tracemalloc
import gc

import csv
import json

np.random.seed(0)

import pdb
import sys
from utils import load_code, syndrome, convert_dense_to_alist, apply_channel, eb_n0_to_snr, calc_sigma, Decoder
import os
import argparse
import matplotlib.pyplot as plt


from .nams import NAMS_net, nams_decode


import torch
torch.manual_seed(0)
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torch.autograd.profiler as profiler
	

def train_nams(nams_code, device, config = None, loaded_weights = None):


	script_dir = os.path.dirname(os.path.abspath(__file__))
	if config is None:
		with open(os.path.join(script_dir, 'train_config.json'), 'r') as f:
			config = json.load(f)

	# get the code params
	H_path = config['H_path']
	G_path = config['G_path']

	code = load_code(H_path, G_path)

	H = code.H
	G = code.G

	var_degrees = code.var_degrees
	chk_degrees = code.chk_degrees
	num_edges = code.num_edges
	u = code.u
	d = code.d
	n = code.n
	m = code.m
	k = code.k

	edges = code.edges
	edges_m = code.edges_m
	edge_order_vc = code.edge_order_vc
	extrinsic_edges_vc = code.extrinsic_edges_vc
	edge_order_cv = code.edge_order_cv
	extrinsic_edges_cv = code.extrinsic_edges_cv

	decoder = Decoder()
	decoder.max_iter = config['max_iter']
	decoder.nn_eq = config['nn_eq']
	
	nams_net = NAMS_net(nams_code, config).to(device)




	# reads config file
	num_epochs = config['num_epochs']
	training_batch_size = config['training_batch_size']
	learning_rate = config['learning_rate']
	channel_type = config['channel_type']
	H_path = config['H_path']
	G_path = config['G_path']

	code = load_code(H_path, G_path)

	H = code.H
	G = code.G

	var_degrees = code.var_degrees
	chk_degrees = code.chk_degrees
	num_edges = code.num_edges
	u = code.u
	d = code.d
	n = code.n
	m = code.m
	k = code.k

	edges = code.edges
	edges_m = code.edges_m
	edge_order_vc = code.edge_order_vc
	extrinsic_edges_vc = code.extrinsic_edges_vc
	edge_order_cv = code.edge_order_cv
	extrinsic_edges_cv = code.extrinsic_edges_cv


		if(args.nn_eq == 0):
			optimizer = optim.Adam([model.B_cv, model.W_cv], lr = learning_rate)
		elif(args.nn_eq == 1):
			optimizer = optim.Adam([model.W_cv], lr = learning_rate)
		elif(args.nn_eq == 2):
			optimizer = optim.Adam([model.B_cv], lr = learning_rate)
	
	

	scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1000, eta_min=0, last_epoch=-1, verbose=False)

	if (args.use_offline_training_data == 1):
		lte_data_filename = "generate_lte_data/lte_data/" + args.coding_scheme + "_" + str(n) + "_" + str(k) + "_"  + str(args.channel_type) + "_data_train_" + str(int(args.eb_n0_train_lo)) + "_" + str(int(args.eb_n0_train_hi)) + ".mat"
		data = mat73.loadmat(lte_data_filename)
		enc_data = torch.tensor(data['enc'])
		llr_data = torch.tensor( data['llr'])

		# # Adding singleton dimension for format matching
		# if (enc_data.dim() == 2):
		# 	enc_data = torch.unsqueeze(enc_data, 2)
		# 	llr_data = torch.unsqueeze(llr_data, 2)
		
	# if adapting, load the previous model for further training
	if (args.adaptivity_training == 1 or args.continue_training == 1):
		print("\n\n********* Loading the saved model for further training *********\n\n",args.saved_model_path,"\n\n")
		sleep(2)
		model.load_state_dict(torch.load(args.saved_model_path))
		# reinitialize the optimizer since the model is reloaded
		if (args.cv_model == 1 and args.vc_model == 0):
			if(args.nn_eq == 0):
				optimizer = optim.Adam([model.B_cv, model.W_cv], lr = learning_rate)
			elif(args.nn_eq == 1):
				optimizer = optim.Adam([model.W_cv], lr = learning_rate)
			elif(args.nn_eq == 2):
				optimizer = optim.Adam([model.B_cv ], lr = learning_rate)
		
		if (args.cv_model == 0 and args.vc_model == 1):
			if(args.nn_eq == 0):
				optimizer = optim.Adam([model.B_vc, model.W_vc], lr = learning_rate)
			elif(args.nn_eq == 1):
				optimizer = optim.Adam([model.W_vc], lr = learning_rate)
			elif(args.nn_eq == 2):
				optimizer = optim.Adam([model.B_vc], lr = learning_rate)
		
		if (args.cv_model == 1 and args.vc_model == 1):
			if(args.nn_eq == 0):
				optimizer = optim.Adam([model.B_cv, model.W_cv, model.B_vc, model.W_vc], lr = learning_rate)
			elif(args.nn_eq == 1):
				optimizer = optim.Adam([model.W_cv, model.W_vc], lr = learning_rate)
			elif(args.nn_eq == 2):
				optimizer = optim.Adam([[model.B_cv, model.B_vc]], lr = learning_rate)
		scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1000, eta_min=0, last_epoch=-1, verbose=False)

		if (args.cv_model == 0 and args.vc_model == 0):
			if(args.nn_eq == 1):
				optimizer = optim.Adam([model.W_gw], lr = learning_rate)
	step = 0
	if (args.continue_training == 1):
		step = 19000+1
  
  

def train_nams(decoder, code, device, config = None, loaded_weights = None):

	script_dir = os.path.dirname(os.path.abspath(__file__))
	if config is None:
		with open(os.path.join(script_dir, 'train_config.json'), 'r') as f:
			config = json.load(f)
   
	# reads config file
	num_epochs = config['num_epochs']
	training_batch_size = config['training_batch_size']
	learning_rate = config['learning_rate']
	channel_type = config['channel_type']
	H_path = config['H_path']
	G_path = config['G_path']
 
	code = load_code(H_path, G_path)

	H = code.H
	G = code.G

	var_degrees = code.var_degrees
	chk_degrees = code.chk_degrees
	num_edges = code.num_edges
	u = code.u
	d = code.d
	n = code.n
	m = code.m
	k = code.k

	edges = code.edges
	edges_m = code.edges_m
	edge_order_vc = code.edge_order_vc
	extrinsic_edges_vc = code.extrinsic_edges_vc
	edge_order_cv = code.edge_order_cv
	extrinsic_edges_cv = code.extrinsic_edges_cv
 
 
	
			
	# create the model
	nams = NAMS_net(decoder, code).to(device)
	
	for epoch in range(num_epochs):
		while step < steps:
			# data generation
			if (args.force_all_zero == 0):
				messages = torch.randint(0,2,(training_batch_size,k))
	
				# encoding
				codewords = torch.matmul(torch.tensor(G,dtype=int), torch.transpose(messages,0,1)) % 2

				# modulation
				BPSK_codewords = (codewords - 0.5) * 2.0

				soft_input = torch.zeros_like(BPSK_codewords)
				received_codewords = torch.zeros_like(BPSK_codewords)
				sigma_vec = torch.zeros_like(BPSK_codewords)
				SNR_vec = torch.zeros_like(BPSK_codewords)
				llr_in = torch.zeros_like(BPSK_codewords)
				
				# create minibatch with codewords from multiple SNRs
				for i in range(0,len(SNRs)):
					sigma = calc_sigma(SNRs[i],rate)
					noise = sigma * np.random.randn(n,training_batch_size//len(SNRs))
					start_idx = training_batch_size*i//len(SNRs)
					end_idx = training_batch_size*(i+1)//len(SNRs)
					
					# Apply channel and noise
					FastFading = False

					received_codewords[:,start_idx:end_idx], soft_input[:,start_idx:end_idx] = apply_channel(BPSK_codewords[:,start_idx:end_idx], sigma, noise, channel_type, FastFading)
					llr_in[:,start_idx:end_idx] = 2*received_codewords[:,start_idx:end_idx]/(sigma**2)
					sigma_vec[:,start_idx:end_idx] = sigma
					SNR_vec[:,start_idx:end_idx] = SNRs[i]

					llr_in = llr_in.to(device)
					codewords = codewords.to(device)

			# training starts
			# perform gradient update for whole snr batch
			soft_output, batch_loss = nn_decode(llr_in,training_batch_size,codewords)
			optimizer.zero_grad()
			batch_loss.backward()

			optimizer.step()

			torch.save(model.state_dict(), filename)

			step += 1

	print("Trained decoder on " + str(step) + " minibatches.\n")
	if (step > 0):
		print ("final prints")
		print (batch_loss)
		if (args.cv_model == 1 and args.vc_model == 0):
			print ("B_cv : ")
			print (model.B_cv)
			print ("W_cv : ")
			print (model.W_cv)
		if (args.cv_model == 0 and args.vc_model == 1):
			print ("B_vc : ")
			print (model.B_vc)
			print ("W_vc : ")
			print (model.W_vc)
			print ("W_ch : ")
			print (model.W_ch)
		if (args.cv_model == 0 and args.vc_model == 0):
			print ("W_gw : ")
			print (model.W_gw)

		if (args.save_torch_model == 1):
			# save in intermediate folder
			if (args.adaptivity_training == 1):
				filename = models_folder_int + "nams_" + str(args.coding_scheme) + "_" + str(n) + "_" + str(k) + "_st_" + str(step) + "_lr_" +str(args.learning_rate) + "_" + str(args.channel_type) + "_adapt_from_" + base_channel + "_ent_" + str(args.entangle_weights) + "_nn_eq_" + str(args.nn_eq) + "_relu_" + str(args.relu) + "_max_iter_" + str(args.num_iterations) + "_" + str(int(args.eb_n0_train_lo)) + "_" + str(int(args.eb_n0_train_hi)) + ".pt"
			else:
				filename = models_folder_int + "nams_" + str(args.coding_scheme) + "_" + str(n) + "_" + str(k) + "_st_" + str(step) + "_lr_" +str(args.learning_rate) + "_" + str(args.channel_type) + "_ent_" + str(args.entangle_weights) + "_nn_eq_" + str(args.nn_eq) + "_relu_" + str(args.relu) + "_max_iter_" + str(args.num_iterations) + "_" + str(int(args.eb_n0_train_lo)) + "_" + str(int(args.eb_n0_train_hi)) + ".pt"
			
			if (args.freeze_weights == 1):
				filename = models_folder_int + "nams_" + str(args.coding_scheme) + "_" + str(n) + "_" + str(k) + "_st_" + str(step) + "_lr_" +str(args.learning_rate) + "_" + str(args.channel_type) + "_ent_" + str(args.entangle_weights) + "_nn_eq_" + str(args.nn_eq) + "_relu_" + str(args.relu) + "_max_iter_" + str(args.num_iterations) + "_" + str(int(args.eb_n0_train_lo)) + "_" + str(int(args.eb_n0_train_hi)) + "_ff_" + str(args.freeze_fraction) + ".pt"
			
			if (args.interf == 1):
				filename = models_folder_int + "nams_" + str(args.coding_scheme) + "_" + str(n) + "_" + str(k) + "_st_" + str(step) + "_lr_" +str(args.learning_rate) + "_" + str(args.channel_type) + "_ent_" + str(args.entangle_weights) + "_nn_eq_" + str(args.nn_eq) + "_relu_" + str(args.relu) + "_max_iter_" + str(args.num_iterations) + "_" + str(int(args.eb_n0_train_lo)) + "_" + str(int(args.eb_n0_train_hi)) + "_alpha_" + str(args.alpha) + ".pt"
			
			torch.save(model.state_dict(), filename) 
			# save in main folder
			if (args.adaptivity_training == 1):
				filename = models_folder + "/nams_" + str(args.coding_scheme) + "_" + str(n) + "_" + str(k) + "_st_" + str(step) + "_lr_" +str(args.learning_rate) + "_" + str(args.channel_type) + "_adapt_from_" + base_channel + "_ent_" + str(args.entangle_weights) + "_nn_eq_" + str(args.nn_eq) + "_relu_" + str(args.relu) + "_max_iter_" + str(args.num_iterations) + "_" + str(int(args.eb_n0_train_lo)) + "_" + str(int(args.eb_n0_train_hi)) + ".pt"
			else:
				filename = models_folder + "/nams_" + str(args.coding_scheme) + "_" + str(n) + "_" + str(k) + "_st_" + str(args.steps) + "_lr_" +str(args.learning_rate) + "_" + str(args.channel_type) + "_ent_" + str(args.entangle_weights) + "_nn_eq_" + str(args.nn_eq) + "_relu_" + str(args.relu) + "_max_iter_" + str(args.num_iterations) + "_" + str(int(args.eb_n0_train_lo)) + "_" + str(int(args.eb_n0_train_hi)) + ".pt"
			if (args.freeze_weights == 1):
				filename = models_folder + "/nams_" + str(args.coding_scheme) + "_" + str(n) + "_" + str(k) + "_st_" + str(args.steps) + "_lr_" +str(args.learning_rate) + "_" + str(args.channel_type) + "_ent_" + str(args.entangle_weights) + "_nn_eq_" + str(args.nn_eq) + "_relu_" + str(args.relu) + "_max_iter_" + str(args.num_iterations) + "_" + str(int(args.eb_n0_train_lo)) + "_" + str(int(args.eb_n0_train_hi)) + "_ff_" + str(args.freeze_fraction) + ".pt"
			
			if (args.interf == 1):
				filename = models_folder + "/nams_" + str(args.coding_scheme) + "_" + str(n) + "_" + str(k) + "_st_" + str(args.steps) + "_lr_" +str(args.learning_rate) + "_" + str(args.channel_type) + "_ent_" + str(args.entangle_weights) + "_nn_eq_" + str(args.nn_eq) + "_relu_" + str(args.relu) + "_max_iter_" + str(args.num_iterations) + "_" + str(int(args.eb_n0_train_lo)) + "_" + str(int(args.eb_n0_train_hi)) + "_alpha_" + str(args.alpha) + ".pt"
			
			torch.save(model.state_dict(), filename)
			# save to matlab for analysis
			if (args.cv_model == 1 and args.vc_model == 0):
				B_cv_temp = model.B_cv.cpu().data.numpy()
				W_cv_temp = model.W_cv.cpu().data.numpy()
				weights_temp = {'B_cv':B_cv_temp, 'W_cv':W_cv_temp}
			elif (args.cv_model == 0 and args.vc_model == 1):
				B_vc_temp = model.B_vc.cpu().data.numpy()
				W_vc_temp = model.W_vc.cpu().data.numpy()
				W_ch_temp = model.W_ch.cpu().data.numpy()
				weights_temp = {'B_vc':B_vc_temp, 'W_vc':W_vc_temp, 'W_ch':W_ch_temp}
			elif (args.cv_model == 1 and args.vc_model == 1):
				B_cv_temp = model.B_cv.cpu().data.numpy()
				W_cv_temp = model.W_cv.cpu().data.numpy()
				B_vc_temp = model.B_vc.cpu().data.numpy()
				W_vc_temp = model.W_vc.cpu().data.numpy()
				W_ch_temp = model.W_ch.cpu().data.numpy()
				weights_temp = {'B_cv':B_cv_temp, 'W_cv':W_cv_temp, 'B_vc':B_vc_temp, 'W_vc':W_vc_temp, 'W_ch':W_ch_temp}
			elif (args.cv_model == 0 and args.vc_model == 0):
				W_gw = model.W_gw.cpu().data.numpy()
				weights_temp = {'W_gw':W_gw}
			if (args.adaptivity_training == 1):
				filename_mat = models_folder_mat_final + "nams_" + str(args.coding_scheme) + "_" + str(n) + "_" + str(k) + "_st_" + str(args.steps) + "_lr_" +str(args.learning_rate) + "_" + str(args.channel_type) + "_adapt_from_" + base_channel + "_ent_" + str(args.entangle_weights)+ "_nn_eq_" + str(args.nn_eq) + "_relu_" + str(args.relu) + "_max_iter_" + str(args.num_iterations) + "_" + str(int(args.eb_n0_train_lo)) + "_" + str(int(args.eb_n0_train_hi)) + ".mat"
			else:
				filename_mat = models_folder_mat_final + "nams_" + str(args.coding_scheme) + "_" + str(n) + "_" + str(k) + "_st_" + str(args.steps) + "_lr_" +str(args.learning_rate) + "_" + str(args.channel_type) + "_ent_" + str(args.entangle_weights)+ "_nn_eq_" + str(args.nn_eq) + "_relu_" + str(args.relu) + "_max_iter_" + str(args.num_iterations) + "_" + str(int(args.eb_n0_train_lo)) + "_" + str(int(args.eb_n0_train_hi)) + ".mat"
			if (args.freeze_weights == 1):
				filename_mat = models_folder_mat_final + "nams_" + str(args.coding_scheme) + "_" + str(n) + "_" + str(k) + "_st_" + str(args.steps) + "_lr_" +str(args.learning_rate) + "_" + str(args.channel_type) + "_ent_" + str(args.entangle_weights)+ "_nn_eq_" + str(args.nn_eq) + "_relu_" + str(args.relu) + "_max_iter_" + str(args.num_iterations) + "_" + str(int(args.eb_n0_train_lo)) + "_" + str(int(args.eb_n0_train_hi)) + "_ff_" + str(args.freeze_fraction) + ".mat"
			sio.savemat(filename_mat,weights_temp)
			if (args.adaptivity_training == 1):
				filename_mat = models_folder_mat + "nams_" + str(args.coding_scheme) + "_" + str(n) + "_" + str(k) + "_st_" + str(args.steps) + "_lr_" +str(args.learning_rate) + "_" + str(args.channel_type) + "_adapt_from_" + base_channel + "_ent_" + str(args.entangle_weights)+ "_nn_eq_" + str(args.nn_eq) + "_relu_" + str(args.relu) + "_max_iter_" + str(args.num_iterations) + "_" + str(int(args.eb_n0_train_lo)) + "_" + str(int(args.eb_n0_train_hi)) + ".mat"
			else:
				filename_mat = models_folder_mat + "nams_" + str(args.coding_scheme) + "_" + str(n) + "_" + str(k) + "_st_" + str(args.steps) + "_lr_" +str(args.learning_rate) + "_" + str(args.channel_type) + "_ent_" + str(args.entangle_weights)+ "_nn_eq_" + str(args.nn_eq) + "_relu_" + str(args.relu) + "_max_iter_" + str(args.num_iterations) + "_" + str(int(args.eb_n0_train_lo)) + "_" + str(int(args.eb_n0_train_hi)) + ".mat"
			if (args.freeze_weights == 1):
				filename_mat = models_folder_mat + "nams_" + str(args.coding_scheme) + "_" + str(n) + "_" + str(k) + "_st_" + str(args.steps) + "_lr_" +str(args.learning_rate) + "_" + str(args.channel_type) + "_ent_" + str(args.entangle_weights)+ "_nn_eq_" + str(args.nn_eq) + "_relu_" + str(args.relu) + "_max_iter_" + str(args.num_iterations) + "_" + str(int(args.eb_n0_train_lo)) + "_" + str(int(args.eb_n0_train_hi)) + "_ff_" + str(args.freeze_fraction) + ".mat"
			sio.savemat(filename_mat,weights_temp)

eb_n0_dB = np.arange(args.eb_n0_lo, args.eb_n0_hi+args.eb_n0_step, args.eb_n0_step)

SNRs = eb_n0_to_snr(eb_n0_dB,rate,mod_bits)

def test_nams(SNRs, model, decoder, code):

	for SNR in SNRs:
		# simulate this SNR
		############ FIX ME###########
		sigma = calc_sigma(SNR,rate)
		frame_count = 0
		bit_errors = 0
		frame_errors = 0
		FE = 0

		testing_batch_size = decoder.testing_batch_size
  
		# simulate frames
		while ((FE < min_frame_errors) or (frame_count <100000)) and (frame_count < max_frames) :
			frame_count += testing_batch_size

			messages = torch.randint(0,2,(testing_batch_size,code.k))
			codewords = torch.matmul(torch.tensor(G,dtype=int), torch.transpose(messages,0,1)) % 2

			# bpsk modulation
			BPSK_codewords = (codewords - 0.5) * 2.0

			BPSK_codewords_interf = torch.zeros_like(BPSK_codewords)
			BPSK_codewords_interf[1:,:] = BPSK_codewords[:-1,:]
			sigma_vec = sigma*torch.ones_like(BPSK_codewords)

			# Pass through channel
			noise = sigma * np.random.randn(BPSK_codewords.shape[0],BPSK_codewords.shape[1])
			FastFading = False
			exact_llr = args.exact_llr

			received_codewords, soft_input = apply_channel(BPSK_codewords, sigma, args.alpha, noise, args.channel_type, FastFading, exact_llr)
			llr_in = 2.0*received_codewords/(sigma*sigma)

			# Phase 2 : decode using wbp
			batch_data = torch.reshape(llr_in,(received_codewords.size(dim=0),received_codewords.size(dim=1))).to(device)
			if (decoder.decoder_type == "undec"):
				received_codewords = (llr_in > 0).to(device)
			else:
				llr_out, received_codewords = nams_decode(soft_input, H, num_iterations)

			# update bit error count and frame error count
			errors = codewords.to(device) != received_codewords
			bit_errors += errors.sum()
			frame_errors += (errors.sum(0) > 0).sum()

			FE = frame_errors
   
	bit_count = frame_count * n
	BER = float(bit_errors) / float(bit_count)
	BERs.append(BER)

	FER = float(frame_errors) / float(frame_count)
	FERs.append(FER)

   
	return SNRs, BERs, FERs

def main(config, gpu = -1):
	pass
