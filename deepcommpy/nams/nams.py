import numpy as np
import torch
import torch.nn as nn
import os


class NAMS_net(nn.Module):
	def __init__(self, decoder, code):
		super(NAMS_net, self).__init__()
		# entanglement type - 0: NAMS, 1: NAMS-RNN-FW, 2: NAMS-RNN-SS
		if(decoder.entangle_weights == 0):
			num_w1 = decoder.max_iter
			num_w2 = code.num_edges				
		elif(decoder.entangle_weights == 1):
			num_w1 = 1
			num_w2 = code.num_edges
		elif(decoder.entangle_weights == 2):
			num_w1 = 1
			num_w2 = 1
		else:
			raise ValueError('Invalid entanglement type')

		# generate the weight init vectors : CV
		B_cv_init = torch.randn([num_w1, num_w2])
		W_cv_init = torch.randn([num_w1, num_w2])

		# augmentation type - 0: NAMS, 1: NNMS, 2: NOMS
		if (decoder.nn_eq == 0):
			self.B_cv = torch.nn.Parameter(B_cv_init)
			self.W_cv = torch.nn.Parameter(W_cv_init)
		elif (decoder.nn_eq == 1):
			self.B_cv = torch.zeros(num_w1, num_w2)
			self.W_cv = torch.nn.Parameter(W_cv_init)
		elif (decoder.nn_eq == 2):
			self.B_cv = torch.nn.Parameter(B_cv_init)
			self.W_cv = torch.ones(num_w1, num_w2)
			
		
def nams_decode(soft_input, H, num_iterations):
    
	num_edges = np.sum(H)
	batch_size = soft_input.shape[0]
	soft_output = soft_input
 
	iteration = 0
	cv = torch.zeros([num_edges,batch_size])
	m_t = torch.zeros([num_edges,batch_size])


	while ( iteration < num_iterations ):
		[soft_input, soft_output, iteration, cv, m_t] = belief_propagation_iteration(soft_input, iteration, cv, m_t, batch_size)

	hard_output = (soft_output > 0).float()
	return soft_output, hard_output


def belief_propagation_iteration(soft_input, iteration, cv, m_t, batch_size):

	vc = compute_vc(cv, soft_input, iteration, batch_size)
	vc_prime = vc

	cv = compute_cv(vc_prime, iteration, batch_size)

	soft_output = marginalize(soft_input, iteration, cv, batch_size)
	iteration += 1

	return soft_input, soft_output, iteration, cv, m_t


# compute messages from variable nodes to check nodes
def compute_vc(model, cv, soft_input, decoder, code, iteration, batch_size):
	weighted_soft_input = soft_input		
	reordered_soft_input = weighted_soft_input[code.edges,:]

	vc = []
	# for each variable node v, find the list of extrinsic edges
	# fetch the LLR values from cv using these indices
	# find the sum and store in temp, finally store these in vc
	count_vc = 0
	for i in range(0, code.n): 
		for j in range(0, code.var_degrees[i]):
			# if the list of extrinsic edges is not empty, add them up
			if code.extrinsic_edges_vc[count_vc]:
				temp = cv[code.extrinsic_edges_vc[count_vc],:]
				temp = torch.sum(temp,0)
			else:
				temp = torch.zeros([batch_size])
			vc.append(temp)
			count_vc = count_vc + 1
	vc = torch.stack(vc)
	vc = vc[code.new_order_vc,:]

	vc = vc + reordered_soft_input

	return vc

# compute messages from check nodes to variable nodes
def compute_cv(model, vc, decoder, code, iteration, batch_size):
	cv_list = torch.tensor([])
	prod_list = torch.tensor([])
	min_list = torch.tensor([])

	if (decoder.decoder_type == "spa"):
		vc = torch.clip(vc, -10, 10)
		tanh_vc = torch.tanh(vc / 2.0)
	count_cv = 0
	for i in range(0, code.m): # for each check node c
		for j in range(0, code.chk_degrees[i]): #edges per check node
			if (decoder.decoder_type == "spa"):
				temp = tanh_vc[code.extrinsic_edges_cv[count_cv],:]
				temp = torch.prod(temp,0)
				temp = torch.log((1+temp)/(1-temp))
				cv_list = torch.cat((cv_list,temp.float()),0)
			elif (decoder.decoder_type == "min_sum" or decoder.decoder_type == "neural_ms"):
				if code.extrinsic_edges_cv[count_cv]:
					temp = vc[code.extrinsic_edges_cv[count_cv],:]
				else:
					temp = torch.zeros([1,batch_size])
				prod_chk_temp = torch.prod(torch.sign(temp),0)
				(sign_chk_temp, min_ind) = torch.min(torch.abs(temp),0)
				prod_list = torch.cat((prod_list,prod_chk_temp.float()),0)
				min_list = torch.cat((min_list,sign_chk_temp.float()),0)
			count_cv = count_cv + 1

	if (decoder.decoder_type == "spa"):
		cv = torch.reshape(cv_list,vc.size())
	elif (decoder.decoder_type == "min_sum"):
		prods = torch.reshape(prod_list,vc.size()) #stack across batch size
		mins = torch.reshape(min_list,vc.size())
		cv = prods * mins
	elif (decoder.decoder_type == "neural_ms"):
		prods = torch.reshape(prod_list,vc.size()) #stack across batch size
		mins = torch.reshape(min_list,vc.size())

		# apply the weights
		# entanglement type - 0 - 5xedges, 1 - 1xedges, 2 - 1xnum_var_nodes, 3 - 1xnum_chk_nodes, 4 - 1xedges_chk_node,  5 - 5xedges_per_chk_node, 6 - 1x1
		if (decoder.entangle_weights == 0):
			idx = iteration
		else:
			idx = 0
   
		# Replicate same weight for all edges	
		if (decoder.entangle_weights == 2):
			B_cv_vec = model.B_cv.repeat([1,code.num_edges])
			W_cv_vec = model.W_cv.repeat([1,code.num_edges])		
		else:
			B_cv_vec = model.B_cv
			W_cv_vec = model.W_cv

		# Replicate the offsets and scaling matrix across batch size
		offsets = torch.tile(torch.reshape(B_cv_vec[idx],[-1,1]),[1,batch_size])
		scaling = torch.tile(torch.reshape(W_cv_vec[idx],[-1,1]),[1,batch_size])
			
		cv = scaling * prods * torch.nn.functional.relu(mins - offsets)


	cv = cv[code.new_order_cv,:]
	return cv

# combine messages to get posterior LLRs
def marginalize(soft_input, cv, code, batch_size):
	weighted_soft_input = soft_input
 
	soft_output =  torch.tensor([])

	for i in range(0,code.n):
		temp = cv[code.edges_m[i],:]

		temp = torch.sum(temp,0) 
		soft_output = torch.cat((soft_output,temp),0)

	soft_output = torch.reshape(soft_output,soft_input.size())

	soft_output = weighted_soft_input + soft_output

	return soft_output