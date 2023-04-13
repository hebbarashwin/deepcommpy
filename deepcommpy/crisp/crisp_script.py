__author__ = 'hebbarashwin'

import torch
import torch.nn as nn
import torch.optim as optim
import time
import os
import csv
import json
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
from tqdm import tqdm

from ..channels import Channel
from ..utils import snr_db2sigma, errors_ber, errors_bler

def crisp_rnn_test(polar, device, net=None, config=None):

    script_dir = os.path.dirname(os.path.abspath(__file__))
    if config is None:
        with open(os.path.join(script_dir, 'test_config.json'), 'r') as f:
            config = json.load(f)

    snr_range = config['snr_range']
    num_test_batches = config['test_size'] // config['test_batch_size']
    noise_type = config['noise_type']
    assert noise_type in ['awgn', 'fading', 'radar', 't-dist'], "Please choose one of these noise types: 'awgn', 'fading', 'radar', 't-dist'"

    channel = Channel(noise_type)
    bers_RNN_test = [0. for ii in snr_range]
    blers_RNN_test = [0. for ii in snr_range]

    bers_SC_test = [0. for ii in snr_range]
    blers_SC_test = [0. for ii in snr_range]

    with torch.no_grad():
        for k in range(num_test_batches):
            msg_bits = 2*torch.randint(0, 2, (config['test_batch_size'], polar.K), dtype=torch.float) - 1
            msg_bits = msg_bits.to(device)
            polar_code = polar.encode(msg_bits)
            for snr_ind, snr in enumerate(snr_range):
                sigma = snr_db2sigma(snr)
                noisy_code = channel.corrupt_signal(polar_code, sigma, vv = config['vv'], radar_power = config['radar_power'], radar_prob = config['radar_prob'])
                noise = noisy_code - polar_code

                SC_llrs, decoded_SC_msg_bits = polar.sc_decode(noisy_code, snr)
                ber_SC = errors_ber(msg_bits, decoded_SC_msg_bits.sign())
                bler_SC = errors_bler(msg_bits, decoded_SC_msg_bits.sign())

                decoded_RNN_msg_bits = polar.crisp_rnn_decode(noisy_code, net)

                ber_RNN = errors_ber(msg_bits, decoded_RNN_msg_bits.sign())
                bler_RNN = errors_bler(msg_bits, decoded_RNN_msg_bits.sign())


                bers_RNN_test[snr_ind] += ber_RNN/num_test_batches
                bers_SC_test[snr_ind] += ber_SC/num_test_batches


                blers_RNN_test[snr_ind] += bler_RNN/num_test_batches
                blers_SC_test[snr_ind] += bler_SC/num_test_batches

    return bers_RNN_test, blers_RNN_test, bers_SC_test, blers_SC_test

def crisp_cnn_test(polar, device, net=None, config=None):

    script_dir = os.path.dirname(os.path.abspath(__file__))
    if config is None:
        with open(os.path.join(script_dir, 'test_config.json'), 'r') as f:
            config = json.load(f)

    snr_range = config['snr_range']
    num_test_batches = config['test_size'] // config['test_batch_size']
    noise_type = config['noise_type']
    assert noise_type in ['awgn', 'fading', 'radar', 't-dist'], "Please choose one of these noise types: 'awgn', 'fading', 'radar', 't-dist'"

    channel = Channel(noise_type)
    bers_CNN_test = [0. for ii in snr_range]
    blers_CNN_test = [0. for ii in snr_range]

    bers_SC_test = [0. for ii in snr_range]
    blers_SC_test = [0. for ii in snr_range]

    with torch.no_grad():
        for k in range(num_test_batches):
            msg_bits = 2*torch.randint(0, 2, (config['test_batch_size'], polar.K), dtype=torch.float) - 1
            msg_bits = msg_bits.to(device)
            polar_code = polar.encode(msg_bits)
            for snr_ind, snr in enumerate(snr_range):
                sigma = snr_db2sigma(snr)
                noisy_code = channel.corrupt_signal(polar_code, sigma, vv = config['vv'], radar_power = config['radar_power'], radar_prob = config['radar_prob'])
                noise = noisy_code - polar_code

                SC_llrs, decoded_SC_msg_bits = polar.sc_decode(noisy_code, snr)
                ber_SC = errors_ber(msg_bits, decoded_SC_msg_bits.sign())
                bler_SC = errors_bler(msg_bits, decoded_SC_msg_bits.sign())

                decoded_CNN_msg_bits = polar.crisp_cnn_decode(noisy_code, net)

                ber_CNN = errors_ber(msg_bits, decoded_CNN_msg_bits.sign())
                bler_CNN = errors_bler(msg_bits, decoded_CNN_msg_bits.sign())


                bers_CNN_test[snr_ind] += ber_CNN/num_test_batches
                bers_SC_test[snr_ind] += ber_SC/num_test_batches


                blers_CNN_test[snr_ind] += bler_CNN/num_test_batches
                blers_SC_test[snr_ind] += bler_SC/num_test_batches

    return bers_CNN_test, blers_CNN_test, bers_SC_test, blers_SC_test

