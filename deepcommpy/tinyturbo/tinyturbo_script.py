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

from .turbo import TurboCode
from .tinyturbo import TinyTurbo
from ..channels import Channel
from ..utils import snr_db2sigma, errors_ber, errors_bler, moving_average

def train_tinyturbo(turbocode, device, config = None, loaded_weights = None):
    """
    Training function
    TinyTurbo training : Use config['target] = 'scale' (default).
    
    If config['target'] == 'LLR', then training proceeds like Turbonet+
    (Y. He, J. Zhang, S. Jin, C.-K. Wen, and G. Y. Li, “Model-driven dnn
    decoder for turbo codes: Design, simulation, and experimental results,”
    IEEE Transactions on Communications, vol. 68, no. 10, pp. 6127–6140)

    Parameters
    ----------
    turbocode : TurboCode
        Turbo code object.
    device : torch.device
        Device to use for computations.
        Eg: torch.device('cuda:0') or torch.device('cpu')
    config : dict, optional
        Configuration dictionary.
        Example config provided as `deepcommpy/tinyturbo/train_config.json`.
    loaded_weights : dict, optional
        Dictionary of weights to load into the model.

    Returns
    -------
    tinyturbo : TinyTurbo
        Trained TinyTurbo model.
    training_losses : list
        List of training losses.
    training_bers : list
        List of training bit error rates.
    step : int
        Number of training steps.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if config is None:
        with open(os.path.join(script_dir, 'train_config.json'), 'r') as f:
            config = json.load(f)

    tinyturbo = TinyTurbo(config['block_len'], config['tinyturbo_iters'], config['init_type'], config['decoding_type'])
    if loaded_weights is not None:
        tinyturbo.load_state_dict(loaded_weights)
    tinyturbo.to(device)

    params = list(tinyturbo.parameters())

    criterion = nn.BCEWithLogitsLoss() if config['loss_type'] == 'BCE' else nn.MSELoss()
    optimizer = optim.Adam(params, lr = config['lr'])

    sigma = snr_db2sigma(config['train_snr'])
    noise_variance = sigma**2

    noise_type = config['noise_type'] #if config['noise_type'] is not 'isi' else 'isi_1'
    assert noise_type in ['awgn', 'fading', 'radar', 't-dist'], "Please choose one of these noise types for training: 'awgn', 'fading', 'radar', 't-dist'"
    channel = Channel(noise_type)
    print("TRAINING")
    training_losses = []
    training_bers = []

    try:
        for step in range(config['num_steps']):
            start = time.time()
            message_bits = torch.randint(0, 2, (config['batch_size'], config['block_len']), dtype=torch.float).to(device)
            coded = turbocode.encode(message_bits, puncture = config['puncture']).to(device)
            # noisy_coded = corrupt_signal(coded, sigma, noise_type, vv = config['vv'], radar_power = config['radar_power'], radar_prob = config['radar_prob'])
            noisy_coded = channel.corrupt_signal(2*coded-1, sigma, vv = config['vv'], radar_power = config['radar_power'], radar_prob = config['radar_prob'])
            if noise_type not in ['EPA', 'EVA', 'ETU', 'MIMO']:
                received_llrs = 2*noisy_coded/noise_variance
            else:
                received_llrs = noisy_coded

            if config['input'] == 'y':
                tinyturbo_llr, decoded_tt = turbocode.tinyturbo_decode(noisy_coded, config['tinyturbo_iters'], tinyturbo = tinyturbo, method = config['tt_bcjr'], puncture = config['puncture'])
            else:
                tinyturbo_llr, decoded_tt = turbocode.tinyturbo_decode(received_llrs, config['tinyturbo_iters'], tinyturbo = tinyturbo, method = config['tt_bcjr'], puncture = config['puncture'])


            if config['target'] == 'LLR':
                #Turbo decode
                log_map_llr, _ = turbocode.turbo_decode(received_llrs, config['turbo_iters'], method='log_MAP', puncture = config['puncture'])
                loss = criterion(tinyturbo_llr, log_map_llr)
            elif config['target'] == 'gt':
                if config['loss_type'] == 'BCE':
                    loss = criterion(tinyturbo_llr, message_bits)
                elif config['loss_type'] == 'MSE':
                    loss = criterion(torch.tanh(tinyturbo_llr/2.), 2*message_bits-1)
            ber = errors_ber(message_bits, decoded_tt)

            training_losses.append(loss.item())
            training_bers.append(ber)


            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (step+1)%10 == 0:
                print('Step : {}, Loss = {:.5f}, BER = {:.5f}, {:.2f} seconds, ID: {}'.format(step+1, loss, ber, time.time() - start, config['id']))

            if (step+1)%config['save_every'] == 0 or step==0:
                torch.save({'weights': tinyturbo.cpu().state_dict(), 'config' : config, 'steps': step+1, 'p_array':turbocode.interleaver.p_array}, os.path.join(script_dir, 'Results', config['id'], 'models/weights.pt'))
                torch.save({'weights': tinyturbo.cpu().state_dict(), 'config' : config, 'steps': step+1, 'p_array':turbocode.interleaver.p_array}, os.path.join(script_dir, 'Results', config['id'], 'models/weights_{}.pt'.format(int(step+1))))
                tinyturbo.to(device)
            if (step+1)%10 == 0:
                plt.figure()
                plt.plot(training_losses)
                plt.plot(moving_average(training_losses, n=10))
                plt.savefig(os.path.join(script_dir, 'Results', config['id'], 'training_losses.png'))
                plt.close()

                plt.figure()
                plt.plot(training_losses)
                plt.plot(moving_average(training_losses, n=10))
                plt.yscale('log')
                plt.savefig(os.path.join(script_dir, 'Results', config['id'], 'training_losses_log.png'))
                plt.close()

                plt.figure()
                plt.plot(training_bers)
                plt.plot(moving_average(training_bers, n=10))
                plt.savefig(os.path.join(script_dir, 'Results', config['id'], 'training_bers.png'))
                plt.close()

                plt.figure()
                plt.plot(training_bers)
                plt.plot(moving_average(training_bers, n=10))
                plt.yscale('log')
                plt.savefig(os.path.join(script_dir, 'Results', config['id'], 'training_bers_log.png'))
                plt.close()

                with open(os.path.join(script_dir, 'Results', config['id'], 'values_training.csv'), 'w') as f:

                    # using csv.writer method from CSV package
                    write = csv.writer(f)

                    write.writerow(list(range(1, step+1)))
                    write.writerow(training_losses)
                    write.writerow(training_bers)

        return tinyturbo, training_losses, training_bers, step+1

    except KeyboardInterrupt:
        print("Exited")

        torch.save({'weights': tinyturbo.cpu().state_dict(), 'config': config, 'steps': step+1, 'p_array':turbocode.interleaver.p_array}, os.path.join(script_dir, 'Results', config['id'], 'models/weights.pt'))
        torch.save({'weights': tinyturbo.cpu().state_dict(), 'config': config, 'steps': step+1, 'p_array':turbocode.interleaver.p_array}, os.path.join(script_dir, 'Results', config['id'], 'models/weights_{}.pt'.format(int(step+1))))
        tinyturbo.to(device)

        with open(os.path.join(script_dir, 'Results', config['id'], 'values_training.csv'), 'w') as f:

             # using csv.writer method from CSV package
             write = csv.writer(f)
             write.writerow(list(range(1, step+1)))
             write.writerow(training_losses)
             write.writerow(training_bers)

        return tinyturbo, training_losses, training_bers, step+1

def test_tinyturbo(turbocode, device, tinyturbo = None, config = None):
    """
    Test TinyTurbo on a test set

    Parameters
    ----------
    turbocode : TurboCode
        Turbo code object
    device : torch.device
        Device to use for training
        Eg. torch.device('cuda:0') or torch.device('cpu')   
    tinyturbo : TinyTurbo, optional
        If None, default TinyTurbo is used from paper)
    config : dict, optional
        If None, default config is used from test_config.json
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if config is None:
        with open(os.path.join(script_dir, 'test_config.json'), 'r') as f:
            config = json.load(f)

    snr_range = config['snr_range']
    num_batches = config['test_size'] // config['test_batch_size']
    noise_type = config['noise_type']
    assert noise_type in ['awgn', 'fading', 'radar', 't-dist', 'EPA', 'EVA', 'ETU'], "Please choose one of these noise types: 'awgn', 'fading', 'radar', 't-dist', 'EPA', 'EVA', 'ETU',"#'MIMO'"

    channel = Channel(noise_type)

    bers_ml = []
    blers_ml = []
    bers_l = []
    blers_l = []
    bers_tt = []
    blers_tt = []
    print("TESTING")
    with torch.no_grad():
        for ii in range(num_batches):
            message_bits = torch.randint(0, 2, (config['test_batch_size'], config['block_len']), dtype=torch.float).to(device)
            coded = turbocode.encode(message_bits, puncture = config['puncture']).to(device)

            for k, snr in tqdm(enumerate(snr_range)):
                sigma = snr_db2sigma(snr)
                noise_variance = sigma**2

                noisy_coded = channel.corrupt_signal(2*coded-1, sigma, vv = config['vv'], radar_power = config['radar_power'], radar_prob = config['radar_prob'])
                if config['noise_type'] not in ['EPA', 'EVA', 'ETU', 'MIMO']:
                    received_llrs = 2*noisy_coded/noise_variance
                else:
                    received_llrs = noisy_coded

                if not only_tt:
                    # Turbo decode
                    ml_llrs, decoded_ml = turbocode.turbo_decode(received_llrs, config['tinyturbo_iters'],
                                                 method='max_log_MAP', puncture = config['puncture'])
                    ber_maxlog = errors_ber(message_bits, decoded_ml)
                    bler_maxlog = errors_bler(message_bits, decoded_ml)

                    if ii == 0:
                        bers_ml.append(ber_maxlog/num_batches)
                        blers_ml.append(bler_maxlog/num_batches)
                    else:
                        bers_ml[k] += ber_maxlog/num_batches
                        blers_ml[k] += bler_maxlog/num_batches

                    l_llrs, decoded_l = turbocode.turbo_decode(received_llrs, config['turbo_iters'],
                                                method='MAP', puncture = config['puncture'])
                    ber_log = errors_ber(message_bits, decoded_l)
                    bler_log = errors_bler(message_bits, decoded_l)

                    if ii == 0:
                        bers_l.append(ber_log/num_batches)
                        blers_l.append(bler_log/num_batches)
                    else:
                        bers_l[k] += ber_log/num_batches
                        blers_l[k] += bler_log/num_batches

                # tinyturbo decode
                if config['input'] == 'y':
                    tt_llrs, decoded_tt = turbocode.tinyturbo_decode(noisy_coded, config['tinyturbo_iters'], tinyturbo = tinyturbo, method = config['tt_bcjr'], puncture = config['puncture'])
                else:
                    tt_llrs, decoded_tt = turbocode.tinyturbo_decode(received_llrs, config['tinyturbo_iters'], tinyturbo = tinyturbo, method = config['tt_bcjr'], puncture = config['puncture'])

                ber_tinyturbo = errors_ber(message_bits, decoded_tt)
                bler_tinyturbo = errors_bler(message_bits, decoded_tt)

                if ii == 0:
                    bers_tt.append(ber_tinyturbo/num_batches)
                    blers_tt.append(bler_tinyturbo/num_batches)
                else:
                    bers_tt[k] += ber_tinyturbo/num_batches
                    blers_tt[k] += bler_tinyturbo/num_batches

    return snr_range, bers_ml, bers_l, bers_tt, blers_ml, blers_l, blers_tt
