import torch
import numpy as np

def snr_db2sigma(train_snr):
    return 10**(-train_snr*1.0/20)

def snr_sigma2db(sigma):
    return -20 * np.log10(sigma)


class Channel:
    """Channel class for adding noise to input signal
    
    Parameters
    ----------
    noise_type : str
        Type of noise to be added to input signal. Valid values are 'awgn', 'fading', 'radar', 't-dist', 'EPA', 'EVA', 'ETU'
        
    """
    def __init__(self, noise_type='awgn'):
        valid_noise_types = ['awgn', 'fading', 'radar', 't-dist', 'EPA', 'EVA', 'ETU']
        assert noise_type in valid_noise_types, "Invalid noise type"
        self.noise_type = noise_type

        if  self.noise_type in ['EPA', 'EVA', 'ETU']: #run from MATLAB
            try:
                assert self.is_matlab_available(), "MATLAB is not available"
                assert self.is_matlab_engine_available(), "MATLAB Engine is not available"
                
                # TO CHECK : START MATLAB ENGINE HERE OR WHEN FUNCTION IS CALLED
                import matlab.engine
                self.eng = matlab.engine.start_matlab()
                s = self.eng.genpath('./matlab_scripts')
                self.eng.addpath(s, nargout=0)
            except:
                # Throw an error if MATLAB is not available
                raise Exception("MATLAB is not available")

    def is_matlab_available():
        import importlib.util
        matlab_spec = importlib.util.find_spec("matlab")
        return matlab_spec is not None

    def is_matlab_engine_available():
        try:
            import matlab.engine
            eng = matlab.engine.start_matlab()
            eng.quit()
            return True
        except:
            return False

    def awgn(self, input_signal, sigma):
        # Add AWGN noise to input_signal
        noise = sigma * torch.randn_like(input_signal)
        corrupted_signal = input_signal + noise
        return corrupted_signal

    def fading(self, input_signal, sigma):
        # Add fading noise to input_signal
        fading_h = torch.sqrt(torch.randn_like(input_signal)**2 +  torch.randn_like(input_signal)**2)/np.sqrt(3.14/2.0)
        noise = self.sigma * torch.randn_like(input_signal) # Define noise
        corrupted_signal = fading_h *(input_signal) + noise
        return corrupted_signal

    def radar(self, input_signal, sigma, radar_power, radar_prob):
        # Add radar noise to input_signal
        data_shape = input_signal.shape
        add_pos = torch.empty(data_shape).bernoulli_(1 - radar_prob)
        corrupted_signal = radar_power * torch.randn(size=data_shape) * add_pos
        noise = sigma * torch.randn_like(input_signal) + corrupted_signal.type(torch.FloatTensor).to(input_signal.device)
        corrupted_signal = input_signal + noise
        return corrupted_signal

    def t_dist(self, input_signal, sigma, vv):
        # Add t-distribution noise to input_signal
        data_shape = input_signal.shape
        noise = sigma * np.sqrt((vv-2)/vv) *np.random.standard_t(vv, size = data_shape)
        corrupted_signal = input_signal + torch.from_numpy(noise).type(torch.FloatTensor).to(input_signal.device)
        return corrupted_signal

    def epa_eva_etu(self, input_signal, snr):
        # Generate LTE data for EPA, EVA, or ETU noise types using MATLAB
        # import matlab.engine
        # eng = matlab.engine.start_matlab()
        # s = eng.genpath('matlab_scripts')
        # eng.addpath(s, nargout=0)

        # calculate closest multiple to num_sym(179)
        num_sym = int(np.floor(input_signal.size(0)/179)) + 1
        code_len = input_signal.shape[-1]
        num_blocks = 179
        SNRs = matlab.double([snr])
        coded_mat = matlab.double(input_signal.numpy().tolist())
        rx_llrs = self.eng.generate_lte_data(coded_mat, code_len, self.noise_type, SNRs, num_blocks, num_sym)
        # convert to numpy
        rx_llrs = np.array(rx_llrs)
        received_llrs = torch.from_numpy(np.transpose(rx_llrs))
        return received_llrs.to(input_signal.device)
    

    def corrupt_signal(self, input_signal, sigma = 1.0, vv =5.0, radar_power = 20.0, radar_prob = 5e-2):
        """Corrupt input signal with noise

        Parameters
        ----------
        input_signal : torch.Tensor
            Input signal to be corrupted
        sigma : float
            Standard deviation of noise
        vv : float (optional)
            Degrees of freedom for t-distribution
        radar_power : float (optional)
            Power of radar noise
        radar_prob : float  (optional)
            Probability of radar noise
        
        Returns
        -------
        corrupted_signal : torch.Tensor
            Corrupted signal
        """
        
        if self.noise_type == 'awgn':
            return self.awgn(input_signal, sigma)
        elif self.noise_type == 'fading':
            return self.fading(input_signal, sigma)
        elif self.noise_type == 'radar':
            return self.radar(input_signal, sigma, radar_power, radar_prob)
        elif self.noise_type == 't-dist':
            return self.t_dist(input_signal, sigma, vv)
        elif self.noise_type in ['EPA', 'EVA', 'ETU']:
            snr = snr_sigma2db(sigma)
            return self.epa_eva_etu(input_signal, snr)
