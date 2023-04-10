import numpy as np
import torch
import os

from .convcode import conv_encode, bcjr_decode, Trellis
from .interleaver import Interleaver
from .tinyturbo import TinyTurbo
from .utils import dec2bitarray


def get_qpp(f1, f2, block_len):
    nums = np.arange(block_len)
    inds = (f1*nums + f2*(nums**2))%block_len

    return inds

class TurboCode():
    def __init__(self, code = 'lte', block_len = 40, interleaver_type = 'qpp', interleaver_seed=0, puncture=False):

        assert code in ['lte', '757'], "Supported codes are 'lte' and '757'"
        self.code = code
        self.block_len = block_len
        self.puncture = puncture
        self.interleaver_seed = interleaver_seed
        self.interleaver_type = interleaver_type

        if self.code == '757':
            # Turbo-757 parameters
            self.M = np.array([2])                         # Number of delay elements in the convolutional encoder
            self.generator_matrix = np.array([[7, 5]])     # Encoder of convolutional encoder
            self.feedback = 7
        else:
            # Turbo-LTE parameters
            self.M = np.array([3])                         # Number of delay elements in the convolutional encoder
            self.generator_matrix = np.array([[11, 13]])     # Encoder of convolutional encoder
            self.feedback = 11

        self.trellis1 = Trellis(self.M, self.generator_matrix, self.feedback, 'rsc')
        self.trellis2 = Trellis(self.M, self.generator_matrix, self.feedback, 'rsc')
        self.interleaver = Interleaver(self.block_len, self.interleaver_seed)

        if self.interleaver_type == 'qpp':
            if self.block_len == 40:
                p_array = get_qpp(3, 10, 40)
                self.interleaver.set_p_array(p_array)
            elif self.block_len == 64:
                p_array = get_qpp(7, 16, 64)
                self.interleaver.set_p_array(p_array)
            elif self.block_len == 104:
                p_array = get_qpp(7, 26, 104)
                self.interleaver.set_p_array(p_array)
            elif self.block_len == 200:
                p_array = get_qpp(13, 50, 200)
                self.interleaver.set_p_array(p_array)
            elif self.block_len == 504:
                p_array = get_qpp(55,84, 504)
                self.interleaver.set_p_array(p_array)
            elif self.block_len == 1008:
                p_array = get_qpp(55, 84, 1008)
                self.interleaver.set_p_array(p_array)
            else:
                print("QPP not yet supported for block length {}".format(self.block_len))
        # interleaver = Interleaver(self.block_len, 0, p_array) # for custom permutation
        print('Using interleaver p-array : {}'.format(list(self.interleaver.p_array)))

        # Load tinyturbo weights
        script_dir = os.path.dirname(os.path.abspath(__file__))
        checkpoint = torch.load(os.path.join(script_dir, "Results", "tinyturbo", "models", "weights.pt"))
        weights = checkpoint['weights']
        self.tinyturbo = TinyTurbo(self.block_len, 3)
        self.tinyturbo.load_state_dict(weights)


    def turbo_encode(self, message_bits, puncture=False):
        """ Turbo Encoder.
        Encode Bits using a parallel concatenated rate-1/3
        turbo code consisting of two rate-1/2 systematic
        convolutional component codes.
        Parameters
        ----------
        message_bits : 2D torch Tensor containing {0, 1} of shape (batch_size, M)
            Stream of bits to be turbo encoded.
        self.trellis1 : self.trellis1 object
            self.trellis1 representation of the
            first code in the parallel concatenation.
        self.trellis2 : self.trellis1 object
            self.trellis1 representation of the
            second code in the parallel concatenation.
        interleaver : Interleaver object
            Interleaver used in the turbo code.
        puncture: Bool
            Currently supports only puncturing pattern '110101'
        Returns
        -------
        stream : torch Tensor of turbo encoded codewords, of shape (batch_size, 3*M + 4*memory)
                where memory is the number of delay elements in the convolutional code, M is the message length.

                First 3*M bits are [sys_1, non_sys1_1, non_sys2_1, . . . . sys_j, non_sys1_j, non_sys2_j, . . . sys_M, non_sys1_M, non_sys2_M]
                Next 2*memory bits are termination bits of sys and non_sys1 : [sys_term_1, non_sys1_term_1, . . . . sys_term_j, non_sys1_term_j, . . . sys_term_M, non_sys1_term_M]
                Next 2*memory bits are termination bits of sys_interleaved and non_sys2 : [sys_inter_term_1, non_sys2_term_1, . . . . sys_inter_term_j, non_sys2_term_j, . . . sys_inter_term_M, non_sys2_term_M]

            Encoded bit streams corresponding
            to the systematic output
            and the two non-systematic
            outputs from the two component codes.
        """

        assert message_bits.shape[1] == self.block_len, "Message length should be equal to block length"
        stream = conv_encode(message_bits, self.trellis1)
        sys_stream = stream[:, ::2]
        non_sys_stream_1 = stream[:, 1::2]

        interlv_msg_bits = self.interleaver.interleave(message_bits)
        #puncture_matrix = np.array([[0, 1]])
        stream_int = conv_encode(interlv_msg_bits, self.trellis2)
        sys_stream_int = stream_int[:, ::2]
        non_sys_stream_2 = stream_int[:, 1::2]

        #Termination bits
        term_sys1 = sys_stream[:, -self.trellis1.total_memory:]
        term_sys2 = sys_stream_int[:, -self.trellis2.total_memory:]
        term_nonsys1 = non_sys_stream_1[:, -self.trellis1.total_memory:]
        term_nonsys2 = non_sys_stream_2[:, -self.trellis2.total_memory:]

        sys_stream = sys_stream[:, :-self.trellis1.total_memory]
        non_sys_stream_1 = non_sys_stream_1[:, :-self.trellis1.total_memory]
        non_sys_stream_2 = non_sys_stream_2[:, :-self.trellis2.total_memory]

        codeword = torch.empty((message_bits.shape[0], message_bits.shape[1]*3), dtype=sys_stream.dtype)
        codeword[:, 0::3] = sys_stream
        codeword[:, 1::3] = non_sys_stream_1
        codeword[:, 2::3] = non_sys_stream_2
        term1 = stream[:, -2*self.trellis1.total_memory:]
        term2 = stream_int[:, -2*self.trellis1.total_memory:]

        if not puncture:
            out = torch.cat((codeword, term1, term2), dim=1)
        else:
            inds = torch.Tensor([1, 1, 0, 1, 0, 1]).repeat(self.block_len//2).byte()
            punctured_codeword = codeword[:, inds]
            out = torch.cat((punctured_codeword, term1, term2), dim=1)
        return out

    def turbo_decode(self, received_llrs, number_iterations, L_int = None, method = 'max_log_MAP', puncture=False):

        """ Turbo Decoder.
        Decode a Turbo code.

        Parameters
        ----------
        received_llrs : LLRs of shape (batch_size, 3*M + 4*memory)
            Received LLRs corresponding to the received Turbo encoded bits
        number_iterations: Int
            Number of iterations of BCJR algorithm
        interleaver : Interleaver object
            Interleaver used in the turbo code.
        L_int : intrinsic LLRs of shape (batch_size, 3*M + 4*memory)
            Intrinsic LLRs (prior). (Set to zeros if no prior)
        method : Turbo decoding method
            max-log-MAP or MAP
        puncture: Bool
            Currently supports only puncturing pattern '110101'

        Returns
        -------
        L_ext : torch Tensor of decoded LLRs, of shape (batch_size, M + memory)

        decoded_bits: L_ext > 0

            Decoded beliefs
        """

        coded = received_llrs[:, :-4*self.trellis1.total_memory]
        term = received_llrs[:, -4*self.trellis1.total_memory:]


        # puncturing to get rate 1/2 . Pattern: '110101'. Can change this later for more patterns
        if puncture:
            # block_len = coded.shape[1]//2
            inds = torch.Tensor([1, 1, 0, 1, 0, 1]).repeat(self.block_len//2).byte()
            zero_inserted = torch.zeros(received_llrs.shape[0], 3*self.block_len, device = received_llrs.device)
            zero_inserted[:, inds] = coded
            coded = zero_inserted.float()
        sys_stream = coded[:, 0::3]
        non_sys_stream1 = coded[:, 1::3]
        non_sys_stream2 = coded[:, 2::3]

        term_sys1 = term[:, :2*self.trellis1.total_memory][:, 0::2]
        term_nonsys1 = term[:, :2*self.trellis1.total_memory][:, 1::2]
        term_sys2 = term[:, 2*self.trellis1.total_memory:][:, 0::2]
        term_nonsys2 = term[:, 2*self.trellis1.total_memory:][:, 1::2]

        sys_llrs = torch.cat((sys_stream, term_sys1), -1)
        non_sys_llrs1 = torch.cat((non_sys_stream1, term_nonsys1), -1)

        sys_stream_inter = self.interleaver.interleave(sys_stream)
        sys_llrs_inter = torch.cat((sys_stream_inter, term_sys2), -1)

        non_sys_llrs2 = torch.cat((non_sys_stream2, term_nonsys2), -1)
        sys_llr = sys_llrs

        if L_int is None:
            L_int = torch.zeros_like(sys_llrs)

        L_int_1 = L_int

        for iteration in range(number_iterations):
            # [L_ext_1, decoded] = bcjr_decode(sys_llrs, non_sys_llrs1, self.trellis1, L_int_1, method=method)
            #
            # L_ext_1 = L_ext_1 - L_int_1 - sys_llr
            # L_int_2 = interleaver.interleave(L_ext_1[:, :sys_stream.shape[1]])
            # L_int_2 = torch.cat((L_int_2, torch.zeros_like(term_sys1)), -1)
            #
            # [L_ext_2, decoded] = bcjr_decode(sys_llrs_inter, non_sys_llrs2, self.trellis1, L_int_2, method=method)
            #
            # L_ext_2 = L_ext_2 - L_int_2
            # L_int_1 = interleaver.deinterleave(L_ext_2[:, :sys_stream.shape[1]])
            # L_int_1 = L_int_1 - sys_llr[:, :sys_stream.shape[1]]
            # L_int_1 = torch.cat((L_int_1, torch.zeros_like(term_sys1)), -1)

            [L_ext_1, decoded] = bcjr_decode(sys_llrs, non_sys_llrs1, self.trellis1, L_int_1, method=method)

            L_ext = L_ext_1 - L_int_1 - sys_llr
            L_e_1 = L_ext_1[:, :sys_stream.shape[1]]
            L_1 = L_int_1[:, :sys_stream.shape[1]]

            L_int_2 = L_e_1 - sys_llr[:, :sys_stream.shape[1]] - L_1
            L_int_2 = self.interleaver.interleave(L_int_2)
            L_int_2 = torch.cat((L_int_2, torch.zeros_like(term_sys1)), -1)

            [L_ext_2, decoded] = bcjr_decode(sys_llrs_inter, non_sys_llrs2, self.trellis1, L_int_2, method=method)

            L_e_2 = self.interleaver.deinterleave(L_ext_2[:, :sys_stream.shape[1]])
            L_2 = self.interleaver.deinterleave(L_int_2[:, :sys_stream.shape[1]])

            L_int_1 = L_e_2 - sys_llr[:, :sys_stream.shape[1]] - L_2
            L_int_1 = torch.cat((L_int_1, torch.zeros_like(term_sys1)), -1)
        LLRs = L_ext + L_int_1 + sys_llr
        decoded_bits = (LLRs > 0).float()

        return LLRs, decoded_bits

    def tinyturbo_decode(self, received_llrs, number_iterations, tinyturbo = None, L_int = None, method = 'max_log_MAP', puncture = False):

        """ Turbo Decoder.
        Decode a Turbo code using TinyTurbo weights.

        Parameters
        ----------
        tinyturbo : instance of decoder class
            Contains normal and interleaved weights for TinyTurbo
        received_llrs : LLRs of shape (batch_size, 3*M + 4*memory)
            Received LLRs corresponding to the received Turbo encoded bits
        trellis : Trellis object
            Trellis representation of the convolutional code
        number_iterations: Int
            Number of iterations of BCJR algorithm
        interleaver : Interleaver object
            Interleaver used in the turbo code.
        L_int : intrinsic LLRs of shape (batch_size, 3*M + 4*memory)
            Intrinsic LLRs (prior). (Set to zeros if no prior)
        method : Turbo decoding method
            max-log-MAP or MAP
        puncture: Bool
            Currently supports only puncturing pattern '110101'

        Returns
        -------
        L_ext : torch Tensor of decoded LLRs, of shape (batch_size, M + memory)

        decoded_bits: L_ext > 0

            Decoded beliefs
        """

        if tinyturbo is None:
            tinyturbo = self.tinyturbo # use default weights
        tinyturbo.to(received_llrs.device)

        coded = received_llrs[:, :-4*self.trellis1.total_memory]
        term = received_llrs[:, -4*self.trellis1.total_memory:]
        if puncture:
            block_len = coded.shape[1]//2
            inds = torch.Tensor([1, 1, 0, 1, 0, 1]).repeat(block_len//2).byte()
            zero_inserted = torch.zeros(received_llrs.shape[0], 3*block_len, device = received_llrs.device)
            zero_inserted[:, inds] = coded
            coded = zero_inserted.float()
        sys_stream = coded[:, 0::3]
        non_sys_stream1 = coded[:, 1::3]
        non_sys_stream2 = coded[:, 2::3]

        term_sys1 = term[:, :2*self.trellis1.total_memory][:, 0::2]
        term_nonsys1 = term[:, :2*self.trellis1.total_memory][:, 1::2]
        term_sys2 = term[:, 2*self.trellis1.total_memory:][:, 0::2]
        term_nonsys2 = term[:, 2*self.trellis1.total_memory:][:, 1::2]

        sys_llrs = torch.cat((sys_stream, term_sys1), -1)
        non_sys_llrs1 = torch.cat((non_sys_stream1, term_nonsys1), -1)

        sys_stream_inter = self.interleaver.interleave(sys_stream)
        sys_llrs_inter = torch.cat((sys_stream_inter, term_sys2), -1)

        non_sys_llrs2 = torch.cat((non_sys_stream2, term_nonsys2), -1)
        sys_llr = sys_llrs

        if L_int is None:
            L_int = torch.zeros_like(sys_llrs).to(coded.device)

        L_int_1 = L_int

        for iteration in range(number_iterations):
            [L_ext_1, decoded] = bcjr_decode(sys_llrs, non_sys_llrs1, self.trellis1, L_int_1, method=method)


            L_ext = L_ext_1 - L_int_1 - sys_llr
            L_e_1 = L_ext_1[:, :sys_stream.shape[1]]
            L_1 = L_int_1[:, :sys_stream.shape[1]]

            L_int_2 = tinyturbo.normal[iteration](L_e_1, sys_llr[:, :sys_stream.shape[1]], L_1)
            L_int_2 = self.interleaver.interleave(L_int_2)
            L_int_2 = torch.cat((L_int_2, torch.zeros_like(term_sys1)), -1)

            [L_ext_2, decoded] = bcjr_decode(sys_llrs_inter, non_sys_llrs2, self.trellis2, L_int_2, method=method)

            L_e_2 = self.interleaver.deinterleave(L_ext_2[:, :sys_stream.shape[1]])
            L_2 = self.interleaver.deinterleave(L_int_2[:, :sys_stream.shape[1]])

            L_int_1 = tinyturbo.interleaved[iteration](L_e_2, sys_llr[:, :sys_stream.shape[1]], L_2)
            L_int_1 = torch.cat((L_int_1, torch.zeros_like(term_sys1)), -1)

        LLRs = torch.cat((L_2, torch.zeros_like(term_sys2)), -1) + L_int_1 + sys_llr

        decoded_bits = (LLRs > 0).float()

        return LLRs, decoded_bits
