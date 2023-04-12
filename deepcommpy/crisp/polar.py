import numpy as np
import torch

from ..utils import snr_db2sigma

def min_sum_log_sum_exp(x, y):

    log_sum_ms = torch.min(torch.abs(x), torch.abs(y))*torch.sign(x)*torch.sign(y)
    return log_sum_ms

class PolarCode:

    def __init__(self, N, K, F = None, rs = None, infty = 1000.):

        assert (N>2 and N%2 == 0)
        self.N = N
        self.n = int(np.log2(N))
        self.K = K
        self.G2 = np.array([[1,0],[1,1]])
        self.G = np.array([1])
        for i in range(self.n):
            self.G = np.kron(self.G, self.G2)
        self.G = torch.from_numpy(self.G).float()
        self.infty = infty
        self.hard_decision = False

        if F is not None:
            assert len(F) == self.N - self.K
            self.frozen_positions = F
            self.unsorted_frozen_positions = self.frozen_positions
            self.frozen_positions.sort()

            self.info_positions = np.array(list(set(self.frozen_positions) ^ set(np.arange(self.N))))
            self.unsorted_info_positions = self.info_positions
            self.info_positions.sort()
        else:
            if rs is None:
                # in increasing order of reliability
                if self.N == 32:
                    rs = np.array([31, 30, 29, 27, 23, 15, 28, 26, 25, 22, 21, 14, 19, 13, 11, 24,  7, 20, 18, 12, 17, 10,  9,  6,  5,  3, 16,  8,  4,  2,  1,  0])

                elif self.N == 16:
                    rs = np.array([15, 14, 13, 11, 7, 12, 10, 9, 6, 5, 3, 8, 4, 2, 1, 0])
                elif self.N == 8:
                    rs = np.array([7, 6, 5, 3, 4, 2, 1, 0])
                elif self.N == 4:
                    rs = np.array([3, 2, 1, 0])

                rs = np.array([256 ,255 ,252 ,254 ,248 ,224 ,240 ,192 ,128 ,253 ,244 ,251 ,250 ,239 ,238 ,247 ,246 ,223 ,222 ,232 ,216 ,236 ,220 ,188 ,208 ,184 ,191 ,190 ,176 ,127 ,126 ,124 ,120 ,249 ,245 ,243 ,242 ,160 ,231 ,230 ,237 ,235 ,234 ,112 ,228 ,221 ,219 ,218 ,212 ,215 ,214 ,189 ,187 ,96 ,186 ,207 ,206 ,183 ,182 ,204 ,180 ,200 ,64 ,175 ,174 ,172 ,125 ,123 ,122 ,119 ,159 ,118 ,158 ,168 ,241 ,116 ,111 ,233 ,156 ,110 ,229 ,227 ,217 ,108 ,213 ,152 ,226 ,95 ,211 ,94 ,205 ,185 ,104 ,210 ,203 ,181 ,92 ,144 ,202 ,179 ,199 ,173 ,178 ,63 ,198 ,121 ,171 ,88 ,62 ,117 ,170 ,196 ,157 ,167 ,60 ,115 ,155 ,109 ,166 ,80 ,114 ,154 ,107 ,56 ,225 ,151 ,164 ,106 ,93 ,150 ,209 ,103 ,91 ,143 ,201 ,102 ,48 ,148 ,177 ,90 ,142 ,197 ,87 ,100 ,61 ,169 ,195 ,140 ,86 ,59 ,32 ,165 ,194 ,113 ,79 ,58 ,153 ,84 ,136 ,55 ,163 ,78 ,105 ,149 ,162 ,54 ,76 ,101 ,47 ,147 ,89 ,52 ,141 ,99 ,46 ,146 ,72 ,85 ,139 ,98 ,31 ,44 ,193 ,138 ,57 ,83 ,30 ,135 ,77 ,40 ,82 ,134 ,161 ,28 ,53 ,75 ,132 ,24 ,51 ,74 ,45 ,145 ,71 ,50 ,16 ,97 ,70 ,43 ,137 ,68 ,42 ,29 ,39 ,81 ,27 ,133 ,38 ,26 ,36 ,131 ,23 ,73 ,22 ,130 ,49 ,15 ,20 ,69 ,14 ,12 ,67 ,41 ,8 ,66 ,37 ,25 ,35 ,34 ,21 ,129 ,19 ,13 ,18 ,11 ,10 ,7 ,65 ,6 ,4 ,33 ,17 ,9 ,5 ,3 ,2 ,1 ]) - 1
                rs = rs[rs<self.N]

                self.rs = self.reliability_seq[self.reliability_seq<self.N]
            else:
                self.reliability_seq = rs
                self.rs = self.reliability_seq[self.reliability_seq<self.N]

                assert len(self.rs) == self.N
            # best K bits
            self.info_positions = self.rs[:self.K]
            self.unsorted_info_positions = self.reliability_seq[self.reliability_seq<self.N][:self.K]
            self.info_positions.sort()
            self.unsorted_info_positions=np.flip(self.unsorted_info_positions)
            # worst N-K bits
            self.frozen_positions = self.rs[self.K:]
            self.unsorted_frozen_positions = self.rs[self.K:]
            self.frozen_positions.sort()


            self.CRC_polynomials = {
            3: torch.Tensor([1, 0, 1, 1]).int(),
            8: torch.Tensor([1, 1, 1, 0, 1, 0, 1, 0, 1]).int(),
            16: torch.Tensor([1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]).int(),
                                    }

    def encode(self, message, custom_info_positions = None):

        # message shape is (batch, k)
        # BPSK convention : 0 -> +1, 1 -> -1
        # Therefore, xor(a, b) = a*b
        if custom_info_positions is not None:
            info_positions = custom_info_positions
        else:
            info_positions = self.info_positions
        u = torch.ones(message.shape[0], self.N, dtype=torch.float).to(message.device)
        u[:, info_positions] = message

        for d in range(0, self.n):
            num_bits = 2**d
            for i in np.arange(0, self.N, 2*num_bits):
                # [u v] encoded to [u xor(u,v)]
                u = torch.cat((u[:, :i], u[:, i:i+num_bits].clone() * u[:, i+num_bits: i+2*num_bits], u[:, i+num_bits:]), dim=1)
        return u

    # def channel(self, code, snr):
    #     sigma = snr_db2sigma(snr)

    #     noise = (sigma* torch.randn(code.shape, dtype = torch.float)).to(code.device)
    #     r = code + noise

    #     return r

    def define_partial_arrays(self, llrs):
        # Initialize arrays to store llrs and partial_sums useful to compute the partial successive cancellation process.
        llr_array = torch.zeros(llrs.shape[0], self.n+1, self.N, device=llrs.device)
        llr_array[:, self.n] = llrs
        partial_sums = torch.zeros(llrs.shape[0], self.n+1, self.N, device=llrs.device)
        return llr_array, partial_sums


    def updateLLR(self, leaf_position, llrs, partial_llrs = None, prior = None):

        #START
        depth = self.n
        decoded_bits = partial_llrs[:,0].clone()
        if prior is None:
            prior = torch.zeros(self.N) #priors
        llrs, partial_llrs, decoded_bits = self.partial_decode(llrs, partial_llrs, depth, 0, leaf_position, prior, decoded_bits)
        return llrs, decoded_bits


    def partial_decode(self, llrs, partial_llrs, depth, bit_position, leaf_position, prior, decoded_bits=None):
        # Function to call recursively, for partial SC decoder.
        # We are assuming that u_0, u_1, .... , u_{leaf_position -1} bits are known.
        # Partial sums computes the sums got through Plotkin encoding operations of known bits, to avoid recomputation.
        # this function is implemented for rate 1 (not accounting for frozen bits in polar SC decoding)

        half_index = 2 ** (depth - 1)
        leaf_position_at_depth = leaf_position // 2**(depth-1) # will tell us whether left_child or right_child

        # n = 2 tree case
        if depth == 1:
            # Left child
            left_bit_position = 2*bit_position
            if leaf_position_at_depth > left_bit_position:
                u_hat = partial_llrs[:, depth-1, left_bit_position:left_bit_position+1]
            elif leaf_position_at_depth == left_bit_position:
                Lu = min_sum_log_sum_exp(llrs[:, depth, left_bit_position*half_index:(left_bit_position+1)*half_index], llrs[:,depth, (left_bit_position+1)*half_index:(left_bit_position+2)*half_index]).sum(dim=1, keepdim=True)
                # Lu = log_sum_avoid_zero_NaN(llrs[:, depth, left_bit_position*half_index:(left_bit_position+1)*half_index], llrs[:,depth, (left_bit_position+1)*half_index:(left_bit_position+2)*half_index]).sum(dim=1, keepdim=True)
                llrs[:, depth-1, left_bit_position*half_index:(left_bit_position+1)*half_index] = Lu + prior[left_bit_position]*torch.ones_like(Lu)
                if self.hard_decision:
                    u_hat = torch.sign(Lu)
                else:
                    u_hat = torch.tanh(Lu/2)

                decoded_bits[:, left_bit_position] = u_hat.squeeze(1)

                return llrs, partial_llrs, decoded_bits

            # Right child
            right_bit_position = 2*bit_position + 1
            if leaf_position_at_depth > right_bit_position:
                pass
            elif leaf_position_at_depth == right_bit_position:
                Lv = u_hat * llrs[:, depth, left_bit_position*half_index:(left_bit_position+1)*half_index] + llrs[:,depth, (left_bit_position+1)*half_index:(left_bit_position+2)*half_index]
                llrs[:, depth-1, right_bit_position*half_index:(right_bit_position+1)*half_index] = Lv + prior[right_bit_position] * torch.ones_like(Lv)
                if self.hard_decision:
                    v_hat = torch.sign(Lv)
                else:
                    v_hat = torch.tanh(Lv/2)
                decoded_bits[:, right_bit_position] = v_hat.squeeze(1)
                return llrs, partial_llrs, decoded_bits

        # General case
        else:
            # LEFT CHILD
            # Find likelihood of (u xor v) xor (v) = u
            # Lu = log_sum_exp(torch.cat([llrs[:, :half_index].unsqueeze(2), llrs[:, half_index:].unsqueeze(2)], dim=2).permute(0, 2, 1))

            left_bit_position = 2*bit_position
            if leaf_position_at_depth > left_bit_position:
                Lu = llrs[:, depth-1, left_bit_position*half_index:(left_bit_position+1)*half_index]
                u_hat = partial_llrs[:, depth-1, left_bit_position*half_index:(left_bit_position+1)*half_index]
            else:

                Lu = min_sum_log_sum_exp(llrs[:, depth, left_bit_position*half_index:(left_bit_position+1)*half_index], llrs[:,depth, (left_bit_position+1)*half_index:(left_bit_position+2)*half_index])
                # Lu = log_sum_avoid_zero_NaN(llrs[:, depth, left_bit_position*half_index:(left_bit_position+1)*half_index], llrs[:,depth, (left_bit_position+1)*half_index:(left_bit_position+2)*half_index])
                llrs[:, depth-1, left_bit_position*half_index:(left_bit_position+1)*half_index] = Lu
                llrs, partial_llrs, decoded_bits = self.partial_decode(llrs, partial_llrs, depth-1, left_bit_position, leaf_position, prior, decoded_bits)

                return llrs, partial_llrs, decoded_bits

            # RIGHT CHILD
            right_bit_position = 2*bit_position + 1

            Lv = u_hat * llrs[:, depth, left_bit_position*half_index:(left_bit_position+1)*half_index] + llrs[:,depth, (left_bit_position+1)*half_index:(left_bit_position+2)*half_index]
            llrs[:, depth-1, right_bit_position*half_index:(right_bit_position+1)*half_index] = Lv
            llrs, partial_llrs, decoded_bits = self.partial_decode(llrs, partial_llrs, depth-1, right_bit_position, leaf_position, prior, decoded_bits)

            return llrs, partial_llrs, decoded_bits

    def updatePartialSums(self, leaf_position, decoded_bits, partial_llrs):

        u = decoded_bits.clone()
        u[:, leaf_position+1:] = 0

        for d in range(0, self.n):
            partial_llrs[:, d] = u
            num_bits = 2**d
            for i in np.arange(0, self.N, 2*num_bits):
                # [u v] encoded to [u xor(u,v)]
                u = torch.cat((u[:, :i], u[:, i:i+num_bits].clone() * u[:, i+num_bits: i+2*num_bits], u[:, i+num_bits:]), dim=1)
        partial_llrs[:, self.n] = u
        return partial_llrs

    def sc_decode(self, corrupted_codewords, snr, use_gt = None):

        # step-wise implementation using updateLLR and updatePartialSums
        sigma = snr_db2sigma(snr)
        llrs = (2/sigma**2)*corrupted_codewords

        priors = torch.zeros(self.N)
        priors[self.frozen_positions] = self.infty

        u_hat = torch.zeros(corrupted_codewords.shape[0], self.N, device=corrupted_codewords.device)
        llr_array, partial_llrs = self.define_partial_arrays(llrs)
        for ii in range(self.N):
            llr_array , decoded_bits = self.updateLLR(ii, llr_array.clone(), partial_llrs, priors)
            if use_gt is None:
                u_hat[:, ii] = torch.sign(llr_array[:, 0, ii])
            else:
                u_hat[:, ii] = use_gt[:, ii]
            partial_llrs = self.updatePartialSums(ii, u_hat, partial_llrs)
        decoded_bits = u_hat[:, self.info_positions]
        return llr_array[:, 0, :].clone(), decoded_bits

    def crisp_rnn_decode(self, y, net=None):

        def get_onehot(actions):
            inds = (0.5 + 0.5*actions).long()
            return torch.eye(2, device = inds.device)[inds].reshape(actions.shape[0], -1)
        
        if net is None:
            # load default if available 
            pass

        onehot_fn = get_onehot
        iter_range = list(range(0, self.N))
        decoded = torch.ones(y.shape[0], self.N, device = y.device)
        net.eval()
        with torch.no_grad():
            if net.y_depth == 0:
                Fy = y
            else:
                Fy = net.get_Fy(y)
            hidden = torch.zeros((int(net.bidirectional) + 1)*net.num_rnn_layers, y.shape[0], net.feature_size, device = y.device)
            for ii, jj in enumerate(iter_range): # don't assume first bit is always frozen
                if ii == 0:
                    out, hidden = net(torch.cat([Fy.unsqueeze(1), onehot_fn(torch.ones(y.shape[0], device = y.device)).view(-1, 1, net.input_size - self.N)], 2), hidden)
                else:
                    out, hidden = net(torch.cat([Fy.unsqueeze(1), onehot_fn(decoded[:, ii-1].sign()).view(-1, 1, net.input_size - self.N).detach().clone()], 2), hidden)
                if jj in self.info_positions:
                    decoded[:, ii] = out.squeeze().sign()

        return decoded[:, self.info_positions]
    
    def crisp_cnn_decode(self, y, net=None):

        if net is None:
            pass 
        with torch.no_grad():
            decoded = net(y)
            return decoded[:, self.info_positions]
        

# if __name__ == '__main__':
#     n = int(np.log2(args.N))


#     # computed for SNR = 0
#     if n == 5:
#         rs = np.array([31, 30, 29, 27, 23, 15, 28, 26, 25, 22, 21, 14, 19, 13, 11, 24,  7, 20, 18, 12, 17, 10,  9,  6,  5,  3, 16,  8,  4,  2,  1,  0])

#     elif n == 4:
#         rs = np.array([15, 14, 13, 11, 7, 12, 10, 9, 6, 5, 3, 8, 4, 2, 1, 0])
#     elif n == 3:
#         rs = np.array([7, 6, 5, 3, 4, 2, 1, 0])
#     elif n == 2:
#         rs = np.array([3, 2, 1, 0])

#     rs = np.array([256 ,255 ,252 ,254 ,248 ,224 ,240 ,192 ,128 ,253 ,244 ,251 ,250 ,239 ,238 ,247 ,246 ,223 ,222 ,232 ,216 ,236 ,220 ,188 ,208 ,184 ,191 ,190 ,176 ,127 ,126 ,124 ,120 ,249 ,245 ,243 ,242 ,160 ,231 ,230 ,237 ,235 ,234 ,112 ,228 ,221 ,219 ,218 ,212 ,215 ,214 ,189 ,187 ,96 ,186 ,207 ,206 ,183 ,182 ,204 ,180 ,200 ,64 ,175 ,174 ,172 ,125 ,123 ,122 ,119 ,159 ,118 ,158 ,168 ,241 ,116 ,111 ,233 ,156 ,110 ,229 ,227 ,217 ,108 ,213 ,152 ,226 ,95 ,211 ,94 ,205 ,185 ,104 ,210 ,203 ,181 ,92 ,144 ,202 ,179 ,199 ,173 ,178 ,63 ,198 ,121 ,171 ,88 ,62 ,117 ,170 ,196 ,157 ,167 ,60 ,115 ,155 ,109 ,166 ,80 ,114 ,154 ,107 ,56 ,225 ,151 ,164 ,106 ,93 ,150 ,209 ,103 ,91 ,143 ,201 ,102 ,48 ,148 ,177 ,90 ,142 ,197 ,87 ,100 ,61 ,169 ,195 ,140 ,86 ,59 ,32 ,165 ,194 ,113 ,79 ,58 ,153 ,84 ,136 ,55 ,163 ,78 ,105 ,149 ,162 ,54 ,76 ,101 ,47 ,147 ,89 ,52 ,141 ,99 ,46 ,146 ,72 ,85 ,139 ,98 ,31 ,44 ,193 ,138 ,57 ,83 ,30 ,135 ,77 ,40 ,82 ,134 ,161 ,28 ,53 ,75 ,132 ,24 ,51 ,74 ,45 ,145 ,71 ,50 ,16 ,97 ,70 ,43 ,137 ,68 ,42 ,29 ,39 ,81 ,27 ,133 ,38 ,26 ,36 ,131 ,23 ,73 ,22 ,130 ,49 ,15 ,20 ,69 ,14 ,12 ,67 ,41 ,8 ,66 ,37 ,25 ,35 ,34 ,21 ,129 ,19 ,13 ,18 ,11 ,10 ,7 ,65 ,6 ,4 ,33 ,17 ,9 ,5 ,3 ,2 ,1 ]) - 1
#     rs = rs[rs<args.N]

#     ###############
#     ### Polar code
#     ##############

#     ### Encoder

#     polar = PolarCode(n, args.K, args, rs = rs)
