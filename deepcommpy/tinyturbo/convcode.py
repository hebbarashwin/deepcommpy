import numpy as np
import torch

from ..utils import dec2bitarray, bitarray2dec

def max_star(metrics):

    assert metrics.shape[1] >= 2, "Number of operants for max* operation must be at least 2"
    temp = metrics[:, 0].clone()
    for ii in range(1, metrics.shape[1]):
        temp = torch.max(temp, metrics[:, ii].clone()) + torch.log(1 + torch.exp(-torch.abs(temp - metrics[:, ii].clone())))

    return temp

class Trellis:
    """
    Class defining a Trellis corresponding to a k/n - rate convolutional code.
    This follow the classical representation. See [1] for instance.
    Input and output are represented as little endian e.g. output = decimal(output[0], output[1] ...).
    Parameters
    ----------
    memory : 1D ndarray of ints
        Number of memory elements per input of the convolutional encoder.
    g_matrix : 2D ndarray of ints (decimal representation)
        Generator matrix G(D) of the convolutional encoder. Each element of G(D) represents a polynomial.
        Coef [i,j] is the influence of input i on output j.
    feedback : 2D ndarray of ints (decimal representation), optional
        Feedback matrix F(D) of the convolutional encoder. Each element of F(D) represents a polynomial.
        Coef [i,j] is the feedback influence of input i on input j.
        *Default* implies no feedback.
        The backwards compatibility version is triggered if feedback is an int.
    code_type : {'default', 'rsc'}, optional
        Use 'rsc' to generate a recursive systematic convolutional code.
        If 'rsc' is specified, then the first 'k x k' sub-matrix of
        G(D) must represent a identity matrix along with a non-zero
        feedback polynomial.
        *Default* is 'default'.
    polynomial_format : {'MSB', 'LSB', 'Matlab'}, optional
        Defines how to interpret g_matrix and feedback. In MSB format, we have 1+D <-> 3 <-> 011.
        In LSB format, which is used in Matlab, we have 1+D <-> 6 <-> 110.
        *Default* is 'MSB' format.
    Attributes
    ----------
    k : int
        Size of the smallest block of input bits that can be encoded using
        the convolutional code.
    n : int
        Size of the smallest block of output bits generated using
        the convolutional code.
    total_memory : int
        Total number of delay elements needed to implement the convolutional
        encoder.
    number_states : int
        Number of states in the convolutional code trellis.
    number_inputs : int
        Number of branches from each state in the convolutional code trellis.
    next_state_table : 2D ndarray of ints
        Table representing the state transition matrix of the
        convolutional code trellis. Rows represent current states and
        columns represent current inputs in decimal. Elements represent the
        corresponding next states in decimal.
    output_table : 2D ndarray of ints
        Table representing the output matrix of the convolutional code trellis.
        Rows represent current states and columns represent current inputs in
        decimal. Elements represent corresponding outputs in decimal.
    Raises
    ------
    ValueError
        polynomial_format is not 'MSB', 'LSB' or 'Matlab'.
    Examples
    --------
    >>> from numpy import array
    >>> import commpy.channelcoding.convcode as cc
    >>> memory = array([2])
    >>> g_matrix = array([[5, 7]]) # G(D) = [1+D^2, 1+D+D^2]
    >>> trellis = cc.Trellis(memory, g_matrix)
    >>> print trellis.k
    1
    >>> print trellis.n
    2
    >>> print trellis.total_memory
    2
    >>> print trellis.number_states
    4
    >>> print trellis.number_inputs
    2
    >>> print trellis.next_state_table
    [[0 2]
     [0 2]
     [1 3]
     [1 3]]
    >>>print trellis.output_table
    [[0 3]
     [3 0]
     [1 2]
     [2 1]]
    References
    ----------
    [1] S. Benedetto, R. Garello et G. Montorsi, "A search for good convolutional codes to be used in the
    construction of turbo codes", IEEE Transactions on Communications, vol. 46, n. 9, p. 1101-1005, spet. 1998
    """
    def __init__(self, memory, g_matrix, feedback=None, code_type='default', polynomial_format='MSB'):

        [self.k, self.n] = g_matrix.shape
        self.code_type = code_type

        self.total_memory = memory.sum()
        self.number_states = pow(2, self.total_memory)
        self.number_inputs = pow(2, self.k)
        self.next_state_table = np.zeros([self.number_states,
                                          self.number_inputs], 'int')
        self.output_table = np.zeros([self.number_states,
                                      self.number_inputs], 'int')

        if isinstance(feedback, int):
            # warn('Trellis  will only accept feedback as a matrix in the future. '
            #      'Using the backwards compatibility version that may contain bugs for k > 1 or with LSB format.',
            #      DeprecationWarning)

            if code_type == 'rsc':
                for i in range(self.k):
                    g_matrix[i][i] = feedback

            # Compute the entries in the next state table and the output table
            for current_state in range(self.number_states):

                for current_input in range(self.number_inputs):
                    outbits = np.zeros(self.n, 'int')

                    # Compute the values in the output_table
                    for r in range(self.n):

                        output_generator_array = np.zeros(self.k, 'int')
                        shift_register = dec2bitarray(current_state,
                                                      self.total_memory)

                        for l in range(self.k):

                            # Convert the number representing a polynomial into a
                            # bit array
                            generator_array = dec2bitarray(g_matrix[l][r],
                                                           memory[l] + 1)

                            # Loop over M delay elements of the shift register
                            # to compute their contribution to the r-th output
                            for i in range(memory[l]):
                                outbits[r] = (outbits[r] + \
                                              (shift_register[i + l] * generator_array[i + 1])) % 2

                            output_generator_array[l] = generator_array[0]
                            if l == 0:
                                feedback_array = (dec2bitarray(feedback, memory[l] + 1)[1:] * shift_register[0:memory[l]]).sum()
                                shift_register[1:memory[l]] = \
                                    shift_register[0:memory[l] - 1]
                                shift_register[0] = (dec2bitarray(current_input,
                                                                  self.k)[0] + feedback_array) % 2
                            else:
                                feedback_array = (dec2bitarray(feedback, memory[l] + 1) *
                                                  shift_register[
                                                  l + memory[l - 1] - 1:l + memory[l - 1] + memory[l] - 1]).sum()
                                shift_register[l + memory[l - 1]:l + memory[l - 1] + memory[l] - 1] = \
                                    shift_register[l + memory[l - 1] - 1:l + memory[l - 1] + memory[l] - 2]
                                shift_register[l + memory[l - 1] - 1] = \
                                    (dec2bitarray(current_input, self.k)[l] + feedback_array) % 2

                        # Compute the contribution of the current_input to output
                        outbits[r] = (outbits[r] + \
                                      (np.sum(dec2bitarray(current_input, self.k) * \
                                              output_generator_array + feedback_array) % 2)) % 2

                    # Update the ouput_table using the computed output value
                    self.output_table[current_state][current_input] = \
                        bitarray2dec(outbits)

                    # Update the next_state_table using the new state of
                    # the shift register
                    self.next_state_table[current_state][current_input] = \
                        bitarray2dec(shift_register)

        else:
            if polynomial_format == 'MSB':
                bit_order = -1
            elif polynomial_format in ('LSB', 'Matlab'):
                bit_order = 1
            else:
                raise ValueError('polynomial_format must be "LSB", "MSB" or "Matlab"')

            if feedback is None:
                feedback = np.identity(self.k, int)
                if polynomial_format in ('LSB', 'Matlab'):
                    feedback *= 2**memory.max()

            max_values_lign = memory.max() + 1  # Max number of value on a delay lign

            # feedback_array[i] holds the i-th bit corresponding to each feedback polynomial.
            feedback_array = np.zeros((max_values_lign, self.k, self.k), np.int8)
            for i in range(self.k):
                for j in range(self.k):
                    binary_view = dec2bitarray(feedback[i, j], max_values_lign)[::bit_order]
                    feedback_array[:max_values_lign, i, j] = binary_view[-max_values_lign-2:]

            # g_matrix_array[i] holds the i-th bit corresponding to each g_matrix polynomial.
            g_matrix_array = np.zeros((max_values_lign, self.k, self.n), np.int8)
            for i in range(self.k):
                for j in range(self.n):
                    binary_view = dec2bitarray(g_matrix[i, j], max_values_lign)[::bit_order]
                    g_matrix_array[:max_values_lign, i, j] = binary_view[-max_values_lign-2:]

            # shift_regs holds on each column the state of a shift register.
            # The first row is the input of each shift reg.
            shift_regs = np.empty((max_values_lign, self.k), np.int8)

            # Compute the entries in the next state table and the output table
            for current_state in range(self.number_states):
                for current_input in range(self.number_inputs):
                    current_state_array = dec2bitarray(current_state, self.total_memory)

                    # Set the first row as the input.
                    shift_regs[0] = dec2bitarray(current_input, self.k)

                    # Set the other rows based on the current_state
                    idx = 0
                    for idx_mem, mem in enumerate(memory):
                        shift_regs[1:mem+1, idx_mem] = current_state_array[idx:idx + mem]
                        idx += mem

                    # Compute the output table
                    outputs_array = np.einsum('ik,ikl->l', shift_regs, g_matrix_array) % 2
                    self.output_table[current_state, current_input] = bitarray2dec(outputs_array)

                    # Update the first line based on the feedback polynomial
                    np.einsum('ik,ilk->l', shift_regs, feedback_array, out=shift_regs[0])
                    shift_regs %= 2

                    # Update current state array and compute next state table
                    idx = 0
                    for idx_mem, mem in enumerate(memory):
                        current_state_array[idx:idx + mem] = shift_regs[:mem, idx_mem]
                        idx += mem
                    self.next_state_table[current_state, current_input] = bitarray2dec(current_state_array)

def conv_encode(message_bits, trellis, puncture_matrix = None):
    """
    Encode bits using a convolutional code.

    Parameters
    ----------
    message_bits : 2D Tensor containing {0, 1}
        Stream of bits to be convolutionally encoded.

    generator_matrix : 2-D ndarray of ints
        Generator matrix G(D) of the convolutional code using which the input
        bits are to be encoded.

    M : 1D ndarray of ints
        Number of memory elements per input of the convolutional encoder.

    Returns
    -------
    coded_bits : 2D Tensor containing {0, 1}
        Encoded bit stream.
    """

    k = trellis.k
    n = trellis.n
    total_memory = trellis.total_memory
    code_type = trellis.code_type
    rate = float(k)/n

    if puncture_matrix is None:
        puncture_matrix = np.ones((trellis.k, trellis.n))

    num_states = pow(2, trellis.total_memory)
    bit_array_states = torch.Tensor(np.array([dec2bitarray(ii, n) for ii in range(num_states)]))
    batch_size, number_message_bits = message_bits.shape

    # Initialize an array to contain the message bits plus the truncation zeros
    if code_type == 'default':
        inbits = torch.zeros(batch_size, number_message_bits + total_memory + total_memory % k, dtype = torch.int)
        number_inbits = number_message_bits + total_memory + total_memory % k

        # Pad the input bits with M zeros (L-th terminated truncation)
        inbits[:, 0:number_message_bits] = message_bits
        number_outbits = int(number_inbits/rate)

    else:
        inbits = message_bits
        number_inbits = number_message_bits
        number_outbits = int((number_inbits + total_memory)/rate)

    outbits = torch.zeros((batch_size, number_outbits), dtype=torch.int)
    next_state_table = torch.from_numpy(trellis.next_state_table)
    output_table = torch.from_numpy(trellis.output_table)

    # Encoding process - Each iteration of the loop represents one clock cycle
    current_states = torch.zeros(batch_size).long()
    j = 0

    # writing for k = 1
    for i in range(number_inbits): # Loop through all input bits

        current_input = inbits[:, i]
        current_outputs = bit_array_states[output_table[current_states.long(), current_input.long()]]
        outbits[:, j*n:(j+1)*n] = current_outputs
        current_states = next_state_table[current_states.long(), current_input.long()]
        j += 1


    if code_type == 'rsc':

        term_bits = torch.stack([torch.from_numpy(dec2bitarray(state, trellis.total_memory)) for state in current_states])
        term_bits = torch.flip(term_bits, [1])
        # print('RSC')
        for i in range(trellis.total_memory):
            states = torch.stack([torch.from_numpy(dec2bitarray(state, trellis.total_memory)) for state in current_states])
            # For LTE
            if trellis.total_memory == 3:
                current_input = (states[:, 1] + states[:, 2])%2
            # For 757
            elif trellis.total_memory == 2:
                current_input = (states[:, 0] + states[:, 1])%2

            current_outputs = bit_array_states[output_table[current_states.long(), current_input.long()]]
            outbits[:, j*n:(j+1)*n] = current_outputs

            current_states = next_state_table[current_states.long(), current_input.long()]
            j += 1

    # PUNCTURING
    p_matrix_size = puncture_matrix.shape[1]
    nonzero_positions = np.nonzero(puncture_matrix[0])
    # In puncturing matrix, if i_th element is 0, drop every ith element from outbits
    inds_list = [np.arange(ii, outbits.shape[1], p_matrix_size) for ii in nonzero_positions[0]]
    inds = np.concatenate(inds_list)
    inds.sort()
    p_outbits = outbits[:, inds]

    return p_outbits.float()

def bcjr_decode(sys_llrs, non_sys_llrs, trellis, L_int, method = 'max_log_MAP'):
    """ BCJR Decoder.
    Decode a rate-1/2 systematic convoluitonal code.

    Parameters
    ----------
    sys_llrs : systematic LLRs of shape (batch_size, 3*M + 4*memory)
        Received LLRs corresponding to systematic bits
    non_sys_llrs : non-systematic LLRs of shape (batch_size, 3*M + 4*memory)
        Received LLRs corresponding to non-systematic parity bits
    trellis : Trellis object
        Trellis representation of the convolutional code
    L_int : intrinsic LLRs of shape (batch_size, 3*M + 4*memory)
        Intrinsic LLRs (prior). (Set to zeros if no prior)
    method : Turbo decoding method
        max-log-MAP or MAP

    Returns
    -------
    L_ext : torch Tensor of decoded LLRs, of shape (batch_size, M + memory)

    decoded_bits: L_ext > 0

        Decoded beliefs
    """

    if method not in ['max_log_MAP', 'MAP']:
        method = 'MAP'

    k = trellis.k
    n = trellis.n
    rate = float(k)/n
    number_states = trellis.number_states
    number_inputs = trellis.number_inputs

    batch_size, msg_length = sys_llrs.shape

    next_state_table = trellis.next_state_table
    output_table = trellis.output_table
    # Initialize forward state metrics (alpha)
    f_state_metrics = -1000* torch.ones((batch_size, number_states, msg_length+1)).to(sys_llrs.device)
    f_state_metrics[:, 0, 0] = 0
    f_state_temp = torch.zeros((batch_size, number_states, number_inputs)).to(sys_llrs.device)

    # Initialize backward state metrics (beta)
    b_state_metrics = -1000* torch.ones((batch_size, number_states, msg_length+1)).to(sys_llrs.device)
    # b_state_metrics[:, :,msg_length] = 0
    b_state_metrics[:, 0,msg_length] = 0
    b_state_temp = torch.zeros((batch_size, number_inputs)).to(sys_llrs.device)

    branch_probs = torch.zeros((batch_size, number_inputs, number_states, msg_length+1)).to(sys_llrs.device)
    L_ext = torch.zeros_like(sys_llrs)
    L_temp = [[], []]
    # Backward recursion:
    for reverse_time_index in range(msg_length, 0, -1):
        for current_state in range(number_states):
            for current_input in range(number_inputs):
                next_state = next_state_table[current_state, current_input]
                code_symbol = output_table[current_state, current_input]
                codeword_array = dec2bitarray(code_symbol, n)
                parity_bit = codeword_array[1]
                msg_bit = codeword_array[0]

                code_symbol_0 = 2*codeword_array[0]-1
                code_symbol_1 = 2*codeword_array[1]-1

                rx_llr_0 = sys_llrs[:, reverse_time_index-1]
                rx_llr_1 = non_sys_llrs[:, reverse_time_index-1]

                # log of branch prob :
                # branch_prob = -(x**2 + y**2)/(2*noise_variance) + torch.log(torch.sigmoid((1 - 2*current_input)*L_int))
                # branch_prob = (code_symbol_0*rx_symbol_0 + code_symbol_1*rx_symbol_1)/noise_variance + 0.5*(2*current_input-1)*L_int[:, reverse_time_index-1]

                branch_prob = 0.5*(code_symbol_0*rx_llr_0 + code_symbol_1*rx_llr_1) + 0.5*(2*current_input-1)*L_int[:, reverse_time_index-1]
                branch_probs[:, current_input, current_state, reverse_time_index-1] = branch_prob

                b_state_temp[:, current_input] = b_state_metrics[:, next_state, reverse_time_index] + branch_prob
            if method == 'max_log_MAP':
                b_state_metrics[:, current_state, reverse_time_index-1] , _ = torch.max(b_state_temp, dim=1)
            elif method == 'MAP':
                b_state_metrics[:, current_state, reverse_time_index-1] = max_star(b_state_temp)

    # Forward recursion:
    for time_index in range(1, msg_length+1):
        L_temp = [[], []]
        for current_state in range(number_states):
            for current_input in range(number_inputs):
                next_state = next_state_table[current_state, current_input]
                branch_prob = branch_probs[:, current_input, current_state, time_index-1]
                f_state_temp[:, next_state, current_input] = f_state_metrics[:, current_state, time_index-1] + branch_prob
        if method == 'max_log_MAP':
            for s in range(number_states):
                f_state_metrics[:, s, time_index] , _ = torch.max(f_state_temp[:, s, :], dim=1)
        elif method == 'MAP':
            for s in range(number_states):
                f_state_metrics[:, s, time_index] = max_star(f_state_temp[:, s, :])
        for current_state in range(number_states):
            for current_input in range(number_inputs):
                next_state = next_state_table[current_state, current_input]
                branch_prob = branch_probs[:, current_input, current_state, time_index-1]
                alpha = f_state_metrics[:, current_state, time_index-1]
                beta = b_state_metrics[:, next_state, time_index]
                L = alpha + beta + branch_prob

                L_temp[current_input].append(L)
        stack0 = torch.stack(L_temp[0], -1)
        stack1 = torch.stack(L_temp[1], -1)

        if method == 'max_log_MAP':
            L0, _ = torch.max(stack0, -1)
            L1, _ = torch.max(stack1, -1)
            L_ext[:, time_index-1] = L1 - L0
        elif method == 'MAP':
            L0 = max_star(stack0)
            L1 = max_star(stack1)
            L_ext[:, time_index-1] = L1 - L0

    decoded = (L_ext> 0).float()
    return L_ext, decoded