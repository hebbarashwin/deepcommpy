Examples
========

This page provides example usage of the `deepcommpy` module.

TinyTurbo Example
-----------------

Here's an example of how to use the `tinyturbo` module:

.. code-block:: python

    import torch
    import deepcommpy
    from deepcommpy.utils import snr_db2sigma
    from deepcommpy.channels import Channel

    # Create a Turbo code object : Turbo-LTE, Block_length = 40
    block_len = 40
    turbocode = deepcommpy.tinyturbo.TurboCode(code='lte', block_len = block_len)

    # Create an AWGN channel object.
    # Channel supports the following channels: 'awgn', 'fading', 't-dist', 'radar'
    # It also supports 'EPA', 'EVA', 'ETU' with matlab dependency.
    channel = Channel('awgn')

    # Generate random message bits for testing
    message_bits = torch.randint(0, 2, (10000, block_len), dtype=torch.float)
    # Turbo encoding and BPSK modulation
    coded = 2 * turbocode.encode(message_bits) - 1

    # Simulate over range of SNRs
    snr_range = [-1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2]
    for snr in snr_range:
        sigma = snr_db2sigma(snr)
        # add noise 
        noisy_coded = channel.corrupt_signal(coded, sigma)
        received_llrs = 2*noisy_coded/sigma**2

        # Max-Log-MAP Turbo decoding with 3 iterations
        _ , decoded_max = turbocode.turbo_decode(received_llrs, number_iterations = 3, method='max_log_MAP')
        # MAP Turbo decoding with 6 iterations
        _ , decoded_map = turbocode.turbo_decode(received_llrs, number_iterations = 6, method='MAP')
        # TinyTurbo decoding with 3 iterations
        _, decoded_tt = turbocode.tinyturbo_decode(received_llrs, number_iterations = 3)

        # Compute the bit error rates
        ber_max = torch.ne(message_bits, decoded_max).float().mean().item()
        ber_map = torch.ne(message_bits, decoded_map).float().mean().item()
        ber_tt = torch.ne(message_bits, decoded_tt).float().mean().item()

