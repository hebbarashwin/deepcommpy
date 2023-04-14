CRISP Module
============
This module provides an interface in Python to perform polar encoding, and decoding using Successive cancellation, as well as the CRISP-CNN and CRISP-RNN decoders.
Based on the paper: ["CRISP: Curriculum based Sequential Neural Decoders for Polar Code Family
"](https://arxiv.org/abs/2210.00313) (ICC 2023)

Functions
---------
.. autoclass:: deepcommpy.crisp.PolarCode
   :members: encode, sc_decode, crisp_rnn_decode, crisp_cnn_decode
   :special-members: __init__
.. autoclass:: deepcommpy.crisp.RNN_Model
    :members: __init__, forward
.. autoclass:: deepcommpy.crisp.convNet
    :members: __init__, forward
.. autofunction:: deepcommpy.crisp.crisp_rnn_test
.. autofunction:: deepcommpy.crisp.crisp_cnn_test
