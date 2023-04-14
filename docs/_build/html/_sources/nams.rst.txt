NAMS Module
============

This module provides an interface in Python to perform neural min-sum decoding, and decoding using the classical decoders as well.
Based on the paper: ["Interpreting Neural Min-Sum Decoders"](https://arxiv.org/abs/2205.10684) (ICC 2023)


Functions
---------
.. autoclass:: deepcommpy.nams.code
   :members: encode, min_sum_decode, nams_decode
   :special-members: __init__
.. autoclass:: deepcommpy.nams.nams
    :members: __init__, forward
.. autofunction:: deepcommpy.nams.train_nams
.. autofunction:: deepcommpy.nams.test_nams