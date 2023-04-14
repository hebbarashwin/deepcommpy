TinyTurbo Module
=================
This module provides an interface in Python to perform Turbo encoding, and decoding using the classical decoders, as well as the TinyTurbo decoder.
Based on the paper: ["TinyTurbo: Efficient Turbo Decoders on Edge"](https://arxiv.org/abs/2209.15614) (ISIT 2022)



Functions
---------
.. autoclass:: deepcommpy.tinyturbo.TurboCode
   :members: encode, turbo_decode, tinyturbo_decode
   :special-members: __init__
.. autoclass:: deepcommpy.tinyturbo.TinyTurbo
    :members: __init__, forward
.. autofunction:: deepcommpy.tinyturbo.train_tinyturbo
.. autofunction:: deepcommpy.tinyturbo.test_tinyturbo
