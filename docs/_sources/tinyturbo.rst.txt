TinyTurbo Module
=================
This module provides an interface in Python to perform Turbo encoding, and decoding using the classical decoders, as well as the TinyTurbo decoder.


Functions
---------
.. autoclass:: deepcommpy.tinyturbo.TurboCode
   :members: encode, turbo_decode, tinyturbo_decode
   :special-members: __init__
.. autoclass:: deepcommpy.tinyturbo.TinyTurbo
    :members: __init__, forward
.. autofunction:: deepcommpy.tinyturbo.train_tinyturbo
.. autofunction:: deepcommpy.tinyturbo.test_tinyturbo
