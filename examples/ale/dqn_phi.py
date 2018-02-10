from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from builtins import *  # NOQA
from future import standard_library
standard_library.install_aliases()

import numpy as np


def dqn_phi(screens):
    """Phi (feature extractor) of DQN for ALE

    Args:
      screens: List of N screen objects. Each screen object must be
      numpy.ndarray whose dtype is numpy.uint8.
    Returns:
      numpy.ndarray
    """
    #print(screens.shape)
    #assert len(screens) == 4
    #assert screens[0].dtype == np.uint8
    raw_values = np.asarray(screens, dtype=np.float32)
    raw_values = raw_values.transpose(2, 0, 1) 
    # [0,255] -> [0, 1]
    #raw_values /= 255.0
    return raw_values
