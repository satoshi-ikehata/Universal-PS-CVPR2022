import os
import sys
import numpy as np

class setup_configuration():
    def __init__(self):
        self.abciroot = '../../../'
        self.img_channels = 3
        self.train_datatype = 'AdobeNPI'
        self.train_suffix= '0*'
        self.train_ext = '.data'
        self.train_imgscale = 1.0
        self.train_maxNumberOfImages = 10
        self.test_datatype = 'RealData'
        self.test_suffix = 'L*'
        self.test_ext = '.data'
        self.test_imgscale = 1.0
        self.test_maxNumberOfImages = 10
