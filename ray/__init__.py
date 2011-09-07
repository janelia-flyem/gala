"""
Ray
===

    Ray is a Python package for nD image segmentation. 
"""

import sys, logging
if sys.version_info[:2] < (2,6):
    logging.warning('Ray has not been tested on Python versions prior to 2.6'+
        ' (%d.%d detected).'%sys.version_info[:2])

__author__ = 'Juan Nunez-Iglesias <jni@janelia.hhmi.org>, '+\
             'Ryan Kennedy <kenry@cis.upenn.edu>'
del sys, logging

import agglo
import morpho
import evaluate
import viz
import imio
import classify

