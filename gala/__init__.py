"""
Gala
===

    Gala is a Python package for nD image segmentation. 
"""

import sys, logging
if sys.version_info[:2] < (2,6):
    logging.warning('Gala has not been tested on Python versions prior to 2.6'+
        ' (%d.%d detected).'%sys.version_info[:2])

__author__ = 'Juan Nunez-Iglesias <jni@janelia.hhmi.org>, '+\
             'Ryan Kennedy <kenry@cis.upenn.edu>'
del sys, logging

__all__ = ['agglo', 'morpho', 'evaluate', 'viz', 'imio', 'classify',
    'stack_np', 'app_logger', 'option_manager', 'features', 'filter']
