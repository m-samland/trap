# -*- coding: utf-8 -*
"""
TRAP init file

@author: samland
"""

__version__ = "1.0.0"
__all__ = ['detection', 'image_coordinates', 'makesource',
           'parameters', 'pca_regression', 'plotting_tools', 'reduction_wrapper',
           'regression', 'template', 'regressor_selection']

from . import (detection, image_coordinates, makesource,
               pca_regression, plotting_tools, reduction_wrapper,
               template, regression, regressor_selection)
from .embed_shell import ipsh
from .utils import (crop_box_from_3D_cube, crop_box_from_4D_cube,
                    crop_box_from_image, derotate_cube,
                    determine_maximum_contrast_for_injection, prepare_psf,
                    resize_cube, resize_image_cube)
