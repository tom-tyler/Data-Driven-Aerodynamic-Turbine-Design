from sklearn.gaussian_process import GaussianProcessRegressor, kernels
from sklearn.metrics import mean_squared_error
import sklearn.preprocessing as pre
import numpy as np
import pandas as pd
import scipy.stats as st
import scipy.optimize as sciop
import matplotlib.pyplot as plt
import matplotlib.colors as mcol
from collections import OrderedDict
import compflow_native as compflow
import os
import sys

length_bounds=(1e-1,1e3)
noise_magnitude=1e-3
noise_bounds=(1e-8,1e-1)

noise_kernel = kernels.WhiteKernel(noise_level=noise_magnitude,
                                         noise_level_bounds=noise_bounds)

matern_kernel = kernels.Matern(length_scale = (1,1,1,1,1),
                    length_scale_bounds=((1e-2,1e2),(1e-2,1e3),(1e-2,1e2),(1e-2,1e2),(1e-2,1e2)),
                    nu=2.5
                    )

kernel_form = matern_kernel + noise_kernel