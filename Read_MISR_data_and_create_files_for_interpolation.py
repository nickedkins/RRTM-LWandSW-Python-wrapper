import numpy as np
import pandas as pd
import xarray as xr
import pynio

# ds=xr.open_dataset('/Users/nickedkins/Dropbox/GitHub_Repositories/RRTM-LWandSW-Python-wrapper/MISR Data/MISR_AM1_CTH_1D_OD_2001_F02_0007.h4')

ds=xr.open_dataset('/Users/nickedkins/Dropbox/GitHub_Repositories/RRTM-LWandSW-Python-wrapper/MISR Data/MISR_AM1_CTH_1D_OD_2001_F02_0007.h4', engine='pynio')

print('done')