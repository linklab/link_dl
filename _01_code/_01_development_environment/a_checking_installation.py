import numpy as np
import torch
import sys
import matplotlib
import sklearn
import pandas

print(f'Python version: {sys.version}')
print(f'Numpy version: {np.version.version}')
print(f'PyTorch version: {torch.version.__version__}')
print(f'Matplotlib version: {matplotlib.__version__}')
print(f'Pandas version is {pandas.__version__}')
print(f'Scikit-learn version: {sklearn.__version__}')
print(f'GPU present: {torch.cuda.is_available()}')
