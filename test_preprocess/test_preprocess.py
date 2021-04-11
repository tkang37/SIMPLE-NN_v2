import sys
import os
sys.path.append('../')

#DataGenerator.py -> datagenerator.py in utils
#DataGenerator Class -> Datagenerator


import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


#os.system('python libsymf_builder.py')

from simple_nn_v2.utils import _make_full_featurelist
from simple_nn_v2.init_inputs import  initialize_inputs
from simple_nn_v2.features.preprocessing import preprocess

# Setting for test preprocessing 

logfile = sys.stdout
logfile = open('LOG', 'w', 10)
inputs = initialize_inputs('input.yaml', logfile)


try:
    print('preprocess & _dump_all test')
    preprocess(inputs, logfile, calc_scale=True, get_atomic_weights=None)
    print('preprocess & _dump_all OK')
    print('')
except:
    print('!!  Error occured preprocess & _dump_all')
    print(sys.exc_info())
    print('')

