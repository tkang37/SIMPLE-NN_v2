import sys
import os
sys.path.append('../')

#DataGenerator.py -> datagenerator.py in utils
#DataGenerator Class -> Datagenerator


import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


#os.system('python libsymf_builder.py')

from simple_nn_v2.utils import _make_full_featurelist
from simple_nn_v2.utils.mpiclass import DummyMPI
from simple_nn_v2.init_inputs import  initialize_inputs
from simple_nn_v2.features.preprocessing import _split_data

# Setting for test preprocessing 

logfile = sys.stdout
logfile = open('LOG', 'w', 10)
inputs = initialize_inputs('input.yaml', logfile)
comm = DummyMPI()


try:
    print('_split_data test')
    print('CONTINUE   :  ', inputs['neural_network']['continue'])
    print('SHUFFLE    :  ', inputs['symmetry_function']['shuffle'])
    print('VALID_RATE :  ', inputs['symmetry_function']['valid_rate'])
    pickle_list = './pickle_list'
    print('NEED pickle_list file  : '+pickle_list)
    _split_data(inputs, pickle_list)
    print('_split_data OK')
    print('TMP_PICKLE_TRAIN  ',inputs['symmetry_function']['train_list'])
    os.system('cat {0}'.format(inputs['symmetry_function']['train_list']))
    print('TMP_PICKLE_VALID  ',inputs['symmetry_function']['valid_list'])
    os.system('cat {0}'.format(inputs['symmetry_function']['train_list']))
    print('')
except:
    print('!!  Error occured _split_data ')
    print(sys.exc_info())
    print('')

