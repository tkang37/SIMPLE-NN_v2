import sys
import os
sys.path.append('../../../')

#DataGenerator.py -> datagenerator.py in utils
#DataGenerator Class -> Datagenerator


import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


#os.system('python libsymf_builder.py')

from simple_nn_v2 import simple_nn
from simple_nn_v2.init_inputs import initialize_inputs
from simple_nn_v2.features.symmetry_function import generate
from simple_nn_v2.utils import features as util_ft
from simple_nn_v2.features import preprocessing

#os.system('rm -r ./data')

# Minimum Setting for Testing feature Descriptor class

logfile = open('LOG', 'w', 10)
inputs = initialize_inputs('./input.yaml', logfile)

try:
    print('generate() test before preprocess')
#    generate(inputs, logfile)
    print('generate() OK')
    print('')
except:
    print('!!  Error occured in generate() ')
    print(sys.exc_info())
    print('')

try:
    print('_split_train_list_and_valid_list test')
    preprocessing._split_train_list_and_valid_list(inputs, data_list='./total_list')
    print('_split_train_list_and_valid_list OK')
    os.system('cat ./train_list')
    print('')
except:
    print('!!  Error occured in _split_train_list_and_valid_list')
    print(sys.exc_info())
    print('')

    print('_calculate_scale test')

try:
    print('_calculate_scale test')
    train_feature_list = util_ft._make_full_featurelist('./train_list', 'x', inputs['atom_types'], pickle_format=False)
    print(train_feature_list.keys()) 

    scale = preprocessing._calculate_scale(inputs, logfile, train_feature_list)
    print('_calculate_scale OK')
    print(scale)
    print('')
except:
    print('!!  Error occured in _calculate_scale')
    print(sys.exc_info())
    print('')

try:
    print('_calculate_pca_matrix test')
    pca = preprocessing._calculate_pca_matrix(inputs, logfile, train_feature_list, scale)
    print(scale)
    print('_calculate_pca_matrix OK')
    print('')
except:
    print('!!  Error occured in _calculate_pca_matrix')
    print(sys.exc_info())
    print('')



try:
    print('preprocess() test')
    preprocessing.preprocess(inputs, logfile)
    print('preprocess() OK')
    print('')
except:
    print('!!  Error occured in preprocess() ')
    print('')
    print(sys.exc_info())


