import sys
import os
sys.path.append('./')
#sys.path.append('../../../')

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




def test():
    # Minimum Setting for Testing feature Descriptor class
    #rootdir='./'
    rootdir='./test_input/'
    logfile = open('LOG', 'w', 10)
    inputs = initialize_inputs(rootdir+'input_SiO.yaml', logfile)

    print("Cleaning current directory")
    os.system('rm LOG total_list train_list valid_list pca scale_factor')

    try:
        print('_split_train_list_and_valid_list test')
        out_list = preprocessing._split_train_list_and_valid_list(inputs, data_list=rootdir+'total_list')
        print('_split_train_list_and_valid_list OK')
        print('total_list : ')
        os.system(f'cat {rootdir}total_list')
        print('train_list : ')
        os.system(f'cat ./train_list')
        print('valid_list : ')
        os.system(f'cat ./valid_list')
        print('')
    except:
        print(sys.exc_info())
        raise Exception('Error occured : _split_train_list_and_valid_list')

    try:
        print('_calculate_scale test')
        train_feature_list = util_ft._make_full_featurelist('./train_list', 'x', inputs['atom_types'], pickle_format=False)
        print('train_feature_list')
        print(train_feature_list) 
        scale = preprocessing._calculate_scale(inputs, logfile, train_feature_list)
        print(scale)
        print('_calculate_scale OK')
        print('')
    except:
        print(sys.exc_info())
        raise Exception('Error occured : _calculate_scale')

    try:
        print('_calculate_pca_matrix test')
        pca = preprocessing._calculate_pca_matrix(inputs, logfile, train_feature_list, scale)
        print(pca)
        print('_calculate_pca_matrix OK')
        print('')
    except:
        print(sys.exc_info())
        raise Exception('Error occured : _calculate_pca_matrix')

    try:
        print('preprocess() test')
        os.system(f'cp {rootdir}total_list ./')
        preprocessing.preprocess(inputs, logfile)
        os.system('cat LOG')
        print('preprocess() OK')
        os.system('rm LOG total_list train_list valid_list pca scale_factor')
        print('')
    except:
        print(sys.exc_info())
        raise('Error occured : preprocess()')




if __name__ == "__main__":
    test()
