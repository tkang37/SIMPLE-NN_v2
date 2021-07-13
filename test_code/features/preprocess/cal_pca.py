import sys
import os
sys.path.append('./')

import torch
import numpy as np

from simple_nn_v2 import simple_nn
from simple_nn_v2.init_inputs import initialize_inputs
from simple_nn_v2.features.symmetry_function import generate
from simple_nn_v2.utils import features as util_ft
from simple_nn_v2.features import preprocessing

rootdir='./test_input/'
logfile = open('LOG', 'w', 10)
inputs = initialize_inputs(rootdir+'input_SiO.yaml', logfile)


print("Load pregenerateded feature list, scale") 
train_feature_list = torch.load(f'{rootdir}/feature_match')
scale = torch.load(f'{rootdir}/scale_match')
print('_calculate_pca_matrix test')

pca = preprocessing._calculate_pca_matrix(inputs, logfile, train_feature_list, scale)
print("pca generate done")
pca_match = torch.load(f"{rootdir}pca_match")
print("Checking generated pca match ")

if np.sum(pca['Si'][0]-pca_match['Si'][0]) < 1E-10:
    print("pca 1st component passed")
else:
    raise Exception(f"pca generated different value at 1st component sum : {np.sum(pca['Si'][0]-pca_match['Si'][0])}")

if np.sum(pca['Si'][1]-pca_match['Si'][1]) < 1E-10:
    print("pca 2nd component passed")
else:
    raise Exception(f"pca generated different value at 2nd component sum : {np.sum(pca['Si'][1]-pca_match['Si'][1])}")

if np.sum(pca['Si'][2]-pca_match['Si'][2]) < 1E-10:
    print(f"pca 3rd component passed")
else:
    raise Exception(f"pca generated different value at 3rd component sum : {np.sum(pca['Si'][2]-pca_match['Si'][2])} ")

print('_calculate_pca_matrix OK')
print('')

