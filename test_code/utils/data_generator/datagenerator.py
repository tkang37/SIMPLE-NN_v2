import sys
import os
sys.path.append('../../../')

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

#os.system('python libsymf_builder.py')

from simple_nn_v2 import simple_nn
from simple_nn_v2.init_inputs import initialize_inputs
from simple_nn_v2.features.symmetry_function import generating
from simple_nn_v2.utils import data_generator

os.system('rm -r ./data')

# Minimum Setting for Testing Symmetry_function methods
# Initialize input file, set Simple_nn object as parent of Symmetry_function object

logfile = open('LOG', 'w', 10)
inputs = initialize_inputs('./input.yaml', logfile)
atom_types = inputs['atom_types']
structure_list = './structure_list'

try:
    print('Checking parameters to use')
    print('INPUTS')
    for key in inputs.keys():
        print('{0}    :'.format(key), inputs[key])
    print('')
    print('Checking parameters DONE')
    print('')
except:
    print('!!! Error occured during checking parameters')

try:
    print('parse_structure_list() test')
    structure_tags, structure_weights, structure_file_list, structure_slicing_list, structure_tag_idx = data_generator.parse_structure_list(logfile, structure_list)
    print('structure_tags   :', structure_tags)
    print('structure_weights   :', structure_weights)
    print('structure_file_list   :', structure_file_list)
    print('structure_slicing_list   :', structure_slicing_list)
    print('structure_tag_idx   :', structure_tag_idx)
    print('parse_structure_list() OK')
    print('')
except:
    print('!!  Error occured in parse_strucrue_list()')
    print('')



try:
    print('_get_tag_and_weight() test')
    test_text = ['OUTCAR1','OUTCAR2:3','OUTCAR3 : 3']
    for text in test_text:
        print('Main text   :   ' + text)
        tag , weight = data_generator._get_tag_and_weight(text)
        print('Tag         :  ' + tag)
        print('Weight      :  ', weight)
        print('')
    print('_get_tag_and_weight() OK')
    print('')
except:
    print('!!  Error occured in _get_tang_and_weight()')
    print('')


try:
    print('load_snapshots() test')
    test_file_list = ['../../test_data/GST/OUTCAR_1', '../../test_data/GST/OUTCAR_2', '../../test_data/GST/OUTCAR_3']
    test_slicing_list = ['::3', '::', '::']
    ### utils.compress_outcar NOT working !!!
    ### ase.__version__ : '3.21.1' 
    ### ase.io.PareseError : Did not find required key "species" in parsed header result
    inputs['descriptor']['compress_outcar'] =  False
    if inputs['descriptor']['compress_outcar']:
        print("Generating ./tmp_comp_OUTCAR")
    for test_file, test_slicing in zip(test_file_list, test_slicing_list):
        print('Main structure  :   ',test_file)
        structures = data_generator.load_structures(inputs, test_file, test_slicing, logfile)
        print('Structures        :  ',structures)
        print('')
    print('load_structures() OK')
    print('')
except:
    print('!!  Error occured in load_structures()')
    print('')


try:
    print('save_to_datafile() test')
    data = {'TMP': 2, 'DATA': 5}
    data_idx = 1
    tag_idx = 1
    print('Test dictrionary data ', data)
    print('Tag_index ', tag_idx)
    tmp_filename = data_generator.save_to_datafile(inputs, data, data_idx, logfile)
    print('Tmp_filename    :', tmp_filename)
    os.system('rm -r ./data')
    print('save_to_pickle() OK')
    print('')
except:
    print('!! Error occured in save_to_pickle()')
    print('')

