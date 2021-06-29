import sys
sys.path.append('../../../../../')

from simple_nn_v2 import simple_nn
from simple_nn_v2.init_inputs import initialize_inputs
from simple_nn_v2.features.symmetry_function import generating

# Minimum Setting for Testing Symmetry_function methods
# Initialize input file, set Simple_nn object as parent of Symmetry_function object

logfile = open('LOG', 'w', 10)
inputs = initialize_inputs('./input.yaml', logfile)
atom_types = inputs['atom_types']

""" Previous setting before test code

    1. load structure from FILE

"""
from ase import io
########## Set This Variable ###########
FILE = '../../../../test_data/SiO2/OUTCAR_comp'
########################################
structures = io.read(FILE, index='::', format='vasp-out', force_consistent=True)
structure = structures[0]

""" Main test code

    Test _get_structure_info()
    1. check if "atom_num" is total atom number
    2. check "type_num" has correct element types and atom number for each elements
    3. check if atom_type_idx has correct values
    4. check "type_atom_idx" has correct atom index 
    5. check lattice parameter
    6. check Cartesian coordination in "cart" (first 5, last 5 coordination)
    7. check Fractional coordination in "scale" (first 5, last 5 coordination)

"""
import numpy as np

atom_num, atom_type_idx, atoms_per_type, atom_idx_per_type = generating._get_atom_types_info(structure, atom_types)

print('1. check if "atom_num" is total atom number')
print('atom num: %s\n'%atom_num)

print('2. check if "atom_type_idx" is total atom number')
print('atom_type_idx: %s\n'%atom_type_idx)

print('3. check "atoms_per_type" has correct element types and atom number for each elements')
print('atoms_per_type: %s\n'%atoms_per_type)

print('4. check if "atom_idx_per_type" is total atom number')
print('atom_idx_per_type: %s\n'%atom_idx_per_type)

