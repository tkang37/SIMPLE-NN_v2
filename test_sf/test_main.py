# Test symmetry function value in OUTCAR or POSCAR to pickled file
# Need module pytorch , PyYaml , Ase , numpy
#
import numpy as np
# All functions in here
import test_sf 



# Load structure
structure = test_sf.open_outcar('OUTCAR_1')


#Generate Symmetry function using parameters
#index = 2 : G2  / 4 : G4 / 5 : G5 symmetry function
test_g2 = test_sf.generate_sf(index = 2 , param_d = [6.0 , 0.003214 , 0.0 , 0.0])  #[cutoff , eta , R_s , Mone]
test_g4 = test_sf.generate_sf(index = 4 , param_d = [6.0 , 0.089277 , 4.0 , 1.0])  #[cutoff , eta , zeta , lambda]


#Get distance information from OUTCAR & POSCAR 
#Distance_atoms use adjust & get distance from structure
distance = test_sf.Distance_atoms(structure)
#Set cufoff radius 
distance.set_cutoff(6.0)



### Validation ###
cal_list = list() 
print('_'*60)
tmp = 0
for i in range(33,73):  ## For Te in OUTCAR_1
    tmp = distance.get_g2_distance(i,1)  #  get_g2_dist(atom number , atom type)
    cal_list.append(test_g2(tmp))  
print('Testing G2 symmetry function')
print('Calculated SF :',np.array(cal_list))
pickle1 = test_sf.load_pickle('data1.pickle')   #From picked data
print('Pickled data  :',pickle1['x']['Te'][:,0])
print('')
### G2 SF validation OK ###


### G4 SF validation ###
cal_list = list()
print('_'*60)
tmp = 0
for i in range(33,73):  ## For Te in OUTCAR_1
    tmp =  distance.get_g4_distance(i,3,3)
    cal_list.append(test_g4(tmp))  # get_g4_dist(atom number , atom type_1 , atom type_2)
print('Testing G4 symmetry function')
print('Calculated SF :',np.array(cal_list))
pickle1 = test_sf.load_pickle('data1.pickle')
print('Pickled data  :',pickle1['x']['Te'][:,131])
### G4 SF validation OK ###


## Use Class to test symmetry function ##

## load data of OUTCAR & yaml & pickled data
load = test_sf.Test_symmetry_function(output_name = 'OUTCAR_1' , yaml_name = 'input.yaml' , data_name = 'data1.pt')

## Can load data seperatly
#load.set_structure(atoms_name = 'OUTCAR_1')
#load.set_yaml(yaml_name = 'inpuy.yaml')
#load.set_data(data_name = 'data1.pickle')

## Also load param_XX seperately
#load.set_params(atom = 'Te', params_dir = 'params_Te')

##Read infomation of loaded data
#load.show_pickled_info()
#load.show_atom_info()


## Calculating symmetry function ##

## Calculate symmetry funvtion of : (atom_type , atom_number , line of symmetry function parameters)

cal = load.calculate_sf(atom = 'Te' , number = 4 , line = 122) # 122th symmetry funtion of params_Te (4th atom of Te)
pic = load.get_sf_from_data(atom = 'Te' , number = 4 , line = 122) ## Same format 

print('Calculated data  :',cal)
print('Pickled    data  :',pic)

# Get all symmetry function of specific atom
cal = load.calculate_sf_by_atom(atom = 'Te' , number = 2)  # Calculate all SF for 7th atom of Te
pic = load.get_sf_from_data_by_atom(atom = 'Te' , number = 2) #same format

print('')
print('Calculated data  :',cal)
print('Pickled    data  :',pic)


# Get value of specific symmetry function of all atom
cal = load.calculate_sf_by_line(atom = 'Sb' , line = 6)  # Calculate 111th SF in param_Sb for all atom of Sb
pic = load.get_sf_from_data_by_line(atom = 'Sb' , line = 6) #same format

print('')
print('Calculated data  :',cal)
print('Pickled    data  :',pic)
