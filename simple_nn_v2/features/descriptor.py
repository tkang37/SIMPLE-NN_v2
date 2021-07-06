## Before the training, pickle data are changed into .tfrecord format. (Later .pt format)
## This class conatin Scale Factor, PCA, GDF, weight calculations
##        
##
##          :param boolean calc_scale: flag to calculate the scale_factor
##          :param boolean use_force: flag to put 'force data' into input
##          :param boolean use_stress: flag to put 'stress data' into the input
##          :param boolean get_atomic_weights: flag to return atomic_weight 
##
##

import os, sys
import tensorflow as tf
import torch

import numpy as np
import six
from six.moves import cPickle as pickle

from ..utils import _generate_scale_file, _make_full_featurelist, _make_data_list, \
                    _make_str_data_list, pickle_load
from ..utils import graph as grp
from ..utils.mpiclass import DummyMPI, MPI4PY

from sklearn.decomposition import PCA

class Descriptor(object):

    def __init__(self):
        pass ##No specific delartion
        
    def preprocess(self, calc_scale=True, use_force=False, use_stress=False, get_atomic_weights=None, **kwargs):
        """
        Before the training, pickle data are changed into .tfrecord format.
        scale_factor, pca, gdf are also calculated here.

        :param boolean calc_scale: flag to calculate the scale_factor
        :param boolean use_force: flag to put 'force data' into input
        :param boolean use_stress: flag to put 'stress data' into the input
        :param boolean get_atomic_weights: flag to return atomic_weight
        """ 
        #Extract save directory information from save_dir
        save_dir = self.parent.descriptor.inputs['save_directory']

        comm = self.get_comm( )
        
        #Split Test, Valid data of str_list
        tmp_pickle_train, tmp_pickle_valid = self._split_data(comm)        
         
        comm.barrier()

        # generate full symmetry function vector
        feature_list_train, idx_list_train = \
            _make_full_featurelist(tmp_pickle_train, 'x', self.parent.inputs['atom_types'], is_ptdata= not self.parent.descriptor.inputs['save_to_pickle'] )
        feature_list_valid, idx_list_valid = \
            _make_full_featurelist(tmp_pickle_valid, 'x', self.parent.inputs['atom_types'], is_ptdata= not self.parent.descriptor.inputs['save_to_pickle'] )

        
        # calculate scale
        scale = self._calc_scale(calc_scale, feature_list_train, comm)
        
        # Fit PCA.
        self._generate_pca(feature_list_train, scale)
        
        # calculate gdf
        aw_tag, atomic_weights_train, atomic_weights_valid = self._calculate_gdf(\
        get_atomic_weights, feature_list_train, idx_list_train, feature_list_valid, idx_list_valid, comm)
            
        #Make tfrecord file from training
        self._train_all(save_dir,tmp_pickle_train, atomic_weights_train, tmp_pickle_valid\
        , atomic_weights_valid, aw_tag, comm, is_tfrecord=True)
        #END of preprocess

        # pickle list -> train / valid
        # This function split data into train, valid set
    def _split_data(self, comm):
        tmp_pickle_train = './pickle_train_list'
        tmp_pickle_valid = './pickle_valid_list'
                
        if comm.rank == 0:
            if not self.inputs['continue']:
                tmp_pickle_train_open = open(tmp_pickle_train, 'w')
                tmp_pickle_valid_open = open(tmp_pickle_valid, 'w')
                for file_list in _make_str_data_list(self.pickle_list):
                    if self.inputs['shuffle']:
                        np.random.shuffle(file_list)
                    num_pickle = len(file_list)
                    num_valid = int(num_pickle * self.inputs['valid_rate'])

                    for i,item in enumerate(file_list):
                        if i < num_valid:
                            tmp_pickle_valid_open.write(item + '\n')
                        else:
                            tmp_pickle_train_open.write(item + '\n')
        
                tmp_pickle_train_open.close()
                tmp_pickle_valid_open.close()

        return tmp_pickle_train, tmp_pickle_valid
    
        #This function calculate scale factor as pickle
    def _calc_scale(self,calc_scale, feature_list_train, comm):
        scale = None
        params_set = dict()
        for item in self.parent.inputs['atom_types']:
            params_set[item] = dict()
            params_set[item]['i'], params_set[item]['d'] = self._read_params(self.inputs['params'][item])
        if calc_scale:
            scale = _generate_scale_file(feature_list_train, self.parent.inputs['atom_types'], 
                                         scale_type=self.inputs['scale_type'],
                                         scale_scale=self.inputs['scale_scale'],
                                         scale_rho=self.inputs['scale_rho'],
                                         params=params_set,
                                         log=self.parent.logfile,
                                         comm=comm)
        else:
            scale = pickle_load('./scale_factor')
        return scale 

        #This function to generate PCA data
    def _generate_pca(self, feature_list_train, scale):
        if self.parent.model.inputs['pca']:
            pca = {}
            for item in self.parent.inputs['atom_types']:
                pca_temp = PCA()
                pca_temp.fit((feature_list_train[item] - scale[item][0:1,:]) / scale[item][1:2,:])
                min_level = self.parent.model.inputs['pca_min_whiten_level']
                # PCA transformation = x * pca[0] - pca[2] (divide by pca[1] if whiten)
                pca[item] = [pca_temp.components_.T,
                             np.sqrt(pca_temp.explained_variance_ + min_level),
                             np.dot(pca_temp.mean_, pca_temp.components_.T)]
            with open("./pca", "wb") as fil:
                pickle.dump(pca, fil, protocol=2)

        #This function for calculate GDF        
    def _calculate_gdf(self, get_atomic_weights, feature_list_train, idx_list_train, featrue_list_valid, idx_list_valid, comm):
        #Define outputs
        atomic_weights_train = atomic_weights_valid = None
        
        #Check get_atomic_weigts exist
        if callable(get_atomic_weights):
            # FIXME: use mpi
            local_target_list = dict()
            local_idx_list = dict()
            #feature_list_train part
            for item in self.parent.inputs['atom_types']:
                q = feature_list_train[item].shape[0] // comm.size
                r = feature_list_train[item].shape[0]  % comm.size

                begin = comm.rank * q + min(comm.rank, r)
                end = begin + q
                if r > comm.rank:
                    end += 1

                local_target_list[item] = feature_list_train[item][begin:end]
                local_idx_list[item] = idx_list_train[item][begin:end]

            atomic_weights_train, dict_sigma, dict_c = get_atomic_weights(feature_list_train, scale, self.parent.inputs['atom_types'], local_idx_list, 
                                                                          target_list=local_target_list, filename='atomic_weights', comm=comm, **kwargs)
            kwargs.pop('sigma')

            if comm.rank == 0:
                self.parent.logfile.write('Selected(or generated) sigma and c\n')
                for item in self.parent.inputs['atom_types']:
                    self.parent.logfile.write('{:3}: sigma = {:4.3f}, c = {:4.3f}\n'.format(item, dict_sigma[item], dict_c[item]))

            local_target_list = dict()
            local_idx_list = dict()
            #feature_list_valid part
            for item in self.parent.inputs['atom_types']:
                q = feature_list_valid[item].shape[0] // comm.size
                r = feature_list_valid[item].shape[0]  % comm.size

                begin = comm.rank * q + min(comm.rank, r)
                end = begin + q
                if r > comm.rank:
                    end += 1

                local_target_list[item] = feature_list_valid[item][begin:end]
                local_idx_list[item] = idx_list_valid[item][begin:end]

            atomic_weights_valid,          _,        _ = get_atomic_weights(feature_list_train, scale, self.parent.inputs['atom_types'], local_idx_list, 
                                                                          target_list=local_target_list, sigma=dict_sigma, comm=comm, **kwargs)
        #Check get_atomic_weights is six.string_types
        elif isinstance(get_atomic_weights, six.string_types):
            atomic_weights_train = pickle_load(get_atomic_weights)
            atomic_weights_valid = 'ones'

        if atomic_weights_train is None:
            aw_tag = False
        else:
            aw_tag = True
            #grp.plot_gdfinv_density(atomic_weights_train, self.parent.inputs['atom_types'])
            # Plot histogram only if atomic weights just have been calculated.
            if comm.rank == 0 and callable(get_atomic_weights):
                grp.plot_gdfinv_density(atomic_weights_train, self.parent.inputs['atom_types'], auto_c=dict_c)
        return aw_tag, atomic_weights_train, atomic_weights_valid

        #This function train data and create .tfrecord file is_tfrecord input is for debugging
    def _train_all(self, save_dir, tmp_pickle_train, atomic_weights_train, tmp_pickle_valid, atomic_weights_valid, aw_tag, comm, is_tfrecord=True):
        # Start of training, validation
        if comm.rank == 0: 
            #Create save directory
            if not os.path.isdir(save_dir):
                os.mkdir(save_dir)

            tmp_pickle_train_list = _make_data_list(tmp_pickle_train)
            #np.random.shuffle(tmp_pickle_train_list)
            num_tmp_pickle_train = len(tmp_pickle_train_list)
            num_tfrecord_train = int(num_tmp_pickle_train / self.inputs['data_per_tfrecord'])
            train_list = open(self.train_data_list, 'w')
 
            random_idx = np.arange(num_tmp_pickle_train)        
            #if not self.inputs['continue']:
            if self.inputs['shuffle']:
                np.random.shuffle(random_idx)
            
            for i,item in enumerate(random_idx):
                ptem = tmp_pickle_train_list[item]
                if is_tfrecord: #Save to .tfrecord file
                    if i == 0:
                        record_name = save_dir+'/training_data_{:0>4}_to_{:0>4}.tfrecord'.format(int(i/self.inputs['data_per_tfrecord']), num_tfrecord_train)
                        writer = tf.python_io.TFRecordWriter(record_name)
                    elif (i % self.inputs['data_per_tfrecord']) == 0:
                        writer.close()
                        self.parent.logfile.write('{} file saved in {}\n'.format(self.inputs['data_per_tfrecord'], record_name))
                        train_list.write(record_name + '\n')
                        record_name = save_dir+'/training_data_{:0>4}_to_{:0>4}.tfrecord'.format(int(i/self.inputs['data_per_tfrecord']), num_tfrecord_train)
                        writer = tf.python_io.TFRecordWriter(record_name)
                else: #Save to pytorch file
                    #TODO: make for torch version to make trrecord file
                    pass
                
                #Check data was saved to .pt file format or .pickle file format
                if self.parent.descriptor.inputs['save_to_pickle']:
                    tmp_res = pickle_load(ptem)
                else:
                    tmp_res = torch.load(ptem)
                    
                tmp_res['pickle_name'] = ptem
                #Check atomic_weights_train exists
                if atomic_weights_train is not None:
                    tmp_aw = dict()
                    for jtem in self.parent.inputs['atom_types']:
                        tmp_idx = (atomic_weights_train[jtem][:,1] == item)
                        tmp_aw[jtem] = atomic_weights_train[jtem][tmp_idx,0]
                    #tmp_aw = np.concatenate(tmp_aw)
                    tmp_res['atomic_weights'] = tmp_aw
                
                #write tfrecord format
                if is_tfrecord:
                    self._write_tfrecords(tmp_res, writer, atomic_weights=aw_tag)
                else:
                    #TODO: make torch version of  _write_tfrecords 
                    pass
 
                if not self.inputs['remain_pickle']:
                    os.remove(ptem)
 
            writer.close()
            self.parent.logfile.write('{} file saved in {}\n'.format((i%self.inputs['data_per_tfrecord'])+1, record_name))
            train_list.write(record_name + '\n')
            train_list.close()
            
            
            #Start validation part
            if self.inputs['valid_rate'] != 0.0:
                # valid
                tmp_pickle_valid_list = _make_data_list(tmp_pickle_valid)
                num_tmp_pickle_valid = len(tmp_pickle_valid_list)
                num_tfrecord_valid = int(num_tmp_pickle_valid / self.inputs['data_per_tfrecord'])
                valid_list = open(self.valid_data_list, 'w')
 
                random_idx = np.arange(num_tmp_pickle_valid)        
                #if not self.inputs['continue']:
                if self.inputs['shuffle']:
                    np.random.shuffle(random_idx)
 
                for i,item in enumerate(random_idx):
                    ptem = tmp_pickle_valid_list[item]
                    if is_tfrecord: #tfrecord version
                        if i == 0:
                            record_name = save_dir+'/valid_data_{:0>4}_to_{:0>4}.tfrecord'.format(int(i/self.inputs['data_per_tfrecord']), num_tfrecord_valid)
                            writer = tf.python_io.TFRecordWriter(record_name)
                        elif (i % self.inputs['data_per_tfrecord']) == 0:
                            writer.close()
                            self.parent.logfile.write('{} file saved in {}\n'.format(self.inputs['data_per_tfrecord'], record_name))
                            valid_list.write(record_name + '\n')
                            record_name = save_dir+'/valid_data_{:0>4}_to_{:0>4}.tfrecord'.format(int(i/self.inputs['data_per_tfrecord']), num_tfrecord_valid)
                            writer = tf.python_io.TFRecordWriter(record_name)
                    else: #pytorch version
                        #TODO: make torch version to save tfrecord
                        pass

                    if self.parent.descriptor.inputs['save_to_pickle']:
                        tmp_res = torch.load(ptem)
                    else:
                        tmp_res = pickle_load(ptem)
                    tmp_res['pickle_name'] = ptem
                    
                    #Check atomic_weights_valid 
                    if atomic_weights_valid == 'ones':
                        tmp_aw = dict()
                        for jtem in self.parent.inputs['atom_types']:
                            tmp_aw[jtem] = np.ones([tmp_res['N'][jtem]]).astype(np.float64)
                        tmp_res['atomic_weights'] = tmp_aw
                    elif atomic_weights_valid is not None:
                        tmp_aw = dict()
                        for jtem in self.parent.inputs['atom_types']:
                            tmp_idx = (atomic_weights_valid[jtem][:,1] == item)
                            tmp_aw[jtem] = atomic_weights_valid[jtem][tmp_idx,0]
                        #tmp_aw = np.concatenate(tmp_aw)
                        tmp_res['atomic_weights'] = tmp_aw
 
                    if is_tfrecord:
                        self._write_tfrecords(tmp_res, writer, atomic_weights=aw_tag)
                    else:
                        #TODO: Make torch version of _write_tfrecords function
                        pass
 
                    if not self.inputs['remain_pickle']:
                        os.remove(ptem)
 
                writer.close()
                self.parent.logfile.write('{} file saved in {}\n'.format((i%self.inputs['data_per_tfrecord'])+1, record_name))
                valid_list.write(record_name + '\n')
                valid_list.close()





    def _write_tfrecords(self, res, writer, atomic_weights=False):
        # TODO: after stabilize overall tfrecor related part,
        # this part will replace the part of original 'res' dict
         
        def _bytes_feature(value):
            return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
        
        def _int64_feature(value):
            return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

        def _gen_1dsparse(arr):
            non_zero = (arr != 0)
            return np.arange(arr.shape[0])[non_zero].astype(np.int32), arr[non_zero], np.array(arr.shape).astype(np.int32)
        
        feature = {
            'E':_bytes_feature(np.array([res['E']]).tobytes()),
            'tot_num':_bytes_feature(res['tot_num'].astype(np.float64).tobytes()),
            'partition':_bytes_feature(res['partition'].tobytes()),
            'struct_type':_bytes_feature(six.b(res['struct_type'])),
            'struct_weight':_bytes_feature(np.array([res['struct_weight']]).tobytes()),
            'pickle_name':_bytes_feature(six.b(res['pickle_name'])),
            'atom_idx':_bytes_feature(res['atom_idx'].tobytes())
        }
    
        try:
            feature['F'] = _bytes_feature(res['F'].tobytes())
        except:
            pass

        try:
            feature['S'] = _bytes_feature(res['S'].tobytes())
        except:
            pass

        for item in self.parent.inputs['atom_types']:
            feature['x_'+item] = _bytes_feature(res['x'][item].tobytes())
            feature['N_'+item] = _bytes_feature(res['N'][item].tobytes())
            feature['params_'+item] = _bytes_feature(res['params'][item].tobytes())

            dx_indices, dx_values, dx_dense_shape = _gen_1dsparse(res['dx'][item].reshape([-1]))
        
            feature['dx_indices_'+item] = _bytes_feature(dx_indices.tobytes())
            feature['dx_values_'+item] = _bytes_feature(dx_values.tobytes())
            feature['dx_dense_shape_'+item] = _bytes_feature(dx_dense_shape.tobytes())

            feature['da_'+item] = _bytes_feature(res['da'][item].tobytes())

            feature['partition_'+item] = _bytes_feature(res['partition_'+item].tobytes())

            if atomic_weights:
                feature['atomic_weights_'+item] = _bytes_feature(res['atomic_weights'][item].tobytes())

            if self.inputs['add_NNP_ref']:
                feature['NNP_E_'+item] = _bytes_feature(res['NNP_E'][item].tobytes())

        example = tf.train.Example(
            features=tf.train.Features(
                feature=feature
            )
        )
        
        writer.write(example.SerializeToString())



    def _tfrecord_input_fn(self, filename_queue, inp_size, batch_size=1, use_force=False, use_stress=False, valid=False, cache=False, full_batch=False, atomic_weights=False):
        dataset = tf.data.TFRecordDataset(filename_queue)
        #dataset = dataset.cache() # for test
        dataset = dataset.map(lambda x: self._parse_data(x, inp_size, use_force=use_force, use_stress=use_stress, atomic_weights=atomic_weights), 
                              num_parallel_calls=self.inputs['num_parallel_calls'])
        if cache:
            dataset = dataset.take(-1).cache() # for test

        batch_dict = dict()
        batch_dict['E'] = [None]
        batch_dict['tot_num'] = [None]
        batch_dict['partition'] = [None]
        batch_dict['struct_type'] = [None]
        batch_dict['struct_weight'] = [None]
        batch_dict['pickle_name'] = [None]
        batch_dict['atom_idx'] = [None, 1]

        if use_force:
            batch_dict['F'] = [None, 3]

        if use_stress:
            batch_dict['S'] = [None]

        for item in self.parent.inputs['atom_types']:
            batch_dict['x_'+item] = [None, inp_size[item]]
            batch_dict['N_'+item] = [None]
            if self.inputs['add_NNP_ref']:
                batch_dict['NNP_E_'+item] = [None, 1]
            if use_force:
                batch_dict['dx_'+item] = [None, inp_size[item], None, 3]
            if use_stress:
                batch_dict['da_'+item] = [None, inp_size[item], 3, 6]
            batch_dict['partition_'+item] = [None]
            if atomic_weights:
                batch_dict['atomic_weights_'+item] = [None]
 
        if valid or full_batch:
            dataset = dataset.padded_batch(batch_size, batch_dict)
            dataset = dataset.prefetch(buffer_size=1)
            #dataset = dataset.cache()
            iterator = dataset.make_initializable_iterator()
        else:
            dataset = dataset.apply(tf.contrib.data.shuffle_and_repeat(200, None))
            dataset = dataset.padded_batch(batch_size, batch_dict)
            # prefetch test
            dataset = dataset.prefetch(buffer_size=1)
            iterator = dataset.make_initializable_iterator()
            
        return iterator  

    def _parse_data(self, serialized, inp_size, use_force=False, use_stress=False, atomic_weights=False):
        features = {
            'E': tf.FixedLenFeature([], dtype=tf.string),
            'tot_num': tf.FixedLenFeature([], dtype=tf.string),
            'partition': tf.FixedLenFeature([], dtype=tf.string),
            'struct_type': tf.FixedLenSequenceFeature([], dtype=tf.string, allow_missing=True),
            'struct_weight': tf.FixedLenFeature([], dtype=tf.string, default_value=np.array([1.0]).tobytes()),
            'pickle_name': tf.FixedLenSequenceFeature([], dtype=tf.string, allow_missing=True),
            'atom_idx': tf.FixedLenFeature([], dtype=tf.string)
        }
 
        for item in self.parent.inputs['atom_types']:
            features['x_'+item] = tf.FixedLenFeature([], dtype=tf.string)
            features['N_'+item] = tf.FixedLenFeature([], dtype=tf.string)
            features['params_'+item] = tf.FixedLenFeature([], dtype=tf.string)
            if use_force:
                features['dx_indices_'+item] = tf.FixedLenFeature([], dtype=tf.string)
                features['dx_values_'+item] = tf.FixedLenFeature([], dtype=tf.string)
                features['dx_dense_shape_'+item] = tf.FixedLenFeature([], dtype=tf.string)
            if use_stress:
                features['da_'+item] = tf.FixedLenFeature([], dtype=tf.string)
            features['partition_'+item] = tf.FixedLenFeature([], dtype=tf.string)
            if atomic_weights:
                features['atomic_weights_'+item] = tf.FixedLenFeature([], dtype=tf.string)

            if self.inputs['add_NNP_ref']:
                features['NNP_E_'+item] = tf.FixedLenFeature([], dtype=tf.string)

        if use_force:
            features['F'] = tf.FixedLenFeature([], dtype=tf.string)

        if use_stress:
            features['S'] = tf.FixedLenFeature([], dtype=tf.string)

        read_data = tf.parse_single_example(serialized=serialized, features=features)
       
 
        res = dict()
 
        res['E'] = tf.decode_raw(read_data['E'], tf.float64)
        res['tot_num'] = tf.decode_raw(read_data['tot_num'], tf.float64)
        res['partition'] = tf.decode_raw(read_data['partition'], tf.int32)
        res['struct_type'] = read_data['struct_type']
        res['struct_weight'] = tf.decode_raw(read_data['struct_weight'], tf.float64)
        res['pickle_name'] = read_data['pickle_name']
        # For backward compatibility...
        res['struct_type'] = tf.cond(tf.equal(tf.shape(res['struct_type'])[0], 0),
                                     lambda: tf.constant(['None']),
                                     lambda: res['struct_type'])
        res['atom_idx'] = tf.reshape(tf.decode_raw(read_data['atom_idx'], tf.int32), [-1, 1])

        for item in self.parent.inputs['atom_types']:
            res['N_'+item] = tf.decode_raw(read_data['N_'+item], tf.int64)

            res['x_'+item] = tf.cond(tf.equal(res['N_'+item][0], 0),
                                     lambda: tf.zeros([0, inp_size[item]], dtype=tf.float64),
                                     lambda: tf.reshape(tf.decode_raw(read_data['x_'+item], tf.float64), [-1, inp_size[item]]))

            if self.inputs['add_NNP_ref']:
                res['NNP_E_'+item] = tf.cond(tf.equal(res['N_'+item][0], 0),
                                         lambda: tf.zeros([0, 1], dtype=tf.float64),
                                         lambda: tf.reshape(tf.decode_raw(read_data['NNP_E_'+item], tf.float64), [-1, 1]))

            if use_force:
                res['dx_'+item] = tf.cond(tf.equal(res['N_'+item][0], 0),
                                          lambda: tf.zeros([0, inp_size[item], 0, 3], dtype=tf.float64),
                                          lambda: tf.reshape(
                                            tf.sparse_to_dense(
                                                sparse_indices=tf.decode_raw(read_data['dx_indices_'+item], tf.int32),
                                                output_shape=tf.decode_raw(read_data['dx_dense_shape_'+item], tf.int32),
                                                sparse_values=tf.decode_raw(read_data['dx_values_'+item], tf.float64)), 
                                            [tf.shape(res['x_'+item])[0], inp_size[item], -1, 3]))

            if use_stress:
                res['da_'+item] = tf.cond(tf.equal(res['N_'+item][0], 0),
                                          lambda: tf.zeros([0, inp_size[item], 3, 6], dtype=tf.float64),
                                          lambda: tf.reshape(tf.decode_raw(read_data['da_'+item], tf.float64), [-1, inp_size[item], 3, 6]))
 
            res['partition_'+item] = tf.decode_raw(read_data['partition_'+item], tf.int32)

            if atomic_weights:
                res['atomic_weights_'+item] = tf.decode_raw(read_data['atomic_weights_'+item], tf.float64)

        if use_force: 
            res['F'] = tf.reshape(tf.decode_raw(read_data['F'], tf.float64), [-1, 3])
 
        if use_stress:
            res['S'] = tf.decode_raw(read_data['S'], tf.float64)
        
        return res

    def _read_params(self, filename):
        params_i = list()
        params_d = list()
        with open(filename, 'r') as fil:
            for line in fil:
                tmp = line.split()
                params_i += [list(map(int, tmp[:3]))]
                params_d += [list(map(float, tmp[3:]))]

        params_i = np.asarray(params_i, dtype=np.intc, order='C')
        params_d = np.asarray(params_d, dtype=np.float64, order='C')

        return params_i, params_d

        #Get mpi module from mpi4py
    def get_comm(self, dummy=False):
        if self.comm is None:
            try:
                import mpi4py
            except ImportError:
                self.comm = DummyMPI()
            else:
                self.comm = MPI4PY()
        if dummy: self.comm = DummyMPI()
        return self.comm


