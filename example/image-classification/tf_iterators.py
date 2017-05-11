import mxnet as mx
import random
from mxnet.io import DataBatch, DataIter
import numpy as np
import os


def ilab_iterator(batch_size):
    # data_shape = input_shape
    train = mx.io.ImageRecordIter(
        path_imglist= "/efs/datasets/users/furlat/ilab/ilab_train_sh.lst",  # you have to specify path_imglist when label_width larger than 2.
        path_imgrec = "/efs/datasets/users/furlat/ilab/ilab_train.rec",
        # mean_img    = "/efs/datasets/users/furlat/ilab_train_mean.bin", # it is OK no such file exists before train, the mxnet will generated it if not exists
        data_shape  = (3, 224, 224),
        batch_size  = batch_size,
        rand_crop   = True,
        rand_mirror = False,
        label_width = 4, 
        preprocess_threads  = 2,
        shuffle             = True,
        # specify label_width = 2 here
        )

    val = mx.io.ImageRecordIter(
       path_imglist= "/efs/datasets/users/furlat/ilab/ilab_test_sh.lst",  # you have to specify path_imglist when label_width larger than 2.
        path_imgrec = "/efs/datasets/users/furlat/ilab/ilab_test.rec",
        # mean_img    = "/efs/datasets/users/furlat/ilab_test_mean.bin", # it is OK no such file exists before train, the mxnet will generated it if not exists
        data_shape  = (3, 224, 224),
        batch_size  = batch_size,
        rand_crop   = False,
        rand_mirror = False,
        label_width = 4, 
        preprocess_threads  = 2,
        shuffle             = True,
        # specify label_width = 2 here
        )

    return (train, val)


class Multi_ilab_iterator(mx.io.DataIter):
    '''multi label ilab iterator'''

    def __init__(self, data_iter,subset):
        super(Multi_ilab_iterator, self).__init__()
        self.data_iter = data_iter
        self.batch_size = self.data_iter.batch_size
        self.labelIdx=subset

    @property
    def provide_data(self):
        return self.data_iter.provide_data

    @property
    def provide_label(self):
        provide_label = self.data_iter.provide_label
        label_names=[]
        batch_size=[]
        for i in range(len(self.labelIdx)):
            label_names.append('softmax%d_label'%(i+1))
            batch_size.append((self.batch_size,))
        return zip(label_names,batch_size)  
           
         #provide_label must have an output like this       
        #return [('softmax1_label', (self.batch_size,)), \
         #       ('softmax2_label', (self.batch_size,)), \
                #('softmax4_label', (self.batch_size,)), \
                #('softmax3_label', (self.batch_size,))]

    def hard_reset(self):
        self.data_iter.hard_reset()

    def reset(self):
        self.data_iter.reset()

    def next(self):
        batch = self.data_iter.next()
        labelnp=[]
        for lab in batch.label[0].T.asnumpy():
            #print lab.shape
            labelnp.append(mx.nd.array(lab))        
        all_label = [labelnp[i] for i in self.labelIdx]
        return mx.io.DataBatch(data=batch.data, label=all_label, \
                pad=batch.pad, index=batch.index)
    
    
class multi_iter_iterator(mx.io.DataIter):
    '''random task ilab iterator'''
    #requires bucketing module, only constraint should be that symgen in the bucketing module must give a single output with name softmax[bucketing_key+1]
    def __init__(self, data_iter_list,iter_active,maxbatch):
        super(multi_iter_iterator, self).__init__()
        self.data_iter_list = data_iter_list
        self.batch_size = self.data_iter_list[0].batch_size
        self.iter_active = iter_active 
        self.iter_id=0
        self.counter = 0
        self.niter=len(data_iter_list)
        self.maxbatch=maxbatch
        print self.niter
        for i,iterator in enumerate(data_iter_list):
            data_iter_list[i] = mx.io.ResizeIter(iterator,self.maxbatch,reset_internal=False)
        
        assert len(iter_active)==len(data_iter_list)
        #self.num_classes = num_cl

    @property
    def provide_data(self):
        return self.data_iter_list[0].provide_data

    @property
    def provide_label(self):
        return self.data_iter_list[0].provide_label

    def hard_reset(self):
        for data_iter in self.data_iter_list:
            data_iter.hard_reset()

    def reset(self):
        for data_iter in self.data_iter_list:
            data_iter.reset()

    def next(self):
        #first a random dataset is selected
        #iter_id = np.random.randint(0,len(self.data_iter_list))
        iter_id = self.iter_id
        #print iter_id
        #change iter at next timestep
        if self.niter==2:
            if iter_id == 0:

                self.iter_id = 1
            elif iter_id == 1:
                self.iter_id = 0
        elif self.niter==3:
            if iter_id == 0:

                self.iter_id = 1

            elif iter_id == 1:
                self.iter_id = 2
            elif iter_id == 2:
                self.iter_id = 0
            
        
        self.counter += 1    
        batch = self.data_iter_list[iter_id].next()
        label_names = []
        batch_size = []
        for i,j in enumerate(self.iter_active[iter_id]):
            if j==1:
                label_names.append('softmax%d_label'%(i+1))
                batch_size.append((self.batch_size,))

        return mx.io.DataBatch(data=batch.data, label=batch.label, \
                   pad=batch.pad, index=batch.index, bucket_key=iter_id, provide_data=self.data_iter_list[iter_id].provide_data,  provide_label=zip(label_names,batch_size))                             
                        
        
def places_iterator(data_dir,batch_size):
    train = mx.io.ImageRecordIter(
            path_imgrec         = os.path.join(data_dir, "places365_train_shuf.rec"),
            label_width         = 1,
            data_name           = 'data',
            label_name          = 'softmax1_label',
            data_shape          = (3, 224, 224),
            batch_size          = batch_size,
            pad                 = 0,
            fill_value          = 127,  # only used when pad is valid
            rand_crop           = True,
            max_random_scale    = 1.0,  # 480 with imagnet, 32 with cifar10
            min_random_scale    = 1,  # 256.0/480.0
            max_aspect_ratio    =  0.25,
            random_h            = 36,  # 0.4*90
            random_s            = 50,  # 0.4*127
            random_l            = 50,  # 0.4*127
            #max_rotate_angle    = 10,
            #max_shear_ratio     = 0.1, #
            rand_mirror         = True,
            shuffle             = True)
            #num_parts           = kv.num_workers,
            #part_index          = kv.rank)
    val = mx.io.ImageRecordIter(
            path_imgrec         = os.path.join(data_dir, "places365_val.rec"),
            label_width         = 1,
            data_name           = 'data',
            label_name          = 'softmax1_label',
            batch_size          = batch_size,
            data_shape          = (3, 224, 224),
            rand_crop           = False,
            rand_mirror         = False)
            #num_parts           = kv.num_workers,
            #part_index          = kv.rank)
    return train, val 

def human_sketches_iterator(data_dir,batch_size):
    train = mx.io.ImageRecordIter(
            path_imgrec         = os.path.join(data_dir, "human_sketches_train.rec"),
            label_width         = 1,
            data_name           = 'data',
            label_name          = 'softmax1_label',
            data_shape          = (3, 224, 224),
            batch_size          = batch_size,
            pad                 = 0,
            fill_value          = 127,  # only used when pad is valid
            rand_crop           = True,
            max_random_scale    = 1.0,  # 480 with imagnet, 32 with cifar10
            min_random_scale    = 1,  # 256.0/480.0
            max_aspect_ratio    =  0.25,
             random_h            = 18,#36,  # 0.4*90
             random_s            = 25,#50,  # 0.4*127
             random_l            = 25,#50,  # 0.4*127
            #max_rotate_angle    = 10,
            #max_shear_ratio     = 0.1, #
            rand_mirror         = True,
            shuffle             = True)
            #num_parts           = kv.num_workers,
            #part_index          = kv.rank)
    val = mx.io.ImageRecordIter(
            path_imgrec         = os.path.join(data_dir, "human_sketches_val.rec"),
            label_width         = 1,
            data_name           = 'data',
            label_name          = 'softmax1_label',
            batch_size          = batch_size,
            data_shape          = (3, 224, 224),
            rand_crop           = False,
            rand_mirror         = False)
            #num_parts           = kv.num_workers,
            #part_index          = kv.rank)
    return train, val

def sketchy_iterator(data_dir,batch_size):
    train = mx.io.ImageRecordIter(
            path_imgrec         = os.path.join(data_dir, "sketchy_sketch_train.rec"),
            label_width         = 1,
            data_name           = 'data',
            label_name          = 'softmax1_label',
            data_shape          = (3, 224, 224),
            batch_size          = batch_size,
            pad                 = 0,
            fill_value          = 127,  # only used when pad is valid
            rand_crop           = True,
            max_random_scale    = 1.0,  # 480 with imagnet, 32 with cifar10
            min_random_scale    = 1,  # 256.0/480.0
            max_aspect_ratio    =  0.25,
            random_h            = 18,  # 0.4*90
            random_s            = 25,  # 0.4*127
            random_l            = 25,  # 0.4*127
            #max_rotate_angle    = 10,
            #max_shear_ratio     = 0.1, #
            rand_mirror         = True,
            shuffle             = True)
            #num_parts           = kv.num_workers,
            #part_index          = kv.rank)
    val = mx.io.ImageRecordIter(
            path_imgrec         = os.path.join(data_dir, "sketchy_sketch_test.rec"),
            label_width         = 1,
            data_name           = 'data',
            label_name          = 'softmax1_label',
            batch_size          = batch_size,
            data_shape          = (3, 224, 224),
            rand_crop           = False,
            rand_mirror         = False)
            #num_parts           = kv.num_workers,
            #part_index          = kv.rank)
    return train, val


def imagenet_iterator(data_dir,batch_size):
    train = mx.io.ImageRecordIter(
            path_imgrec         = os.path.join(data_dir, "imagenet_small_train.rec"),
            label_width         = 1,
            data_name           = 'data',
            label_name          = 'softmax1_label',
            data_shape          = (3, 224, 224),
            batch_size          = batch_size,
            pad                 = 0,
            fill_value          = 127,  # only used when pad is valid
            rand_crop           = True,
            max_random_scale    = 1.0,  # 480 with imagnet, 32 with cifar10
            min_random_scale    = 0.533,  # 256.0/480.0
            max_aspect_ratio    =  0.25,
            random_h            = 36,  # 0.4*90
            random_s            = 50,  # 0.4*127
            random_l            = 50,  # 0.4*127
            #max_rotate_angle    = 10,
            #max_shear_ratio     = 0.1, #
            rand_mirror         = True,
            shuffle             = True)
            #num_parts           = kv.num_workers,
            #part_index          = kv.rank)
    val = mx.io.ImageRecordIter(
            path_imgrec         = os.path.join(data_dir, "imagenet_small_val.rec"),
            label_width         = 1,
            data_name           = 'data',
            label_name          = 'softmax1_label',
            batch_size          = batch_size,
            max_random_scale    = 0.533,  # 480 with imagnet, 32 with cifar10
            min_random_scale    = 0.533,  # 256.0/480.
            data_shape          = (3, 224, 224),
            rand_crop           = False,
            rand_mirror         = False)
            #num_parts           = kv.num_workers,
            #part_index          = kv.rank)
    return train, val
