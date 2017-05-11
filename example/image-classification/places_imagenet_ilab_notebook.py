
# coding: utf-8

# In[1]:

import os
import argparse
import logging
logging.basicConfig(level=logging.DEBUG)
from common import find_mxnet, data, fit
from common.util import download_file
from ilab_iterator import ilab_iterator, Multi_ilab_iterator, Single_ilab_iterator, random_task_iterator
import mxnet as mx
import numpy as np
import numpy 

class Cross_Entropy(mx.metric.EvalMetric):
    """Calculate accuracies of multi label"""

    def __init__(self):
        super(Cross_Entropy, self).__init__('cross-entropy')
    def update(self, labels, preds):
        mx.metric.check_label_shapes(labels, preds)
        label = labels[0].asnumpy()
        pred = preds[0].asnumpy()
        for i in range(label.shape[0]):
            prob = pred[i,numpy.int64(label[i])]
            if len(labels) == 1:
                self.sum_metric += (-numpy.log(prob)).sum()
        self.num_inst += label.shape[0]
        
class Multi_Entropy(mx.metric.EvalMetric):
    """Calculate accuracies of multi label"""

    def __init__(self, num=None):
        super(Multi_Entropy, self).__init__('multi-entropy', num)

    def update(self, labels, preds):
        mx.metric.check_label_shapes(labels, preds)

        if self.num != None:
            assert len(labels) == self.num
        
        for i in range(len(labels)):
                #pred_label = mx.nd.argmax_channel(preds[i]).asnumpy()
                #label = label.asnumpy()
                pred = preds[i].asnumpy()
                #pred = pred(pred_label)
                #prb = pred.ravel()
                label = labels[i].asnumpy().ravel()
                assert label.shape[0] == pred.shape[0]
               

                prob = pred[numpy.arange(label.shape[0]), numpy.int64(label)]
                self.sum_metric[i] += (-numpy.log(prob)).sum()
                self.num_inst[i] += label.shape[0]   
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
        for i,iterator in enumerate(data_iter_list):
            data_iter_list[i] = mx.io.ResizeIter(iterator,maxbatch,reset_internal=False)
        
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

        return mx.io.DataBatch(data=batch.data, label=batch.label,                    pad=batch.pad, index=batch.index, bucket_key=iter_id, provide_data=self.data_iter_list[iter_id].provide_data,  provide_label=zip(label_names,batch_size))                             
                    


# In[ ]:




# In[2]:

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
            max_rotate_angle    = 10,
            max_shear_ratio     = 0.1,
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
            data_shape          = (3, 224, 224),
            rand_crop           = False,
            rand_mirror         = False)
            #num_parts           = kv.num_workers,
            #part_index          = kv.rank)
    return train, val


# In[3]:

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
            min_random_scale    = 1.0, #0.533,  # 256.0/480.0
            max_aspect_ratio    =  0.25,
            random_h            = 36,  # 0.4*90
            random_s            = 50,  # 0.4*127
            random_l            = 50,  # 0.4*127
            max_rotate_angle    = 10,
            max_shear_ratio     = 0.1,
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


# In[4]:

#prepare the single_dataset iterators
#ilab
batch_size = 1280
data_dir_imag='/efs/datasets/users/furlat/imagenet'
train_ilab, val_ilab = ilab_iterator(batch_size)
train_ilab = Multi_ilab_iterator(train_ilab,subset=[0,2,3])
#train_ilab2= mx.io.ResizeIter(train_ilab,1000,reset_internal=False)
val_ilab = Multi_ilab_iterator(val_ilab,subset=[0,2,3])
data_dir_places='/efs/datasets/users/furlat/places'
#places
train_places, val_places = places_iterator(data_dir_places,batch_size)
train_imag, val_imag = imagenet_iterator(data_dir_imag,batch_size)
train_multi = multi_iter_iterator([train_imag,train_ilab,train_places],[[1,0,0,0,0],[0,1,1,1,0],[0,0,0,0,1]],1000)
val_multi = multi_iter_iterator([val_imag,val_ilab,val_places],[[1,0,0,0],[0,1,1,1],[0,0,0,0,1]],100)


# In[ ]:

from importlib import import_module
net = import_module('symbols.resnet_md')
#batch_size = 16
image_shape = '3,224,224'

def sym_gen(bucket_key):
    num_classes=[1000,10,11,8,365]
    active = [[1,0,0,0,0],[0,1,1,1,0],[0,0,0,0,1]]
    rescale_grad=[1,0.5,1]
    if bucket_key == 3:
        return net.get_vmt_symbol(num_classes, 50, image_shape, conv_workspace=256)
    else:
        return net.get_mt_symbol(num_classes,active[bucket_key],rescale_grad[bucket_key], 50, image_shape, conv_workspace=256)

ctx=[mx.gpu(0),mx.gpu(1),mx.gpu(2),mx.gpu(3),mx.gpu(4),mx.gpu(5),mx.gpu(6),mx.gpu(7),mx.gpu(8),mx.gpu(9),mx.gpu(10),mx.gpu(11),mx.gpu(12),mx.gpu(13),mx.gpu(14),mx.gpu(15)]
mod = mx.mod.BucketingModule(sym_gen,default_bucket_key=3, context=ctx)

#mod = mx.mod.BucketingModule(sym_gen,default_bucket_key=2, context=[mx.gpu(0)])


model_prefix = '/home/ubuntu/results/imagenet_ilab_places'
#model_prefix2load=/efs/datasets/users/furlat/imagenet/imagenet_r50-0013.params

checkpoint = mx.callback.module_checkpoint(mod,model_prefix)




#lr_schedule it isimilar to the CIFAR100 schedule but half length of the steps
schedule = [90000,180000,240000]


# In[ ]:

mod.fit(train_multi,
        eval_data=val_multi,
        eval_metric=[Cross_Entropy(),Multi_Entropy(num=3),Cross_Entropy()],#(num=4),
        batch_end_callback = mx.callback.log_train_metric(5),
        epoch_end_callback=checkpoint,
        allow_missing=False,
        optimizer_params={'learning_rate':0.1, 'momentum': 0.9,'wd':0.0001, 'lr_scheduler': mx.lr_scheduler.MultiFactorScheduler(step=schedule,factor=0.1) },
        num_epoch=20, log_prefix = model_prefix)


# In[ ]:



