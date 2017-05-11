
# coding: utf-8

# In[ ]:

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

batch_size = 1280
data_dir='/efs/datasets/users/furlat/imagenet'
#imagenet
train_imag, val_imag = imagenet_iterator(data_dir,batch_size)

from importlib import import_module
net = import_module('symbols.resnet_md')
image_shape = '3,224,224'
label_names = [train_imag.provide_label[0][0]]
num_classes=[1000]
sym = net.get_vmt_symbol(num_classes, 50, image_shape, conv_workspace=256)  
sym=sym[0]
ctx=[mx.gpu(0),mx.gpu(1),mx.gpu(2),mx.gpu(3),mx.gpu(4),mx.gpu(5),mx.gpu(6),mx.gpu(7),mx.gpu(8),mx.gpu(9),mx.gpu(10),mx.gpu(11),mx.gpu(12),mx.gpu(13),mx.gpu(14),mx.gpu(15)]

mod = mx.mod.Module(sym, label_names=label_names, context=ctx)

model_prefix = '/home/ubuntu/results/imagenet_r50-lr05'

checkpoint = mx.callback.module_checkpoint(mod,model_prefix)
begepoch = 90
#lr_schedule it isimilar to the CIFAR100 schedule but half length of the steps
schedule = [40000,70000,100000]
sym, arg_params, aux_params = mx.model.load_checkpoint(model_prefix, begepoch)
mod.bind(data_shapes=train_imag.provide_data,
             label_shapes=val_imag.provide_label)        
mod.set_params(arg_params, aux_params)


# In[ ]:

mod.fit(train_imag,
        eval_data=val_imag,
        eval_metric=[Cross_Entropy()],
        batch_end_callback = mx.callback.log_train_metric(1),
        epoch_end_callback=checkpoint,
        allow_missing=False,
        begin_epoch=begepoch,
        log_prefix = model_prefix,
        optimizer_params={'learning_rate':0.0005, 'momentum': 0.9,'wd':0.0001},
        num_epoch=200)


# In[ ]:

mod.bind(data_shapes=train_imag.provide_data,
             label_shapes=val_imag.provide_label)
for i in range(20,25):
    
    sym, arg_params, aux_params =             mx.model.load_checkpoint(model_prefix, i+1)
        
    mod.set_params(arg_params, aux_params)
    res_train= mod.score(train_imag, mx.metric.Accuracy(),num_batch=10)
    res_val= mod.score(val_imag, mx.metric.Accuracy(),num_batch=10)
    epoch = i
    for name, val in res_train:
        print 'Epoch[%d] Training-%s=%f' %(epoch, name, val)
    for name, val in res_val:
        print 'Epoch[%d] Validation-%s=%f' %(epoch, name, val)


# In[ ]:




# In[ ]:




# In[ ]:



