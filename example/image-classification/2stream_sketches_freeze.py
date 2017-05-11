
# coding: utf-8

# In[6]:

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
from mxnet import nd

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


# In[7]:

def sketches_iterator(data_dir,batch_size):
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


# In[8]:

batch_size = 256
data_dir='/efs/datasets/users/furlat/human_sketches'
#places
train_sketches, val_sketches = sketches_iterator(data_dir,batch_size)
image_shape = '3,224,224'
label_names = [train_sketches.provide_label[0][0]]
num_classes=[365]
# ctx=[mx.gpu(0),mx.gpu(1),mx.gpu(2),mx.gpu(3),mx.gpu(4),mx.gpu(5),mx.gpu(6),mx.gpu(7),mx.gpu(8),mx.gpu(9),mx.gpu(10),mx.gpu(11),mx.gpu(12),mx.gpu(13),mx.gpu(14),mx.gpu(15)]
ctx=[mx.gpu(6),mx.gpu(7),mx.gpu(8),mx.gpu(9),mx.gpu(10),mx.gpu(11)]


# In[9]:

from importlib import import_module
net = import_module('symbols.resnet_factory')
sym,data,labels=net.get_symbol_2branch(num_classes=365,active=1,gate_prefix=None,rescale_grad=1, num_layers=50,gated=True, image_shape=image_shape)


# In[10]:

imagenet_weights= '/efs/datasets/users/furlat/results_imagenet/imagenet_r50-lr05'
#model_prefix= 'tryhard-resnet'
prefix = imagenet_weights
#prefix = model_prefix
epoch=120
save_dict = nd.load('%s-%04d.params' % (prefix, epoch))

arg_params_imag = {}
aux_params_imag = {}
for k, v in save_dict.items():
    tp, name = k.split(':', 1)
    if tp == 'arg':
        arg_params_imag[name] = v
    if tp == 'aux':
        aux_params_imag[name] = v
del arg_params_imag['fc1_bias']
del arg_params_imag['fc1_weight']
symlist=sym.list_arguments()
gatelist=[s for s in symlist if 'gate' in s]

model_prefix = '/efs/datasets/users/furlat/results/human_sketches/resnet-2streams-gated-humansketch-freeze-09gate' 

# In[ ]:
#epoch=18
#save_dict = nd.load('%s-%04d.params' % (model_prefix, epoch))
#save_dict = nd.load('resnet-2streams-gated-humansketch-no-freeze-0019.params')
#arg_params_load = {}
#aux_params_load = {}
#for k, v in save_dict.items():
#    tp, name = k.split(':', 1)
#    if tp == 'arg':
#        arg_params_load[name] = v
#    if tp == 'aux':
#        aux_params_load[name] = v
        
arg_params_init= arg_params_imag
aux_params_init= aux_params_imag
mod = mx.mod.Module(sym, label_names=label_names,fixed_param_names=arg_params_imag.keys()
,context=ctx)
checkpoint = mx.callback.module_checkpoint(mod,model_prefix)
mod.bind(data_shapes=train_sketches.provide_data, label_shapes=val_sketches.provide_label)
mod.init_params(initializer=mx.initializer.Uniform(0.01), arg_params=arg_params_init, aux_params=aux_params_init,
                    allow_missing=True, force_init=False)




checkpoint = mx.callback.module_checkpoint(mod,model_prefix)


# In[7]:
mod.fit(train_sketches,
        eval_data=val_sketches,
        eval_metric=[Cross_Entropy()],
        #eval_metric=[mx.metric.Accuracy()],#

        batch_end_callback = [mx.callback.log_train_metric(1),mx.callback.Speedometer(batch_size,1)],
        epoch_end_callback=checkpoint,
        allow_missing=False,
        begin_epoch=0,
        log_prefix = model_prefix,
        optimizer_params={'learning_rate':0.1, 'momentum': 0.9,'wd':0.0001 },
        num_epoch=30)


# In[9]:

mod.get_params()[0]['_a_stage3_unit1_conv3_gate'].asnumpy()


# In[ ]:

train_score=[]
val_score=[]
epoch=[]
for i in range(30):
    
    sym, arg_params, aux_params =             mx.model.load_checkpoint(model_prefix, i+1)
        
    mod.set_params(arg_params, aux_params)
    res_train = mod.score(train_sketches, mx.metric.Accuracy(),num_batch=40)
    res_val= mod.score(val_sketches, mx.metric.Accuracy(),num_batch=40)
    epoch.append(i+1)
    for name, value in res_train:
        print 'Epoch[%d] Training-%s=%f' %(i+1, name, value)
        train_score.append(value)
    for name, value in res_val:
        print 'Epoch[%d] Validation-%s=%f' %(i+1, name, value)
        val_score.append(value)
        


# In[ ]:

logfile_url= '%s-eval-metric-log-accuracy.txt' % (model_prefix)
print 'saving logfiles  at %s' % (logfile_url)
logfile = open(logfile_url, 'a')
for  epoch,train_metric, val_metric in zip(epoch,train_score,val_score):
    #logfile.write("%s\n" % item)


    logfile.write(str(epoch)+"\t"+str(train_metric)+"\t"+ str(val_metric)+"\n")
logfile.close()

