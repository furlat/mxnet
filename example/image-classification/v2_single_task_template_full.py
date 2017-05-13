
# coding: utf-8

# In[1]:

import os
import argparse
import logging
logging.basicConfig(level=logging.DEBUG)
from common import find_mxnet, data, fit
from common.util import download_file
from tf_iterators import *
import mxnet as mx
import numpy as np
import numpy 
from mxnet import nd
#from mxnet.module.module_tf import *


# In[2]:

def birds_iterator(data_dir,batch_size):
    train = mx.io.ImageRecordIter(
            path_imgrec         = os.path.join(data_dir, "birds_train.rec"),
            label_width         = 1,
            data_name           = 'data',
            label_name          = 'softmax1_label',
            data_shape          = (3, 444, 444),
            batch_size          = batch_size,
            pad                 = 0,
            fill_value          = 127,  # only used when pad is valid
            #rand_crop           = True,
            max_random_scale    = 1.0,  # 480 with imagnet, 32 with cifar10 448 with birds 0.93
            min_random_scale    = 0.93,  # 256.0/480.0
            #max_aspect_ratio    =  0.25,
            #random_h            = 36,  # 0.4*90
            #random_s            = 50,  # 0.4*127
            #random_l            = 50,  # 0.4*127
            #max_rotate_angle    = 10,
            #max_shear_ratio     = 0.1, #
            rand_mirror         = True,
            shuffle             = True)
            #num_parts           = kv.num_workers,
            #part_index          = kv.rank)
    val = mx.io.ImageRecordIter(
            path_imgrec         = os.path.join(data_dir, "birds_test.rec"),
            label_width         = 1,
            data_name           = 'data',
            label_name          = 'softmax1_label',
            batch_size          = batch_size,
            max_random_scale    = 0.93,  # 480 with imagnet, 32 with cifar10
            min_random_scale    = 0.93,  # 256.0/480.
            data_shape          = (3, 444, 444),
            rand_crop           = False,
            rand_mirror         = False)
            #num_parts           = kv.num_workers,
            #part_index          = kv.rank)
    return train, val


# In[3]:

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


# In[4]:

image_shape = '3,448,448'
batch_size = 16
results_prefix='/home/ubuntu/results/'

#results_prefix='/efs/users/furlat/v1_results/'http://127.0.0.1:8888/notebooks/example/image-classification/v2_single_task_template.ipynb#


# In[5]:

dataset='birds'
nstream=1
init_1='imagenet'
init_2='imagenet'
freeze=1



# In[6]:

if dataset == 'birds':
    data_dir_birds='/home/ubuntu/data/birds'
    train_birds, val_birds = birds_iterator(data_dir_birds,batch_size)
    num_classes=201
    print "mona"
    


# In[7]:



    
#elif dataset =='sketchy':
#    data_dir_sketchy='/efs/datasets/users/furlat/sketchy_database'
#    train_sketches, val_sketches = sketchy_iterator(data_dir_sketchy,batch_size)
#    num_classes=125
label_names = [train_birds.provide_label[0][0]]
    
from importlib import import_module
net = import_module('symbols.resnet_factory')

if nstream == 1:
    print 'plain resnet'

    ctx=[mx.gpu(0)]    
    #ctx=[mx.gpu(0+(3*gpu_block)),mx.gpu(1+(3*gpu_block)),mx.gpu(2+(3*gpu_block))]
    #print 'gpu bloc%2d : using gpu %2d to %2d' %(gpu_block,0+(3*gpu_block),2+(3*gpu_block))

    arch='resnet'    
    sym,data,labels=net.get_symbol(num_classes=num_classes,active=[1], gate_prefix=None,rescale_grad=1, num_layers=50,gated=False, image_shape=image_shape)
elif nstream==2:
    print 'multi branch resnet - Good luck daddy ;)'
    #with batch size 256  5*9686M +1*10550 ~6 full k80 Speed: 52.49 samples/sec
    #if gpu_block == 1:
        #print 'gpu block1: using gpu 0 to 6'
        
        #ctx=[mx.gpu(0),mx.gpu(1),mx.gpu(2),mx.gpu(3),mx.gpu(4),mx.gpu(5),mx.gpu(6)]
#     elif gpu_block == 2:
#         print 'gpu block2: using gpu 6 to 11'
#         ctx=[mx.gpu(7),mx.gpu(8),mx.gpu(9),mx.gpu(10),mx.gpu(11),mx.gpu(12),mx.gpu(13)]
#     elif gpu_block > 3:
#         print 'gpu bloc3 is too small for this taks enjoy the error soon'
    ctx=[mx.gpu(0)]     
    arch='mpath'
    sym,data,labels=net.get_symbol_2branch(num_classes=250,active=1, gate_prefix=None,rescale_grad=1, num_layers=50,gated=True, image_shape=image_shape,coupled=True)
    


# In[8]:

#mx.viz.plot_network(symbol=sym)

imagenet_weights= '/home/ubuntu/models/imagenet_r50-lr05'
#model_prefix= 'tryhard-resnet'
prefix = imagenet_weights
#prefix = model_prefix
epoch=120
save_dict = nd.load('%s-%04d.params' % (prefix, epoch))
arg_params_imag = {}
aux_params_imag = {}
ext_check=['sc','fc1','data']
imagenet_par=[]
exact_check=['bn1_beta','bn1_gamma']
for k, v in save_dict.items():
    tp, name = k.split(':', 1)

    if tp == 'arg':
        arg_params_imag[name] = v
        
        
        #print name
        if not any(ext in name for ext in ext_check):
            if not any(ext == name for ext in exact_check):
                imagenet_par.append(name)
                if init_2=='imagenet':
                    arg_params_imag['_a_'+name] = v


    if tp == 'aux':
        aux_params_imag[name] = v
        if init_2=='imagenet':
            aux_params_imag['_a_'+name] = v
del arg_params_imag['fc1_bias']
del arg_params_imag['fc1_weight']



#arg_params_imag.list_arguments


#_a_bn_data_beta

symlist=sym.list_arguments()
gatelist=[s for s in symlist if 'gate' in s]


# In[9]:

#imagenet_par


# In[10]:



if init_1=='imagenet':
    arg_params=arg_params_imag
    aux_params=aux_params_imag
    print 'initizalizing left stream from imagenet'
    if init_2=='imagenet':
        print 'initizalizing right stream from imagenet'
else:
    arg_params=None
    aux_params=None
    print 'initizalizing left stream from random'
    
    
if nstream == 2:
    model_prefix = arch+'-fix'+str(freeze)+'-'+init_1+'-'+init_2+'-'+dataset
    if init_2=='rand':
        print 'initizalizing right stream from random'
    print model_prefix
    if freeze==1:
        
        fixed=imagenet_par
    else:
        fixed=None
else:
    #only the parameters from the original 1stream imagenet network
    model_prefix =  arch+'-fix'+str(freeze)+'-'+init_1+'-'+dataset
    print model_prefix
    if freeze==1:
        
        fixed=imagenet_par
    else:
        fixed=None

mod = mx.mod.Module(sym, label_names=label_names,fixed_param_names=fixed,context=ctx)
#
checkpoint_path=results_prefix+model_prefix
#checkpoint = mx.callback.module_checkpoint(mod,model_prefix)
mod.bind(data_shapes=train_birds.provide_data, label_shapes=train_birds.provide_label)
mod.init_params(initializer=mx.initializer.Uniform(0.01), arg_params=arg_params, aux_params=aux_params,
                    allow_missing=True, force_init=False)

checkpoint = mx.callback.module_checkpoint(mod,checkpoint_path,period=5)

# for gate in gatelist:
#     print mod.get_params()[0][gate].asnumpy(), gate

# In[7]:




# In[11]:

begin_epoch=0
if begin_epoch>0:
    sym, arg_params, aux_params =mx.model.load_checkpoint(checkpoint_path, begin_epoch)

    mod.set_params(arg_params, aux_params)
    
logistic_init=False    
if logistic_init:
    logistic_path='/home/ubuntu/results/resnet-fix1-imagenet-birds'
    _, arg_params, aux_params =mx.model.load_checkpoint(logistic_path, 120)

    mod.set_params(arg_params, aux_params)    


# In[16]:

mod.fit(train_birds,
         eval_data=val_birds,
         eval_metric=[Cross_Entropy()],
         #eval_metric=[mx.metric.Accuracy()],

         batch_end_callback = [mx.callback.log_train_metric(50),mx.callback.Speedometer(batch_size,100)],
         epoch_end_callback=checkpoint,
         allow_missing=False,
         begin_epoch=begin_epoch,
         log_prefix = model_prefix,
         optimizer_params={'learning_rate':0.001, 'momentum': 0.9,'wd':0.0004 },
         num_epoch=90)


# In[ ]:

checkpoint_path


# In[ ]:




# In[ ]:



# In[ ]:
# 
train_score=[]
val_score=[]
epoch=[]
for i in range(5,91,5):
    
    sym, arg_params, aux_params =             mx.model.load_checkpoint(checkpoint_path, i)
        
    mod.set_params(arg_params, aux_params)
    
    #for gate in gatelist:
        #print mod.get_params()[0][gate].asnumpy(), gate
    res_train = mod.score(train_birds, mx.metric.Accuracy(),num_batch=93)
    res_val= mod.score(val_birds, mx.metric.Accuracy(),num_batch=93)
    epoch.append(i+1)
    for name, value in res_train:
        print 'Epoch[%d] Training-%s=%f' %(i, name, value)
        train_score.append(value)
    for name, value in res_val:
        print 'Epoch[%d] Validation-%s=%f' %(i, name, value)
        val_score.append(value)
        
#for gate in gatelist:
    #print mod.get_params()[0][gate].asnumpy(), gate


# In[ ]:

logfile_url= '%s-eval-metric-log-accuracy.txt' % (results_prefix+'eval/'+model_prefix)
print 'saving logfiles  at %s' % (logfile_url)
logfile = open(logfile_url, 'w')
for  epoch,train_metric, val_metric in zip(epoch,train_score,val_score):
    #logfile.write("%s\n" % item)


    logfile.write(str(epoch)+"\t"+str(train_metric)+"\t"+ str(val_metric)+"\n")
logfile.close()



# In[ ]:




# In[ ]:



