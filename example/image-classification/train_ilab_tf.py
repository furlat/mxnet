import os
import argparse
import logging
logging.basicConfig(level=logging.DEBUG)
from common import find_mxnet, data, fit
from common.util import download_file
from ilab_iterator import ilab_iterator, Multi_ilab_iterator, Single_ilab_iterator, random_task_iterator
import mxnet as mx
import numpy

# class AccuracyIlab(mx.metric.EvalMetric):
#     """Calculate accuracies of multi label"""

#     def __init__(self):
#         super(Multi_Accuracy, self).__init__('multi-accuracy', num)

#     def update(self, labels, preds):
#         mx.metric.check_label_shapes(labels, preds)

# #         if self.num != None:
# #             assert len(labels) == self.num

#         for i in range(len(labels)):
#             pred_label = mx.nd.argmax_channel(preds[i]).asnumpy().astype('int32')
#             label = labels[i].asnumpy().astype('int32')

#             mx.metric.check_label_shapes(label, pred_label)

#             if i == None:
#                 self.sum_metric += (pred_label.flat == label.flat).sum()
#                 self.num_inst += len(pred_label.flat)
#             else:
#                 self.sum_metric[i] += (pred_label.flat == label.flat).sum()
#                 self.num_inst[i] += len(pred_label.flat)
            
class CrossEntropyIlab(mx.metric.EvalMetric):
    """Calculate Cross Entropy loss"""
    def __init__(self,task=0):
        super(CrossEntropyIlab, self).__init__('cross-entropy', task)
        self.task = task

    def update(self, labels, preds):
        mx.metric.check_label_shapes(labels, preds)

        for label, pred in zip(labels, preds):
            label = label[self.task].asnumpy().astype('int32')
            #label = label.asnumpy()
            pred = pred.asnumpy()

            label = label.ravel()
            assert label.shape[0] == pred.shape[0]
            prob = pred[numpy.arange(label.shape[0]), numpy.int64(label)]
            self.sum_metric += (-numpy.log(prob)).sum()
            self.num_inst += label.shape[0]
            f.open('trainig_stats.txt')
            f.write(self.sum_metric)
            f.close()

class Multi_Accuracy(mx.metric.EvalMetric):
    """Calculate accuracies of multi label"""

    def __init__(self, num=None):
        super(Multi_Accuracy, self).__init__('multi-accuracy', num)

    def update(self, labels, preds):
        #print labels 
        #print preds
        mx.metric.check_label_shapes(labels, preds)

        if self.num != None:
            assert len(labels) == self.num

        for i in range(len(labels)):
            pred_label = mx.nd.argmax_channel(preds[i]).asnumpy().astype('int32')
            label = labels[i].asnumpy().astype('int32')

            mx.metric.check_label_shapes(label, pred_label)

            if i == None:
                self.sum_metric += (pred_label.flat == label.flat).sum()
                self.num_inst += len(pred_label.flat)
            else:
                self.sum_metric[i] += (pred_label.flat == label.flat).sum()
                self.num_inst[i] += len(pred_label.flat)

class Multi_Entropy(mx.metric.EvalMetric):
    """Calculate accuracies of multi label"""

    def __init__(self, num=None):
        super(Multi_Entropy, self).__init__('multi-entropy', num)

    def update(self, labels, preds):
        #mx.metric.check_label_shapes(labels, preds)

        if self.num != None:
            print labels
            print len(labels)
            assert len(labels) == self.num

        for i in range(len(labels)):
                #pred_label = mx.nd.argmax_channel(preds[i]).asnumpy()
                #label = label.asnumpy()
                pred = preds[i].asnumpy()
                #pred = pred(pred_label)
                #prb = pred.ravel()
                label = labels[i].asnumpy().ravel()
                assert label.shape[0] == pred.shape[0]
                print label.shape, pred.shape
                prob = pred[numpy.arange(label.shape[0]), numpy.int64(label)]
                self.sum_metric[i] += (-numpy.log(prob)).sum()
                self.num_inst[i] += label.shape[0]         

class Single_Entropy(mx.metric.EvalMetric):
    """Calculate accuracies of multi label"""

    def __init__(self):
        super(Single_Entropy, self).__init__('single-entropy')

    def update(self, labels, preds):
        mx.metric.check_label_shapes(labels, preds)

      
        label = labels[0].asnumpy()
        pred = preds[0].asnumpy()
        #print pred.shape
        for i in range(label.shape[0]):
            prob = pred[i,numpy.int64(label[i])]
            self.sum_metric += (-numpy.log(prob)).sum()
            #print self.sum_metric
#                 #print len(labels)
#                 #pred_label = mx.nd.argmax_channel(preds[i]).asnumpy()
#                 #label = label.asnumpy()
#                 pred = preds[i].asnumpy()
#                 #pred = pred(pred_label)
#                 #prb = pred.ravel()
                
#                 assert label.shape[0] == pred.shape[0]
#                 print numpy.arange(label.shape[0]), label.shape, pred.shape
#                 prob = pred[:,numpy.int64(label)]
#                 print prob, prob.shape
#                 self.sum_metric += (-numpy.log(prob)).sum()
#                 self.num_inst += label.shape[0]  
                #print self.sum_metric
# if __name__ == '__main__':
# download data

# data setup
# parse args

image_shape = '3,224,224'
# load network
from importlib import import_module
#num_classes = [[10,0,0],[0,11,0],[0,0,8]]
num_classes=[10,11,8]

net = import_module('symbols.resnet_md')
batch_size = 128

train, val_raw = ilab_iterator(batch_size=batch_size)
train=random_task_iterator(train,subset=[0,2,3])#,num_cl=num_classes)
val=random_task_iterator(val_raw,subset=[0,2,3])#,num_cl=num_classes)
multi_val = Multi_ilab_iterator(val_raw,subset=[0,2,3])


def sym_gen(bucket_key):
    num_classes=[10,11,8]
    #num_classes=[10,10,10]
    active = [[1,0,0],[0,1,0],[0,0,1]]
    #num_classes = [[10,0],[0,8]]
    if bucket_key == 3:
        return net.get_vmt_symbol(num_classes, 50, image_shape, conv_workspace=256)
    else:
        return net.get_mt_symbol(num_classes,active[bucket_key], 50, image_shape, conv_workspace=256)

def shared_sym_gen():
    num_classes=[10,11,8]
    #num_classes=[10,118]
    #active = [[1,0,0],[0,1,0],[0,0,1]]
    #num_classes = [[10,0,0],[0,11,0],[0,0,8]]
    return net.get_vmt_symbol(num_classes, 50, image_shape, conv_workspace=256)
#num_classes = [10,91,11,8]
num_classes=[10,11,8]
#num_classes=[10]
#sym, label_names = net.get_vmt_symbol(num_classes, 50, image_shape, conv_workspace=256)
#mod = mx.mod.Module(sym, label_names=(label_names), context=[mx.gpu(0)])
mod = mx.mod.BucketingModule(sym_gen,default_bucket_key=3, context=[mx.gpu(0),mx.gpu(1)])
#sym_gen, default_bucket_key=None
#mod = mx.mod.Module(sym, context=[mx.gpu(0)])



#val=random_task_iterator(mx.io.DataIter)
#train = Multi_ilab_iterator(train,subset=[0,2,3])
#val = Multi_ilab_iterator(val,subset=[0,2,3])
# train = Single_ilab_iterator(train,labid=1)
# val = Single_ilab_iterator(val,labid=1)

model_prefix = '/home/ubuntu/results/ilab10_rand'
#model_prefix = '/home/ubuntu/results/ilab_cat50'
checkpoint = mx.callback.do_checkpoint(model_prefix)




#lr_schedule it isimilar to the CIFAR100 schedule but half length of the steps
schedule = [20000,30000,40000]

mod.fit(train,
        eval_data=val,
        eval_metric=Single_Entropy(),
        batch_end_callback = mx.callback.log_train_metric(100),
        epoch_end_callback=checkpoint,
        allow_missing=False,
        optimizer_params={'learning_rate':0.05, 'momentum': 0.0,'wd':0.0004, 'lr_scheduler': mx.lr_scheduler.MultiFactorScheduler(step=schedule,factor=0.1) },
        num_epoch=20)
# train loading form nepoch
# mod.fit(train, eval_data=val, epoch_end_callback=checkpoint, optimizer_params={'learning_rate':0.1, 'momentum': 0.9,'wd':0.0004}, num_epoch=300, arg_params=arg_params, aux_params=aux_params,
#         begin_epoch=n_epoch_load)
