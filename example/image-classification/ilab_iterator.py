import mxnet as mx
import random
from mxnet.io import DataBatch, DataIter
import numpy as np


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

#it takes as input a multilabel data iter generatec from ilab_iterator and a vector called subset, subset <= label_width
#subset defines which of the tasks to train on, the symbol must have the same number of outputs as label_width.
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

class Merge_binaryclass_iterator(mx.io.DataIter):
    '''from a multitask dataset select a single multinomial task\
    merge a multinomial classifier into a binary classifier '''

    def __init__(self, data_iter,merge):
        super(Merge_binaryclass_iterator, self).__init__()
        self.data_iter = data_iter
        self.batch_size = self.data_iter.batch_size
        self.labelIdx=subset #for andrea's case this len(self.labelIdx) must be 1
        self.merge= merge #labels to transform into a 1, assumes all other must become 0
        assert len(labelIdx)==1
        
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
 
    def hard_reset(self):
        self.data_iter.hard_reset()

    def reset(self):
        self.data_iter.reset()

    def next(self):
        batch = self.data_iter.next()
        labelnp=[]
        for lab in batch.label[0].T.asnumpy():
            lab_merged = [1 if x in self.merge else 0 for x in lab]
            labelnp.append(mx.nd.array(lab_merged))        
        all_label = [labelnp[i] for i in self.labelIdx]
        return mx.io.DataBatch(data=batch.data, label=all_label, \
                pad=batch.pad, index=batch.index)

    
class random_task_iterator(mx.io.DataIter):
    '''random task ilab iterator'''
    #requires bucketing module, only constraint should be that symgen in the bucketing module must give a single output with name softmax[bucketing_key+1]
    def __init__(self, data_iter,subset):
        super(random_task_iterator, self).__init__()
        self.data_iter = data_iter
        self.batch_size = self.data_iter.batch_size
        self.labelIdx=subset
        #self.num_classes = num_cl

    @property
    def provide_data(self):
        return self.data_iter.provide_data

    @property
    def provide_label(self):
        provide_label = self.data_iter.provide_label
        label_names=[]
        batch_size=[]
        for i in range(2):
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
        #label_id = np.random.randint(0,len(self.labelIdx))
        label_id = np.random.randint(0,len(self.labelIdx))
        #print label_id
         #num_classes= self.num_classes[label_id] #the original num_clases is a list of lists saying which labels are active for each id, only one in this case
        all_label=[]
        # prepare all the labels
        for lab in  batch.label[0].T.asnumpy():
            all_label.append(mx.nd.array(lab))
        # take the subset, in this case the single label_id   
        subset_label = [all_label[i] for i in self.labelIdx]
        final_label = [subset_label[i] for i in [label_id]]

        #print i, len(final_label), final_label[0].asnumpy(), 
        # generates the provide label adequate to the current label
        label_names=[]
        batch_size=[]
        for i in [label_id]:
            label_names.append('softmax%d_label'%(i+1))
            batch_size.append((self.batch_size,))
        
        #print zip(label_names,batch_size)
        return mx.io.DataBatch(data=batch.data, label=final_label, \
                   pad=batch.pad, index=batch.index, bucket_key=label_id, provide_data=self.data_iter.provide_data,  provide_label=zip(label_names,batch_size))                             
    
    


class Single_ilab_iterator(mx.io.DataIter):
    '''multi label mnist iterator'''

    def __init__(self, data_iter, labid):
        super(Single_ilab_iterator, self).__init__()
        self.data_iter = data_iter
        self.batch_size = self.data_iter.batch_size
        self.labid = labid

    @property
    def provide_data(self):
        return self.data_iter.provide_data

    @property
    def provide_label(self):
        provide_label = self.data_iter.provide_label[0]
        # Different labels should be used here for actual application
        return [('softmax1_label', (self.batch_size,))]

    def hard_reset(self):
        self.data_iter.hard_reset()

    def reset(self):
        self.data_iter.reset()

    def next(self):
        batch = self.data_iter.next()
        label = batch.label[0]
        label1,label2,label3,label4 =  label.T.asnumpy() 
        label1=mx.nd.array(label1)
        label2=mx.nd.array(label2)
        label3=mx.nd.array(label3)
        label4=mx.nd.array(label4)
        all_label=[label1,label2,label3,label4]
        #labeli = label.T.asnumpy()
        #single_label = mx.nd.array(labeli[self.labid,:])
        single_label = all_label[self.labid]
        
        return mx.io.DataBatch(data=batch.data, label=[single_label], \
                pad=batch.pad, index=batch.index)    

class Multi_noid_ilab_iterator(mx.io.DataIter):
    '''multi label mnist iterator'''

    def __init__(self, data_iter):
        super(Multi_ilab_iterator, self).__init__()
        self.data_iter = data_iter
        self.batch_size = self.data_iter.batch_size

    @property
    def provide_data(self):
        return self.data_iter.provide_data

    @property
    def provide_label(self):
        provide_label = self.data_iter.provide_label
        # Different labels should be used here for actual application
        return [('softmax1_label', (self.batch_size,)), \
                ('softmax2_label', (self.batch_size,)), \
                ('softmax3_label', (self.batch_size,)), \
                ('softmax4_label', (self.batch_size,)), \
                ('softmax5_label', (self.batch_size,)), \
                ('softmax6_label', (self.batch_size,))]

    def hard_reset(self):
        self.data_iter.hard_reset()

    def reset(self):
        self.data_iter.reset()

    def next(self):
        batch = self.data_iter.next()
        label = batch.label[0]
        labeli = label.T.asnumpy()
        label = mx.nd.array(np.delete(labeli[1,:]))
        # label1,label2,label3,label4,label5,label6,label7 =  
        # label1=mx.nd.array(label1)
        # label2=mx.nd.array(label2)
        # label3=mx.nd.array(label3)
        # label4=mx.nd.array(label4)
        # label5=mx.nd.array(label5)
        # label6=mx.nd.array(label6)
        # label7=mx.nd.array(label7)
        # [label1, label2,label3,label4,label5,label6, label7]
        return mx.io.DataBatch(data=batch.data, label=label, \
                pad=batch.pad, index=batch.index)                  

