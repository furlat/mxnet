import os
import argparse
import logging
logging.basicConfig(level=logging.DEBUG)
from common import find_mxnet, data, fit
from common.util import download_file
import mxnet as mx

def download_cifar10():
    data_dir="data"
    fnames = (os.path.join(data_dir, "cifar10_train.rec"),
              os.path.join(data_dir, "cifar10_val.rec"))
    download_file('http://data.mxnet.io/data/cifar10/cifar10_val.rec', fnames[1])
    download_file('http://data.mxnet.io/data/cifar10/cifar10_train.rec', fnames[0])
    return fnames

# if __name__ == '__main__':
# download data
(train_fname, val_fname) = download_cifar10()

# data setup
# parse args
parser = argparse.ArgumentParser(description= "train cifar10",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
#fit.add_fit_args(parser)
data.add_data_args(parser)
data.add_data_aug_args(parser)
data.set_data_aug_level(parser, 2)
parser.set_defaults(
    # network
    network        = 'resnet',
    num_layers     = 110,
    # data
    data_train     = train_fname,
    data_val       = val_fname,
    num_classes    = 10,
    num_examples  = 50000,
    image_shape    = '3,28,28',
    pad_size       = 4,
    # train
    batch_size     = 128,
    num_epochs     = 300,
    lr             = .05,
    lr_step_epochs = '200,250',
)
args = parser.parse_args()
image_shape = '3,28,28'
# load network
from importlib import import_module
net = import_module('symbols.resnet')
sym = net.get_symbol(10, 38, image_shape, conv_workspace=256)

mod = mx.mod.Module(sym, context=[mx.gpu(0)])
train, val = data.get_rec_iter(args)
model_prefix = '/home/ubuntu/results/cif10_r38'
checkpoint = mx.callback.do_checkpoint(model_prefix)


## this part only to resume training

# sym, arg_params, aux_params = mx.model.load_checkpoint(model_prefix, n_epoch_load)

#  # assign parameters
# mod.set_params(arg_params, aux_params)
schedule = [40000,60000,80000]
#lr_schedule=

 
# train
mod.fit(train, eval_data=val, epoch_end_callback=checkpoint, optimizer_params={'learning_rate':0.1, 'momentum': 0.9,'wd':0.0004, 'lr_scheduler': mx.lr_scheduler.MultiFactorScheduler(step=schedule,factor=0.1) }, num_epoch=300)
# train loading form nepoch
# mod.fit(train, eval_data=val, epoch_end_callback=checkpoint, optimizer_params={'learning_rate':0.1, 'momentum': 0.9,'wd':0.0004}, num_epoch=300, arg_params=arg_params, aux_params=aux_params,
#         begin_epoch=n_epoch_load)
