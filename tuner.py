import os
import argparse
import numpy as np
import pandas as pd
import random
import time

import matplotlib.pyplot as plt
plt.style.use('science')
from statistics import mean, stdev

import logging
logging.basicConfig(level=logging.CRITICAL)


## set visible devices
parser = argparse.ArgumentParser(description='Blocktype benchmarking')
parser.add_argument('--GPU', default='1', type=str)
parser.add_argument('--seed', default=1, type=int)

parser.add_argument('--warmup_iters', default=10, type=int)
parser.add_argument('--iters', default=500, type=int)

args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.GPU

import torch
import torch.nn as nn

import tvm
from tvm import te
from tvm import autotvm
from tvm import relay
import tvm.relay.testing
from tvm.autotvm.tuner import XGBTuner, GATuner, RandomTuner, GridSearchTuner
from tvm.contrib.util import tempdir
import tvm.contrib.graph_runtime as runtime

from tvm import autotvm

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

import torchvision.models as models
import timm ### https://github.com/rwightman/pytorch-image-models

df = pd.read_csv('results/table.csv')

configs = {
    'EfficientNet-B0': timm.create_model('efficientnet_b0',scriptable=True),
    'EfficientNet-B1': timm.create_model('efficientnet_b1',scriptable=True),
    'EfficientNet-B2': timm.create_model('efficientnet_b2',scriptable=True),
}

IMAGE_SIZE = (1,3,224,224)
inf_times = []

# You can skip the implementation of this function for this tutorial.
def tune_tasks(tasks,
               measure_option,
               tuner='xgb',
               n_trial=1000,
               early_stopping=None,
               log_filename='tuning.log',
               use_transfer_learning=True):
    # create tmp log file
    tmp_log_file = log_filename + ".tmp"
    if os.path.exists(tmp_log_file):
        os.remove(tmp_log_file)

    for i, tsk in enumerate(reversed(tasks)):
        prefix = "[Task %2d/%2d] " %(i+1, len(tasks))

        tuner_obj = XGBTuner(tsk, loss_type='rank')

        # do tuning
        tsk_trial = min(n_trial, len(tsk.config_space))
        tuner_obj.tune(n_trial=tsk_trial,
                       early_stopping=early_stopping,
                       measure_option=measure_option,
                       callbacks=[
                       autotvm.callback.progress_bar(tsk_trial, prefix=prefix),
                       autotvm.callback.log_to_file(tmp_log_file)
                       ])

    # pick best records to a cache file
    autotvm.record.pick_best(tmp_log_file, log_filename)
    os.remove(tmp_log_file)

for name, config in configs.items():
    if 'Efficient' in name or 'Mix' in name:
        net = config.eval()
    else:
        net = config().eval()

    x = torch.rand(IMAGE_SIZE)

    ## script the model
    scripted_model = torch.jit.trace(net, x).eval()
    img = x.detach().cpu().numpy()

    input_name = 'input0'
    shape_list = [(input_name, img.shape)]
    mod, params = relay.frontend.from_pytorch(scripted_model,shape_list)

    target = 'cuda'
    target_host = 'llvm'
    ctx = tvm.gpu()
    with tvm.transform.PassContext(opt_level=3):
        graph, lib, params = relay.build(mod,
                                         target=target,
                                         target_host=target_host,
                                         params=params)

    #### DEVICE CONFIG ####
    target = tvm.target.cuda()

    #### TUNING OPTION ####
    network = name
    log_file = "logs/%s.log" % network
    dtype = 'float32'

    tuning_opt = {
        'log_filename': log_file,

        'tuner': 'xgb',
        'n_trial': 100,
        'early_stopping': 60,

        'measure_option': autotvm.measure_option(
            builder=autotvm.LocalBuilder(timeout=10),
            runner=autotvm.LocalRunner(number=20, repeat=3, timeout=4, min_repeat_ms=150),
        ),
    }

    tasks = autotvm.task.extract_from_program(mod["main"], target=target,
                                              params=params,
                                              ops=(relay.op.get("nn.conv2d"), relay.op.get("nn.dense")))

    tune_tasks(tasks, **tuning_opt)


df['torchscript'] = pd.Series(inf_times, index=df.index)

df.to_pickle('results/table_with_speeds.pd')
