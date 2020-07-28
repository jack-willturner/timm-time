import os
import argparse
import numpy as np
import pandas as pd
import random
import time
from utils import *

import matplotlib.pyplot as plt
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


torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

import torchvision.models as models
import timm ### https://github.com/rwightman/pytorch-image-models

df = pd.read_pickle('results/table_with_speeds.pd')

IMAGE_SIZE = (1,3,224,224)

inf_times = []
flops     = []
params    = []

df = pd.read_pickle('results/table_with_speeds.pd')


print("AutoTVM ")

configs = {
    'EfficientNet-B0': timm.create_model('efficientnet_b0',scriptable=True),
    'EfficientNet-B1': timm.create_model('efficientnet_b1',scriptable=True),
    'EfficientNet-B2': timm.create_model('efficientnet_b2',scriptable=True),
}

inf_times = {k: [] for k in configs.keys()}

import tvm
from tvm import relay
from tvm.contrib.util import tempdir

from tvm import autotvm

for name, config in configs.items():
    if ('EfficientNet' not in name) and ('MixNet' not in name):
        net = config().eval()
    else:
        net = config.eval()

    x   = torch.rand((1,3,224,224))

    ## script the model
    scripted_model = torch.jit.trace(net, x).eval()

    img = x.detach().numpy()

    input_name = 'input0'
    shape_list = [(input_name, img.shape)]
    mod, params = relay.frontend.from_pytorch(scripted_model,
                                              shape_list)

    target = 'cuda'
    target_host = 'llvm'
    ctx = tvm.gpu()


    tasks = autotvm.task.extract_from_program(mod["main"], target=target,
                                              params=params,
                                              ops=(relay.op.get("nn.conv2d"), relay.op.get("nn.dense")))


    from tvm.contrib import graph_runtime
    dtype = 'float32'

    with autotvm.apply_history_best(f'logs/{name}.log'):
        with tvm.transform.PassContext(opt_level=3):
            graph, lib, params = relay.build_module.build(mod,
                                             target=target,
                                             target_host=target_host,
                                             params=params)

        # export library
        tmp = tempdir()
        filename = "net.tar"
        lib.export_library(tmp.relpath(filename))

        m = graph_runtime.create(graph, lib, ctx)

        # Set inputs
        m.set_input(input_name, tvm.nd.array(img.astype(dtype)))
        m.set_input(**params)

        for i in range(args.warmup_iters):
            # Execute
            m.run()
            # Get outputs
            tvm_output = m.get_output(0)


        times = []
        for i in range(args.iters):
            t1 = time.time()
            # Execute
            m.run()
            # Get outputs
            tvm_output = m.get_output(0)
            t2 = time.time()

            times.append(t2-t1)

        inf_times[name] = [t*1000 for t in times]
        print(f"\t{name}:\t{mean(times):.2f} +- {stdev(times):.2f}")

print(inf_times)

df['TVM'] = inf_times.values()
print('pickling')

print(df)

df.to_csv('results/table_with_speeds.csv')
df.to_pickle('results/table_with_speeds.pd')
