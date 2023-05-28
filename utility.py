# -*- coding: utf-8 -*-
# @Desc  : setup utility, such as gpu, out_dir, etc.


def set_bestgpu(servertype='2080', parallel=False):
    from numpy import argmax, array, argwhere, argsort
    import os
    os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
    memory_gpu = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]

    memory_indexes = argsort(memory_gpu).tolist()
    mygpus = [1, 2, 3, 4, 5, 6, 7,0]
    all_gpus = ''
    for i in range(len(mygpus)):
        if i == 0:
            all_gpus += str(mygpus[i])
        else:
            all_gpus += ','
            all_gpus += str(mygpus[i])
    if parallel:
        os.environ[
            'CUDA_VISIBLE_DEVICES'] = all_gpus
    else:
        for i in reversed(memory_indexes):
            # if i == 0 or i == 2 or i == 3 or i == 4 or i == 5 or i == 6 or i == 7:
            if i in mygpus:
                os.environ['CUDA_VISIBLE_DEVICES'] = str(i)
                break

    print(f'gpu:{os.environ["CUDA_VISIBLE_DEVICES"]}')
    os.system('rm tmp')


def get_currenttime():
    import datetime
    return datetime.datetime.now().strftime("%b%d_%H-%M-%S")


def set_outdir(args):
    import sys
    sp = '/'
    args_vars = vars(args)
    current_file = sys.argv[0].split(".")[0].split('/')[-1]

    out_dir = f'runs/{current_file}'
    if args_vars.get('debug') == True:
        out_dir = f'runs/debug/{current_file}'
    for attr in args.__dict__:
        out_dir += f'{sp}{attr}{args_vars.get(attr)}'
    out_dir += f'{sp}{get_currenttime()}'
    return out_dir
