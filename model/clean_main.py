import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from segan.models import *
from segan.datasets import *
import soundfile as sf
from scipy.io import wavfile
from torch.autograd import Variable
import numpy as np
import random
import librosa
import matplotlib
import timeit
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
import glob
import os


'''
    set the input and output folder, then run the clean.py to generatethe clean wav files.
'''



class ArgParser(object):

    def __init__(self, args):
        for k, v in args.items():
            setattr(self, k, v)

def main(opts):
    assert opts.cfg_file is not None
    assert opts.test_files is not None
    assert opts.g_pretrained_ckpt is not None

    with open(opts.cfg_file, 'r') as cfg_f:
        args = ArgParser(json.load(cfg_f))
        # print('Loaded train config: ')  # xudong hide this line.
        # print(json.dumps(vars(args), indent=2)) # xudong hide this line.
    args.cuda = opts.cuda
    print('------------ 0 ------------')
    if hasattr(args, 'wsegan') and args.wsegan:
        segan = WSEGAN(args)
        print('------------ 0.1 ------------')
    else:
        segan = SEGAN(args) # <--- 'ckpt_segan+/train.opts'
        print('------------ 0.2 ------------')
    print('------------ 1 ------------')
    segan.G.load_pretrained(opts.g_pretrained_ckpt, True)
    print('------------- 2 -----------')
    if opts.cuda:
        segan.cuda()
    segan.G.eval()


    # process every wav in the test_files
    print('-----> 1: ', type(opts.test_files))
    print('-----> 2: ', opts.test_files)
    if len(opts.test_files) == 1:
        # assume we read directory
        twavs = glob.glob(os.path.join(opts.test_files[0], '*.wav'))
        print('-----> L50, clean.py: ', twavs)
    else:
        # assume we have list of files in input
        twavs = opts.test_files
        print('-----> L54, clean.py: ', twavs)

    print('Cleaning {} wavs'.format(len(twavs)))
    beg_t = timeit.default_timer()
    for t_i, twav in enumerate(twavs, start=1):
        tbname = os.path.basename(twav)
        # print('-----> L57, clean.py: ', twav)   # ./noisy_testset/p287_003.wav
        rate, wav = wavfile.read(twav)
        wav = normalize_wave_minmax(wav)
        wav = pre_emphasize(wav, args.preemph)
        pwav = torch.FloatTensor(wav).view(1,1,-1)
        # print('------ 1 --------')
        if opts.cuda:
            pwav = pwav.cuda()
        g_wav, g_c = segan.generate(pwav)
        # print('------ 2 --------')
        out_path = os.path.join(opts.synthesis_path, tbname)
        # print('------ 3 --------')
        if opts.soundfile:
            sf.write(out_path, g_wav, 16000)
        else:
            wavfile.write(out_path, 16000, g_wav)
        end_t = timeit.default_timer()
        print('Cleaned {}/{}: {} in {} s'.format(t_i, len(twavs), twav,
                                                 end_t-beg_t))
        beg_t = timeit.default_timer()
        print('------ finish --------')

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--g_pretrained_ckpt', type=str, default= 'ckpt_segan+/weights.ckpt')
    parser.add_argument('--cfg_file', type=str, default='ckpt_segan+/train.opts')

    parser.add_argument('--test_files', type=list, nargs='+', default= list(['../Speech Datasets/objectiveEvaluation/downSample_noisy_testset_wav8/']))
    parser.add_argument('--seed', type=int, default=111, help="Random seed (Def: 111).")
    parser.add_argument('--synthesis_path', type=str, default='../Speech Datasets/objectiveEvaluation/segan/', help='Path to save output samples (Def: segan_samples).')
    parser.add_argument('--cuda', action='store_true', default=False)
    parser.add_argument('--soundfile', action='store_true', default=False)


    opts = parser.parse_args()

    if not os.path.exists(opts.synthesis_path):
        os.makedirs(opts.synthesis_path)
    
    # seed initialization
    random.seed(opts.seed)
    np.random.seed(opts.seed)
    torch.manual_seed(opts.seed)
    if opts.cuda:
        torch.cuda.manual_seed_all(opts.seed)

    main(opts)
