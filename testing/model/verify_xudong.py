
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
import glob
import os
from scipy import signal
import scipy.signal as sps
# from lib import wave
import wave
import audioop

'''
    set the input and output folder, then run the clean.py to generatethe clean wav files.
'''


# https://stackoverflow.com/questions/30619740/downsampling-wav-audio-file
def main(opts):
    # assert opts.cfg_file is not None
    assert opts.original_files_folder is not None
    assert opts.samplerate is not None
    assert opts.outPutFolder is not None
    # process every wav in the original_files_folder
    # print('-----> 1: ', type(opts.original_files_folder))
    # print('-----> 2: ', opts.original_files_folder)
    if len(opts.original_files_folder) == 1:
        # assume we read directory
        twavs = glob.glob(os.path.join(opts.original_files_folder[0], '*.wav'))
        outwavs = glob.glob(os.path.join(opts.outPutFolder[0], '*.wav'))
        print('-----> L50, clean.py: ', len(opts.original_files_folder))
        print('-----> L50, clean.py: ', len(opts.outPutFolder))
    else:
        # assume we have list of files in input
        twavs = opts.original_files_folder
        outwavs = opts.outPutFolder
        print('-----> L54, clean.py: ', twavs)

    print('DownSampling {} wavs'.format(len(twavs)))

    # root = opts.outPutFolder + '/'
    for twav, outwav in zip(twavs, outwavs):
        # tbname = os.path.basename(twav)
        #
        # # filename = root + tbname
        # print(twav)
        # print(outwav)
        sampling_rate1, data1 = wavfile.read(twav)
        sampling_rate2, data2 = wavfile.read(outwav)
        # assert sampling_rate1 == sampling_rate2
        # assert data1.shape == data2.shape
        print('sampling_rate1: ', sampling_rate1, ', sampling_rate2: ', sampling_rate2)
        print('data1.shape: ', data1.shape, ', data2.shape: ', data2.shape)

        #
        # # Your new sampling rate
        # new_rate = 16000
        # sampling_rate, data = wavfile.read(twav)
        #
        # # Resample data
        # number_of_samples = round(len(data) * float(new_rate) / sampling_rate)
        # data = sps.resample(data, number_of_samples)
        #
        # wavfile.write(filename, opts.samplerate, data.astype(np.int16))
        # # rate, wav = wavfile.read(filename)
        # # print(rate)
        # # print('------------------')
        # # if t_i == 4:
        # #     break


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    # parser.add_argument('--outPutFolder', type=str, default= '../Speech Datasets/downSample_noisy_testset_wav')
    parser.add_argument('--samplerate', type=int, default=16000, help=" Sample Rate")
    parser.add_argument('--original_files_folder', type=list, nargs='+', default=list(['../Speech Datasets/output_noisy_testset_wav/']))

    parser.add_argument('--outPutFolder', type=list, nargs='+',
                        default=list(['../Speech Datasets/downSample_noisy_testset_wav1/']))

    opts = parser.parse_args()
    main(opts)
