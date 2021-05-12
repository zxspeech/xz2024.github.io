
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
def downsampleWav(src, dst, inrate=44100, outrate=16000, inchannels=2, outchannels=1):
    if not os.path.exists(src):
        print( 'Source not found!')
        return False

    if not os.path.exists(os.path.dirname(dst)):
        os.makedirs(os.path.dirname(dst))

    try:
        s_read = wave.open(src, 'r')
        s_write = wave.open(dst, 'w')
    except:
        print( 'Failed to open files!')
        return False

    n_frames = s_read.getnframes()
    data = s_read.readframes(n_frames)

    try:
        converted = audioop.ratecv(data, 2, inchannels, inrate, outrate, None)
        if outchannels == 1:
            converted = audioop.tomono(converted[0], 2, 1, 0)
    except:
        print( 'Failed to downsample wav')
        return False

    try:
        s_write.setparams((outchannels, 2, outrate, 0, 'NONE', 'Uncompressed'))
        s_write.writeframes(converted)
    except:
        print( 'Failed to write wav')
        return False

    try:
        s_read.close()
        s_write.close()
    except:
        print( 'Failed to close wav files')
        return False

    return True


# https://stackoverflow.com/questions/30619740/downsampling-wav-audio-file
def main(opts):
    # assert opts.cfg_file is not None
    # assert opts.inputFolder is not None
    assert opts.samplerate is not None
    assert opts.outPutFolder is not None
    # process every wav in the inputFolder
    # print('-----> 1: ', type(opts.inputFolder))
    # print('-----> 2: ', opts.inputFolder)
    if len(opts.inputFolder) == 1:
        # assume we read directory
        twavs = glob.glob(os.path.join(opts.inputFolder[0], '*.wav'))
        # print('-----> L50, clean.py: ', twavs)
    else:
        # assume we have list of files in input
        twavs = opts.inputFolder
        print('-----> L54, clean.py: ', twavs)

    print('DownSampling {} wavs'.format(len(twavs)))
    beg_t = timeit.default_timer()
    root = opts.outPutFolder + '/'
    for t_i, twav in enumerate(twavs, start=1):
        tbname = os.path.basename(twav)

        filename = root + tbname
        sampling_rate, data = wavfile.read(twav)
        # downsampleWav(twav, filename, inrate=48000, outrate=16000, inchannels=1, outchannels=1)

        # Your new sampling rate
        new_rate = 16000
        sampling_rate, data = wavfile.read(twav)

        # Resample data
        number_of_samples = round(len(data) * float(new_rate) / sampling_rate)
        data = sps.resample(data, number_of_samples)

        wavfile.write(filename, opts.samplerate, data.astype(np.int16))
        # rate, wav = wavfile.read(filename)
        # print(rate)
        # print('------------------')
        # if t_i == 4:
        #     break


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--outPutFolder', type=str, default= '../Speech Datasets/downSample_clean_testset_wav')
    parser.add_argument('--samplerate', type=int, default=16000, help=" Sample Rate")
    parser.add_argument('--inputFolder', type=list, nargs='+', default=list(['../Speech Datasets/clean_testset_wav/']))


    opts = parser.parse_args()
    main(opts)
