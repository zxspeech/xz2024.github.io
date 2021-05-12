from __future__ import print_function
import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import default_collate
import os
import glob
import json
import gzip
import pickle
import timeit
import scipy.io.wavfile as wavfile
import numpy as np
import multiprocessing as mp
import random
import librosa
from ahoproc_tools.io import *
from ahoproc_tools.interpolate import *
import h5py
# from ..utils import *
from segan.utils import *
from torch.utils.data import DataLoader
import numpy as np
from scipy.io import wavfile
from matplotlib import pyplot as plt


def collate_fn(batch):
    # first we have utt bname, then tensors
    data_batch = []
    uttname_batch = []
    for sample in batch:
        uttname_batch.append(sample[0])
        data_batch.append(sample[1:])
    data_batch = default_collate(data_batch)
    return [uttname_batch] + data_batch

def slice_signal(signal, window_sizes, stride=0.5):
    """ Slice input signal

        # Arguments
            window_sizes: list with different sizes to be sliced
            stride: fraction of sliding window per window size

        # Returns
            A list of numpy matrices, each one being of different window size
    """
    assert signal.ndim == 1, signal.ndim
    n_samples = signal.shape[0]
    slices = []
    for window_size in window_sizes:
        offset = int(window_size * stride)
        slices.append([])
        for beg_i in range(n_samples + offset, offset):
            end_i = beg_i + offset
            if end_i > n_samples:
                # last slice is offset to past to fit full window
                beg_i = n_samples - offset
                end_i = n_samples
            slice_ = signal[beg_i:end_i]
            assert slice_.shape[0] == window_size, slice_.shape[0]
            slices[-1].append(slice_)
        slices[-1] = np.array(slices[-1], dtype=np.int32)
    return slices

def slice_index_helper(args):
    return slice_signal_index(*args)

def slice_signal_index(path, window_size, stride):
    """ Slice input signal into indexes (beg, end) each

        # Arguments
            window_size: size of each slice
            stride: fraction of sliding window per window size

        # Returns
            A list of tuples (beg, end) sample indexes
    """
    signal, rate = librosa.load(path, 16000)    # xudong: does not return signal, return indices.
    assert stride <= 1, stride
    assert stride > 0, stride
    assert signal.ndim == 1, signal.ndim
    n_samples = signal.shape[0]
    slices = []
    offset = int(window_size * stride)
    for beg_i in range(0, n_samples - (offset), offset):
        end_i = beg_i + window_size
        slice_ = (beg_i, end_i)
        slices.append(slice_)
    return slices

def abs_normalize_wave_minmax(x):
    x = x.astype(np.int32)
    imax = np.max(np.abs(x))
    x_n = x / imax
    return x_n

def abs_short_normalize_wave_minmax(x):
    imax = 32767.
    x_n = x / imax
    return x_n

def dynamic_normalize_wave_minmax(x):   # 归一化到 0 ～ +1。
    x = x.astype(np.int32)
    imax = np.max(x)
    imin = np.min(x)
    x_n = (x - np.min(x)) / (float(imax) - float(imin))
    return x_n * 2 - 1

def normalize_wave_minmax(x):   # 归一化到 -1 ～ +1。
    return (2./65535.) * (x - 32767.) + 1.

def reverse_normalize_wave_minmax(x):   # 归一化到 -1 ～ +1。
    return np.int16((x+1)*65535./2 - 32767.)

def pre_emphasize(x, coef=0.95):
    if coef <= 0:
        return x
    # print(x.shape)
    x0 = np.reshape(x[0], (1,))


    diff = x[1:] - coef * x[:-1]
    concat = np.concatenate((x0, diff), axis=0)
    print(x.shape)
    print(diff.shape)
    print(concat.shape)
    # return concat
    return concat

def de_emphasize(y, coef=0.95):
    if coef <= 0:
        return y
    x = np.zeros(y.shape[0], dtype=np.float32)
    x[0] = y[0]
    for n in range(1, y.shape[0], 1):
        x[n] = coef * x[n - 1] + y[n]
    return x

class SEDataset(Dataset):
    """ Speech enhancement dataset """
    def __init__(self, clean_dir, noisy_dir,
                 preemph, cache_dir='.',
                 split='train', slice_size=2**14,
                 stride = 0.5, max_samples=None, do_cache=False, verbose=False,
                 slice_workers=2, preemph_norm=False,
                 random_scale=[1]):
        super(SEDataset, self).__init__()
        print('1. Creating {} split out of data in {}'.format(split, clean_dir))
        self.clean_names = glob.glob(os.path.join(clean_dir, '*.wav'))
        self.noisy_names = glob.glob(os.path.join(noisy_dir, '*.wav'))
        print('2. Found {} clean names and {} noisy names'.format(len(self.clean_names), len(self.noisy_names)))


        self.slice_workers = slice_workers
        if len(self.clean_names) != len(self.noisy_names) or \
           len(self.clean_names) == 0:
            raise ValueError('No wav data found! Check your data path please')
        if max_samples is not None:
            assert isinstance(max_samples, int), type(max_samples)
            self.clean_names = self.clean_names[:max_samples]
            self.noisy_names = self.noisy_names[:max_samples]
        # path to store pairs of wavs
        self.cache_dir = cache_dir
        self.slice_size = slice_size
        self.stride = stride
        self.split = split
        self.verbose = verbose
        self.preemph = preemph
        # order is preemph + norm (rather than norm + preemph)
        self.preemph_norm = preemph_norm
        # random scaling list, selected per utterance
        self.random_scale = random_scale
        #self.read_wavs()
        cache_path = cache_dir#os.path.join(cache_dir, '{}_chunks.pkl'.format(split))
        #if os.path.exists(cache_path):
        #    with open(cache_path ,'rb') as ch_f:
        #        self.slicings = pickle.load(ch_f)
        #else:

        debugging("169", 'se_dataset.py')
        if not os.path.exists(cache_path):
            os.makedirs(cache_path)


        debugging("173", 'se_dataset.py')
        if not os.path.exists(os.path.join(cache_path, '{}_idx2slice.pkl'.format(split))):
            # make the slice indexes given slice_size and stride
            self.prepare_slicing()
            debugging("177", 'se_dataset.py')


            #with open(os.path.join(cache_path, '{}_cache.cfg'.format(split)), 'w') as cfg_f:
            #    cfg_f.write(json.dumps({'num_slicings':len(self.slicings)}))
            with open(os.path.join(cache_path, '{}_idx2slice.pkl'.format(split)), 'wb') as i2s_f:
                pickle.dump(self.idx2slice, i2s_f)
            #if do_cache:

            for s_i, slicing in self.slicings.items():
                with open(os.path.join(cache_path, '{}_{}.pkl'.format(split, s_i)), 'wb') as ch_f:
                    # store slicing results
                    pickle.dump(slicing, ch_f)
            self.num_samples = len(self.idx2slice)
            self.slicings = None
            debugging("188", 'se_dataset.py')
        else:
            #with open(os.path.join(cache_path, '{}_cache.cfg'.format(split)), 'r') as cfg_f:
            #    self.num_samples = json.load(cfg_f)
            with open(os.path.join(cache_path, '{}_idx2slice.pkl'.format(split)), 'rb') as i2s_f:
                self.idx2slice = pickle.load(i2s_f)
            print('Loaded {} idx2slice items'.format(len(self.idx2slice)))
            debugging("195", 'se_dataset.py')

    def read_wav_file(self, wavfilename):
        rate, wav = wavfile.read(wavfilename)
        # print('----', wav.dtype)
        if self.preemph_norm:
            wav = pre_emphasize(wav, self.preemph)
            wav = normalize_wave_minmax(wav)
        else:
            wav = normalize_wave_minmax(wav)
            wav = pre_emphasize(wav, self.preemph)
            # print('-----', rate)    # ----- 16000
            # print(type(wav))    # <class 'numpy.ndarray'>
        return rate, wav

    def read_wavs(self):
        #self.clean_wavs = []
        self.clean_paths = []
        #self.noisy_wavs = []
        self.noisy_paths = []
        clen = len(self.clean_names)
        nlen = len(self.noisy_names)
        assert clen == nlen, clen
        if self.verbose:
            print('< Reading {} wav files... >'.format(clen))
        beg_t = timeit.default_timer()
        for i, (clean_name, noisy_name) in enumerate(zip(self.clean_names, self.noisy_names), start=1):
            self.clean_paths.append(clean_name)

            #n_rate, n_wav = self.read_wav_file(noisy_name)
            #self.noisy_wavs.append(n_wav)
            self.noisy_paths.append(noisy_name)
        end_t = timeit.default_timer()
        if self.verbose:
            print('> Loaded files in {} s <'.format(end_t - beg_t))

    def read_wavs_and_cache(self):
        """ Read in all clean and noisy wavs """
        cache_path = os.path.join(self.cache_dir, 'cached_pair.pkl')
        try:
            with open(cache_path) as f_in:
                cache = pickle.load(f_in)
                if self.verbose:
                    print('Reading clean and wav pair from ', cache_path)
                self.clean_wavs = cache['clean']
                self.noisy_wavs = cache['noisy']
        except IOError:
            self.read_wavs()
            cache = {'noisy':self.noisy_wavs, 'clean':self.clean_wavs}
            if not os.path.exists(self.cache_dir):
                os.makedirs(self.cache_dir)
            with open(cache_path, 'wb') as f_out:
                pickle.dump(cache, f_out)
                if self.verbose:
                    print('Cached clean and wav pair into ', cache_path)

    def prepare_slicing(self):
        """ Make a dictionary containing, for every wav file, its
            slices performed sequentially in steps of stride and
            sized slice_size
        """
        slicings = {}
        idx2slice = []
        verbose = self.verbose
        if verbose:
            print('< Slicing all signals with window {} and stride {}... >'.format(self.slice_size, self.stride))


        beg_t = timeit.default_timer()
        pool = mp.Pool(self.slice_workers)

        clean_args = [(self.clean_names[i], self.slice_size, self.stride) for \
                      i in range(len(self.clean_names))]
        c_slices = pool.map(slice_index_helper, clean_args)

        noisy_args = [(self.noisy_names[i], self.slice_size, self.stride) for \
                      i in range(len(self.noisy_names))]
        n_slices = pool.map(slice_index_helper, noisy_args)

        if len(n_slices) != len(c_slices):
            raise ValueError('n_slices and c_slices have different lengths:'
                             '{} != {}'.format(len(n_slices), len(c_slices)))

        for w_i, (c_slice, n_slice) in enumerate(zip(c_slices, n_slices)):
            c_path = self.clean_names[w_i]
            n_path = self.noisy_names[w_i]
            if w_i not in slicings:
                slicings[w_i] = []
            for t_i, (c_ss, n_ss) in enumerate(zip(c_slice, n_slice)):
                if c_ss[1] - c_ss[0] < 1024:
                    # decimate less than 4096 samples window
                    continue
                slicings[w_i].append({'c_slice':c_ss,
                                      'n_slice':n_ss,
                                      'c_path':c_path,
                                      'n_path':n_path,
                                      'slice_idx':t_i})
                idx2slice.append((w_i, t_i))

        self.slicings = slicings
        self.idx2slice = idx2slice
        end_t = timeit.default_timer()
        if verbose:
            print('Sliced all signals in {} s'.format(end_t - beg_t))

    # extract_slice ---> read_wav_file
    def extract_slice(self, index): # use wavefile.read to load wav file, then slice it.
        # load slice
        s_i, e_i = self.idx2slice[index]
        #print('selected item: ', s_i, e_i)
        slice_file = os.path.join(self.cache_dir, '{}_{}.pkl'.format(self.split, s_i))
        #print('reading slice file: ', slice_file)
        with open(slice_file, 'rb') as s_f:
            slice_ = pickle.load(s_f)
            #print('slice_: ', slice_)
            slice_ = slice_[e_i]
            c_slice_, n_slice_ = slice_['c_slice'], slice_['n_slice']
            slice_idx = slice_['slice_idx']
            n_path = slice_['n_path']
            bname = os.path.splitext(os.path.basename(n_path))[0]
            met_path = os.path.join(os.path.dirname(n_path), bname + '.met')
            ssnr = None
            pesq = None
            if os.path.exists(met_path):
                metrics = json.load(open(met_path, 'r'))
                pesq = metrics['pesq']
                ssnr = metrics['ssnr']
            #c_signal, rate = librosa.load(slice_['c_path'])    # 其中的load函数就是用来读取音频的。当然，读取之后，转化为了numpy的格式储存，而不再是音频的格式了。
            #n_signal, rate = librosa.load(slice_['n_path'])
            c_signal = self.read_wav_file(slice_['c_path'])[1]
            n_signal = self.read_wav_file(slice_['n_path'])[1]
            #c_signal = self.clean_wavs[idx_]
            #n_signal = self.noisy_wavs[idx_]
            c_slice = c_signal[c_slice_[0]:c_slice_[1]]
            n_slice = n_signal[n_slice_[0]:n_slice_[1]]
            if n_slice.shape[0] > c_slice.shape[0]:
                n_slice = n_slice[:c_slice.shape[0]]
            if c_slice.shape[0] > n_slice.shape[0]:
                c_slice = c_slice[:n_slice.shape[0]]
            #print('c_slice[0]: {} c_slice[1]: {}'.format(c_slice_[0],
            #                                             c_slice_[1]))
            if c_slice.shape[0] < self.slice_size:
                pad_t = np.zeros((self.slice_size - c_slice.shape[0],))
                c_slice = np.concatenate((c_slice, pad_t))
                n_slice = np.concatenate((n_slice, pad_t))
            #print('c_slice shape: ', c_slice.shape)
            #print('n_slice shape: ', n_slice.shape)
            bname = os.path.splitext(os.path.basename(n_path))[0]
            return c_slice, n_slice, pesq, ssnr, slice_idx, bname

# 1.  __getitem__(self, index) --->  self.extract_slice(index).
    def __getitem__(self, index):
        c_slice, n_slice, pesq, ssnr, slice_idx, bname = self.extract_slice(index)
        rscale = random.choice(self.random_scale)
        if rscale != 1:

            # n_slice = rscale * (n_slice - c_slice)
            n_slice = rscale * n_slice
            c_slice = rscale * c_slice
        n_slice = n_slice - c_slice # xudong add here!!!
        # print("xudong type: ", type(c_slice), type(n_slice))    #-> xudong type:  <class 'numpy.ndarray'> <class 'numpy.ndarray'>
        returns = [bname, torch.FloatTensor(c_slice), torch.FloatTensor(n_slice), slice_idx]
        # print("------", pesq, ssnr) # None None.
        if pesq is not None:
            returns.append(torch.FloatTensor([pesq]))
        if ssnr is not None:
            returns.append(torch.FloatTensor([ssnr]))
        # print('idx: {} c_slice shape: {}'.format(index, c_slice.shape))
        return returns

    def __len__(self):
        return len(self.idx2slice)

if __name__ == '__main__':

    cleanFile = '../../trainingDataSet/clean_trainset_wav'
    noisyFile = '../../trainingDataSet/noisy_trainset_wav'
    wavname = '/p287_002.wav'
    clean = cleanFile + wavname
    noisy = noisyFile + wavname

    dset = SEDataset(cleanFile, noisyFile, 0.95,
                     cache_dir='../../trainingDataSet/cache', max_samples=100, verbose=True)
    dloader = DataLoader(dset, batch_size= 100,
                         shuffle=True, num_workers=1,
                         pin_memory= False,
                         collate_fn=collate_fn)
    # sample_0 = dset.__getitem__(0)
    # print('sample_0: ', sample_0)

    for bidx, batch in enumerate(dloader, start=1):
        uttname, clean, noisy, slice_idx = batch
        print("uttname: ", uttname)  # -> <class 'list'>, ['p287_002', 'p287_001', 'p287_002', 'p287_002', 'p287_002', 'p287_001', 'p287_001', 'p287_002', 'p287_002']
        print("clean: ", type(clean))  # -> <class 'torch.Tensor'>
        print("noisy: ", type(noisy))  # -> <class 'torch.Tensor'>
        # clean_numpy = clean.numpy()
        # print("clean_numpy: ", type(clean_numpy))
        # print("clean_numpy.shape: ", clean_numpy.shape)
        # print(clean_numpy[0][1000:1005])

        # a = torch.ones(5)
        # print(type(a))  # <class 'torch.Tensor'>
        # b = a.numpy()
        # print(type(b))  # <class 'numpy.ndarray'>

        numpy_clean = clean.numpy()
        numpy_noise = noisy.numpy()
        print(type(numpy_clean))
        print(type(numpy_noise))
        print(numpy_clean.shape)
        print(numpy_noise.shape)

        # # y, sr = librosa.load(noisy, sr=None)
        # # librosa.display.waveplot(y, sr)
        # #
        # # y, sr = librosa.load(numpy_clean[0], sr=None)
        # # librosa.display.waveplot(y, sr)
        # plt.plot(numpy_clean[0])
        # plt.title('wavform')
        # plt.show()
        # plt.plot(numpy_noise[0])
        # plt.title('wavform1')
        # plt.show()
        scaled = reverse_normalize_wave_minmax(numpy_clean[0])
        wavfile.write('numpy_clean.wav', 16000, scaled)

        scaled = reverse_normalize_wave_minmax(numpy_noise[0])
        wavfile.write('numpy_noise.wav', 16000, scaled)
        break

'''
    # def read_wav_file(self, wavfilename):
    #     rate, wav = wavfile.read(wavfilename)
    #     print('----', wav.dtype)
    #     if self.preemph_norm:
    #         wav = pre_emphasize(wav, self.preemph)
    #         wav = normalize_wave_minmax(wav)
    #     else:
    #         wav = normalize_wave_minmax(wav)
    #         wav = pre_emphasize(wav, self.preemph)
    #         print('-----', rate)
    #         print(type(wav))
    #     return rate, wav
    
    # https: // docs.scipy.org / doc / scipy / reference / generated / scipy.io.wavfile.read.html
    rate, wav_clean = wavfile.read(clean)
    wav_clean = normalize_wave_minmax(wav_clean)
    wav_clean_em = pre_emphasize(wav_clean, 0.95)
    # print('1 ', wav_clean.dtype)    # float32, -1.0 ~ +1.0
    #
    rate, wav_noisy = wavfile.read(noisy)
    wav_noisy = normalize_wave_minmax(wav_noisy)
    wav_noisy_em = pre_emphasize(wav_noisy, 0.95)
    # # print('2 ', wav_noisy.dtype)    # int16, -32768 ~ +32767
    # # print(type(wav_noisy))  # <class 'numpy.ndarray'>
    # # print(wav_noisy.shape)
    # # plt.plot(wav_noisy)
    # plt.plot(x, 'b')
    # plt.show()
    # plt.plot(wav_noisy_em, 'r')
    # plt.show()

    # data = np.random.uniform(-1, 1, 44100)  # 44100 random samples between -1 and 1
    # scaled = np.int16(data / np.max(np.abs(data)) * 32767)
    scaled = reverse_normalize_wave_minmax(wav_noisy_em)
    wavfile.write('wav_noisy_em.wav', 16000, scaled)

    scaled = reverse_normalize_wave_minmax(wav_noisy_em - wav_clean_em)
    wavfile.write('x.wav', 16000, scaled)

'''







    # dset = RandomChunkSEF0Dataset('../../data/silent/clean_trainset',
    #                              '../../data/silent/lf0_trainset', 0.)
    # sample_0 = dset.__getitem__(0)
    # print('len sample_0: ', len(sample_0))



    # dset = SEH5Dataset('../../data/widebandnet_h5/speaker1', 'train',
    #                    0.95, verbose=True)
    # print(len(dset))
