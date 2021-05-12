import librosa
import numpy as np
from segan.utils import *
import glob
import timeit
import argparse
from scipy.io import wavfile

# eval expanded noisy testset with composite metrics
#NOISY_TEST_PATH = 'data/expanded_segan1_additive/noisy_testset'

def main(opts):
    NOISY_TEST_PATH = opts.test_wavs
    CLEAN_TEST_PATH = opts.clean_wavs
    print(NOISY_TEST_PATH)
    noisy_wavs = glob.glob(os.path.join(NOISY_TEST_PATH, '*.wav'))
    metrics = {'csig':[], 'cbak':[], 'covl':[], 'pesq':[], 'ssnr':[], 'stoi':[]}
    timings = []
    #out_log = open('eval_noisy.log', 'w')
    out_log = open(opts.logfile, 'w')
    out_log.write('FILE \t      CSIG  CBAK  COVL  PESQ  SSNR\n')
    # print('--- 0 -----')
    # print(type(noisy_wavs))
    # print(noisy_wavs)
    for n_i, noisy_wav in enumerate(noisy_wavs, start=1):
        # print('--- 1 -----')
        bname = os.path.splitext(os.path.basename(noisy_wav))[0]
        # print('--- 2 -----')
        clean_wav = os.path.join(CLEAN_TEST_PATH, bname + '.wav')
        # print('--- 3 -----')
        # noisy, rate = librosa.load(noisy_wav, 16000)
        # clean, rate = librosa.load(clean_wav, 16000)
        # print(noisy_wav)
        # print(clean_wav)
        rate, noisy = wavfile.read(noisy_wav)
        rate, clean = wavfile.read(clean_wav)
        # print(type(noisy)) # <class 'numpy.ndarray'>
        # print(type(clean))  # <class 'numpy.ndarray'>

        beg_t = timeit.default_timer()
        # print('ok 1')
        csig, cbak, covl, pesq, ssnr, stoi = CompositeEval(clean, noisy, True)
        # print('ok 2')
        end_t = timeit.default_timer()
        timings.append(end_t - beg_t)
        metrics['csig'].append(csig)
        metrics['cbak'].append(cbak)
        metrics['covl'].append(covl)
        metrics['pesq'].append(pesq)
        metrics['ssnr'].append(ssnr)
        metrics['stoi'].append(stoi)
        out_log.write('{} {:.3f} {:.3f} {:.3f} {:.3f} {:.3} {:.3}\n'.format(bname + '.wav',
                                                                      csig, 
                                                                      cbak, 
                                                                      covl,
                                                                      pesq,
                                                                      ssnr,
                                                                      stoi))
        print('Processed {}/{} wav, -> CSIG:{:.3f}, CBAK:{:.3f}, COVL:{:.3f}, '
              'PESQ:{:.3f}, SSNR:{:.3f}, STOI:{:.3f}, -> '
              'total time: {:.2f} seconds, mproc: {:.2f}'
              ' seconds'.format(n_i, len(noisy_wavs), csig, cbak, covl,
                                pesq, ssnr, stoi,
                                np.sum(timings),
                                np.mean(timings)))
        # print('ok 3')
        # if n_i ==100:
        #     break
    out_log.close()

    print('mean PESQ: ', np.mean(metrics['pesq']))
    print('mean CSIG: ', np.mean(metrics['csig']))
    print('mean CBAK: ', np.mean(metrics['cbak']))
    print('mean COVL: ', np.mean(metrics['covl']))

    print('mean SSNR: ', np.mean(metrics['ssnr']))
    print('mean STOI: ', np.mean(metrics['stoi']))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--test_wavs', type=str, default= './noisy_testset_wav' )
    # parser.add_argument('--clean_wavs', type=str, default= './clean_testset_wav')

    parser.add_argument('--test_wavs', type=str, default='../SpeechDatasets/objectiveEvaluation/topogan/')
    parser.add_argument('--clean_wavs', type=str, default='../SpeechDatasets/objectiveEvaluation/downSample_clean_testset_wav/')
    # parser.add_argument('--test_wavs', type=str, default='../Speech Datasets/noisy_testest_wave/')
    # parser.add_argument('--clean_wavs', type=str, default='../Speech Datasets/clean_testest_wave/')
    parser.add_argument('--logfile', type=str, default= './dataset/evaluation_log.txt')

    opts = parser.parse_args()

    assert opts.test_wavs is not None
    assert opts.clean_wavs is not None
    print(opts.logfile)
    assert opts.logfile is not None

    main(opts)

'''
https://wkcosmology.github.io/2018/11/06/note-cython/Cython-2/

python setup.py build_ext --inplace

'''