import math
import numpy as np
from scipy.fftpack import dct


def mfcc(signal,sampleRate,winSize=0.025,winStep=0.01,cepCount=13,filterCount=26,fftSize=512,lowFreq=0,highFreq=None,preemph=0.97,ceplifter=22):
    """
    Compute Mel Frequency Cepstral Coefficients features from an audio signal.
    
    :param signal     : the audio signal from which to compute features. Should be an N*1 array
    :param sampleRate : the samplerate of the signal we are working with.
    :param winSize    : the length of the analysis window in seconds. Default is 0.025s (25 milliseconds)    
    :param winStep    : the step between successive windows in seconds. Default is 0.01s (10 milliseconds)    
    :param cepCount   : the number of cepstrum to return, default 13    
    :param filterCount: the number of filters in the filterbank, default 26.
    :param fftSize    : the FFT size. Default is 512.
    :param lowFreq    : lowest band edge of mel filters. In Hz, default is 0.
    :param highFreq   : highest band edge of mel filters. In Hz, default is samplerate/2
    :param preemph    : apply preemphasis filter with preemph as coefficient. 0 is no filter. Default is 0.97. 
    :param ceplifter  : apply a lifter to final cepstral coefficients. 0 is no lifter. Default is 22. 
    
    :returns: A np array of size (NUMFRAMES by numcep) containing features. Each row holds 1 feature vector.
    """     
    
    signal = preemphasis(signal, preemph)
    
    frames = frameSignal(signal, winSize*sampleRate, winStep*sampleRate, np.hamming)
    
    pspec = powerSpectrum(frames, fftSize)
    
    fb = melFilterBank(filterCount, fftSize, sampleRate, lowFreq, highFreq)
    features = np.dot(pspec, fb.T)
    
    features = np.where(features == 0,np.finfo(float).eps,features) # if feat is zero, we get problems with log
    features = np.log(features)
    
    features = dct(features, type=2, axis=1, norm='ortho')[:,:cepCount]
    
    features = lifter(features,ceplifter)
           
    return features

    
def freq2mel(hz):
    """
    Convert a value in Hertz to Mels
    
    :param hz: a value in Hz. This can also be a np array, conversion proceeds element-wise.
    
    :returns: a value in Mels. If an array was passed in, an identical sized array is returned.
    """
    return 1125 * np.log(1+hz/700.0)

    
def mel2freq(mel):
    """
    Convert a value in Mels to Hertz
    
    :param mel: a value in Mels. This can also be a np array, conversion proceeds element-wise.
    
    :returns: a value in Hertz. If an array was passed in, an identical sized array is returned.
    """
    return 700*(np.exp(mel/1125.0)-1)


def melFilterBank(filterCount=20,fftSize=512,sampleRate=16000,lowFreq=0,highFreq=None):
    """
    Compute a Mel-filterbank. The filters are stored in the rows, the columns correspond
    to fft bins. The filters are returned as an array of size nfilt * (nfft/2 + 1)
    
    :param filterCount: the number of filters in the filterbank, default 20.
    :param fftSize    : the FFT size. Default is 512.
    :param sampleRate : the samplerate of the signal we are working with. Affects mel spacing.
    :param lowFreq    : lowest band edge of mel filters, default 0 Hz
    :param highFreq   : highest band edge of mel filters, default samplerate/2
    
    :returns: A np array of size nfilt * (nfft/2 + 1) containing filterbank. Each row holds 1 filter.
    """
    highFreq= highFreq or sampleRate/2
    assert highFreq <= sampleRate/2, "highfreq is greater than samplerate/2"
    
    # compute points evenly spaced in mels
    lowMel    = freq2mel(lowFreq)
    highMel   = freq2mel(highFreq)
    melPoints = np.linspace(lowMel, highMel, filterCount + 2)
    
    # our points are in Hz, but we use fft bins, so we have to convert
    #  from Hz to fft bin number
    bins = np.floor((fftSize + 1) * mel2freq(melPoints) / sampleRate)

    filterBank = np.zeros([filterCount, fftSize/2 + 1])
    for i in xrange(0, filterCount):
        
        for j in xrange(int(bins[i]), int(bins[i+1])):
            filterBank[i,j] = (j - bins[i]) / (bins[i+1] - bins[i])
            
        for j in xrange(int(bins[i+1]), int(bins[i+2])):
            filterBank[i,j] = (bins[i+2] - j) / (bins[i+2] - bins[i+1])
            
    return filterBank    


def frameSignal(signal, frameSize, frameStep, winFunc=None):
    """
    Frame a signal into overlapping frames.
    
    :param signal   : the audio signal to frame.
    :param frameSize: length of each frame measured in samples.
    :param frameStep: number of samples after the start of the previous frame that the next frame should begin.
    :param winFunc  : the analysis window to apply to each frame. By default no window is applied.    
    
    :returns: an array of frames. Size is NUMFRAMES by frame_len.
    """
    signalSize = len(signal)
    
    frameSize = int(round(frameSize))
    frameStep = int(round(frameStep))
    
    frameCount = 1    
    if signalSize > frameSize: 
        frameCount += int(math.ceil((1.0*signalSize - frameSize)/frameStep))
        
    padSize = int( (frameCount - 1)*frameStep + frameSize )
    
    padSignal = np.concatenate([signal, np.zeros([padSize - signalSize])])
    
    indices = np.tile(np.arange(0,frameSize), [frameCount,1]) + np.tile(np.arange(0, frameCount*frameStep, frameStep), [frameSize, 1]).T
    indices = np.array(indices, dtype = np.int32)
    
    frames = padSignal[indices]
    
    if winFunc:
        frames *= np.tile(winFunc(frameSize), [frameCount, 1])
    
    return frames

          
def powerSpectrum(frames, fftSize):
    """
    Compute the power spectrum of each frame in frames. If frames is an NxD matrix, output will be NxNFFT.
     
    :param frames : the array of frames. Each row is a frame.
    :param fftSize: the FFT length to use. If fftSize > frame_len, the frames are zero-padded. 
    
    :returns: If frames is an NxD matrix, output will be NxfftSize. Each row will be the power spectrum of the corresponding frame.
    """        
    return 1.0/fftSize * np.square( np.absolute( np.fft.rfft(frames, fftSize) ) )

    
def preemphasis(signal, coeff = 0.95):
    """
    perform preemphasis on the input signal.
    
    :param signal: The signal to filter.
    :param coeff : The preemphasis coefficient. 0 is no filter, default is 0.95.
    
    :returns: the filtered signal.
    """    
    return np.append(signal[0], signal[1:] - coeff*signal[:-1])


def lifter(cepstra,L=22):
    """
    Apply a cepstral lifter the the matrix of cepstra. This has the effect of increasing the
    magnitude of the high frequency DCT coeffs.
    
    :param cepstra: the matrix of mel-cepstra, will be numframes * numcep in size.
    :param L: the liftering coefficient to use. Default is 22. L <= 0 disables lifter.
    """
    if L > 0:
        coeffCount = cepstra.shape[1]
        n = np.arange(coeffCount)
        lift = 1 + (L/2) * np.sin(np.pi * n/L)
        return lift * cepstra
    else:
        return cepstra