import os
import numpy as np
import scipy.io.wavfile as wav
import featureExtraction as fe
from HMM import HMM



def main():
    
    (rate, signal) = wav.read("data/yes/yes0.wav")
    
    features = fe.mfcc(signal[:,0], rate)
    
    hmm = HMM(20)
    
    hmm.train(features)

        
if __name__ == "__main__":
    main()