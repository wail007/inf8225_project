import os
import numpy as np
import scipy.io.wavfile as wav
import featureExtraction as fe
from HMM import HMM



def main():
    files = os.listdir("data/yes")
    
    observations = []
    
    for f in files:
        (rate, signal) = wav.read(os.path.join("data/yes", f))
        observations.append( fe.mfcc(signal[:,0], rate) )
    
    hmm = HMM(20)
    
    hmm.train(observations)

        
if __name__ == "__main__":
    main()