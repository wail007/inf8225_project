import os
import scipy.io.wavfile as wav
import featureExtraction as fe
from GMHMM import GMHMM



def main():
    files = os.listdir("data")
    
    observations = []
    
    for f in files:
        (rate, signal) = wav.read(os.path.join("data", f))
        observations.append( fe.mfcc(signal, rate) )
    
    hmm = GMHMM(20, 10, 13)
    
    hmm.train(observations)

        
if __name__ == "__main__":
    main()