import os
import scipy.io.wavfile as wav
import featureExtraction as fe
from HMM import HMM
import CodeBook as cb



def main():
    files = os.listdir("data")
    
    observations = []
    
    for f in files:
        (rate, signal) = wav.read(os.path.join("data", f))
        observations.append( fe.mfcc(signal, rate) )
    
    codeBook = cb.makeCodeBook(observations, 256)
    obsCodes = cb.getCodes(observations, codeBook)
    
    hmm = HMM(512, 256)
    
    hmm.train(obsCodes)

        
if __name__ == "__main__":
    main()