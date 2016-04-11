import os
import gzip
import cPickle
import scipy.io.wavfile as wav
import featureExtraction as fe
from HMM import HMM
import CodeBook as cb
from IWR import IWR
from MakeData import Data
from numpy import append

def load(filename):
    file = gzip.GzipFile(filename, 'rb')
    object = cPickle.load(file)
    file.close()
    return object

def main():
    iwr = IWR(load("data_train_256.cpkl.gz"))    
    
    iwr.train()
    
    signals = []
    rates   = []
    y       = []
    for word in os.listdir("data/test"):
        path = os.path.join("data/test", word)
        for sample in os.listdir(path):
            (rate, signal) = wav.read(os.path.join(path, sample))
            
            signals.append((signal[:,0]+signal[:,1])/2)
            rates  .append(rate)
            
    p = iwr.predict(signals, rates)
    print p

        
if __name__ == "__main__":
    main()