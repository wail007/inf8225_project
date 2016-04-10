import numpy as np
import CodeBook as cb
from HMM import HMM
from featureExtraction import mfcc
from MakeData import Data

class IWR:
    
    def __init__(self, data):
        self.data    = data
        self.hmmList = []
        
        for i in xrange(len(self.data.words)):
            self.hmmList.append(HMM(64, len(self.data.codeBooks[i])))        
        
    def train(self):
        
        for i in xrange(len(self.data.words)):
            print "Training: {}".format(self.data.words[i])
            self.hmmList[i].train(self.data.samples[i])
            
    def predict(self, signals, sampleRates):
        features = []
        for i in xrange(len(signals)):
            features.append( mfcc(signals[i], sampleRates[i]) )
        
        y = np.zeros( [len(self.data.words), len(features)] )
        
        for i in xrange(len(self.data.words)):
                observations = cb.getCodes(features, self.data.codeBooks[i])
                
                for j in xrange(len(observations)):
                    y[i][j] = self.hmmList[i].cost([observations[j]])
                
        return np.argmin(y, axis=0)
            
        