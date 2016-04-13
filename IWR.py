import numpy as np
import CodeBook as cb
from HMM import HMM
from featureExtraction import mfcc

class Data:
    def __init__(self):
        self.samples   = []
        self.codeBooks = []
        self.words     = []

class IWR:
    
    def __init__(self, data):
        self.data    = data
        self.hmmList = []
        
    def train(self, n):
        self.hmmList = []
        for i in xrange(len(self.data.words)):
            self.hmmList.append(HMM(n, len(self.data.codeBooks[i])))
        
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
    
    def precision(self, signals, sampleRates, y):
        return np.sum(y == self.predict(signals, sampleRates)) / float(len(signals))
        
        