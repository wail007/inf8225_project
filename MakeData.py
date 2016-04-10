import os
import gzip
import cPickle
import CodeBook as cb
import scipy.io.wavfile as wav
from featureExtraction import mfcc

class Data:
    def __init__(self):
        self.samples   = []
        self.codeBooks = []
        self.words     = []

def save(object, filename, protocol = -1):
    file = gzip.GzipFile(filename, 'wb')
    cPickle.dump(object, file, protocol)
    file.close()
    
def main():
    data = Data()
    for word in os.listdir("data/train"):
        path = os.path.join("data/train", word)
        
        data.words.append(word)
        
        features = []
        
        for sample in os.listdir(path):
            (rate, signal) = wav.read(os.path.join(path, sample))
            features.append( mfcc((signal[:,0]+signal[:,1])/2, rate) )
        
        data.codeBooks.append( cb.makeCodeBook(features, 64                ) )
        data.samples  .append( cb.getCodes    (features, data.codeBooks[-1]) )
        
            
    save(data, "data_train_64.cpkl.gz")
            
        
    
if __name__ == "__main__":
    main()