import os
import sys
import gzip
import getopt
import cPickle
import CodeBook          as cb
import scipy.io.wavfile  as wav
import featureExtraction as fe
from IWR import Data

def save(object, filename, protocol = -1):
    file = gzip.GzipFile(filename, 'wb')
    cPickle.dump(object, file, protocol)
    file.close()
    
def printOptions():
    print "-s : Number of observable symbols to use for each word (default: 64)"
    print "-f : Path to the folder containing all the training data. That folder should contain a folder for each word. (default: data/train)"
    
def main():
    symbolCount = 64
    dataPath    = "data/train"
    
    try:
        opts, args = getopt.getopt(sys.argv[1:],"s:h")
    except getopt.GetoptError:
        print "Error: unknown options"
        printOptions()
        sys.exit(2)
        
    for opt, arg in opts:
        if opt == '-h':
            printOptions()
            sys.exit()
        elif opt == "-s":
            symbolCount = int(arg)
    
    data = Data()
    for word in os.listdir(dataPath):
        path = os.path.join(dataPath, word)
        
        print "Working on: {}".format(path)
        
        data.words.append(word)
        features = []
        
        for sample in os.listdir(path):
            (rate, signal) = wav.read(os.path.join(path, sample))
            features.append( fe.mfcc((signal[:,0]+signal[:,1])/2, rate) )
        
        data.codeBooks.append( cb.makeCodeBook(features, symbolCount, verbose=True) )
        data.samples  .append( cb.getCodes    (features, data.codeBooks[-1]) )
        
    
    fileName = "data_train_{}.cpkl.gz".format(symbolCount)
    print "saving: {}".format(fileName)
    
    save(data, fileName)
    print "Done"
            

if __name__ == "__main__":
    main()