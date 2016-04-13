import os
import sys
import getopt
import gzip
import cPickle
import scipy.io.wavfile as wav
from IWR import *

def load(filename):
    file = gzip.GzipFile(filename, 'rb')
    object = cPickle.load(file)
    file.close()
    return object

def save(object, filename, protocol = -1):
    file = gzip.GzipFile(filename, 'wb')
    cPickle.dump(object, file, protocol)
    file.close()
    
def printOptions():
    print "-i : Input file containing the trained HMMs"
    print "-f : Path to the folder containing all the test WAV files. That folder should contain a folder for each word. (default: data/test)"

def main():
    
    input    = None
    dataPath = "data/test"
    
    try:
        opts, args = getopt.getopt(sys.argv[1:],"i:f:h")
    except getopt.GetoptError:
        print "Error: unknown options"
        printOptions()
        sys.exit(2)
        
    for opt, arg in opts:
        if opt == '-h':
            printOptions()
            sys.exit()
        elif opt == "-i":
            input = arg
        elif opt == "-f":
            dataPath = arg
    
    if input is not None:
        iwr = load(input)
    
        if os.path.isdir(dataPath):            
            signals = []
            rates   = []
            y       = []
            for word in os.listdir(dataPath):
                path = os.path.join(dataPath, word)
                
                id = iwr.data.words.index(word)
                
                for sample in os.listdir(path):
                    (rate, signal) = wav.read(os.path.join(path, sample))
                    
                    signals.append((signal[:,0]+signal[:,1])/2)
                    rates  .append(rate)
                    y      .append(id)
                    
            print "Precision: {}".format( iwr.precision(signals, rates, y))
            
        elif os.path.isfile(dataPath):
            (rate, signal) = wav.read(dataPath)
            
            id = iwr.predict( [(signal[:,0]+signal[:,1])/2], [rate])
            
            print iwr.data.words[id]
        
        
if __name__ == "__main__":
    main()