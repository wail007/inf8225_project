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
    print "-n : Number of states in the HMMs (default: 64)"
    print "-i : Input file containing the training data"
    print "-o : Output file containing the trained HMMs"

def main():
    
    n      = 64
    input  = None
    output = None
    
    try:
        opts, args = getopt.getopt(sys.argv[1:],"n:i:o:h")
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
        elif opt == "-o":
            output = arg
        elif opt == "-n":
            n = int(arg)
    
    if input is not None:
        iwr = IWR(load(input))
        iwr.train(n)
        
        if output is not None:
            save(iwr, output)
        
if __name__ == "__main__":
    main()