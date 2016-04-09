import scipy.cluster.vq as vq
import numpy as np


def makeCodeBook(observations, symbolCount):
    obs = observations[0]
    for i in xrange(1, len(observations)):
        obs = np.vstack([obs, observations[i]]) 
        
    obs = vq.whiten(obs)
    
    [codebook, distortion] = vq.kmeans(obs, symbolCount)
    
    return codebook


def getCodes(observations, codeBook):
    obsCode = []
    
    for i in xrange(len(observations)):
        obs = vq.whiten(observations[i])
        
        [code, dist] = vq.vq(obs, codeBook)
        
        obsCode.append(code)
    
    return obsCode