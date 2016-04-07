from HMM import _HMM
import numpy as np

class GMHMM(_HMM):
    '''
    Gaussian Mixture Hidden Markov Model
    '''
    
    def __init__(self, n, m, d, min_std=0.01, precision=np.double, verbose=False):
        _HMM.__init__(self, n, precision, verbose)
        
        self.m = m
        self.d = d
        self.min_std = min_std

        self.reset()
    
    
    def reset(self):
        _HMM.reset(self)
        
        self.w      = np.ones ([self.n, self.m        ], dtype=self.precision) * (1.0/self.m)            
        self.means  = np.zeros([self.n, self.m, self.d], dtype=self.precision)
        
        self.covars = [[ np.matrix(np.ones([self.d, self.d], dtype=self.precision)) for j in xrange(self.m)] for i in xrange(self.n)]
    
    def _B(self, observation, probMap=None):
        B = np.zeros([len(observation), self.n], dtype=self.precision)
        
        if probMap is None:
            probMap = np.zeros([len(observation), self.n, self.m], dtype=self.precision)
        
        for t in xrange(len(observation)):
            for i in xrange(self.n):
                for j in xrange(self.m):
                    probMap[t][i][j] = self._gaussianPDF(observation[t], self.means[i][j], self.covars[i][j])
                    
            B[t] = np.sum(probMap[t] * self.w, axis=1)            
        
        return B
    
    def _Bs(self, observations):
        Bs = []
        self.probMaps = []
        
        for o in observations:
            self.probMaps.append(np.zeros([len(o), self.n, self.m], dtype=self.precision))
            Bs.append(self._B(o, self.probMaps[-1]))
            
        return Bs
    
    #ref:http://research.cs.tamu.edu/prism/lectures/pr/pr_l24.pdf
    def _gammaGM(self, observation, gamma, probMap):
        gammaGM  = np.tile(self.w, [len(observation),1,1]) * probMap
        gammaGM /= np.sum(gammaGM, axis=2)
        gammaGM *= gamma
        
        return gammaGM
    
    
    def _updateW(self, gammaGM):
        numer = np.sum(gammaGM[0], axis=0)
        denom = np.sum(numer     , axis=1)
        
        for i in xrange(1, len(gammaGM)):
            a = np.sum(gammaGM[1], axis=0)
            numer += a
            denom += np.sum(a, axis=1)
            
        self.w = numer / denom
        
        
    def _updateMeans(self, observations, gammaGM):
        numer = np.dot(gammaGM[0], observations[0])
        denom = np.sum(gammaGM[0], axis=0)
        
        for i in xrange(1, len(observations)):
            numer += np.dot(gammaGM[i], observations[0])
            denom += np.sum(gammaGM[i], axis=0)
        
        self.means = numer / denom
        
        
    def _updateCovars(self, observations, means, gammaGM):
        #TODO
        cov_prior = [[ np.matrix(self.min_std*np.eye((self.d), dtype=self.precision)) for j in xrange(self.m)] for i in xrange(self.n)]
        
        denom = np.sum(gammaGM[0], axis=0)
        
        for j in xrange(self.n):
            for m in xrange(self.m):
                numer = np.matrix(np.zeros( (self.d,self.d), dtype=self.precision))
                for t in xrange(len(observations)):
                    vector_as_mat = np.matrix( observations[t]-self.means[j][m], dtype=self.precision )
                    numer += gammaGM[t][j][m] * np.dot( vector_as_mat.T, vector_as_mat)
                self.covars[j][m] = numer/denom[j][m]
                self.covars[j][m] = self.covars[j][m] + cov_prior[j][m]
    
    
    def _EStep(self, observations, Bs):
        stats = _HMM._EStep(self, observations, Bs)
        
        stats["gamma_gm"] = []
        for i in xrange(len(observations)):
            stats["gamma_gm"].append( self._gammaGM(observations[i], stats["gamma"], self.probMap[i]) )
            
        return stats
    
    def _MStep(self, observations, stats):
        _HMM._MStep(self, observations, stats)
            
            
    def _gaussianPDF(self, x, mean, covar):        
        return (1 / ( (2.0*np.pi)**(self.d/2.0) * np.linalg.det(covar)**0.5 )) * np.exp(-0.5 * np.dot( np.dot((x-mean),covar.I), (x-mean)) )
    
    