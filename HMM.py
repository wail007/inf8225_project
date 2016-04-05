import numpy as np

class   HMM:
    '''
    Gaussian Mixture Hidden Markov Model    
    
    P(X(1),X(2),...,X(t),W(1),W(2),...,W(t)) = P(W(1))P(X(1)|W(1)) * product[P(W(i)|W(i-1))P(X(i)|W(i)), i=2...t]
    
    W(i): state of hidden variable W at time i. 
    X(i): observed symbol at time i
    
    '''
    def __init__(self, n, precision = np.double, verbose = False):
        """
        Constructor
        
        :param n        : number of hidden states of the hidden variable W
        :param precision: precision of the parameters, default is double
        :param verbose  : flag for printing progress information when learning, deafault is False
        """     
        self.n = n
        
        self.precision = precision
        self.verbose   = verbose
        
        self.reset()
        
        
    def reset(self):  
        self.pi     = np.ones ([self.n]        , dtype = self.precision) * (1.0 / self.n)
        self.A      = np.ones ([self.n, self.n], dtype = self.precision) * (1.0 / self.n)
       
    
    def logLikelihood(self, observations):
        alpha = self._alpha(observations, self._B(observations))
        return np.log(sum(alpha[-1]))
        
    
    def forwardbackward(self, observations):
        B = self._B(observations)
        p = self._alpha(observations, B) * self._beta(observations, B);
        return p / np.sum(p, axis=1, keepdims=True)
    
    
    def train(self, observations, eps=0.0001):
        B = self._B(observations)
        
        previousCost = float("inf");
        
        while True:
            self._baumWelch(observations, B)
            
            cost = self.logLikelihood(observations)
            
            if abs(cost - previousCost) < eps:
                break;
            
            previousCost = cost            
            
    
    def _B(self, observations):
        """
        Computes the emission probability 'B'. B[t][i] = P(X{t}=x | W(t)=i).
        It is the probability of observing symbol 'x' at time 't' given that we are in state 'i'.
        
        :param observations: X(1), ..., X(t)
        
        :returns: (TxN) numpy array. All probabilities are uniform. Deriving classes should override this method.
        """
        B = np.random.rand(len(observations), self.n)
        return B / np.sum(B, axis=1, keepdims=True)
    
    
    def _alpha(self, observations, B):
        alpha = np.zeros([len(observations), self.n], dtype = self.precision)
        
        alpha[0,:] = self.pi * B[0,:]
        
        for t in xrange(1, len(observations)):
            alpha[t,:] = np.dot(alpha[t-1,:], self.A) * B[t,:]
                
        return alpha
    
    
    def _beta(self, observations, B):      
        beta = np.zeros([len(observations),self.n], dtype = self.precision)
        
        beta[-1,:] = 1.
        
        for t in xrange(len(observations)-2,-1,-1):
            for i in xrange(self.n):
                beta[t][i] = sum(self.A[i,:]*B[t+1,:]*beta[t+1,:])
                    
        return beta
    
    
    def _xi(self, observations, B, alpha, beta):       
        xi = np.zeros([len(observations), self.n, self.n], dtype=self.precision)
        
        for t in xrange(len(observations)-1):
            numerator = self.A * np.tile(B[t+1]*beta[t+1], [self.n,1]) * np.tile(alpha[t], [self.n,1]).T
            xi[t] = numerator / np.sum(numerator)
        
        return xi


    def _gamma(self, observations, xi):        
        return np.sum(xi, axis=2)
        
    
    def _updatePi(self, gamma):
        self.pi = gamma[0]
    
    
    def _updateA(self, xi, gamma):
        self.A = np.sum(xi , axis=0) / np.tile(np.sum(gamma, axis=0), [self.n, 1]).T
                        
                
    def _EStep(self, observations, B):
        stats = {}
        
        stats['alpha'] = self._alpha(observations, B)
        stats['beta']  = self._beta (observations, B)
        stats['xi']    = self._xi   (observations, B, stats['alpha'],stats['beta'])
        stats['gamma'] = self._gamma(observations, stats['xi'])
        
        return stats
    
    
    def _MStep(self, observations, stats):
        self._updatePi(stats['gamma'])
        self._updateA(stats['xi'], stats['gamma'])
        
    
    def _baumWelch(self, observations, B):
        stats = self._EStep(observations, B)
        self._MStep(observations, stats)
    
            