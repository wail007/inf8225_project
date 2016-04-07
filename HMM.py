import numpy as np

class _HMM:
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


    def reset(self):  
        self.pi     = np.ones ([self.n]        , dtype = self.precision) * (1.0 / self.n)
        self.A      = np.ones ([self.n, self.n], dtype = self.precision) * (1.0 / self.n)


    def cost(self, observations, Bs=None):
        if Bs is None:
            Bs = self._Bs(observations)
        
        L = 0      
        for i in xrange(len(observations)):
            alpha = self._alpha(observations[i], Bs[i])
            L += np.log(sum(alpha[-1]))
        
        return -L / len(observations)


    def forwardbackward(self, observation, B=None):
        if B is None:
            B = self._B(observation)
        
        p = self._alpha(observation, B) * self._beta(observation, B);
        return p / np.sum(p, axis=1, keepdims=True)


    def train(self, observations, eps=0.0001):
        self._initParam(observations)
        Bs = self._Bs(observations)
        
        previousCost = float("inf");
        
        while True:
            self._baumWelch(observations, Bs)
            
            cost = self.cost(observations, Bs)
            
            if abs(previousCost - cost) < eps:
                break;
            
            previousCost = cost
            
            
    def _initParam(self, observations):
        return


    def _B(self, observation):
        """
        Computes the emission probability 'B'. B[t][i] = P(X{t}=x | W(t)=i).
        It is the probability of observing symbol 'x' at time 't' given that we are in state 'i'.
        
        :param observations: X(1), ..., X(t)
        
        :returns: (TxN) numpy array. All probabilities are uniform. Deriving classes should override this method.
        """
        B = np.random.rand(len(observation), self.n)
        return B / np.sum(B, axis=1, keepdims=True)
    
    
    def _Bs(self, observations):
        Bs = []
        for o in observations:
            Bs.append( self._B(o) )
            
        return Bs
    

    def _alpha(self, observation, B):
        alpha = np.zeros([len(observation), self.n], dtype = self.precision)
        
        alpha[0]  = self.pi * B[0]
        alpha[0] /= np.sum(alpha[0])
        
        for t in xrange(1, len(observation)):
            alpha[t]  = np.dot(alpha[t-1], self.A) * B[t]
            alpha[t] /= np.sum(alpha[t])
        
        return alpha


    def _beta(self, observation, B):      
        beta = np.zeros([len(observation),self.n], dtype = self.precision)
        
        beta[-1] = 1.0/self.n
        
        for t in xrange(len(observation)-2,-1,-1):
            beta[t] = np.dot(self.A, B[t+1] * beta[t+1])
            beta[t] /= np.sum(beta[t])
        
        return beta


    def _xi(self, observation, B, alpha, beta):       
        xi = np.zeros([len(observation), self.n, self.n], dtype=self.precision)
        
        for t in xrange(len(observation)-1):
            numerator = self.A * np.tile(B[t+1]*beta[t+1], [self.n,1]) * np.tile(alpha[t], [self.n,1]).T
            xi[t] = numerator / np.sum(numerator)
        
        return xi


    def _gamma(self, xi):
        return np.sum(xi, axis=2)


    def _updatePi(self, gamma):
        self.pi = gamma[0][0]
        
        for i in xrange(1, len(gamma)):
            self.pi += gamma[i][0]
            
        self.pi /= len(gamma)


    def _updateA(self, xi, gamma):
        #self.A = np.sum(xi , axis=0) / np.tile(np.sum(gamma, axis=0), [self.n, 1]).T
        numer = np.sum(xi   [0], axis=0)
        denom = np.sum(gamma[0], axis=0)
        
        for i in xrange(1, len(gamma)):
            numer += np.sum(xi   [i], axis=0)
            denom += np.sum(gamma[i], axis=0)
        
        self.A = numer / np.tile(denom, [self.n, 1]).T


    def _EStep(self, observations, Bs):
        stats = {}
        
        stats['alpha'] = []
        stats['beta']  = []
        stats['xi']    = []
        stats['gamma'] = []
        
        for i in xrange(len(observations)):            
            stats['alpha'].append( self._alpha(observations[i], Bs[i]) )
            stats['beta'] .append( self._beta (observations[i], Bs[i]) )
            stats['xi']   .append( self._xi   (observations[i], Bs[i], stats['alpha'][i], stats['beta'][i]) )
            stats['gamma'].append( self._gamma(stats['xi'][i]) )

        return stats


    def _MStep(self, observations, stats):
        self._updatePi(stats['gamma'])
        self._updateA(stats['xi'], stats['gamma'])
        

    def _baumWelch(self, observations, Bs):
        stats = self._EStep(observations, Bs)
        self._MStep(observations, stats)