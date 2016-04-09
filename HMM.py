import numpy as np

class HMM:
    '''
    Gaussian Mixture Hidden Markov Model    
    
    P(X(1),X(2),...,X(t),W(1),W(2),...,W(t)) = P(W(1))P(X(1)|W(1)) * product[P(W(i)|W(i-1))P(X(i)|W(i)), i=2...t]
    
    W(i): state of hidden variable W at time i. 
    X(i): observed symbol at time i
    '''
    def __init__(self, n, m, precision=np.double, verbose=False):
        """
        Constructor
        
        :param n        : number of hidden states of the hidden variable W
        :param precision: precision of the parameters, default is double
        :param verbose  : flag for printing progress information when learning, deafault is False
        """
        self.n = n
        self.m = m
        
        self.precision = precision
        self.verbose   = verbose
        
        self.reset()


    def reset(self):  
        self.pi  = np.random.rand(self.n)
        self.pi /= np.sum(self.pi, keepdims=True)

        self.A  = np.random.rand(self.n, self.n)
        self.A /= np.sum(self.A, axis=1, keepdims=True)
        
        self.B  = np.random.rand(self.n, self.m)
        self.B /= np.sum(self.B, axis=1, keepdims=True)


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
            B = self._Bmap(observation)
        
        p = self._alpha(observation, B) * self._beta(observation, B);
        return p / np.sum(p, axis=1, keepdims=True)


    def train(self, observations, eps=0.00001):
        Bs = self._Bs(observations)
        
        previousCost = float("inf");
        
        while True:
            self._baumWelch(observations, Bs)
            
            cost = self.cost(observations, Bs)
            print cost
            
            if abs(previousCost - cost) < eps:
                break;
            
            previousCost = cost


    def _Bmap(self, observation):
        """
        Computes the emission probability 'B'. B[t][i] = P(X{t}=x | W(t)=i).
        It is the probability of observing symbol 'x' at time 't' given that we are in state 'i'.
        
        :param observations: X(1), ..., X(t)
        
        :returns: (TxN) numpy array. All probabilities are uniform. Deriving classes should override this method.
        """
        return self.B[:, observation].T


    def _Bs(self, observations):
        Bs = []
        for o in observations:
            Bs.append( self._Bmap(o) )
            
        return Bs
    

    def _alpha(self, observation, B):
        alpha = np.zeros([len(observation), self.n], dtype=self.precision)
        
        alpha[0]  = self.pi * B[0]
        
        for t in xrange(1, len(observation)):
            alpha[t]  = np.dot(alpha[t-1], self.A) * B[t]
        
        return alpha


    def _beta(self, observation, B):      
        beta = np.zeros([len(observation),self.n], dtype = self.precision)
        
        beta[-1] = 1.0
        
        for t in xrange(len(observation)-2,-1,-1):
            beta[t] = np.dot(self.A, B[t+1] * beta[t+1])
        
        return beta


    def _xi(self, observation, B, alpha, beta):       
        xi = np.zeros([len(observation), self.n, self.n], dtype=self.precision)
        
        for t in xrange(len(observation)-1):
            numerator = self.A * np.tile(B[t+1]*beta[t+1], [self.n,1]) * np.tile(alpha[t], [self.n,1]).T
            xi[t] = numerator / np.sum(numerator)
        
        return xi


    def _gamma(self, xi):
        return np.sum(xi, axis=2)


    def _updatePi(self, gammas):
        self.pi = gammas[0][0]
        
        for i in xrange(1, len(gammas)):
            self.pi += gammas[i][0]
            
        self.pi /= len(gammas)


    def _updateA(self, xis, gammas):
        #self.A = np.sum(xi , axis=0) / np.tile(np.sum(gamma, axis=0), [self.n, 1]).T
        numer = np.sum(xis   [0], axis=0)
        denom = np.sum(gammas[0], axis=0)
        
        for i in xrange(1, len(gammas)):
            numer += np.sum(xis   [i], axis=0)
            denom += np.sum(gammas[i], axis=0)
        
        self.A = numer / np.tile(denom, [self.n, 1]).T
        
    def _updateB(self, observations, gammas):
        symbols = np.tile(np.arange(self.m), [len(observations[0]), 1])
        y = symbols == observations[0]
        numer = np.dot(gammas[0], y)
        
        denom = np.sum(gammas[0], axis=0)
        
        for i in xrange(1, len(observations)):
            denom += np.sum(gammas[i], axis=0)
        
        self.B = np.zeros( (self.n,self.m) ,dtype=self.precision)
        
        for j in xrange(self.n):
            for k in xrange(self.m):
                numer = 0.0
                denom = 0.0
                for i in xrange(len(observations)):
                    for t in xrange(len(observations[i])):
                        if observations[i][t] == k:
                            numer += gammas[i][t][j]
                        denom += gammas[i][t][j]
                self.B[j][k] = numer/denom


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
        self._updateB(observations, stats['gamma'])
        

    def _baumWelch(self, observations, Bs):
        stats = self._EStep(observations, Bs)
        self._MStep(observations, stats)