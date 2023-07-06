from itertools import combinations

from causallearn.utils.cit import *
from numpy import ndarray
import numpy as np
from scipy.stats import norm
from .utils import CustomThread
from .wrapper.wrapper import fastResiduals

cci = 'cci'

def CITCustom(data, method='fisherz', **kwargs):
    '''
    Parameters
    ----------
    data: numpy.ndarray of shape (n_samples, n_features)
    method: str, in ["fisherz", "mv_fisherz", "mc_fisherz", "kci", "chisq", "gsq"]
    kwargs: placeholder for future arguments, or for KCI specific arguments now
        TODO: utimately kwargs should be replaced by explicit named parameters.
              check https://github.com/cmu-phil/causal-learn/pull/62#discussion_r927239028
    '''
    if method == fisherz:
        return FisherZ(data, **kwargs)
    elif method == kci:
        return KCI(data, **kwargs)
    elif method in [chisq, gsq]:
        return Chisq_or_Gsq(data, method_name=method, **kwargs)
    elif method == mv_fisherz:
        return MV_FisherZ(data, **kwargs)
    elif method == mc_fisherz:
        return MC_FisherZ(data, **kwargs)
    elif method == d_separation:
        return D_Separation(data, **kwargs)
    elif method == cci:
        return CCI(data, **kwargs)
    else:
        raise ValueError("Unknown method: {}".format(method))
    

class CCI(CIT_Base):
    def __init__(self, data, **kwargs):
        super().__init__(data, **kwargs)
        self.method = 'cci'
        self.alpha = kwargs['alpha']
        self.assert_input_data_is_valid()
        

    @staticmethod
    def polyF(X: ndarray, order):
        return np.power(X,order)

        
    def residuals(self, X, Z):
        residuals = []
        kernel = Kernel(Z)
     
        residuals = fastResiduals(X,Z,kernel.bandwidth)

        return residuals

    def _independent(self, X, Y, order, draw = False):

        fx = self.polyF(X,order[0])
        gy = self.polyF(Y,order[1])

        cov_matrix = np.cov(fx, gy)
                
        r = abs(cov_matrix[0,1]/sqrt(np.var(fx) * np.var(gy)))
        if r>=1:
            r=0.99
        z = 0.5 * np.log((1 + r) / (1 - r))
    
        zscore = sqrt(self.sample_size) * abs(z)
        # if put_tau:
        #     X_normal = (fx-np.mean(fx))/np.std(fx)
        #     Y_normal = (gy-np.mean(gy))/np.std(gy)
        #     tau2 = np.sum(np.multiply(np.power(X_normal,2),np.power(Y_normal,2)))
        # else: 
        #     tau2 = 1

        p = 2 * (1 - norm.cdf(zscore, scale=1))

        if draw:
                print(f'\n order {order}')
                print(f'cov ij: {cov_matrix[0,1]}')
                print(f'var fx {np.var(fx)} var gy {np.var(gy)}')
                print(f'r {r}')
                print(f'z {z}')
                print(f'zscore {zscore}')
                # print(f'tau {np.sqrt(tau2)}')
                # print(f'tau2 {tau2}')
                print(f'p valor {p}')
                time.sleep(3)
        
        return p

    def independent(self, X, Y, S, draw = False):
        
        pvalues = []
        threads = list()

        for order in combinations(range(1,5), 2):
           
            t = CustomThread(target=self._independent, args=(X, Y, order))
            threads.append(t)
            t.start()
            t.join()

        for index, t in enumerate(threads):
            pval = t.value()
            pvalues.append(pval)

               
        pvalues.sort()
        pvalues = np.array(pvalues)
        pvalues_fdr = np.divide(np.array(pvalues)*len(pvalues), np.arange(1,len(pvalues)+1))   
       
        check_pvals = np.where(pvalues_fdr < self.alpha)[0]
        if len(check_pvals) > 0: #dependent
            dependent_pvals = pvalues_fdr[check_pvals]
            meanPval = np.mean(dependent_pvals)
        else: #indepenent
            meanPval = np.mean(pvalues)
        
        if draw:
            print(f'p values {pvalues}')
            print(f'p values fdr  {pvalues_fdr}')
            print(f'meanPval {meanPval}')
            time.sleep(2)
        
        return meanPval
        

    def __call__(self, X, Y, condition_set=None):
        # Conditonal Correlation Independence test.
        Xs, Ys, condition_set, cache_key = self.get_formatted_XYZ_and_cachekey(X, Y, condition_set)

        if len(condition_set) == 0:
            return self.independent(self.data[:, Xs].ravel(), self.data[:, Ys].ravel(), [])
        
        threads=list()
        args = [(self.data[:, Xs], self.data[:, condition_set]), 
                (self.data[:, Ys], self.data[:, condition_set])]
        
        for arg in args:
            t = CustomThread(target=self.residuals, args=arg)
            threads.append(t)
            t.start()
            t.join()
       
        rx = threads[0].value()
        ry = threads[1].value()

        p = self.independent(rx.ravel(), ry.ravel(), self.data[:, condition_set].T)
        return p
    



class Kernel:
    def __init__(self, data) -> None:
        self.data = data
        self.N,self.M = self.data.shape
        

        self.bandwidth = self.prepare_kernel()


    def prepare_kernel(self):
        median_diff = abs(self.data-np.median(self.data, axis=0))
        MADs = np.median(median_diff, axis=0)

        h = 1.4826 * MADs * ((4/3)/self.N)**0.2 
        h_fused = np.sqrt(self.M)*np.max(h)

        return h_fused
        


    def kernel_weight(self, d):
      
        if abs(d) <= self.bandwidth:
            return 1/(2*self.bandwidth)
        else:
            return 0
            
   