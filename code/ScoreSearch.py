from pgmpy.estimators import StructureScore
import numpy as np
from typing import List



class GaussianBIC(StructureScore):
    def __init__(self, data, **kwargs):

        super(GaussianBIC, self).__init__(data, **kwargs)

        self.nodemap = {}
        for nn, node in enumerate(data.columns):
            self.nodemap[node] = nn

    
    def local_score(self, variable: str, parents: List[str], parameters=None) -> float:
        """
        Calculate the *negative* local score with BIC for the linear Gaussian continue data case

        Parameters
        ----------
        Data: ndarray, (sample, features)
        i: current index
        PAi: parent indexes
        parameters: lambda_value, the penalty discount of bic

        Returns
        -------
        score: local BIC score
        """

    
        i = self.nodemap[variable]
        PAi = [self.nodemap[pi] for pi in parents]
        data = np.array(self.data)


        cov = np.corrcoef(data.T)
        n = data.shape[0]
        # cov, n = Data

        if parameters is None:
            lambda_value = 1
        else:
            lambda_value = parameters["lambda_value"]

        if len(PAi) == 0:
            return n * np.log(cov[i, i])

        yX = np.mat(cov[np.ix_([i], PAi)])
        XX = np.mat(cov[np.ix_(PAi, PAi)])
        H = np.log(cov[i, i] - yX * np.linalg.inv(XX) * yX.T)

        return n * H + np.log(n) * len(PAi) * lambda_value
    
    