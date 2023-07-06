import numpy as np
import scipy
from scipy.stats import norm
from scipy.special import kl_div
import pandas as pd
from copy import deepcopy
import pybnesian as pbn

class ParameterTL:
    def    __init__(self, target: dict, auxModels: list, dag, auxDags:list, node_types) -> None:

        self.auxModels = auxModels # list of auxiliary models
        self.target = target # target
        self.dag = dag
        self.auxDags = auxDags
        self.node_types = node_types
        self.nodemap = {}

        for nn, node in enumerate(self.dag.nodes()):
            self.nodemap[node] = nn
        


    def get_maximum_discrepancy(self, taskid, model: dict, node: str):

        data_k = model['DATA']
        data_t = self.target['DATA']
  
        # Since we have continuous data, we cannot use the PDF because the probability of each value is essentially 0.
        # Therefore, we use the cumulative density function.
        qt = self.dag.cpd(node).cdf(data_t)
        qk = self.auxDags[taskid].cpd(node).cdf(data_t)
        
        kld = kl_div(qk, qt)
        return 1/max(kld)


    def discrepancy_weight(self, task: int, node: str) -> float:

        kl_discrepances= list()
        for nm, model in enumerate(self.auxModels):

            discrepancy_k = self.get_maximum_discrepancy(nm, model, node)
            kl_discrepances.append(discrepancy_k)

        wk = kl_discrepances[task] / np.sum(kl_discrepances)
        return wk

    def gaussian_loglinearpool(self, ci):
        meanParamBetas = np.ones(shape=(len(self.dag.nodes()), len(self.dag.nodes())))
        meanParamVariances = np.ones(shape=(len(self.dag.nodes()), 1))
        for n, dta in enumerate(self.auxModels):
            for nn, node in enumerate(self.dag.nodes()):
                wk = self.discrepancy_weight(n, node)

                meanParamBetas[nn] = np.abs(meanParamBetas[nn])*np.abs(self.auxModels[n]['BETAS'][nn])**wk
                meanParamBetas[nn] = meanParamBetas[nn]*np.sign(self.auxModels[n]['BETAS'][nn])

                meanParamVariances[nn] = meanParamVariances[nn]*np.abs(self.auxModels[n]['VARIANCES'][nn])**wk
                     
        for nn, node in enumerate(self.dag.nodes()):
            
            meanParamBetas[nn] = (np.abs(meanParamBetas[nn])**(1-ci)) * (np.abs(self.target['BETAS'][nn])**ci)
            meanParamBetas[nn] = meanParamBetas[nn]*np.sign(self.target['BETAS'][nn])
            meanParamVariances[nn] = (np.abs(meanParamVariances[nn])**(1-ci)) * (np.abs(self.target['VARIANCES'][nn])**ci)

        return meanParamBetas, meanParamVariances
    

    def spbn_loglinearpool(self, ci):

        meanParameters = self.prepare_spbn()
        for n, _ in enumerate(self.auxModels):
            for node, nodetype in self.node_types:
                    
                wk = self.discrepancy_weight(n, node)
                if type(nodetype)==type(pbn.CKDEType()):
                    Hi_joint = self.auxModels[n]['joint_kde'][node] 
                    Hi_marg = self.auxModels[n]['marg_kde'][node] 

                    meanParameters['joint_kde'][node] = np.abs(meanParameters['joint_kde'][node]) * np.abs(Hi_joint)**wk
                    meanParameters['joint_kde'][node] = meanParameters['joint_kde'][node] * np.sign(Hi_joint)
                    meanParameters['marg_kde'][node] = np.abs(meanParameters['marg_kde'][node]) * np.abs(Hi_marg)**wk
                    meanParameters['marg_kde'][node] = meanParameters['marg_kde'][node] * np.sign(Hi_marg)
                else:
                    betas = self.auxModels[n]['betas'][node] 
                    variances = self.auxModels[n]['variances'][node] 

                    meanParameters['betas'][node] = np.abs(meanParameters['betas'][node]) * np.abs(betas)**wk
                    meanParameters['betas'][node] = meanParameters['betas'][node] * np.sign(betas)
                    meanParameters['variances'][node] = meanParameters['variances'][node] * variances**wk

        for node, nodetype in self.node_types:
            
            if type(nodetype)==type(pbn.CKDEType()):
                Hi_joint = self.target['joint_kde'][node] 
                Hi_marg = self.target['marg_kde'][node] 

                meanParameters['joint_kde'][node] = (np.abs(meanParameters['joint_kde'][node])**(1-ci)) * (np.abs(Hi_joint)**ci)
                meanParameters['joint_kde'][node] = meanParameters['joint_kde'][node] * np.sign(Hi_joint)
                meanParameters['marg_kde'][node] = (np.abs(meanParameters['marg_kde'][node])**(1-ci)) * (np.abs(Hi_marg)**ci)
                meanParameters['marg_kde'][node] = meanParameters['marg_kde'][node] * np.sign(Hi_marg)
            else:
                betas = self.target['betas'][node] 
                variances = self.target['variances'][node] 

                meanParameters['betas'][node] =  (np.abs(meanParameters['betas'][node])**(1-ci)) * (np.abs(betas)**ci)
                meanParameters['betas'][node] = meanParameters['betas'][node] * np.sign(betas)
                meanParameters['variances'][node] = (meanParameters['variances'][node]**(1-ci)) * (variances**ci)

        return meanParameters
    
    
    def gaussian_linearpool(self, ci):
        meanParamBetas = np.zeros(shape=(len(self.dag.nodes()), len(self.dag.nodes())))
        meanParamVariances = np.zeros(shape=(len(self.dag.nodes()), 1))
        for n, dta in enumerate(self.auxModels):
            for nn, node in enumerate(self.dag.nodes()):
                wk = self.discrepancy_weight(n, node)

                meanParamBetas[nn] = meanParamBetas[nn]+self.auxModels[n]['BETAS'][nn]*wk
                meanParamVariances[nn] = meanParamVariances[nn]+self.auxModels[n]['VARIANCES'][nn]*wk

                
        for nn, node in enumerate(self.dag.nodes()):
            
            meanParamBetas[nn] = meanParamBetas[nn]*(1-ci) + self.target['BETAS'][nn]*ci
            meanParamVariances[nn] = meanParamVariances[nn]*(1-ci) + self.target['VARIANCES'][nn]*ci
        
        return meanParamBetas, meanParamVariances

    def gaussian_fuse_params(self, ci, Log=True):

        if Log:
            meanParamBetas, meanParamVariances = self.gaussian_loglinearpool(ci)
        else:
            meanParamBetas, meanParamVariances = self.gaussian_linearpool(ci)
        
        target_new = deepcopy(self.dag)
        for ni, node in enumerate(self.dag.nodes()):
            new_betas = []
            new_betas.append(meanParamBetas[ni,ni])
            new_cpd = pbn.LinearGaussianCPD(node, self.dag.parents(node))
            for pa in self.dag.parents(node):
                pa_j = self.nodemap[pa]
                new_betas.append(meanParamBetas[ni,pa_j])
            
            new_cpd.beta = np.array(new_betas)
            new_cpd.variance = meanParamVariances[ni]
            target_new.add_cpds([new_cpd])

        return target_new
    
    def spbn_fuse_params(self, ci, Log=True):

        if Log:
            meanParameters = self.spbn_loglinearpool(ci)
        

        target_new = pbn.SemiparametricBN(node_types = self.node_types, nodes=self.dag.nodes())
        for arc in self.dag.arcs():
            target_new.add_arc(arc[0],arc[1])
        target_new.fit(self.auxModels[0]['DATA'])

        for node, nodetype in self.node_types:
            if type(nodetype)==type(pbn.CKDEType()):
                joint_kde = meanParameters['joint_kde'][node] 
                marg_kde = meanParameters['marg_kde'][node]
                target_new.cpd(node).kde_joint().bandwidth = joint_kde
                target_new.cpd(node).kde_marg().bandwidth = marg_kde
            else:
                betas = meanParameters['betas'][node] 
                variances = meanParameters['variances'][node]
                target_new.cpd(node).beta = betas
                target_new.cpd(node).variance = variances

        
        return target_new

    def prepare_spbn(self, log=True):
        meanParameters = {}
        mean_jointbw = {}
        mean_margbw = {}
        mean_betas = {}
        mean_variances = {}

        for node, nodetype in self.node_types:

            cpd = self.dag.cpd(node)
            
            if type(nodetype)==type(pbn.CKDEType()):
                
                joint_bw = cpd.kde_joint().bandwidth
                marg_bw = cpd.kde_marg().bandwidth
                if log:
                    mean_jointbw[node] = np.ones_like(joint_bw)
                    mean_margbw[node] = np.ones_like(marg_bw)
                else:
                    mean_jointbw[node] = np.zeros_like(joint_bw)
                    mean_margbw[node] = np.zeros_like(marg_bw)
            
            else:
                bet = cpd.beta
                if log:
                    mean_betas[node] = np.ones_like(bet)
                    mean_variances[node] = 1
                else:
                    mean_betas[node] = np.zeros_like(bet)
                    mean_variances[node] = 0

        
        meanParameters['joint_kde'] = mean_jointbw
        meanParameters['marg_kde'] = mean_margbw
        meanParameters['betas'] = mean_betas
        meanParameters['variances'] = mean_variances
        
        return meanParameters
    
    @staticmethod
    def get_gaussian_parameters(model):
        
        nodes = model.nodes()
        bn_BETAS = np.zeros(shape=(len(nodes), len(nodes)))
        bn_VARS = np.zeros(shape=(len(nodes), 1))
        
        nodes_map = {}
        parameters = {}

        for nn, node in enumerate(nodes):
            nodes_map[node] = nn

        for i, node in enumerate(nodes):
            betas = model.cpd(node).beta
            variances =  model.cpd(node).variance
            parents = model.parents(node)
            
            bn_BETAS[i,i] =  betas[0] # ordenada
            bn_VARS[i] = variances # variance
            for pa, beta_pa in zip(parents, betas[1:]):
                j = nodes_map[pa]
                bn_BETAS[i,j] =  beta_pa # beta_ij

        parameters['BETAS'] = bn_BETAS
        parameters['VARIANCES'] = bn_VARS
        return parameters

    @staticmethod
    def get_spbn_parameters(model):
        
        nodes = model.nodes()
        parameters = {}
        jointbw = {}
        margbw = {}
        betas = {}
        variances = {}


        for i, node in enumerate(nodes):
            
            cpd = model.cpd(node)
            if type(cpd)==type(pbn.CKDE('x',[])):
                
                joint_bw = cpd.kde_joint().bandwidth
                marg_bw = cpd.kde_marg().bandwidth
                jointbw[node] = joint_bw.copy()
                margbw[node] = marg_bw.copy()
            
            else:
                bet = cpd.beta
                var = cpd.variance
                betas[node] = bet.copy()
                variances[node] = var
                
            
        
        parameters['joint_kde'] = jointbw
        parameters['marg_kde'] = margbw
        parameters['betas'] = betas
        parameters['variances'] = variances
        return parameters

        