from __future__ import annotations
from typing import List
from itertools import combinations
import itertools
from copy import deepcopy


from causallearn.utils.PCUtils.BackgroundKnowledge import BackgroundKnowledge
from causallearn.utils.cit import *
from causallearn.utils.PCUtils import Meek, SkeletonDiscovery


import pandas as pd 
import pybnesian as pbn
from scipy.linalg import sqrtm
import networkx as nx

from .Orient import OrientMAX
from .Skeleton import Skeleton
from .LearnParams import ParameterTL
from .CIT import CITCustom
from .utils import CustomThread
from .blacklist import forbiden_poles



class PCMaxTL:
    def __init__(self, 
                alpha: float, indep_test: str, datasAux: List[ pd.DataFrame] = [], node_names: List[str] = [], 
                stable: bool = True,
                background_knowledge: BackgroundKnowledge | None = None,
                npoles = None,
                verbose: bool = False,
                show_progress: bool = False) -> None:
        
        self.alpha = alpha
        self.indep_test = indep_test
      
        self.datasAux = datasAux
        self.node_names = node_names

        self.stable = stable
        self.bgKnowledge = background_knowledge
        self.npoles = npoles
        self.verbose = verbose
        self.show_progress = show_progress

        self.skeletonTL = Skeleton(alpha, stable, background_knowledge=self.bgKnowledge, verbose=verbose, show_progress=show_progress,)
        self.similars_count = {}

        self.hc = pbn.GreedyHillClimbing()
        self.cache = {'cycles':[], 'lastModel': None}
        


    def gaussian_parameter_learning(self, model: pbn, data0DF: pd.DataFrame,
                                Nt, Nmin, Nmax = 1500):
        target = deepcopy(model)
        arcs_t = target.arcs()
        nodes_t = target.nodes()

        ## learn target params
        target.fit(data0DF)
        paramsBN_target = ParameterTL.get_gaussian_parameters(target)
        paramsBN_target['DATA'] = data0DF
        
        
        ## learn auxiliary data params
        paramsBN_Aux = list()
        auxiliar = list()
        for datAux in self.datasAux:
            modeli = pbn.GaussianNetwork(nodes = nodes_t, arcs=arcs_t)
            datAux.columns = nodes_t
            modeli.fit(datAux)

            paramsBNi = ParameterTL.get_gaussian_parameters(modeli)
            paramsBNi['DATA'] = datAux
            paramsBN_Aux.append(paramsBNi.copy())
            auxiliar.append(modeli)


        paramLearning = ParameterTL(paramsBN_target, paramsBN_Aux, target, auxiliar, list(model.node_types().items()))

        ## function to fuse the target params with the auxiliary data params with a confidence coefficient
        ci = (Nt-Nmin)/(Nmax-Nmin)
        if ci>0.75:
            ci=0.75
        elif ci<0.25:
                ci=0.25
        
        target_new = paramLearning.gaussian_fuse_params(ci=ci)

        return target_new, target


    def spbn_parameter_learning(self, model: pbn, data0DF: pd.DataFrame,
                                Nt, Nmin, Nmax = 1500):
        
        nodes_t = model.nodes()
        node_types_t = list(model.node_types().items())

        ## learn target params
        target = pbn.SemiparametricBN(node_types = node_types_t, nodes=nodes_t)
        for arc in model.arcs():
            target.add_arc(arc[0],arc[1])
        target.fit(data0DF)
        paramsBN_target = ParameterTL.get_spbn_parameters(target)
        paramsBN_target['DATA'] = data0DF

        
        
        ## learn auxiliary data params
        paramsBN_Aux = list()
        auxiliar = list()
        for datAux in self.datasAux:

            modeli = pbn.SemiparametricBN(node_types = node_types_t, nodes=nodes_t)
            for arc in model.arcs():
                modeli.add_arc(arc[0],arc[1])
            datAux.columns = nodes_t
            modeli.fit(datAux)

            paramsBNi = ParameterTL.get_spbn_parameters(modeli)
            paramsBNi['DATA'] = datAux
            paramsBN_Aux.append(paramsBNi.copy())
            auxiliar.append(modeli)


        paramLearning = ParameterTL(paramsBN_target, paramsBN_Aux, target, auxiliar, node_types_t)

        ## function to fuse the target params with the auxiliary data params with a confidence coefficient
        ci = (Nt-Nmin)/(Nmax-Nmin)
        if ci>0.75:
            ci=0.75
        elif ci<0.25:
                ci=0.25
        
        target_new = paramLearning.spbn_fuse_params(ci=ci)

        return target_new, target      
    

    def structure_learning(self,  
                        data0DF: pd.DataFrame,
                        uc_rule: int = 1,
                        uc_priority: int = 3,
                        **kwargs
                        ):
        
        self.node_names = list(data0DF.columns) if len(self.node_names)==0 else self.node_names
        data0 = np.array(data0DF).copy()
        datasAux = [np.array(dataAux).copy() for dataAux in self.datasAux]

        # SKELETON 
        indep_test0 = CITCustom(data0, self.indep_test, alpha = self.alpha, **kwargs)
        if len(datasAux) > 0:
            indep_testAux_list = []
            for aux in datasAux:
                indep_testAux = CITCustom(aux, self.indep_test, alpha = self.alpha, **kwargs)
                indep_testAux_list.append(indep_testAux)


            cg_1 = self.skeletonTL.skeleton_discovery_TL(data0, indep_test0, 
                                                        datasAux, indep_testAux_list,
                                                        self.node_names)
            global_similarity = self.skeletonTL.global_similarities
            self.similars_count = self.skeletonTL.similars_count

            self.orient = OrientMAX(cg_1, data0, datasAux, global_similarity, self.similars_count,
                                alpha = self.alpha, background_knowledge = self.bgKnowledge)
            
        else:
            cg_1 = SkeletonDiscovery.skeleton_discovery(data0, self.alpha, indep_test0,
                                                        background_knowledge=self.bgKnowledge, verbose=self.verbose,
                                                        show_progress=self.show_progress, node_names=self.node_names)
            
            self.orient = OrientMAX(cg_1, data0, datasAux, 
                                alpha = self.alpha, background_knowledge = self.bgKnowledge)




        # FIRST ORIENTATION 
        if self.bgKnowledge is not None:
            self.orient.orient_by_background_knowledge()

        if uc_rule == 1:
            if uc_priority != -1:
                cg_2 = self.orient.maxpTL(uc_priority)
                self.similars_count = self.orient.similars_count
            else:
                cg_2 = self.orient.maxp(priority = 3) 
            cg = Meek.meek(cg_2, background_knowledge=self.bgKnowledge)

        elif uc_rule == 2:
            if uc_priority != -1:
                cg_2 = self.orient.definite_maxpTL(uc_priority)
                self.similars_count = self.orient.similars_count
            else:
                cg_2 = self.orient.definite_maxp(priority = 4)
            cg_before = Meek.definite_meek(cg_2, background_knowledge=self.bgKnowledge)
            cg = Meek.meek(cg_before, background_knowledge=self.bgKnowledge)

        else:
            raise ValueError("uc_rule should be in [1, 2]")
        
        

        # HANDLE CYCLES AND UNORIENTED EDGES
        
        simil = list(self.similars_count.values())
        if len(simil)>0:
            
            mostSimilarTask = np.argmax(simil)
            secondSimilarTask = np.argmax(simil.pop(mostSimilarTask))

            auxData1 = self.datasAux[mostSimilarTask]
            auxData2 = self.datasAux[secondSimilarTask]
            avgData = (auxData1.shape[0] + auxData2.shape[0])//2
            auxData1 = auxData1.sample(int(avgData*0.1))
            auxData2 = auxData2.sample(int(avgData*0.1))

            ## == reweight ==
            # data1_new, data2_new = self.transform_data(data0DF, auxData1, auxData2)
            # newData = pd.concat([data0DF, data1_new, data2_new ], axis=0, ignore_index=True)

            ## == 10%x2 similar ==
            newData = pd.concat([data0DF, auxData1, auxData2], axis=0, ignore_index=True)

            ## == alone ==
            # newData = data0DF
        
            model = self.orientate_remaining(cg, newData, verbose = False)
            return model, cg
            
        else:
            model = self.orientate_remaining(cg, data0DF, verbose = False)
            return model, cg

    def normalize_dataframe(self, dataf):
        maxval = max(dataf.max())
        minval = min(dataf.min())
        dataf0 = dataf.copy()
        for col in dataf.columns:
            dataf0[col] = (dataf0[col]-minval)/(maxval-minval)
        return dataf0

    def transform_data(self, data0, data1, data2):

        cs1 = data1.cov() + np.eye(data1.shape[1])
        cs2 = data2.cov()  + np.eye(data2.shape[1])
        ct = data0.cov() + np.eye(data0.shape[1])
        

        data1_new = data1 @ np.linalg.inv(sqrtm(cs1))  # weightening source
        data1_new = data1_new @ sqrtm(ct) # Re-coloring with target covariance
        data1_new.columns = data1.columns
        data2_new = data2 @ np.linalg.inv(sqrtm(cs2))  
        data2_new = data2_new @ sqrtm(ct) 
        data2_new.columns = data2.columns

        return data1_new, data2_new
        
    
            
    
    def get_restrictions(self, cg):
        pag = cg.G.graph
        depsx, depsy = np.where(pag==1)
        indsx, indsy = np.where(pag==0)
        whitelist, blacklist = [],[]
        whitebans = []
        for i,j in zip(depsx,depsy):
                if (self.node_names[j], self.node_names[i]) not in forbiden_poles[self.npoles]:

                    whitelist.append((self.node_names[j], self.node_names[i]))
                    whitebans.append((self.node_names[i], self.node_names[j]))

        for i,j in zip(indsx,indsy):
            if i!=j:
                blacklist.append((self.node_names[j],self.node_names[i]))

        undirected = set(combinations(self.node_names,2)) - set(whitelist) - set(blacklist) - set(whitebans)
        undirected = [(x,y) for x,y in undirected]
    
        
        return whitelist, blacklist,undirected
            
    def cycles_combinations(self, cycles):

        total_cycles = {}
        for n, cycle in enumerate(cycles):
            cuples_cycle = []
            
            for nvar in range(0, len(cycle)-1):
                cuples_cycle.append((cycle[nvar], cycle[nvar+1]))
            
            cuples_cycle.append((cycle[len(cycle)-1], cycle[0]))
            total_cycles[f'cycle{n}'] = cuples_cycle.copy()
        
        nkeys = len(total_cycles.keys())
        
        # if there are + than 1 cycles, make combinations of edges removals from the different ones
        if nkeys>1:
            total_combs = list(itertools.product(*list(total_cycles.values())))
            for n, comb in enumerate(total_combs):      
                uniq =  np.unique(comb, axis=0)
                if len(uniq) < 2:
                    total_combs.pop(n)
        else:
            total_combs = list(total_cycles.values())[0]
        
        return total_combs, nkeys

    def _test_cycles(self, arclist, blacklist, dataf, vl):
   
        
        # start_model = pbn.GaussianNetwork(nodes = self.node_names, arcs=list(arclist))
        start_model = pbn.SemiparametricBN(nodes = self.node_names, arcs=list(arclist))
        pool = pbn.OperatorPool([pbn.ArcOperatorSet(), 
                                 pbn.ChangeNodeTypeSet()
                                 ])
        model = self.hc.estimate(operators = pool, score = vl, start = start_model,
                                 arc_blacklist=blacklist, 
                                 )
        model.fit(dataf)

        score = np.mean(model.logl(dataf))
        return score, model
        


    def orientate_remaining(self, cg, dataf, verbose = False):
        
        vl = pbn.CVLikelihood(dataf, k=5,seed=0)
        lastCycles = self.cache['cycles']

        whitelist, blacklist, undirected = self.get_restrictions(cg)

        if self.npoles is not None:
            blacklist += forbiden_poles[self.npoles]
            blacklist = np.unique(blacklist,axis=0).tolist()
            blacklist = [(nod[0],nod[1]) for nod in blacklist]
        


        # DETECT CYCLES
        G = nx.DiGraph(whitelist)
        cycles = [cycle for cycle in nx.simple_cycles(G)]
        if len(cycles) > 0 and cycles!=lastCycles:
            
            total_combs, nkeys = self.cycles_combinations(cycles)
            scores, models = [], []

            # Remove the edge, or cobinations of edges, from the whitelist and score which removal increases the mean log-likehood
            threads = list()
            for edge in total_combs:   
                if nkeys>1:
                    arclist = set(whitelist) - set(edge) 
                else:
                    arclist = set(whitelist) - set([edge]) 
            
           
                t = CustomThread(target=self._test_cycles, args=(arclist, blacklist, dataf, vl))
                threads.append(t)
                t.start()
                t.join()

            for index, t in enumerate(threads):
                sc, mdel = t.value()
                scores.append(sc)
                models.append(deepcopy(mdel))

            best_opt = np.argmax(scores)
            
            if verbose:
                print(f'Edges with cycles: {total_combs}')
                print(f'Scores excluding 1: {scores}')
                print(f'Best {best_opt} with mean value: {scores[best_opt]}')
                print(f'{total_combs[best_opt]} Edge will be removed\n' )

            model = models[best_opt]
            self.cache['cycles'] = cycles
            self.cache['lastModel'] = deepcopy(model)
            return model

        elif len(lastCycles) > 0 and cycles == lastCycles:
            return self.cache['lastModel']
    

        # start_model = pbn.GaussianNetwork(nodes = self.node_names, arcs=whitelist)
        start_model = pbn.SemiparametricBN(nodes = self.node_names, arcs=whitelist)
 
        if len(undirected)<1:
               return start_model
    
        
        pool = pbn.OperatorPool([pbn.ArcOperatorSet(), 
                                    pbn.ChangeNodeTypeSet()
                                    ])
        model = self.hc.estimate(operators = pool, score = vl, start = start_model,
                                 arc_blacklist=blacklist,
                                 )
       
        
        return model


    


