from __future__ import annotations
from itertools import combinations

from typing import List
import warnings
import io 

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from numpy import ndarray
from tqdm.auto import tqdm

from causallearn.graph.GraphClass import CausalGraph
from causallearn.utils.PCUtils.BackgroundKnowledge import BackgroundKnowledge
from causallearn.utils.GraphUtils import GraphUtils
from causallearn.utils.cit import *
from causallearn.utils.PCUtils import SkeletonDiscovery
from causallearn.utils.PCUtils.Helper import append_value
from causallearn.utils.PCUtils.BackgroundKnowledge import BackgroundKnowledge

from .utils import CustomThread



class CausalGraphCustom(CausalGraph):
    def __init__(self, no_of_var: int, alpha:float = 0.05, node_names: List[str] | None = None):
        super().__init__(no_of_var, node_names)
        self.alpha = alpha
    
    def set_ind_test0(self, indep_test0:CIT):
        """Set the Target conditional independence test that will be used"""
        self.test0 = indep_test0

    def set_ind_testAux(self, indep_testAux_list:List[CIT]):
        """Set the Auxiliar conditional independence tests that will be used"""
        self.testAux = indep_testAux_list

    def ci_test0(self, i: int, j: int, S) -> float:
        """Define the Target conditional independence test"""
        # assert i != j and not i in S and not j in S
        if self.test0.method == 'mc_fisherz': return self.test0(i, j, S, self.nx_skel, self.prt_m)

        return self.test0(i,j,S)
    
    def ci_testAux(self, d:int, i: int, j: int, S) -> float:
        """Define the specific auxiliar conditional independence test (d: id of the auxiliar data)"""
        # assert i != j and not i in S and not j in S
        if self.testAux[d].method == 'mc_fisherz': return self.testAux[d](i, j, S, self.nx_skel, self.prt_m)
        return self.testAux[d](i,j,S)
        
    
    def draw_pydot_graph(self, labels: List[str] | None = None):
        """Draw nx_graph if skel = False and draw nx_skel otherwise"""
        warnings.filterwarnings("ignore", category=UserWarning)
        pyd = GraphUtils.to_pydot(self.G, labels=labels)
        tmp_png = pyd.create_png(f="png")
        fp = io.BytesIO(tmp_png)
        img = mpimg.imread(fp, format='png')
        plt.rcParams["figure.figsize"] = [15, 7]
        plt.rcParams["figure.autolayout"] = True
        plt.axis('off')
        plt.imshow(img)
        plt.show()




class Skeleton:
    def __init__(self,
                alpha: float, 
                stable: bool = True,
                background_knowledge: BackgroundKnowledge | None = None, 
                verbose: bool = False,
                show_progress: bool = False,
                ) -> None:
        
        self.alpha = alpha
        self.stable = stable
        self.bgKnowledge = background_knowledge
        self.verbose = verbose
        self.show_progress = show_progress


        self.global_similarities = {}
        self.similars_count = {}


    @staticmethod
    def _similarity(cg, task_id, x, y, S, global_similarity:List):
        local_similarity_j =  Skeleton.local_similarity(cg, task_id, x, y, S)
        combined_similarity = global_similarity[task_id] * local_similarity_j

        return combined_similarity
        
        
    @staticmethod
    def get_most_similar_task(cg: CausalGraphCustom, global_similarity:List , similars_count:dict, datasAux: List[ndarray], x: int, y: int, S) -> int:

        combined_similarities = list()
        threads = list()
        for task_id, dataAux in enumerate(datasAux):
            # calculate local similarity
            t = CustomThread(target=Skeleton._similarity, args=(cg, task_id, x, y, S, global_similarity))
            threads.append(t)
            t.start()
            t.join()
            # local_similarity_j =  Skeleton.local_similarity(cg, task_id, x, y, S)
            # combined_similarity = global_similarity[task_id] * local_similarity_j
            # combined_similarities.append(combined_similarity)

        for index, t in enumerate(threads):
            combined_similarity = t.value()
            combined_similarities.append(combined_similarity)
        

        maxTaskID = np.argmax(combined_similarities)
        maxTasklist = [i for i, j in enumerate(combined_similarities) if j == max(combined_similarities)]
        if len(list(similars_count.keys())) == 0:
            for taskid, dta in enumerate(datasAux):
                similars_count[taskid] = 0
        
        for taskid in maxTasklist:
            similars_count[taskid] += 1
         
        
        return maxTaskID, similars_count

    @staticmethod
    def local_similarity(cg:CausalGraphCustom, task_id:int, x:int, y:int, S) -> float:

        pval0 = cg.ci_test0(x, y, S)
        pvalAux = cg.ci_testAux(task_id, x, y, S)        

        if (pval0 > 0.05 and pvalAux > 0.05) or (pval0 < 0.05 and pvalAux < 0.05):
            return 1
        else: 
            return 0.5
        
    def global_similarity(self, data, datasAux):
        indep_test0 = CIT(data, fisherz)
        
        cg0 = SkeletonDiscovery.skeleton_discovery(data, 0.05, indep_test0, show_progress=False, node_names=self.node_names)
        cg0Matrix = cg0.G.graph

        global_similarities = {}
        for taskId, dataAux in enumerate(datasAux):
            indep_testAux = CIT(dataAux, fisherz)
            cg = SkeletonDiscovery.skeleton_discovery(dataAux, 0.05, indep_testAux, show_progress=False, node_names=self.node_names)

            cgAuxMatrix = cg.G.graph
            equalmat = np.equal(cg0Matrix, cgAuxMatrix)
            pos_commons = np.where(equalmat==True)[0]
            deps_indeps = (len(pos_commons)- data.shape[1])*0.5
            global_similarities[taskId] = deps_indeps

        return global_similarities
        


    def skeleton_discovery_TL(self,
                            data0: ndarray, 
                            indep_test0: CIT,
                            datasAux: List[ndarray],
                            indep_testAux_list: List[CIT], 
                            node_names = List[str]
        ) -> CausalGraph:
        """
        Perform skeleton discovery

        Parameters
        ----------
        data : data set (numpy ndarray), shape (n_samples, n_features). The input data, where n_samples is the number of
                samples and n_features is the number of features.
        alpha: float, desired significance level of independence tests (p_value) in (0,1)
        indep_test : class CIT, the independence test being used
                [fisherz, chisq, gsq, mv_fisherz, kci]
            - fisherz: Fisher's Z conditional independence test
            - chisq: Chi-squared conditional independence test
            - gsq: G-squared conditional independence test
            - mv_fisherz: Missing-value Fishers'Z conditional independence test
            - kci: Kernel-based conditional independence test
        stable : run stabilized skeleton discovery if True (default = True)
        background_knowledge : background knowledge
        verbose : True iff verbose output should be printed.
        show_progress : True iff the algorithm progress should be show in console.
        node_names: Shape [n_features]. The name for each feature (each feature is represented as a Node in the graph, so it's also the node name)

        Returns
        -------
        cg : a CausalGraph object. Where cg.G.graph[j,i]=0 and cg.G.graph[i,j]=1 indicates  i -> j ,
                        cg.G.graph[i,j] = cg.G.graph[j,i] = -1 indicates i -- j,
                        cg.G.graph[i,j] = cg.G.graph[j,i] = 1 indicates i <-> j.

        """
        assert type(data0) == ndarray
        assert 0 < self.alpha < 1

        self.node_names = node_names
        no_of_var = data0.shape[1]
        cg = CausalGraphCustom(no_of_var, self.alpha, self.node_names)
        cg.set_ind_test0(indep_test0)
        cg.set_ind_testAux(indep_testAux_list)

        

        depth = -1
        pbar = tqdm(total=no_of_var) if self.show_progress else None


        # calculate global similarity
        self.global_similarities = self.global_similarity(data0, datasAux)

        while cg.max_degree() - 1 > depth:
            depth += 1
            edge_removal = []
            if self.show_progress:
                pbar.reset()
            for x in range(no_of_var):
                if self.show_progress:
                    pbar.update()
                if self.show_progress:
                    pbar.set_description(f'Depth={depth}, working on node {x}')

                Neigh_x = cg.neighbors(x)
                if len(Neigh_x) < depth - 1:
                    continue

                for y in Neigh_x:
                    knowledge_ban_edge = False
                    sepsets = set()
                    if self.bgKnowledge is not None and (
                            self.bgKnowledge.is_forbidden(cg.G.nodes[x], cg.G.nodes[y])
                            and self.bgKnowledge.is_forbidden(cg.G.nodes[y], cg.G.nodes[x])):
                        knowledge_ban_edge = True
                    if knowledge_ban_edge:
                        if not self.stable:
                            edge1 = cg.G.get_edge(cg.G.nodes[x], cg.G.nodes[y])
                            if edge1 is not None:
                                cg.G.remove_edge(edge1)
                            edge2 = cg.G.get_edge(cg.G.nodes[y], cg.G.nodes[x])
                            if edge2 is not None:
                                cg.G.remove_edge(edge2)
                            append_value(cg.sepset, x, y, ())
                            append_value(cg.sepset, y, x, ())
                            break
                        else:
                            edge_removal.append((x, y))  # after all conditioning sets at
                            edge_removal.append((y, x))  # depth l have been considered

                    Neigh_x_noy = np.delete(Neigh_x, np.where(Neigh_x == y))
                    for S in combinations(Neigh_x_noy, depth):

                        pval0 = cg.ci_test0(x, y, S)

                        # Find the most similar auxiliary task, k, and its similarity measure Sk_XY
                        task_id, self.similars_count = self.get_most_similar_task(cg, self.global_similarities, self.similars_count, 
                                                                                  datasAux, x, y, S)

                        # Determine the confidence measures alpha(X,Y|S) auxiliary task
                        pvalAux = cg.ci_testAux(task_id, x, y, S)
                        alpha0 = data0.shape[0]/(data0.shape[0] + datasAux[task_id].shape[0])
                        pfused = alpha0 * pval0 + (1-alpha0) * pvalAux


                        if pfused > self.alpha:
                    
                            if self.verbose:
                                print(f'Argmax:{task_id}')
                                print(f'pval0: {pval0}, alpha0: {alpha0}')
                                print(f'pvalAux: {pvalAux} alphaAux: {1-alpha0}')
                                print('%d ind %d | %s with p-value %f and alpha %f\n' % (x, y, S, pfused, self.alpha))
                            if not self.stable:
                                edge1 = cg.G.get_edge(cg.G.nodes[x], cg.G.nodes[y])
                                if edge1 is not None:
                                    cg.G.remove_edge(edge1)
                                edge2 = cg.G.get_edge(cg.G.nodes[y], cg.G.nodes[x])
                                if edge2 is not None:
                                    cg.G.remove_edge(edge2)
                                append_value(cg.sepset, x, y, S)
                                append_value(cg.sepset, y, x, S)
                                break

                            else:
                                edge_removal.append((x, y))  # after all conditioning sets at
                                edge_removal.append((y, x))  # depth l have been considered
                                for s in S:
                                    sepsets.add(s)
                        else:
                            
                            if self.verbose:
                                print(f'Argmax:{task_id}')
                                print(f'pval0: {pval0}, alpha0: {alpha0}')
                                print(f'pvalAux: {pvalAux} alphaAux: {1-alpha0}')
                                print('%d dep %d | %s with p-value %f and alpha %f\n' % (x, y, S, pfused, self.alpha))

                    append_value(cg.sepset, x, y, tuple(sepsets))
                    append_value(cg.sepset, y, x, tuple(sepsets))

            if self.show_progress:
                pbar.refresh()

            for (x, y) in list(set(edge_removal)):
                edge1 = cg.G.get_edge(cg.G.nodes[x], cg.G.nodes[y])
                if edge1 is not None:
                    cg.G.remove_edge(edge1)

        if self.show_progress:
            pbar.close()

        return cg