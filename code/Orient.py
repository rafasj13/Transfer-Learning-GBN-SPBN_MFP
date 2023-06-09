from copy import deepcopy
from typing import List
from numpy import ndarray

from causallearn.graph.Edge import Edge
from causallearn.graph.Endpoint import Endpoint
from causallearn.utils.PCUtils.Helper import sort_dict_ascending
from causallearn.utils.PCUtils.BackgroundKnowledge import BackgroundKnowledge
from causallearn.utils.PCUtils import UCSepset
from causallearn.graph.GraphClass import CausalGraph

from .Skeleton import CausalGraphCustom, Skeleton


class OrientMAX:
    def __init__(self, cg: CausalGraphCustom, data0:ndarray, datasAux: List[ndarray], 
                 global_similarity: List[int] = None, similars_count: dict = None, 
                 alpha: float = 0.05, background_knowledge:BackgroundKnowledge = None) -> None:
        
        self.cg = deepcopy(cg)
        self.data0 = data0
        self.datasAux = datasAux
        self.global_similarity = global_similarity
        self.similars_count = similars_count

        self.alpha = alpha
        self.bgKnowledge = background_knowledge


    def maxp(self, priority):
        """
        Run (MaxP) to orient unshielded colliders

        Parameters
        ----------
        cg : a CausalGraph object
        priority : rule of resolving conflicts between unshielded colliders (default = 3)
            0: overwrite
            1: orient bi-directed
            2. prioritize existing colliders
            3. prioritize stronger colliders
            4. prioritize stronger* colliers
        background_knowledge : artificial background background_knowledge

        Returns
        -------
        cg_new : a CausalGraph object. Where cg_new.G.graph[j,i]=1 and cg_new.G.graph[i,j]=-1 indicates  i --> j ,
                        cg_new.G.graph[i,j] = cg_new.G.graph[j,i] = -1 indicates i --- j,
                        cg_new.G.graph[i,j] = cg_new.G.graph[j,i] = 1 indicates i <-> j.
        """
        return UCSepset.maxp(self.cg, priority = priority, background_knowledge=self.bgKnowledge)
    
    def definite_maxp(self, priority):
        """
        Run (Definite_MaxP) to orient unshielded colliders

        Parameters
        ----------
        cg : a CausalGraph object
        priority : rule of resolving conflicts between unshielded colliders (default = 3)
            0: overwrite
            1: orient bi-directed
            2. prioritize existing colliders
            3. prioritize stronger colliders
            4. prioritize stronger* colliers
        background_knowledge : artificial background background_knowledge

        Returns
        -------
        cg_new : a CausalGraph object. Where cg_new.G.graph[j,i]=1 and cg_new.G.graph[i,j]=-1 indicates  i --> j ,
                        cg_new.G.graph[i,j] = cg_new.G.graph[j,i] = -1 indicates i --- j,
                        cg_new.G.graph[i,j] = cg_new.G.graph[j,i] = 1 indicates i <-> j.
        """
        return UCSepset.definite_maxp(self.cg, self.alpha, priority = priority, background_knowledge=self.bgKnowledge)
    
    def maxpTL(self,
            priority: int = 3, 
            ):
       

        assert priority in [0, 1, 2, 3, 4]

        
        gs = deepcopy(self.global_similarity)
        similars_count = deepcopy(self.similars_count)
        UC_dict = {}
        UT = [(i, j, k) for (i, j, k) in self.cg.find_unshielded_triples() if i < k]  # Not considering symmetric triples

        for (x, y, z) in UT:
            if (self.bgKnowledge is not None) and \
                    (self.bgKnowledge.is_forbidden(self.cg.G.nodes[x], self.cg.G.nodes[y]) or
                    self.bgKnowledge.is_forbidden(self.cg.G.nodes[z], self.cg.G.nodes[y]) or
                    self.bgKnowledge.is_required(self.cg.G.nodes[y], self.cg.G.nodes[x]) or
                    self.bgKnowledge.is_required(self.cg.G.nodes[y], self.cg.G.nodes[z])):
                continue

            cond_with_y = self.cg.find_cond_sets_with_mid(x, z, y)
            cond_without_y = self.cg.find_cond_sets_without_mid(x, z, y)
            
            max_p_contain_y,max_p_not_contain_y = [],[]
            for S in cond_with_y: 
                
                pval0 = self.cg.ci_test0(x, z, S)   
                task_id, self.similars_count = Skeleton.get_most_similar_task(self.cg, gs, similars_count, self.datasAux, x, z, S)
                pvalAux = self.cg.ci_testAux(task_id, x, z, S)

                alpha0 = self.data0.shape[0]/(self.data0.shape[0] + self.datasAux[task_id].shape[0])
                pfused = alpha0 * pval0 + (1-alpha0) * pvalAux

                max_p_contain_y.append(pfused)
            
            for S in cond_without_y: 
                
                pval0 = self.cg.ci_test0(x, z, S)   
                task_id, self.similars_count = Skeleton.get_most_similar_task(self.cg, gs, similars_count, self.datasAux, x, z, S)
                pvalAux = self.cg.ci_testAux(task_id, x, z, S)

                alpha0 = self.data0.shape[0]/(self.data0.shape[0] + self.datasAux[task_id].shape[0])
                pfused = alpha0 * pval0 + (1-alpha0) * pvalAux
                
                max_p_not_contain_y.append(pfused)



            max_p_contain_y = max(max_p_contain_y)
            max_p_not_contain_y = max(max_p_not_contain_y)


            if max_p_not_contain_y > max_p_contain_y:
                if priority == 0:  # 0: overwrite
                    edge1 = self.cg.G.get_edge(self.cg.G.nodes[x], self.cg.G.nodes[y])
                    if edge1 is not None:
                        self.cg.G.remove_edge(edge1)
                    edge2 = self.cg.G.get_edge(self.cg.G.nodes[y], self.cg.G.nodes[x])
                    if edge2 is not None:
                        self.cg.G.remove_edge(edge2)
                    # Fully orient the edge irrespective of what have been oriented
                    self.cg.G.add_edge(Edge(self.cg.G.nodes[x], self.cg.G.nodes[y], Endpoint.TAIL, Endpoint.ARROW))

                    edge3 = self.cg.G.get_edge(self.cg.G.nodes[y], self.cg.G.nodes[z])
                    if edge3 is not None:
                        self.cg.G.remove_edge(edge3)
                    edge4 = self.cg.G.get_edge(self.cg.G.nodes[z], self.cg.G.nodes[y])
                    if edge4 is not None:
                        self.cg.G.remove_edge(edge4)
                    self.cg.G.add_edge(Edge(self.cg.G.nodes[z], self.cg.G.nodes[y], Endpoint.TAIL, Endpoint.ARROW))

                elif priority == 1:  # 1: orient bi-directed
                    edge1 = self.cg.G.get_edge(self.cg.G.nodes[x], self.cg.G.nodes[y])
                    if edge1 is not None:
                        if self.cg.G.graph[x, y] == Endpoint.TAIL.value and self.cg.G.graph[y, x] == Endpoint.TAIL.value:
                            self.cg.G.remove_edge(edge1)
                            self.cg.G.add_edge(Edge(self.cg.G.nodes[x], self.cg.G.nodes[y], Endpoint.TAIL, Endpoint.ARROW))
                        elif self.cg.G.graph[x, y] == Endpoint.ARROW.value and self.cg.G.graph[y, x] == Endpoint.TAIL.value:
                            self.cg.G.remove_edge(edge1)
                            self.cg.G.add_edge(Edge(self.cg.G.nodes[x], self.cg.G.nodes[y], Endpoint.ARROW, Endpoint.ARROW))
                    else:
                        self.cg.G.add_edge(Edge(self.cg.G.nodes[x], self.cg.G.nodes[y], Endpoint.TAIL, Endpoint.ARROW))

                    edge2 = self.cg.G.get_edge(self.cg.G.nodes[z], self.cg.G.nodes[y])
                    if edge2 is not None:
                        if self.cg.G.graph[z, y] == Endpoint.TAIL.value and self.cg.G.graph[y, z] == Endpoint.TAIL.value:
                            self.cg.G.remove_edge(edge2)
                            self.cg.G.add_edge(Edge(self.cg.G.nodes[z], self.cg.G.nodes[y], Endpoint.TAIL, Endpoint.ARROW))
                        elif self.cg.G.graph[z, y] == Endpoint.ARROW.value and self.cg.G.graph[y, z] == Endpoint.TAIL.value:
                            self.cg.G.remove_edge(edge2)
                            self.cg.G.add_edge(Edge(self.cg.G.nodes[z], self.cg.G.nodes[y], Endpoint.ARROW, Endpoint.ARROW))
                    else:
                        self.cg.G.add_edge(Edge(self.cg.G.nodes[z], self.cg.G.nodes[y], Endpoint.TAIL, Endpoint.ARROW))

                elif priority == 2:  # 2: prioritize existing
                    if (not self.cg.is_fully_directed(y, x)) and (not self.cg.is_fully_directed(y, z)):
                        edge1 = self.cg.G.get_edge(self.cg.G.nodes[x], self.cg.G.nodes[y])
                        if edge1 is not None:
                            self.cg.G.remove_edge(edge1)
                        # Orient only if the edges have not been oriented the other way around
                        self.cg.G.add_edge(Edge(self.cg.G.nodes[x], self.cg.G.nodes[y], Endpoint.TAIL, Endpoint.ARROW))

                        edge2 = self.cg.G.get_edge(self.cg.G.nodes[z], self.cg.G.nodes[y])
                        if edge2 is not None:
                            self.cg.G.remove_edge(edge2)
                        self.cg.G.add_edge(Edge(self.cg.G.nodes[z], self.cg.G.nodes[y], Endpoint.TAIL, Endpoint.ARROW))

                elif priority == 3:
                    UC_dict[(x, y, z)] = max_p_contain_y

                elif priority == 4:
                    UC_dict[(x, y, z)] = max_p_not_contain_y

        if priority in [0, 1, 2]:
            return self.cg

        else:
            if priority == 3:  # 3. Order colliders by p_{xz|y} in ascending order
                UC_dict = sort_dict_ascending(UC_dict)
            else:  # 4. Order colliders by p_{xz|not y} in descending order
                UC_dict = sort_dict_ascending(UC_dict, True)

            for (x, y, z) in UC_dict.keys():
                if (self.bgKnowledge is not None) and \
                        (self.bgKnowledge.is_forbidden(self.cg.G.nodes[x], self.cg.G.nodes[y]) or
                        self.bgKnowledge.is_forbidden(self.cg.G.nodes[z], self.cg.G.nodes[y]) or
                        self.bgKnowledge.is_required(self.cg.G.nodes[y], self.cg.G.nodes[x]) or
                        self.bgKnowledge.is_required(self.cg.G.nodes[y], self.cg.G.nodes[z])):
                    continue

                if (not self.cg.is_fully_directed(y, x)) and (not self.cg.is_fully_directed(y, z)):
                    edge1 = self.cg.G.get_edge(self.cg.G.nodes[x], self.cg.G.nodes[y])
                    if edge1 is not None:
                        self.cg.G.remove_edge(edge1)
                    # Orient only if the edges have not been oriented the other way around
                    self.cg.G.add_edge(Edge(self.cg.G.nodes[x], self.cg.G.nodes[y], Endpoint.TAIL, Endpoint.ARROW))

                    edge2 = self.cg.G.get_edge(self.cg.G.nodes[z], self.cg.G.nodes[y])
                    if edge2 is not None:
                        self.cg.G.remove_edge(edge2)
                    self.cg.G.add_edge(Edge(self.cg.G.nodes[z], self.cg.G.nodes[y], Endpoint.TAIL, Endpoint.ARROW))

            return self.cg
        
    def definite_maxpTL(self, priority: int = 4,
                   ) -> CausalGraphCustom:
       
        assert 1 > self.alpha >= 0
        assert priority in [2, 3, 4]

       
        gs = deepcopy(self.global_similarity)
        similars_count = deepcopy(self.similars_count)

        UC_dict = {}
        UT = [(i, j, k) for (i, j, k) in self.cg.find_unshielded_triples() if i < k]  # Not considering symmetric triples

        for (x, y, z) in UT:
            cond_with_y = self.cg.find_cond_sets_with_mid(x, z, y)
            cond_without_y = self.cg.find_cond_sets_without_mid(x, z, y)
            max_p_contain_y = 0
            max_p_not_contain_y = 0
            uc_bool = True
            nuc_bool = True

            for S in cond_with_y:
                pval0 = self.cg.ci_test0(x, z, S)   
                task_id, self.similars_count = Skeleton.get_most_similar_task(self.cg, gs, similars_count, self.datasAux, x, z, S)
                pvalAux = self.cg.ci_testAux(task_id, x, z, S)

                alpha0 = self.data0.shape[0]/(self.data0.shape[0] + self.datasAux[task_id].shape[0])
                pfused = alpha0 * pval0 + (1-alpha0) * pvalAux

                if pfused > self.alpha:
                    uc_bool = False
                    break
                elif pfused > max_p_contain_y:
                    max_p_contain_y = pfused

            for S in cond_without_y:
                pval0 = self.cg.ci_test0(x, z, S)   
                task_id, self.similars_count = Skeleton.get_most_similar_task(self.cg, gs, similars_count, self.datasAux, x, z, S)
                pvalAux = self.cg.ci_testAux(task_id, x, z, S)

                alpha0 = self.data0.shape[0]/(self.data0.shape[0] + self.datasAux[task_id].shape[0])
                pfused = alpha0 * pval0 + (1-alpha0) * pvalAux

                if pfused > self.alpha:
                    nuc_bool = False
                    if not uc_bool:
                        break  # ambiguous triple
                if pfused > max_p_not_contain_y:
                    max_p_not_contain_y = pfused

            if uc_bool:
                if nuc_bool:
                    if max_p_not_contain_y > max_p_contain_y:
                        if priority in [2, 3]:
                            UC_dict[(x, y, z)] = max_p_contain_y
                        if priority == 4:
                            UC_dict[(x, y, z)] = max_p_not_contain_y
                    else:
                        self.cg.definite_non_UC.append((x, y, z))
                else:
                    if priority in [2, 3]:
                        UC_dict[(x, y, z)] = max_p_contain_y
                    if priority == 4:
                        UC_dict[(x, y, z)] = max_p_not_contain_y

            elif nuc_bool:
                self.cg.definite_non_UC.append((x, y, z))

        if priority == 3:  # 3. Order colliders by p_{xz|y} in ascending order
            UC_dict = sort_dict_ascending(UC_dict)
        elif priority == 4:  # 4. Order colliders by p_{xz|not y} in descending order
            UC_dict = sort_dict_ascending(UC_dict, True)

        for (x, y, z) in UC_dict.keys():
            if (self.bgKnowledge is not None) and \
                    (self.bgKnowledge.is_forbidden(self.cg.G.nodes[x], self.cg.G.nodes[y]) or
                    self.bgKnowledge.is_forbidden(self.cg.G.nodes[z], self.cg.G.nodes[y]) or
                    self.bgKnowledge.is_required(self.cg.G.nodes[y], self.cg.G.nodes[x]) or
                    self.bgKnowledge.is_required(self.cg.G.nodes[y], self.cg.G.nodes[z])):
                continue

            if (not self.cg.is_fully_directed(y, x)) and (not self.cg.is_fully_directed(y, z)):
                edge1 = self.cg.G.get_edge(self.cg.G.nodes[x], self.cg.G.nodes[y])
                if edge1 is not None:
                    self.cg.G.remove_edge(edge1)
                # Orient only if the edges have not been oriented the other way around
                self.cg.G.add_edge(Edge(self.cg.G.nodes[x], self.cg.G.nodes[y], Endpoint.TAIL, Endpoint.ARROW))

                edge2 = self.cg.G.get_edge(self.cg.G.nodes[z], self.cg.G.nodes[y])
                if edge2 is not None:
                    self.cg.G.remove_edge(edge2)
                self.cg.G.add_edge(Edge(self.cg.G.nodes[z], self.cg.G.nodes[y], Endpoint.TAIL, Endpoint.ARROW))

                self.cg.definite_UC.append((x, y, z))

        return self.cg