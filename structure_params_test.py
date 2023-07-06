import os
import threading

from causallearn.utils.cit import *

from code.utils import get_df_filtered, draw_graph
from code.PCTL import PCMaxTL
from code.CIT import *
from code.blacklist import blacklist_knowledge

from tqdm.auto import tqdm
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import pybnesian as pbn
from copy import deepcopy
matplotlib.use('agg')



def train(datasAux, poles, VANO, jumps, 
          save_path_structures, save_path_std_structures, 
          save_path_params, save_path_std_params, 
          savebool_structures, savebool_params):
    

    score_fused_structures, score_alone_structures = [],[]
    score_fused_std_structures, score_alone_std_structures = [], []

    scores_fused_params, scores_alone_params, scores_structure_params = [], [], []
    scores_fused_std_params, scores_alone_std_params, scores_structure_std_params = [], [], []

    instances = []
    similarsdf0 = []

    pbar = tqdm(total=2000//jumps)
    

    pctl = PCMaxTL(alpha=0.05, indep_test = fisherz, datasAux = datasAux, npoles=keypoles,verbose=False)
    pc = PCMaxTL(alpha=0.05, indep_test = fisherz, npoles=keypoles)
    for niter in range(20,2000+jumps,jumps):

        datat0 = dfs_bilbao[VANO].iloc[:niter, poles]
        data0_train = dfs_bilbao[VANO].iloc[:3000, poles]
        data0_test = dfs_bilbao[VANO].iloc[3000:, poles]

        
        #transfer learning
        model_fused_s,_ = pctl.structure_learning(datat0, uc_rule = 2, uc_priority = 4)
        similarsdf0.append(pd.DataFrame(pctl.similars_count, index=[0]).copy())
        model_fused_sp, model_fused_s = pctl.gaussian_parameter_learning(model_fused_s, datat0, Nt = niter, Nmin = jumps, Nmax = 1500)
        # model_fused_sp, model_fused_s = pctl.spbn_parameter_learning(model_fused_s, datat0, Nt = niter, Nmin = jumps, Nmax = 1500) 

        #alone
        model_alone,_ = pc.structure_learning(datat0, uc_rule=2, uc_priority=-1) 


        ## =========== SCORE STRUCTURES ====================

        # model fused
        nodes_t = model_fused_s.nodes()
        node_types_t = list(model_fused_s.node_types().items())
        model_fused_structure = pbn.SemiparametricBN(node_types = node_types_t, nodes=nodes_t)
        for arc in model_fused_s.arcs():
            model_fused_structure.add_arc(arc[0],arc[1])

        model_fused_structure.fit(data0_train)
        scorefused = model_fused_structure.logl(data0_test)
        score_fused_structures.append(np.mean(scorefused))
        score_fused_std_structures.append(np.std(scorefused))

        # model alone
        model_alone_structures = deepcopy(model_alone)
        model_alone_structures.fit(data0_train)
        scorealone = model_alone_structures.logl(data0_test)
        score_alone_structures.append(np.mean(scorealone))
        score_alone_std_structures.append(np.std(scorealone))
      


        

        ## =========== SCORE PARAMETERS ====================
        ## only structure
        scorestructure = model_fused_s.logl(data0_test)
        scores_structure_params.append(np.mean(scorestructure))
        scores_structure_std_params.append(np.std(scorestructure))
        
        ## structure and params
        scorefused = model_fused_sp.logl(data0_test)
        scores_fused_params.append(np.mean(scorefused))
        scores_fused_std_params.append(np.std(scorefused))
        
        # model alone
        model_alone.fit(datat0)
        scorealone = model_alone.logl(data0_test)
        scores_alone_params.append(np.mean(scorealone))
        scores_alone_std_params.append(np.std(scorealone))
        
        
        

        ## ========= UPDATE =========
        instances.append(niter)
        pbar.update()


        if savebool_structures:

            if not os.path.exists(save_path_structures + 'similars'):
                os.mkdir(save_path_structures + 'similars')
            similarsdf = pd.concat(similarsdf0, ignore_index=True)
            similarsdf.max().to_frame().T.to_csv(save_path_structures + f'similars/target_{VANO}.csv', index=False)
    
            scoresfPD_structures = pd.DataFrame({'score_fused': score_fused_structures, 'score_alone':score_alone_structures})
            if not os.path.exists(save_path_structures + 'scores'):
                os.mkdir(save_path_structures + 'scores')

            scoresfPD_std_structures = pd.DataFrame({'score_fused': score_fused_std_structures,'score_alone':score_alone_std_structures})
            if not os.path.exists(save_path_std_structures + 'scores'):
                os.mkdir(save_path_std_structures + 'scores')
            
            scoresfPD_structures.to_csv(save_path_structures + f'scores/target_{VANO}.csv', index=False)    
            scoresfPD_std_structures.to_csv(save_path_std_structures + f'scores/target_{VANO}.csv', index=False)  
            
   

            plt.figure(figsize=(15,7))
            plt.title(f'Target {VANO}')
            plt.plot(instances, score_fused_structures,  label='Tranfer Learning')
            plt.plot(instances, score_alone_structures, label='Alone')
            plt.xlabel('Instances')
            plt.ylabel('Log-Likelihood')
            plt.legend()
            plt.savefig(save_path_structures + f'target_{VANO}.png')
            plt.close('all')

            plt.figure(figsize=(15,7))
            plt.title(f'Target {VANO}')
            plt.plot(instances, score_fused_std_structures,  label='Tranfer Learning')
            plt.plot(instances, score_alone_std_structures, label='Alone')
            plt.xlabel('Instances')
            plt.ylabel('STD Log-Likelihood')
            plt.legend()
            plt.savefig(save_path_std_structures + f'target_{VANO}.png')
            plt.close('all')



        if savebool_params:

            if not os.path.exists(save_path_params + 'similars'):
                os.mkdir(save_path_params + 'similars')
            similarsdf = pd.concat(similarsdf0, ignore_index=True)
            similarsdf.max().to_frame().T.to_csv(save_path_params + f'similars/target_{VANO}.csv', index=False)
    
            scoresfPD_params = pd.DataFrame({'score_fused': scores_fused_params, 'score_structure': scores_structure_params,'score_alone':scores_alone_params})
            if not os.path.exists(save_path_params + 'scores'):
                os.mkdir(save_path_params + 'scores')
            
            scoresfPD_std_params = pd.DataFrame({'score_fused': scores_fused_std_params, 'score_structure': scores_structure_std_params,'score_alone':scores_alone_std_params})
            if not os.path.exists(save_path_std_params + 'scores'):
                os.mkdir(save_path_std_params + 'scores')
            
            scoresfPD_params.to_csv(save_path_params + f'scores/target_{VANO}.csv', index=False)    
            scoresfPD_std_params.to_csv(save_path_std_params + f'scores/target_{VANO}.csv', index=False)  
        
            plt.figure(figsize=(15,7))
            plt.title(f'Target {VANO}')
            plt.plot(instances, scores_fused_params,  label='Tranfer Learning S+P')
            plt.plot(instances, scores_structure_params, label='Tranfer Learning S')
            plt.plot(instances, scores_alone_params, label='Alone')
            plt.xlabel('Instances')
            plt.ylabel('Log-Likelihood')
            plt.legend()
            plt.savefig(save_path_params + f'target_{VANO}.png')
            plt.close('all')

            plt.figure(figsize=(15,7))
            plt.title(f'Target {VANO}')
            plt.plot(instances, scores_fused_std_params,  label='Tranfer Learning S+P')
            plt.plot(instances, scores_structure_std_params, label='Tranfer Learning S')
            plt.plot(instances, scores_alone_std_params, label='Alone')
            plt.xlabel('Instances')
            plt.ylabel('STD Log-Likelihood')
            plt.legend()
            plt.savefig(save_path_std_params + f'target_{VANO}.png')
            plt.close('all')

    pbar.close()


if __name__ == '__main__':


    dfs_bilbao = {}
    colsnames = ['NAT_FREQ_1','NAT_FREQ_2','NAT_FREQ_3','NAT_FREQ_4','NAT_FREQ_5',
                 'DAMP_1','DAMP_2','DAMP_3','DAMP_4','DAMP_5',]

    path_bilbao = [os.path.join("data/bilbao2023/new",fold) for fold in os.listdir("data/bilbao2023/new")]
    for pth in path_bilbao:
        pth_csv = os.path.join(pth,'result.csv')
        df = get_df_filtered(pth_csv,';','ms')
        df = df.iloc[:,[6,7,8,9,10,1,2,3,4,5]]
        df.columns = colsnames
        dfs_bilbao[pth.split('/')[-1]] = df

    
    poles3 = [0,1,2,5,6,7]
    poles4 = [0,1,2,3,5,6,7,8]
    poles5 = [0,1,2,3,4,5,6,7,8,9]
    keypoles = 'p5'

    
    jumps = 40
    threads = list()
    for VANO in [
                'vano1','vano2',
                'vano3','vano4',
                'vano5','vano6'
                ]:
        
        datasAux = []
        for n, key in enumerate(dfs_bilbao.keys()):
            if key != VANO:
                dataf = dfs_bilbao[key].iloc[:3000, poles5]
                datasAux.append(dataf.copy())  
                

        save_path_params='pictures/Param&Structure/gaussian/LogLPool/75-25-1500/2_4/expert/5p_targets/'
        save_path_std_params='pictures/Param&Structure/gaussian/LogLPool/75-25-1500/2_4/expert_std/5p_targets/'
       
        save_path_structures = f'pictures/Structures/gaussian/2_4/expert/5p_targets/'
        save_path_std_structures =  f'pictures/Structures/gaussian/2_4/expert_std/5p_targets/'
        savebool_structures = True
        savebool_params = True



        t = threading.Thread(target=train, args=(datasAux, poles5, VANO, jumps, 
                                                save_path_structures, save_path_std_structures, 
                                                save_path_params, save_path_std_params, 
                                                savebool_structures, savebool_params))
        threads.append(t)
        t.start()

    for index, t in enumerate(threads):
        t.join()
        




        

        


        















    


  
