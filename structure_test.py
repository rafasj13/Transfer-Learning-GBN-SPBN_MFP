import threading
import os

from causallearn.utils.cit import *

from code.utils import get_df_filtered, draw_graph
from code.PCTL import PCMaxTL
from code.CIT import *


from tqdm.auto import tqdm
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('agg')



def train(datasAux, poles, VANO, jumps, save_path, save):
    
    score_fused, score_alone = [],[]
    instances = []
    similarsdf0 = []

    pbar = tqdm(total=2000//jumps)
    

    pctl = PCMaxTL(alpha=0.08, indep_test = cci, datasAux = datasAux, verbose=False)
    pc = PCMaxTL(alpha=0.08, indep_test = cci)  
    for n in range(20,2000+jumps,jumps):

        # datat0 = dfs_andoain['abril2022'].iloc[:n,1:]
        # data0_train = dfs_andoain['abril2022'].iloc[:3000,1:]
        # data0_test = dfs_andoain['abril2022'].iloc[3000:,1:]

        datat0 = dfs_bilbao[VANO].iloc[:n, poles]
        data0_train = dfs_bilbao[VANO].iloc[:3000, poles]
        data0_test = dfs_bilbao[VANO].iloc[3000:, poles]


        # model of transfer learning
        model_fused,_ = pctl.structure_learning(datat0, uc_rule = 2, uc_priority = 4)#, est_width='median', cache_path = "cache/cache_TL_kci.json")
        model_fused.fit(data0_train)
        scorefused = model_fused.logl(data0_test)
        similarsdf0.append(pd.DataFrame(pctl.similars_count, index=[0]).copy())
        score_fused.append(np.mean(scorefused))

        # model alone
        model_alone,_ = pc.structure_learning(datat0, uc_rule=2, uc_priority=-1)#, est_width='median', cache_path = "cache/cache_Alone_kci.json") 
        model_alone.fit(data0_train)
        scorealone = model_alone.logl(data0_test)
        score_alone.append(np.mean(scorealone))

        instances.append(n)
    
        ## ========= UPDATE =========
        pbar.update()

        if save:

            if not os.path.exists(save_path + 'similars'):
                os.mkdir(save_path + 'similars')
            similarsdf = pd.concat(similarsdf0, ignore_index=True)
            similarsdf.max().to_frame().T.to_csv(save_path + f'similars/target_{VANO}.csv', index=False)
    
            scoresfPD = pd.DataFrame({'score_fused': score_fused, 'score_alone':score_alone})
            if not os.path.exists(save_path + 'scores'):
                os.mkdir(save_path + 'scores')
            
            scoresfPD.to_csv(save_path + f'scores/target_{VANO}.csv', index=False)    
   

            plt.figure(figsize=(15,7))
            plt.title(f'Target {VANO}')
            plt.plot(instances, score_fused,  label='Tranfer Learning')
            plt.plot(instances, score_alone, label='Alone')
            plt.xlabel('Instances')
            plt.ylabel('Log-Likelihood')
            plt.legend()
            plt.savefig(save_path + f'target_{VANO}.png')
            plt.close('all')




    pbar.close()
    


    

if __name__ == '__main__':

    main_path='./data'
    dfs_andoain = {}
    dfs_bilbao = {}
    colsnames = ['NAT_FREQ_1','NAT_FREQ_2','NAT_FREQ_3','NAT_FREQ_4','NAT_FREQ_5',
                 'DAMP_1','DAMP_2','DAMP_3','DAMP_4','DAMP_5',]

    path = [fold for fold in os.listdir(main_path)]
    path_andoain = [os.path.join(main_path,path[0],fold) for fold in os.listdir(os.path.join(main_path,path[0]))]
    path_bilbao = [os.path.join("data/bilbao2023/new",fold) for fold in os.listdir("data/bilbao2023/new")]
    path_choco = [os.path.join(main_path,path[2],fold) for fold in os.listdir(os.path.join(main_path,path[2]))]

    for pth in path_andoain:
        pth_csv = os.path.join(pth,'result.csv')
        df = get_df_filtered(pth_csv,';','ms')
        dfs_andoain[pth.split('/')[-1]] = df

    for pth in path_bilbao:
        pth_csv = os.path.join(pth,'result.csv')
        df = get_df_filtered(pth_csv,';','ms')
        df = df.iloc[:,[6,7,8,9,10,1,2,3,4,5]]
        df.columns = colsnames
        dfs_bilbao[pth.split('/')[-1]] = df

    for pth in path_choco:
        df_choco= get_df_filtered(pth,';','ms')


    
    poles3 = [0,1,2,5,6,7]
    poles4 = [0,1,2,3,5,6,7,8]

    jumps = 40
    threads = list()
    for VANO in [
                'vano1','vano2','vano3',
                'vano4','vano5','vano6'
                #'andoain'
                ]:
        
        datasAux = []
        for n, key in enumerate(dfs_bilbao.keys()):
            if key != VANO:
                dataf = dfs_bilbao[key].iloc[:3000, poles4]
                datasAux.append(dataf.copy())  
        
        # andoain = dfs_andoain['abril2022'].iloc[:3000, 1:] #andoain abril 2022
        # datasAux.append(andoain.to_numpy())
        # choco = df_choco.iloc[:1000,1:]
        # datasAux.append(choco.to_numpy())


        save_path = f'pictures/Structures/semiparametric/4p_targets_0.1x2_0.08/'
        save = True
        #train(datasAux, colsvanos, VANO, jumps, save_path, save)

    #     
        t = threading.Thread(target=train, args=(datasAux, poles4, VANO, jumps, save_path, save))
        threads.append(t)
        t.start()

    for index, t in enumerate(threads):
        t.join()
        
        
