import os
import threading

from causallearn.utils.cit import *

from code.utils import get_df_filtered, draw_graph
from code.PCTL import PCMaxTL
from code.CIT import *
from causallearn.utils.GraphUtils import GraphUtils
from tqdm.auto import tqdm
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.image as mpimg
import io
matplotlib.use('agg')

def get_dag(datasAux, VANO, N, poles):
    main = 'pictures/Collage/structures/semiparametric/DAGS/aux0.10/N100'
    pctl = PCMaxTL(alpha=0.08, indep_test = cci, datasAux = datasAux, verbose=False)
    pc = PCMaxTL(alpha=0.08, indep_test = cci)  

    print(f'init {VANO} with {N} instances')
    datat0 = dfs_bilbao[VANO].iloc[:N, poles]



    # model of transfer learning
    model_fused, cg = pctl.structure_learning(datat0, uc_rule = 2, uc_priority = 4)
    # model alone
    model_alone,_ = pc.structure_learning(datat0, uc_rule=2, uc_priority=-1)

    pyd = GraphUtils.to_pydot(cg.G)
    tmp_png = pyd.create_png(f="png")
    fp = io.BytesIO(tmp_png)
    img = mpimg.imread(fp, format='png')
    mpimg.imsave(main + f'{VANO}_{N}_fused_1st.png',img)

    draw_graph(model_fused, save_path=main+ f'{VANO}_{N}_fused_final.png')
    draw_graph(model_alone, save_path=main+ f'{VANO}_{N}_alone.png')
    

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

    N = 100
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
        # datasAux.append(andoain)
        # choco = df_choco.iloc[:1000,1:]
        # datasAux.append(choco.to_numpy())


        t = threading.Thread(target=get_dag, args=(datasAux, VANO, N, poles4))
        threads.append(t)
        t.start()

    for index, t in enumerate(threads):
        t.join()

        
        
        
