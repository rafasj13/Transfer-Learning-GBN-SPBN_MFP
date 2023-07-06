import os
import threading

from causallearn.utils.cit import *

from code.utils import get_df_filtered
from code.PCTL import PCMaxTL
from code.CIT import *

import pybnesian as pbn
import subprocess
import networkx as nx


def draw_model(model, filename):

    DG = nx.DiGraph()
    DG.add_nodes_from(model.nodes())
    DG.add_edges_from(model.arcs())

    if isinstance(model, pbn.BayesianNetworkBase):
        for node in DG.nodes:
            if model.node_type(node) == pbn.CKDEType():
                DG.nodes[node]['style'] = 'filled'
                DG.nodes[node]['fillcolor'] = 'gray'

    a = nx.nx_agraph.to_agraph(DG)
    if filename[-4:] != '.dot':
        filename += '.dot'
    a.write(filename)
    a.clear()

    pdf_out = filename[:-4] + '.pdf'

    subprocess.run(["dot", "-Tpdf", filename, "-o", pdf_out])


def get_dag(datasAux, VANO, N, poles, save):      
    
    pctl = PCMaxTL(alpha=0.08, indep_test = cci, datasAux = datasAux, npoles=keypoles,verbose=False)
    pc = PCMaxTL(alpha=0.08, indep_test = cci, npoles=keypoles)

    print(f'init {VANO} with {N} instances')

    if not os.path.exists(os.path.join(save, str(N))):
        os.mkdir(os.path.join(save, str(N)))   

    datat0 = dfs_bilbao[VANO].iloc[:N, poles]



    # model of transfer learning
    model_fused, cg = pctl.structure_learning(datat0, uc_rule = 2, uc_priority = 4)
    draw_model(model_fused, os.path.join(save, str(N),f'{VANO}_fused_final') )

    # model alone
    model_alone,_ = pc.structure_learning(datat0, uc_rule=2, uc_priority=-1)
    draw_model(model_alone, os.path.join(save, str(N),f'{VANO}_alone') )


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

    N = 400
    threads = list()
    for VANO in [
                'vano1','vano2','vano3',
                'vano4','vano5','vano6'
                ]:
        
        datasAux = []
        for n, key in enumerate(dfs_bilbao.keys()):
            if key != VANO:
                dataf = dfs_bilbao[key].iloc[:3000, poles5]
                datasAux.append(dataf.copy())  
        

        save = 'pictures/Collage/final_experiments/semiparametric/dags/'
        t = threading.Thread(target=get_dag, args=(datasAux, VANO, N, poles5,save))
        threads.append(t)
        t.start()

    for index, t in enumerate(threads):
        t.join()

        
        
        
