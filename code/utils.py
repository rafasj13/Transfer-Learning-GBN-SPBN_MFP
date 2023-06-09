from __future__ import annotations
import pandas as pd

from causallearn.graph.GeneralGraph import GeneralGraph
from causallearn.utils.GraphUtils import GraphUtils
from causallearn.graph.Edge import Edge
from causallearn.graph.GraphNode import GraphNode
from causallearn.graph.Endpoint import Endpoint
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import io
import threading
import warnings

class CustomThread(threading.Thread):
    
    def __init__(self, group=None, target=None, name=None,
                 args=(), kwargs={}, Verbose=None):
        threading.Thread.__init__(self, group, target, name, args, kwargs)
        self._return = None

    def run(self):
        if self._target is not None:
            self._return = self._target(*self._args,
                                                **self._kwargs)
    def join(self, *args):
        threading.Thread.join(self, *args)
        
    
    def value(self):
        return self._return



def get_df_filtered(path, separation, frequency):
    df = pd.read_csv(path, sep=separation)
    fact = 10**6 if df['timestamp'][0]>9*10**9 else 1
    df['timestamp'] = df['timestamp']/fact 
    df['timestamp'] = pd.to_datetime(df['timestamp'],unit='s')
    return df


def draw_graph(model, title='DAG', save_path='pictures/Collage/structures/DAGS/dag.png'):
    nodes = []
    for name in model.nodes():
            node = GraphNode(name)
            nodes.append(node)

    Gdraw = GeneralGraph(nodes)
    for arc in model.arcs():
        Gdraw.add_edge(Edge(GraphNode(arc[0]), GraphNode(arc[1]), Endpoint.TAIL, Endpoint.ARROW))

    warnings.filterwarnings("ignore", category=UserWarning)
    pyd = GraphUtils.to_pydot(Gdraw)
    tmp_png = pyd.create_png(f="png")
    fp = io.BytesIO(tmp_png)
    img = mpimg.imread(fp, format='png')
    # plt.rcParams["figure.figsize"] = [15, 7]
    # plt.rcParams["figure.autolayout"] = True
    # plt.axis('off')
    # plt.title(title)
    # plt.show()
    mpimg.imsave(save_path,img)
