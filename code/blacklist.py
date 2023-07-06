
from causallearn.graph.GraphNode import GraphNode
from causallearn.utils.PCUtils.BackgroundKnowledge import BackgroundKnowledge


forbiden_poles = dict(
    p5 = [
    ('DAMP_1', 'NAT_FREQ_1'), ('DAMP_1', 'NAT_FREQ_2'), ('DAMP_1', 'NAT_FREQ_3'), ('DAMP_1', 'NAT_FREQ_4'), ('DAMP_1', 'NAT_FREQ_5'), 
    ('DAMP_2', 'NAT_FREQ_1'), ('DAMP_2', 'NAT_FREQ_2'), ('DAMP_2', 'NAT_FREQ_3'), ('DAMP_2', 'NAT_FREQ_4'), ('DAMP_2', 'NAT_FREQ_5'), 
    ('DAMP_3', 'NAT_FREQ_1'), ('DAMP_3', 'NAT_FREQ_2'), ('DAMP_3', 'NAT_FREQ_3'), ('DAMP_3', 'NAT_FREQ_4'), ('DAMP_3', 'NAT_FREQ_5'), 
    ('DAMP_4', 'NAT_FREQ_1'), ('DAMP_4', 'NAT_FREQ_2'), ('DAMP_4', 'NAT_FREQ_3'), ('DAMP_4', 'NAT_FREQ_4'), ('DAMP_4', 'NAT_FREQ_5'), 
    ('DAMP_5', 'NAT_FREQ_1'), ('DAMP_5', 'NAT_FREQ_2'), ('DAMP_5', 'NAT_FREQ_3'), ('DAMP_5', 'NAT_FREQ_4'), ('DAMP_5', 'NAT_FREQ_5')
    ],
    p4 = [
        ('DAMP_1', 'NAT_FREQ_1'), ('DAMP_1', 'NAT_FREQ_2'), ('DAMP_1', 'NAT_FREQ_3'), ('DAMP_1', 'NAT_FREQ_4'), 
        ('DAMP_2', 'NAT_FREQ_1'), ('DAMP_2', 'NAT_FREQ_2'), ('DAMP_2', 'NAT_FREQ_3'), ('DAMP_2', 'NAT_FREQ_4'), 
        ('DAMP_3', 'NAT_FREQ_1'), ('DAMP_3', 'NAT_FREQ_2'), ('DAMP_3', 'NAT_FREQ_3'), ('DAMP_3', 'NAT_FREQ_4'), 
        ('DAMP_4', 'NAT_FREQ_1'), ('DAMP_4', 'NAT_FREQ_2'), ('DAMP_4', 'NAT_FREQ_3'), ('DAMP_4', 'NAT_FREQ_4')
        ], 
    p3 = [
        ('DAMP_1', 'NAT_FREQ_1'), ('DAMP_1', 'NAT_FREQ_2'), ('DAMP_1', 'NAT_FREQ_3'),  
        ('DAMP_2', 'NAT_FREQ_1'), ('DAMP_2', 'NAT_FREQ_2'), ('DAMP_2', 'NAT_FREQ_3'), 
        ('DAMP_3', 'NAT_FREQ_1'), ('DAMP_3', 'NAT_FREQ_2'), ('DAMP_3', 'NAT_FREQ_3')
        ]
    )

blacklist_knowledge = {}
for key, forbid in forbiden_poles.items():
    knowledge = BackgroundKnowledge()
    for  arc in (forbid):
        node1 = GraphNode(arc[0])
        node2 = GraphNode(arc[1])
        knowledge.add_forbidden_by_node(node1, node2)
        
    blacklist_knowledge[key] = knowledge