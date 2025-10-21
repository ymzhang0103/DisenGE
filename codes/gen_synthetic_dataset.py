import networkx as nx
import torch
import numpy as np
import matplotlib.pyplot as plt
import random
from torch_geometric.data import Data
import pickle
from torch_geometric.utils import dense_to_sparse


def ba(start, width, role_start=0, m=5):
    """Builds a BA preferential attachment graph, with index of nodes starting at start
    and role_ids at role_start
    INPUT:
    -------------
    start       :    starting index for the shape
    width       :    int size of the graph
    role_start  :    starting index for the roles
    OUTPUT:
    -------------
    graph       :    a BA shape graph, with ids beginning at start
    roles       :    list of the roles of the nodes (indexed starting at
                     role_start)
    """
    graph = nx.barabasi_albert_graph(width, m)
    graph.add_nodes_from(range(start, start + width))
    nids = sorted(graph)
    mapping = {nid: start + i for i, nid in enumerate(nids)}
    graph = nx.relabel_nodes(graph, mapping)
    roles = [role_start for i in range(width)]
    return graph, roles


def star(start, nb_branches, role_start=0):
    """Builds a star graph, with index of nodes starting at start
    and role_ids at role_start
    INPUT:
    -------------
    nb_branches :    int correspondingraph to the nb of star branches
    start       :    starting index for the shape
    role_start  :    starting index for the roles
    OUTPUT:
    -------------
    graph       :    a star shape graph, with ids beginning at start
    roles       :    list of the roles of the nodes (indexed starting at
                     role_start)
    """
    graph = nx.Graph()
    graph.add_nodes_from(range(start, start + nb_branches + 1))
    for k in range(1, nb_branches + 1):
        graph.add_edges_from([(start, start + k)])
    roles = [role_start + 1] * (nb_branches + 1)
    roles[0] = role_start
    return graph, roles


def house(start, role_start=0):
    """Builds a house-like  graph, with index of nodes starting at start
    and role_ids at role_start
    INPUT:
    -------------
    start       :    starting index for the shape
    role_start  :    starting index for the roles
    OUTPUT:
    -------------
    graph       :    a house shape graph, with ids beginning at start
    roles       :    list of the roles of the nodes (indexed starting at
                     role_start)
    """
    graph = nx.Graph()
    graph.add_nodes_from(range(start, start + 5))
    graph.add_edges_from(
        [
            (start, start + 1),
            (start + 1, start + 2),
            (start + 2, start + 3),
            (start + 3, start),
        ]
    )
    # graph.add_edges_from([(start, start + 2), (start + 1, start + 3)])
    graph.add_edges_from([(start + 4, start), (start + 4, start + 1)])
    roles = [role_start, role_start, role_start + 1, role_start + 1, role_start + 2]
    return graph, roles


def grid(start, dim=2, role_start=0):
    """ Builds a 2by2 grid
    """
    grid_G = nx.grid_graph([dim, dim])
    grid_G = nx.convert_node_labels_to_integers(grid_G, first_label=start)
    roles = [role_start for i in grid_G.nodes()]
    return grid_G, roles


def cycle(start, len_cycle, role_start=0):
    """Builds a cycle graph, with index of nodes starting at start
    and role_ids at role_start
    INPUT:
    -------------
    start       :    starting index for the shape
    role_start  :    starting index for the roles
    OUTPUT:
    -------------
    graph       :    a house shape graph, with ids beginning at start
    roles       :    list of the roles of the nodes (indexed starting at
                     role_start)
    """
    graph = nx.Graph()
    graph.add_nodes_from(range(start, start + len_cycle))
    for i in range(len_cycle - 1):
        graph.add_edges_from([(start + i, start + i + 1)])
    graph.add_edges_from([(start + len_cycle - 1, start)])
    roles = [role_start] * len_cycle
    return graph, roles


def clique(start, nb_nodes, nb_to_remove=0, role_start=0):
    """ Defines a clique (complete graph on nb_nodes nodes,
    with nb_to_remove  edges that will have to be removed),
    index of nodes starting at start
    and role_ids at role_start
    INPUT:
    -------------
    start       :    starting index for the shape
    nb_nodes    :    int correspondingraph to the nb of nodes in the clique
    role_start  :    starting index for the roles
    nb_to_remove:    int-- numb of edges to remove (unif at RDM)
    OUTPUT:
    -------------
    graph       :    a house shape graph, with ids beginning at start
    roles       :    list of the roles of the nodes (indexed starting at
                     role_start)
    """
    a = np.ones((nb_nodes, nb_nodes))
    np.fill_diagonal(a, 0)
    graph = nx.from_numpy_array(a)
    edge_list = graph.edges().keys()
    roles = [role_start] * nb_nodes
    if nb_to_remove > 0:
        lst = np.random.choice(len(edge_list), nb_to_remove, replace=False)
        print(edge_list, lst)
        to_delete = [edge_list[e] for e in lst]
        graph.remove_edges_from(to_delete)
        for e in lst:
            print(edge_list[e][0])
            print(len(roles))
            roles[edge_list[e][0]] += 1
            roles[edge_list[e][1]] += 1
    mapping_graph = {k: (k + start) for k in range(nb_nodes)}
    graph = nx.relabel_nodes(graph, mapping_graph)
    return graph, roles


def diamond(start, role_start=0):
    """Builds a diamond graph, with index of nodes starting at start
    and role_ids at role_start
    INPUT:
    -------------
    start       :    starting index for the shape
    role_start  :    starting index for the roles
    OUTPUT:
    -------------
    graph       :    a house shape graph, with ids beginning at start
    roles       :    list of the roles of the nodes (indexed starting at
                     role_start)
    """
    graph = nx.Graph()
    graph.add_nodes_from(range(start, start + 6))
    graph.add_edges_from(
        [
            (start, start + 1),
            (start + 1, start + 2),
            (start + 2, start + 3),
            (start + 3, start),
        ]
    )
    graph.add_edges_from(
        [
            (start + 4, start),
            (start + 4, start + 1),
            (start + 4, start + 2),
            (start + 4, start + 3),
        ]
    )
    graph.add_edges_from(
        [
            (start + 5, start),
            (start + 5, start + 1),
            (start + 5, start + 2),
            (start + 5, start + 3),
        ]
    )
    roles = [role_start] * 6
    return graph, roles


def fan(start, nb_branches, role_start=0):
    """Builds a fan-like graph, with index of nodes starting at start
    and role_ids at role_start
    INPUT:
    -------------
    nb_branches :    int correspondingraph to the nb of fan branches
    start       :    starting index for the shape
    role_start  :    starting index for the roles
    OUTPUT:
    -------------
    graph       :    a house shape graph, with ids beginning at start
    roles       :    list of the roles of the nodes (indexed starting at
                     role_start)
    """
    graph, roles = star(start, nb_branches, role_start=role_start)
    for k in range(1, nb_branches):
        roles[k] += 1
        roles[k + 1] += 1
        graph.add_edges_from([(start + k, start + k + 1)])
    return graph, roles


def plot_graph(edge_index, k=0):
    g = nx.Graph()
    g.add_edges_from(edge_index.T.numpy().tolist())
    nx.draw(g)
    plt.savefig(str(k)+".png")
    plt.show()


def add_motif_to_graph(edge_indices, node_masks, edge_masks, ys, base_conn_nodeid, num_nodes, motif_edge_index, motif_num_nodes, motif_num_edges):
    # Add motif to the graph.
    edge_indices.append(motif_edge_index + num_nodes)
    node_masks.append(torch.ones(motif_num_nodes))
    edge_masks.append(torch.ones(motif_num_edges))

    # Add random motif connection to the graph.
    j = int(torch.randint(0, motif_num_nodes, (1, ))) + num_nodes
    edge_indices.append(torch.tensor([[base_conn_nodeid, j], [j, base_conn_nodeid]]))
    edge_masks.append(torch.zeros(2))
    ys.append(torch.ones(motif_num_nodes, dtype=torch.long))
    num_nodes += motif_num_nodes
    return edge_indices, node_masks, edge_masks, ys, num_nodes



def main_gen_BA_4motifs():
    num_graphs = 2000
    adjs = []
    feats = []
    feat_dim = 10
    labels = []
    for k in range(num_graphs):
        if k < 500:
            graph_label=[1,0]
            basis_width = 20
            motif, role_id = eval("house")(start=basis_width)
        elif k >= 500 and k<1000:
            graph_label=[1,0]
            basis_width = 18
            motif, role_id = eval("star")(start=basis_width, nb_branches=6)
        elif k>=1000 and k<1500:
            graph_label = [0,1]
            basis_width = 19
            motif, role_id = eval("cycle")(start=basis_width, len_cycle=6)
        elif k>=1500 and k<2000:
            graph_label = [0,1]
            basis_width = 16
            motif, role_id = eval("grid")(start=basis_width,dim=3)
        motif_num_nodes = nx.number_of_nodes(motif)
        motif_edges = np.array(motif.edges).T.tolist()
        motif_edge_index = torch.cat((torch.tensor(motif_edges), torch.tensor([motif_edges[1], motif_edges[0]])), dim=1)
        motif_num_edges = motif_edge_index.shape[1]
        #generate basis  BA graph
        basis, role_id = eval("ba")(start=0, width=basis_width, m=5)
        num_nodes = nx.number_of_nodes(basis)
        edge_index = torch.tensor(np.array(basis.edges).T)
        ba_graph = Data(edge_index=edge_index, num_nodes=num_nodes)
        node_masks = [torch.zeros(ba_graph.num_nodes)]
        edge_masks = [torch.zeros(ba_graph.num_edges)]
        ys = [torch.zeros(num_nodes, dtype=torch.long)]
        #generate connecting node between basis graph and motif graph
        connecting_node = random.randint(0, num_nodes-1)
        
        #motif_type_dic = {0:'house', 1:'star', 2:'cycle', 3:'grid'}
        #motif_type = motif_type_dic[random.randint(0,3)]  #0:house, 1:star, 3:cycle, 4:grid
        #conn motif to basis
        j = int(torch.randint(0, motif_num_nodes, (1, ))) + num_nodes
        edge_index = torch.cat((edge_index,torch.tensor([[connecting_node, j], [j, connecting_node]])), dim=1)
        edge_masks.append(torch.zeros(2))
        #Add motif to basis
        ys.append(torch.ones(motif_num_nodes, dtype=torch.long))
        edge_index = torch.cat((edge_index,motif_edge_index), dim=1)
        edge_masks.append(torch.ones(motif_num_edges))
        node_masks.append(torch.ones(motif_num_nodes))
        num_nodes += motif_num_nodes
        #Adj, feature, graph label
        adj = torch.sparse_coo_tensor(indices=edge_index, values= torch.tensor([1]*edge_index.shape[1]), size=(num_nodes, num_nodes)).to_dense()
        adjs.append(adj.unsqueeze(0))
        #feat = torch.ones(num_nodes, feat_dim)/feat_dim
        feat = torch.ones(num_nodes, feat_dim)
        feats.append(feat.unsqueeze(0))
        labels.append(graph_label)
        print("k", k, "num_nodes", num_nodes)
    adjs =  torch.cat(adjs, dim=0)
    feats = torch.cat(feats, dim=0)
    labels = np.array(labels)

    #data = {"adj": adjs.numpy(), "feat":feats, "labels":labels}
    #data = (adjs.numpy(), feats.numpy(), np.array(labels))

    data_list = []
    for graph_idx in range(adjs.shape[0]):
        data_list.append(Data(x=feats[graph_idx].float(),
                                edge_index=dense_to_sparse(adjs[graph_idx])[0],
                                y=torch.from_numpy(np.where(labels[graph_idx])[0]), y_type = ""))
        
    with open("/mnt/8T/torch_projects/datasets/BA_4Motifs_1/raw/BA_4Motifs.pkl", "wb") as file:
        pickle.dump(data_list, file)
    print("generate complete")


def main_gen_BA_6motifs():
    num_graphs = 3000
    adjs = []
    feats = []
    feat_dim = 10
    labels = []
    motif_types = []
    for k in range(num_graphs):
        if k < 500:
            graph_label=[1, 0, 0]
            basis_width = 20
            motif, role_id = eval("house")(start=basis_width)
            motif_types.append(0)
        elif k >= 500 and k<1000:
            graph_label=[1, 0, 0]
            basis_width = 18
            motif, role_id = eval("star")(start=basis_width, nb_branches=6)
            motif_types.append(1)
        elif k>=1000 and k<1500:
            graph_label = [0, 1, 0]
            basis_width = 19
            motif, role_id = eval("cycle")(start=basis_width, len_cycle=6)
            motif_types.append(2)
        elif k>=1500 and k<2000:
            graph_label = [0, 1, 0]
            basis_width = 16
            motif, role_id = eval("grid")(start=basis_width,dim=3)
            motif_types.append(3)
        elif k>=2000 and k<2500:
            graph_label=[0, 0, 1]
            basis_width = 18
            motif, role_id = eval("fan")(start=basis_width, nb_branches=6)
            motif_types.append(4)
        elif k>=2500 and k<3000:
            graph_label=[0, 0, 1]
            basis_width = 19
            motif, role_id = eval("diamond")(start=basis_width)
            motif_types.append(5)
        motif_num_nodes = nx.number_of_nodes(motif)
        motif_edges = np.array(motif.edges).T.tolist()
        motif_edge_index = torch.cat((torch.tensor(motif_edges), torch.tensor([motif_edges[1], motif_edges[0]])), dim=1)
        motif_num_edges = motif_edge_index.shape[1]
        #generate basis  BA graph
        basis, role_id = eval("ba")(start=0, width=basis_width, m=5)
        num_nodes = nx.number_of_nodes(basis)
        edge_index = torch.tensor(np.array(basis.edges).T)
        ba_graph = Data(edge_index=edge_index, num_nodes=num_nodes)
        node_masks = [torch.zeros(ba_graph.num_nodes)]
        edge_masks = [torch.zeros(ba_graph.num_edges)]
        ys = [torch.zeros(num_nodes, dtype=torch.long)]
        #generate connecting node between basis graph and motif graph
        connecting_node = random.randint(0, num_nodes-1)
        
        #motif_type_dic = {0:'house', 1:'star', 2:'cycle', 3:'grid'}
        #motif_type = motif_type_dic[random.randint(0,3)]  #0:house, 1:star, 3:cycle, 4:grid
        #conn motif to basis
        j = int(torch.randint(0, motif_num_nodes, (1, ))) + num_nodes
        edge_index = torch.cat((edge_index,torch.tensor([[connecting_node, j], [j, connecting_node]])), dim=1)
        edge_masks.append(torch.zeros(2))
        #Add motif to basis
        ys.append(torch.ones(motif_num_nodes, dtype=torch.long))
        edge_index = torch.cat((edge_index,motif_edge_index), dim=1)
        edge_masks.append(torch.ones(motif_num_edges))
        node_masks.append(torch.ones(motif_num_nodes))
        num_nodes += motif_num_nodes
        #Adj, feature, graph label
        adj = torch.sparse_coo_tensor(indices=edge_index, values= torch.tensor([1]*edge_index.shape[1]), size=(num_nodes, num_nodes)).to_dense()
        adjs.append(adj.unsqueeze(0))
        #feat = torch.ones(num_nodes, feat_dim)/feat_dim
        feat = torch.ones(num_nodes, feat_dim)
        feats.append(feat.unsqueeze(0))
        labels.append(graph_label)
        print("k", k, "num_nodes", num_nodes)
    adjs =  torch.cat(adjs, dim=0)
    feats = torch.cat(feats, dim=0)
    labels = np.array(labels)
    #data = {"adj": adjs.numpy(), "feat":feats, "labels":labels}
    #data = (adjs.numpy(), feats.numpy(), np.array(labels))

    data_list = []
    for graph_idx in range(adjs.shape[0]):
        data_list.append(Data(x=feats[graph_idx].float(),
                                edge_index=dense_to_sparse(adjs[graph_idx])[0],
                                y=torch.from_numpy(np.where(labels[graph_idx])[0]), motif_types=motif_types[graph_idx]))
        
    with open("/mnt/8T/torch_projects/datasets/BA_6Motifs/raw/BA_6Motifs.pkl", "wb") as file:
        pickle.dump(data_list, file)
    print("generate complete")


def gen_random_motifs(basis, motif_slices, node_label, edge_label, node_masks, edge_masks):
    #motifs_arr = []
    start = nx.number_of_nodes(basis)
    for motif_index in range(len(motif_slices)):
        motif_type =  motif_slices[motif_index]
        if motif_type==0:
            motif, role_id = eval("house")(start=start)
            node_label.append(torch.ones(nx.number_of_nodes(motif), dtype=torch.long))
            edge_label.append(torch.ones(nx.number_of_edges(motif)*2, dtype=torch.long))
        elif motif_type==1:
            motif, role_id = eval("star")(start=start, nb_branches=6)
            node_label.append(torch.tensor([2.0] * nx.number_of_nodes(motif), dtype=torch.long))
            edge_label.append(torch.tensor([2.0] * nx.number_of_edges(motif)*2, dtype=torch.long))
        elif motif_type==2:
            motif, role_id = eval("cycle")(start=start, len_cycle=6)
            node_label.append(torch.tensor([3.0] * nx.number_of_nodes(motif), dtype=torch.long))
            edge_label.append(torch.tensor([3.0] * nx.number_of_edges(motif)*2, dtype=torch.long))
        elif motif_type==3:
            motif, role_id = eval("grid")(start=start,dim=3)
            node_label.append(torch.tensor([4.0] * nx.number_of_nodes(motif), dtype=torch.long))
            edge_label.append(torch.tensor([4.0] * nx.number_of_edges(motif)*2, dtype=torch.long))
        elif motif_type==4:
            motif, role_id = eval("fan")(start=start, nb_branches=6)
            node_label.append(torch.tensor([5.0] * nx.number_of_nodes(motif), dtype=torch.long))
            edge_label.append(torch.tensor([5.0] * nx.number_of_edges(motif)*2, dtype=torch.long))
        elif motif_type==5:
            motif, role_id = eval("diamond")(start=start)
            node_label.append(torch.tensor([6.0] * nx.number_of_nodes(motif), dtype=torch.long))
            edge_label.append(torch.tensor([6.0] * nx.number_of_edges(motif)*2, dtype=torch.long))

        motif_num_nodes = nx.number_of_nodes(motif)
        start = start + motif_num_nodes
        #motifs_arr.append(motif)

        #motif = motifs_arr[motif_index]
        #merge nodes
        motif_num_nodes = nx.number_of_nodes(motif)
        node_masks.append(torch.ones(motif_num_nodes))
        
        #merge edges
        motif_edges = np.array(motif.edges).T.tolist()
        motif_edge_index = torch.cat((torch.tensor(motif_edges), torch.tensor([motif_edges[1], motif_edges[0]])), dim=1)
        edge_index = torch.cat((edge_index, motif_edge_index), dim=1)
        motif_num_edges = motif_edge_index.shape[1]
        edge_masks.append(torch.ones(motif_num_edges))

        #connecting
        basis_conn_node = random.sample(list(basis.nodes), 1)[0]
        motif_conn_node = random.sample(list(motif.nodes), 1)[0]
        edge_index = torch.cat((edge_index, torch.tensor([[basis_conn_node, motif_conn_node], [motif_conn_node, basis_conn_node]])), dim=1)
        edge_masks.append(torch.zeros(2))
        edge_label.append(torch.zeros(2, dtype=torch.long))

    num_nodes = start - 1

    #return motifs_arr, node_label, edge_label
    return num_nodes, edge_index, node_label, edge_label, node_masks, edge_masks


def main_gen_BA_6motifs_2class(random_motif = True):
    data_list = []
    num_graphs = 2000
    #adjs = []
    #feats = []
    #labels = []
    feat_dim = 10
    motif_slices = []
    for k in range(num_graphs):
        basis_width =20
        #generate basis  BA graph
        basis, role_id = eval("ba")(start=0, width=basis_width, m=5)
        num_nodes = nx.number_of_nodes(basis)
        edges_arr = np.array(basis.edges).T
        edge_index = torch.cat((torch.tensor(edges_arr), torch.tensor(np.array([edges_arr[1],edges_arr[0]]))), dim=1)
        #edge_index = torch.tensor(np.array(basis.edges).T)
        ba_graph = Data(edge_index=edge_index, num_nodes=num_nodes)
        node_masks = [torch.zeros(ba_graph.num_nodes)]
        edge_masks = [torch.zeros(ba_graph.num_edges)]
        node_label = torch.zeros(num_nodes, dtype=torch.long)
        edge_label = torch.zeros(edge_index.shape[1], dtype=torch.long)

        if k <1000:
            graph_label=[1, 0]
            if random_motif == True:
                motif_count = np.random.randint(1, 4)
            else:
                motif_count = 3
            motif_slices =  random.sample(list(range(3)), motif_count)
            #print(motif_slices)
        else:
            graph_label=[0, 1]
            if random_motif == True:
                motif_count = np.random.randint(1, 4)
            else:
                motif_count = 3
            motif_slices =  random.sample(list(range(3,6)), motif_count)
            #print(motif_slices)
        
        #gen motifs and connecting motifs to basis
        #num_nodes, edge_index, node_label, edge_label, node_masks, edge_masks = gen_random_motifs(basis, motif_slices, node_label, edge_label, node_masks, edge_masks)
        for motif_index in range(len(motif_slices)):
            #gen motif
            motif_type =  motif_slices[motif_index]
            if motif_type==0:
                motif, role_id = eval("house")(start=num_nodes)
            elif motif_type==1:
                motif, role_id = eval("star")(start=num_nodes, nb_branches=6)
            elif motif_type==2:
                motif, role_id = eval("cycle")(start=num_nodes, len_cycle=6)
            elif motif_type==3:
                motif, role_id = eval("grid")(start=num_nodes,dim=3)
            elif motif_type==4:
                motif, role_id = eval("fan")(start=num_nodes, nb_branches=6)
            elif motif_type==5:
                motif, role_id = eval("diamond")(start=num_nodes)

            #connecting motif to basis
            motif_num_nodes = nx.number_of_nodes(motif)
            node_masks.append(torch.ones(motif_num_nodes))
            node_label = torch.cat((node_label, torch.tensor([motif_type+1] * motif_num_nodes, dtype=torch.long)))
            
            #merge edges
            motif_edges = np.array(motif.edges).T.tolist()
            motif_edge_index = torch.cat((torch.tensor(motif_edges), torch.tensor([motif_edges[1], motif_edges[0]])), dim=1)
            edge_index = torch.cat((edge_index, motif_edge_index), dim=1)
            motif_num_edges = motif_edge_index.shape[1]
            edge_masks.append(torch.ones(motif_num_edges))
            edge_label = torch.cat((edge_label, torch.tensor([motif_type+1] * motif_num_edges, dtype=torch.long)))

            #connecting
            basis_conn_node = random.sample(list(basis.nodes), 1)[0]
            motif_conn_node = random.sample(list(motif.nodes), 1)[0]
            edge_index = torch.cat((edge_index, torch.tensor([[basis_conn_node, motif_conn_node], [motif_conn_node, basis_conn_node]])), dim=1)
            edge_masks.append(torch.zeros(2))
            edge_label = torch.cat((edge_label, torch.zeros(2, dtype=torch.long)))

            num_nodes = num_nodes + motif_num_nodes

        #feat = torch.ones(num_nodes, feat_dim)/feat_dim
        feat = torch.ones(num_nodes, feat_dim)
        data_list.append(Data(x=feat.float(),
                                edge_index=edge_index,
                                y=torch.from_numpy(np.where(graph_label)[0]), node_label= node_label, edge_label = edge_label, node_mask= torch.cat(node_masks), edge_mask = torch.cat(edge_masks)))
            
    if random_motif:
        filename = "/mnt/8T/torch_projects/datasets/BA_6Motifs_2class_random_withedgelabel/raw/BA_6Motifs_2class_random_withedgelabel.pkl"
    else:
        filename = "/mnt/8T/torch_projects/datasets/BA_6Motifs_2class_withedgelabel/raw/BA_6Motifs_2class_withedgelabel.pkl"
    with open(filename, "wb") as file:
        pickle.dump(data_list, file)
    print("generate complete")



def main_gen_BA_3motifs_7class():
    data_list = []
    num_graphs = 3500
    feat_dim = 10
    motif_slices = []
    for k in range(num_graphs):
        basis_width =20
        #generate basis  BA graph
        basis, role_id = eval("ba")(start=0, width=basis_width, m=5)
        num_nodes = nx.number_of_nodes(basis)
        edges_arr = np.array(basis.edges).T
        edge_index = torch.cat((torch.tensor(edges_arr), torch.tensor(np.array([edges_arr[1],edges_arr[0]]))), dim=1)
        #edge_index = torch.tensor(np.array(basis.edges).T)
        ba_graph = Data(edge_index=edge_index, num_nodes=num_nodes)
        node_masks = [torch.zeros(ba_graph.num_nodes)]
        edge_masks = [torch.zeros(ba_graph.num_edges)]
        node_label = torch.zeros(num_nodes, dtype=torch.long)
        edge_label = torch.zeros(edge_index.shape[1], dtype=torch.long)

        if k <500:
            graph_label=[1, 0, 0, 0, 0, 0, 0]
            #motif_slices =  [1,2]
            motif_slices =  [4,2]
        elif k >=500 and k <1000:
            graph_label=[0, 1, 0, 0, 0, 0, 0]
            #motif_slices =  [1,3]
            motif_slices =  [4,3]
        elif k>=1000 and k <1500:
            graph_label=[0, 0, 1, 0, 0, 0, 0]
            motif_slices =  [2,3]
        elif k>=1500 and k <2000:
            graph_label=[0, 0, 0, 1, 0, 0, 0]
            #motif_slices =  [1,2,3]
            motif_slices =  [4,2,3]
        elif k>=2000 and k <2500:
            graph_label=[0, 0, 0, 0, 1, 0, 0]
            #motif_slices =  [1]
            motif_slices =  [4]
        elif k>=2500 and k <3000:
            graph_label=[0, 0, 0, 0, 0, 1, 0]
            motif_slices =  [2]
        elif k>=3000 and k <3500:
            graph_label=[0, 0, 0, 0, 0, 0, 1]
            motif_slices =  [3]

        #gen motifs and connecting motifs to basis
        for motif_type in motif_slices:
            #gen motif
            if motif_type==1:
                motif, role_id = eval("house")(start=num_nodes)
            elif motif_type==2:
                motif, role_id = eval("star")(start=num_nodes, nb_branches=6)
            elif motif_type==3:
                motif, role_id = eval("cycle")(start=num_nodes, len_cycle=6)
            elif motif_type==4:
                motif, role_id = eval("grid")(start=num_nodes,dim=3)
            elif motif_type==5:
                motif, role_id = eval("fan")(start=num_nodes, nb_branches=6)
            elif motif_type==6:
                motif, role_id = eval("diamond")(start=num_nodes)

            #connecting motif to basis
            motif_num_nodes = nx.number_of_nodes(motif)
            node_masks.append(torch.ones(motif_num_nodes))
            node_label = torch.cat((node_label, torch.tensor([motif_type] * motif_num_nodes, dtype=torch.long)))
            
            #merge edges
            motif_edges = np.array(motif.edges).T.tolist()
            motif_edge_index = torch.cat((torch.tensor(motif_edges), torch.tensor([motif_edges[1], motif_edges[0]])), dim=1)
            edge_index = torch.cat((edge_index, motif_edge_index), dim=1)
            motif_num_edges = motif_edge_index.shape[1]
            edge_masks.append(torch.ones(motif_num_edges))
            edge_label = torch.cat((edge_label, torch.tensor([motif_type] * motif_num_edges, dtype=torch.long)))

            #connecting
            basis_conn_node = random.sample(list(basis.nodes), 1)[0]
            motif_conn_node = random.sample(list(motif.nodes), 1)[0]
            edge_index = torch.cat((edge_index, torch.tensor([[basis_conn_node, motif_conn_node], [motif_conn_node, basis_conn_node]])), dim=1)
            edge_masks.append(torch.zeros(2))
            edge_label = torch.cat((edge_label, torch.zeros(2, dtype=torch.long)))

            num_nodes = num_nodes + motif_num_nodes

        #feat = torch.ones(num_nodes, feat_dim)/feat_dim
        feat = torch.ones(num_nodes, feat_dim)
        data_list.append(Data(x=feat.float(),
                                edge_index=edge_index,
                                y=torch.from_numpy(np.where(graph_label)[0]), node_label= node_label, edge_label = edge_label, node_mask= torch.cat(node_masks), edge_mask = torch.cat(edge_masks)))
            
    filename = "/mnt/8T/torch_projects/datasets/BA_3Motifs_4class_423/raw/BA_3Motifs_7class_423.pkl"
    with open(filename, "wb") as file:
        pickle.dump(data_list, file)
    print("generate complete")



main_gen_BA_3motifs_7class()



'''
star, role_id = eval("fan")(start=0, nb_branches=5)
num_nodes = nx.number_of_nodes(star)
edge_index = torch.tensor(np.array(star.edges).T)
plot_graph(edge_index, 0)
print("test")
'''

'''
basis, role_id = eval("diamond")(start=0)
num_nodes = nx.number_of_nodes(basis)
edge_index = torch.tensor(np.array(basis.edges).T)
plot_graph(edge_index)
'''