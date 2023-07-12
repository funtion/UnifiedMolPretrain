from torch_geometric.data import InMemoryDataset
import numpy as np
import torch
from tqdm import tqdm

from matbench.bench import MatbenchBenchmark
from pymatgen.optimization.neighbors import find_points_in_spheres
from pretrain3d.utils.graph import get_face_of_radius_graph
from pretrain3d.data.pcqm4m import DGData
from torch_geometric.nn import radius_graph
import networkx as nx

import multiprocessing as mp
import os

TRAIN_PARTITIONS = 10

def process_sinlge(args):
    z = torch.LongTensor(args[0])
    pos = torch.FloatTensor(args[1])
    target = torch.FloatTensor([args[2]])
    crystal_structure = args[3]
    lattice_matrix = crystal_structure.lattice.matrix.astype(float)
    # print(crystal_structure.lattice.as_dict())
    cutoff_radius = 5.0
    numerical_tol = 1e-8

    data = DGData(z=z, pos=pos, target=target)
    # edge_index0= radius_graph(pos, r=5)

    pbc_ = np.array([1, 1, 1], dtype=int)
    # pbc_ = np.array([0, 0, 0], dtype=int)
    # pbc_ = np.array(crystal_structure.pbc, dtype=int)
    center_indices, neighbor_indices, offset_vectors, distances = find_points_in_spheres(
        crystal_structure.cart_coords,
        crystal_structure.cart_coords,
        r=cutoff_radius,
        pbc=pbc_,
        lattice=lattice_matrix,
        tol=numerical_tol,
    )

    center_indices = center_indices.astype(np.int64)
    neighbor_indices = neighbor_indices.astype(np.int64)
    offset_vectors = offset_vectors.astype(np.int64)
    distances = distances.astype(float)
    exclude_self = (center_indices != neighbor_indices) | (distances > numerical_tol)
    sent_index = center_indices[exclude_self]
    receive_index = neighbor_indices[exclude_self]
    pbc_vectors = torch.FloatTensor(offset_vectors[exclude_self])
    distances = distances[exclude_self]
    edge_index = torch.from_numpy(np.array([sent_index,receive_index]))
    
    edge_attr = pos[edge_index[0]] - (pos[edge_index[1]] +
                                      torch.einsum("bi, ij->bj", pbc_vectors, torch.FloatTensor(lattice_matrix)))
    
    data.__num_nodes__ = int(pos.size(0))


    G = nx.Graph()
    G.add_nodes_from(range(data.num_nodes))
    G.add_edges_from(edge_index.t().tolist())

    edge_index = torch.cat((edge_index,  torch.from_numpy(np.array([receive_index, sent_index]))), dim=1)
    edge_attr = torch.cat((edge_attr, -edge_attr),dim=0)
    # print(edge_attr[:5])
    # print(edge_attr[120:126])
    # edges_list = []
    # edge_features_list = []

    # num_bond_features = 3 # dx, dy, dz 
    # if len(G.edges()) > 0:
    #     for bond in G.edges():
    #         s = bond[0]
    #         t = bond[1]

    #         edge_feature = pos[s] - pos[t]
                            
    #         # add edges in both directions
    #         edges_list.append((s, t))
    #         edge_features_list.append(edge_feature)
    #         edges_list.append((t, s))
    #         edge_features_list.append(-edge_feature)

    #     edge_index = torch.LongTensor(edges_list).T
    #     edge_attr = torch.stack(edge_features_list)

    #     faces, left, _ = get_face_of_radius_graph(G)  # ring left ?

    #     num_faces = len(faces)
    #     face_mask = [False] * num_faces
    #     face_index = [[-1, -1]] * len(edges_list)  # ? 表示什么？
    #     face_mask[0] = True
    #     for ii in range(len(edges_list)):
    #         inface = left[ii ^ 1]
    #         outface = left[ii]
    #         face_index[ii] = [inface, outface]

    #     nf_node = []
    #     nf_ring = []
    #     for ii, face in enumerate(faces):
    #         face = list(set(face))
    #         nf_node.extend(face)
    #         nf_ring.extend([ii] * len(face))

    #     face_mask = torch.BoolTensor(face_mask)
    #     face_index = torch.LongTensor(face_index).T
    #     n_nfs = len(nf_node)
    #     nf_node = torch.LongTensor(nf_node).reshape(1, -1)
    #     nf_ring = torch.LongTensor(nf_ring).reshape(1, -1)
    # else:
    #     edge_index = torch.zeros((2, 0), dtype=torch.long)
    #     edge_attr = torch.zeros((0, num_bond_features), dtype=torch.long)
    #     face_mask = torch.zeros((0), dtype=torch.bool)
    #     face_index = torch.zeros((2, 0), dtype=torch.long)
    #     num_faces = 0
    #     n_nfs = 0
    #     nf_node = torch.zeros((1, 0), dtype=torch.long)
    #     nf_ring = torch.zeros((1, 0), dtype=torch.long)

    
    
    face_mask = torch.zeros((0), dtype=torch.bool)
    face_index = torch.zeros((2, 0), dtype=torch.long)
    num_faces = 0
    n_nfs = 0
    nf_node = torch.zeros((1, 0), dtype=torch.long)
    nf_ring = torch.zeros((1, 0), dtype=torch.long)

    n_src = list()
    n_tgt = list()
    for atom in G.nodes():
        n_ids = list(G.neighbors(atom))
        if len(n_ids) > 1:
            n_src.append(atom)
            n_tgt.append(n_ids[:6])
    nums_neigh = len(n_src)
    nei_src_index = torch.LongTensor(n_src).reshape(1, -1)
    nei_tgt_index = torch.zeros((6, nums_neigh), dtype=torch.long)
    nei_tgt_mask = torch.ones((6, nums_neigh), dtype=bool)
    

    for i, n_ids in enumerate(n_tgt):
        nei_tgt_index[: len(n_ids), i] = torch.LongTensor(n_ids)
        nei_tgt_mask[: len(n_ids), i] = False

    data.edge_index = edge_index
    data.edge_attr = edge_attr
    data.x = z.unsqueeze(-1)
    data.num_nodes = len(z)
    data.n_edges = len(edge_attr)
    data.n_nodes = len(z)

    data.ring_mask = face_mask
    data.ring_index = face_index
    data.num_rings = num_faces
    data.n_nfs = n_nfs
    data.nf_node = nf_node
    data.nf_ring = nf_ring

    data.nei_src_index = nei_src_index
    data.nei_tgt_index = nei_tgt_index
    data.nei_tgt_mask = nei_tgt_mask

    return data



class MATBENCH(InMemoryDataset):
    def __init__(self, root, data, split:str, target:str, cutoff_radius=5.,transform=None, pre_transform=None, pre_filter=None):
        # assert split in ['train', 'validation', 'test']
        self.data = data
        self.split = split
        self.cutoff_radius = cutoff_radius
        self.target = target
        # self.matbench = MatbenchBenchmark(autoload=False)
        super().__init__(root, transform, pre_transform, pre_filter)

        self.data, self.slices = torch.load(self.processed_paths[0])
        print(f"{self.split} dataset: ",self.data)
        print("************************************************************")

    @property
    def raw_file_names(self):
        return [self.split + '.npz']
    
    @property
    def processed_file_names(self):
        return [self.split + '.pt']

    # Strangly, the processed_path in the parent class cannot be accessed
    @property
    def processed_paths(self):
        r"""The absolute filepaths that must be present in order to skip
        processing."""
        files = self.processed_file_names
        return [os.path.join(self.processed_dir, f) for f in list(files)]

    def process(self):
        # task = getattr(self.matbench, self.dataset_name)
        # task.load()
        # if self.split == "train":
        #     data = task.get_train_and_val_data(0, as_type="df")  # fold == 0
        # elif self.split == "test":
        #     data = task.get_test_data(
        #             0, include_target=True, as_type="df"
        #         )
            
        # target_name = [ col for col in data.columns
        #            if col not in ("id", "structure", "composition")
        #         ][0]
        
        
        z_list = []
        pos_list = []
        target_list = []
        crystal_structure_list = []

        for _, j in self.data.iterrows():
            z_list.append(np.array(j.structure.atomic_numbers))
            pos_list.append(j.structure.cart_coords)
            target_list.append(getattr(j, self.target))
            crystal_structure_list.append(j.structure.copy())

        data_list = []
        for result in map(
            process_sinlge,
            tqdm(zip(z_list, pos_list, target_list, crystal_structure_list), total=len(z_list), desc=f'Processing {self.processed_paths[0]}'),
        ):
            data_list.append(result)
        print("Processed data success!")
        data, slices = self.collate(data_list)
        print("Collated data success!")
        torch.save((data, slices), self.processed_paths[0])
