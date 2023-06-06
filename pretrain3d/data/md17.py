import torch
import numpy as np
import re
from torch_sparse import SparseTensor
import networkx as nx

from torch_geometric.datasets.md17 import MD17
from torch_geometric.data import Data
from torch_geometric.nn import radius_graph

from typing import Callable, List, Optional
from tqdm import tqdm

# also see pretrain3d/utils/graph.py
def get_face_of_radius_graph(G):

    bond2id = dict()
    for i, bond in enumerate(G.edges()):
        bond2id[tuple(bond)] = len(bond2id)
        bond2id[tuple(reversed(bond))] = len(bond2id)

    ssr = []
    for cycle in nx.cycle_basis(G):
        if len(cycle) > 2:
            ssr.append(cycle)

    num_edge = len(bond2id)
    left = [0] * num_edge
    face = [[]]
    for ring in ssr:
        ring = list(ring)

        bond_list = []
        for i, atom in enumerate(ring):
            bond_list.append((ring[i - 1], atom))

        exist = False
        if any([left[bond2id[bond]] != 0 for bond in bond_list]):
            exist = True
        if exist:
            ring = list(reversed(ring))
        face.append(ring)
        for i, atom in enumerate(ring):
            bond = (ring[i - 1], atom)
            if left[bond2id[bond]] != 0:
                bond = (atom, ring[i - 1])
            bondid = bond2id[bond]
            if left[bondid] == 0:
                left[bondid] = len(face) - 1

    return face, left, bond2id
    


class MD17Dataset(MD17):
    def __init__(
        self,
        root: str,
        name: str,
        train: Optional[bool] = None,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
    ):
        super(MD17Dataset, self).__init__(root, name, train, transform, pre_transform, pre_filter)
    

    def process(self):
        it = zip(self.raw_paths, self.processed_paths)
        for raw_path, processed_path in it:
            raw_data = np.load(raw_path)

            if self.revised:
                z = torch.from_numpy(raw_data['nuclear_charges']).long()
                pos = torch.from_numpy(raw_data['coords']).float()
                energy = torch.from_numpy(raw_data['energies']).float()
                force = torch.from_numpy(raw_data['forces']).float()
            else:
                z = torch.from_numpy(raw_data['z']).long() # [21]
                pos = torch.from_numpy(raw_data['R']).float() # [100000, 21, 3]
                energy = torch.from_numpy(raw_data['E']).float() # [100000]
                force = torch.from_numpy(raw_data['F']).float() # [100000, 21, 3]

            data_list = []
            for i in tqdm(range(pos.size(0)), desc=f'Processing {processed_path}'):
                data = DGData(z=z, pos=pos[i], energy=energy[i], force=force[i])
                edge_index = radius_graph(pos[i], r=5)

                data.__num_nodes__ = int(pos[i].size(0))

                G = nx.Graph()
                G.add_nodes_from(range(data.num_nodes))
                G.add_edges_from(edge_index.t().tolist())

                edges_list = []
                edge_features_list = []

                num_bond_features = 3 # dx, dy, dz
                if len(G.edges()) > 0:
                    for bond in G.edges():
                        s = bond[0]
                        t = bond[1]

                        edge_feature = pos[i][s] - pos[i][t]
                                        
                        # add edges in both directions
                        edges_list.append((s, t))
                        edge_features_list.append(edge_feature)
                        edges_list.append((t, s))
                        edge_features_list.append(-edge_feature)

                    edge_index = torch.LongTensor(edges_list).T
                    edge_attr = torch.stack(edge_features_list)

                    faces, left, _ = get_face_of_radius_graph(G)

                    num_faces = len(faces)
                    face_mask = [False] * num_faces
                    face_index = [[-1, -1]] * len(edges_list)
                    face_mask[0] = True
                    for i in range(len(edges_list)):
                        inface = left[i ^ 1]
                        outface = left[i]
                        face_index[i] = [inface, outface]

                    nf_node = []
                    nf_ring = []
                    for i, face in enumerate(faces):
                        face = list(set(face))
                        nf_node.extend(face)
                        nf_ring.extend([i] * len(face))

                    face_mask = torch.BoolTensor(face_mask)
                    face_index = torch.LongTensor(face_index).T
                    n_nfs = len(nf_node)
                    nf_node = torch.LongTensor(nf_node).reshape(1, -1)
                    nf_ring = torch.LongTensor(nf_ring).reshape(1, -1)
                else:
                    edge_index = torch.zeros((2, 0), dtype=torch.long)
                    edge_attr = torch.zeros((0, num_bond_features), dtype=torch.long)
                    face_mask = torch.zeros((0), dtype=torch.bool)
                    face_index = torch.zeros((2, 0), dtype=torch.long)
                    num_faces = 0
                    n_nfs = 0
                    nf_node = torch.zeros((1, 0), dtype=np.int64)
                    nf_ring = torch.zeros((1, 0), dtype=np.int64)

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
                # data.edge_feat = 
                data.edge_attr = edge_attr
                # data.node_feat = 
                data.x = z.unsqueeze(-1)
                data.num_nodes = len(z)

                data.ring_mask = face_mask
                data.ring_index = face_index
                data.num_rings = num_faces
                data.n_edges = len(edge_attr)
                data.n_nodes = len(z)

                data.n_nfs = n_nfs
                data.nf_node = nf_node
                data.nf_ring = nf_ring

                data.nei_src_index = nei_src_index
                data.nei_tgt_index = nei_tgt_index
                data.nei_tgt_mask = nei_tgt_mask

                if self.pre_filter is not None and not self.pre_filter(data):
                    continue
                if self.pre_transform is not None:
                    data = self.pre_transform(data)
                data_list.append(data)

            torch.save(self.collate(data_list), processed_path)

class DGData(Data):
    def __cat_dim__(self, key, value, *args, **kwargs):
        if isinstance(value, SparseTensor):
            return (0, 1)
        elif bool(re.search("(index|face)", key)):
            return -1
        elif bool(re.search("(nf_node|nf_ring|nei_tgt_mask)", key)):
            return -1
        return 0

    def __inc__(self, key, value, *args, **kwargs):
        if bool(re.search("(ring_index|nf_ring)", key)):
            return int(self.num_rings.item())
        elif bool(re.search("(index|face|nf_node)", key)):
            return self.num_nodes
        else:
            return 0


