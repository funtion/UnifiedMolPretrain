import numpy as np
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from pretrain3d.utils.graph import get_face_of_radius_graph
from pretrain3d.data.pcqm4m import DGData
from torch_geometric.nn import radius_graph
import networkx as nx

import multiprocessing as mp
from torch_geometric.data.collate import collate

from concurrent.futures import ProcessPoolExecutor  
  
def process_sinlge(args):
    z = torch.LongTensor(args[0])
    pos = torch.FloatTensor(args[1])
    energy = args[2]
    force = torch.FloatTensor(args[3])

    data = DGData(z=z, pos=pos, energy=energy, force=force)
    edge_index = radius_graph(pos, r=5)

    data.__num_nodes__ = int(pos.size(0))

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

            edge_feature = pos[s] - pos[t]
                            
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
        for ii in range(len(edges_list)):
            inface = left[ii ^ 1]
            outface = left[ii]
            face_index[ii] = [inface, outface]

        nf_node = []
        nf_ring = []
        for ii, face in enumerate(faces):
            face = list(set(face))
            nf_node.extend(face)
            nf_ring.extend([ii] * len(face))

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

    return data

def main():
    data_home = Path('./dataset/SPICE')

    for split in ['train', 'validation', 'test'][::-1]:
        output_path = data_home / 'processed' / f'{split}.pt'
        if output_path.exists():
            print(f'{split} already processed')
            continue

        data = np.load(data_home / 'raw' / f'{split}.npz', allow_pickle=True, mmap_mode='r')

        z_list = data['z']
        pos_list = data['R']
        energy_list = data['E']
        force_list = data['F']

        data_list = []
        for result in map(
            process_sinlge,
            tqdm(zip(z_list, pos_list, energy_list, force_list), total=len(z_list), desc=f'Processing {split}'),
            ):
            data_list.append(result)
        

        # num_processes = max(1, mp.cpu_count() // 2)  
        # with ProcessPoolExecutor(max_workers=num_processes) as executor:  

        #     for result in executor.map(
        #             process_sinlge,
        #             tqdm(zip(z_list, pos_list, energy_list, force_list), total=len(z_list), desc=f'Processing {split}'),
        #             timeout=60):
        #         data_list.append(result)

        # with mp.Pool(processes=mp.cpu_count()) as pool:
        #     data_list = pool.map(
        #         process_sinlge,
        #         tqdm(zip(z_list, pos_list, energy_list, force_list), total=len(z_list), desc=f'Processing {split}'),
        #         )

            # batch_size = math.ceil(len(data_list) // TRAIN_PARTITIONS)
            # print('Batch size:', batch_size, 'Total data:', len(data_list), 'Total partitions:', TRAIN_PARTITIONS)
            # batchs = [data_list[i * batch_size : (i + 1) * batch_size] for i in range(TRAIN_PARTITIONS)]
        print("Processed data success!")
        colldated_data = collate(
            data_list[0].__class__,
            data_list=data_list,
            increment=False,
            add_batch=False
        )
        print("Collated data success!")
        torch.save(colldated_data,  output_path)

if __name__ ==  '__main__':
    main()
