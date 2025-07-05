import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
import networkx as nx
import argparse
import os
from datetime import datetime

# Suppress RDKit warnings (including deprecation warnings)
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

# Définitions de mol_features.py
SYMBOLS = ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg',
           'Na', 'Ca', 'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl',
           'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn',
           'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn',
           'Zr', 'Cr', 'Pt', 'Hg', 'Pb', 'W', 'Ru', 'Nb', 'Re',
           'Te', 'Rh', 'Tc', 'Ba', 'Bi', 'Hf', 'Mo', 'U', 'Sm',
           'Os', 'Ir', 'Ce', 'Gd', '*', 'UNK']  # 64 éléments
FORMAL_CHARGES = [-2, -1, 0, 1, 2]
BOND_TYPES = [Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE,
              Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.AROMATIC, None]
MAX_NEIGHBORS = 10
DEGREES = list(range(MAX_NEIGHBORS))
EXPLICIT_VALENCES = [0, 1, 2, 3, 4, 5, 6]
IMPLICIT_VALENCES = [0, 1, 2, 3, 4, 5]
N_ATOM_FEATS = len(SYMBOLS) + len(FORMAL_CHARGES) + len(DEGREES) + len(EXPLICIT_VALENCES) + len(IMPLICIT_VALENCES) + 1  # 93
N_BOND_FEATS = len(BOND_TYPES) + 1 + 1  # 7

def onek_unk_encoding(x, set):
    if x not in set:
        x = 'UNK'
    return [int(x == s) for s in set]

def get_atom_features(atom):
    symbol = onek_unk_encoding(atom.symbol, SYMBOLS)
    fc = onek_unk_encoding(atom.fc, FORMAL_CHARGES)
    degree = onek_unk_encoding(atom.degree, DEGREES)
    exp_valence = onek_unk_encoding(atom.exp_valence, EXPLICIT_VALENCES)
    imp_valence = onek_unk_encoding(atom.imp_valence, IMPLICIT_VALENCES)
    aro = [atom.aro]
    return np.array(symbol + fc + degree + exp_valence + imp_valence + aro, dtype=np.float32)

def get_bond_features(bond):
    if bond is None:
        bond_type = onek_unk_encoding(None, BOND_TYPES)
        conj = [0]
        ring = [0]
    else:
        bond_type = onek_unk_encoding(bond.bond_type, BOND_TYPES)
        conj = [bond.is_conjugated]
        ring = [bond.is_in_ring]
    return np.array(bond_type + conj + ring, dtype=np.float32)

# Définitions de path_utils.py
def get_num_path_features(max_path_length, p_embed=True, ring_embed=True):
    num_features = max_path_length * N_BOND_FEATS
    if p_embed:
        num_features += max_path_length + 2
    if ring_embed:
        num_features += 5
    return num_features

def get_path_features(rd_mol, path_atoms, path_length, max_path_length, p_embed=True):
    path_bonds = []
    for path_idx in range(len(path_atoms) - 1):
        atom_1 = path_atoms[path_idx]
        atom_2 = path_atoms[path_idx + 1]
        bond = rd_mol.GetBondBetweenAtoms(atom_1, atom_2)
        path_bonds.append(bond)
    
    features = []
    for path_idx in range(max_path_length):
        bond = path_bonds[path_idx] if path_idx < len(path_bonds) else None
        features.append(get_bond_features(Bond(0, 0, 0, bond) if bond else None))
    if p_embed:
        position_feature = np.zeros(max_path_length + 2)
        position_feature[path_length] = 1
        features.append(position_feature)
    return np.concatenate(features, axis=0)

def get_ring_features(ring_dict, atom_pair):
    ring_features = np.zeros(5)
    if atom_pair in ring_dict:
        ring_features[0] = 1
        rings = ring_dict[atom_pair]
        for ring_size, aromatic in rings:
            if ring_size == 5 and not aromatic:
                ring_features[1] = 1
            elif ring_size == 5 and aromatic:
                ring_features[2] = 1
            elif ring_size == 6 and not aromatic:
                ring_features[3] = 1
            elif ring_size == 6 and aromatic:
                ring_features[4] = 1
    return ring_features

def ordered_pair(a1, a2):
    return (min(a1, a2), max(a1, a2))

# Classes de mol_graph.py
class Atom:
    def __init__(self, idx, rd_atom=None, is_dummy=False):
        self.idx = idx
        self.bonds = []
        self.is_dummy = is_dummy
        if is_dummy:
            self.symbol = '*'
        if rd_atom is not None:
            self.symbol = rd_atom.GetSymbol()
            self.fc = rd_atom.GetFormalCharge()
            self.degree = rd_atom.GetDegree()
            try:
                self.exp_valence = rd_atom.GetExplicitValence()
                self.imp_valence = rd_atom.GetImplicitValence()
            except Exception as e:
                print(f"Erreur lors de l'accès aux valences: {str(e)}")
                self.exp_valence = 0
                self.imp_valence = 0
            self.aro = int(rd_atom.GetIsAromatic())

    def add_bond(self, bond):
        self.bonds.append(bond)

class Bond:
    def __init__(self, idx, out_atom_idx, in_atom_idx, rd_bond=None):
        self.idx = idx
        self.out_atom_idx = out_atom_idx
        self.in_atom_idx = in_atom_idx
        if rd_bond is not None:
            self.bond_type = rd_bond.GetBondType()
            self.is_conjugated = int(rd_bond.GetIsConjugated())
            self.is_in_ring = int(rd_bond.IsInRing())

class Molecule:
    def __init__(self, atoms, bonds):
        self.atoms = atoms
        self.bonds = bonds

class MolGraph:
    def __init__(self, atoms, bonds, scope, path_input, path_mask, device):
        self.mols = [Molecule(atoms, bonds)]
        self.scope = scope
        self.path_input = path_input
        self.path_mask = path_mask
        self.device = device

    def get_atom_inputs(self, output_tensors=True):
        fatoms = [get_atom_features(atom) for atom in self.mols[0].atoms]
        fatoms = np.stack(fatoms, axis=0)
        if output_tensors:
            fatoms = torch.tensor(fatoms, device=self.device).float()
        return fatoms, self.scope

    def get_graph_inputs(self, output_tensors=True):
        return None, self.scope  # Non utilisé pour MolTransformer

# Utilitaires pour le modèle
class StatsTracker:
    def __init__(self):
        self.stats = {}

    def add_stat(self, name, value, count):
        if name not in self.stats:
            self.stats[name] = {'sum': 0, 'count': 0}
        self.stats[name]['sum'] += value
        self.stats[name]['count'] += count

    def get_stat(self, name):
        if name in self.stats and self.stats[name]['count'] > 0:
            return self.stats[name]['sum'] / self.stats[name]['count']
        return 0

class ModelUtils:
    @staticmethod
    def compute_max_atoms(scope):
        return max([le for _, le in scope])

    @staticmethod
    def convert_to_3D(tensor, scope, max_atoms, device, self_attn=False):
        batch_sz = len(scope)
        output = torch.zeros(batch_sz, max_atoms, tensor.size(1), device=device)
        atom_mask = torch.zeros(batch_sz, max_atoms, max_atoms, device=device)
        for mol_idx, (st, le) in enumerate(scope):
            output[mol_idx, :le] = tensor[st:st+le]
            atom_mask[mol_idx, :le, :le] = 1
            if not self_attn:
                for i in range(le):
                    atom_mask[mol_idx, i, i] = 0
        return output, atom_mask

    @staticmethod
    def convert_to_2D(tensor, scope):
        output = []
        for mol_idx, (st, le) in enumerate(scope):
            output.append(tensor[mol_idx, :le])
        return torch.cat(output, dim=0)

# Dataset
class PAGTN_Dataset(Dataset):
    def __init__(self, Data_List, max_distance=5, return_energies=True, device='cpu', p_embed=True, ring_embed=True, self_attn=False, no_truncate=False):
        self.Data_List = Data_List
        self.N = len(Data_List)
        self.max_distance = max_distance
        self.return_energies = return_energies
        self.device = torch.device(device)
        self.p_embed = p_embed
        self.ring_embed = ring_embed
        self.self_attn = self_attn
        self.no_truncate = no_truncate
        
        self.n_atom_feats = N_ATOM_FEATS  # 93
        self.n_path_feats = get_num_path_features(max_distance, p_embed, ring_embed)  # 47

        if self.N == 0:
            raise ValueError("La liste de données est vide.")

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        data = self.Data_List[idx]
        id_mol = data['Id']
        symbols = data['Atoms_DataFrame'].iloc[:, 0].values
        positions = data['Atoms_DataFrame'].iloc[:, 1:4].values
        bonds = data.get('Bonds', None)
        energy = data['Energy'] if self.return_energies else 0.0

        if len(symbols) != len(positions):
            raise ValueError(f"Molécule {id_mol}: le nombre de symboles ({len(symbols)}) ne correspond pas au nombre de positions ({len(positions)})")

        try:
            mol, atoms, bonds = self._create_molecule(symbols, positions, bonds)
        except Exception as e:
            print(f"Erreur lors de la création de la molécule {id_mol}: {str(e)}")
            raise
        
        paths_dict, pointer_dict, ring_dict = self._compute_shortest_paths(mol)
        
        n_atoms = mol.GetNumAtoms()
        path_input = np.zeros((n_atoms, n_atoms, self.n_path_feats), dtype=np.float32)
        path_mask = np.zeros((n_atoms, n_atoms), dtype=np.float32)
        
        for atom_1 in range(n_atoms):
            for atom_2 in range(n_atoms):
                path_atoms, path_length, mask_ind = self._get_path_atoms(
                    atom_1, atom_2, paths_dict, pointer_dict, self.max_distance,
                    truncate=not self.no_truncate, self_attn=self.self_attn)
                path_features = get_path_features(
                    mol, path_atoms, path_length, self.max_distance, self.p_embed)
                if self.ring_embed:
                    ring_features = get_ring_features(ring_dict, ordered_pair(atom_1, atom_2))
                    path_features = np.concatenate([path_features, ring_features], axis=0)
                path_input[atom_1, atom_2] = path_features
                path_mask[atom_1, atom_2] = mask_ind
        
        scope = [(0, n_atoms)]
        
        path_input = torch.tensor(path_input, dtype=torch.float, device=self.device)
        path_mask = torch.tensor(path_mask, dtype=torch.float, device=self.device)
        energy = torch.tensor([energy], dtype=torch.float, device=self.device) if self.return_energies else None

        mol_graph = MolGraph(atoms, bonds, scope, path_input, path_mask, self.device)

        return id_mol, mol_graph, energy

    def _create_molecule(self, symbols, positions, bonds=None):
        mol = Chem.RWMol()
        atom_map = {}
        atoms = []
        
        for i, symbol in enumerate(symbols):
            rd_atom = Chem.Atom(symbol)
            rd_atom.SetDoubleProp('x', positions[i, 0])
            rd_atom.SetDoubleProp('y', positions[i, 1])
            rd_atom.SetDoubleProp('z', positions[i, 2])
            atom_idx = mol.AddAtom(rd_atom)
            atom_map[i] = atom_idx
            atoms.append(Atom(idx=atom_idx, rd_atom=rd_atom))
        
        if bonds is not None:
            for i, j, bond_type in bonds:
                if i not in atom_map or j not in atom_map:
                    raise ValueError(f"Molécule: indices d'atomes invalides dans Bonds ({i}, {j})")
                if bond_type in [1, 2, 3, 1.5]:
                    bond_type = {1: Chem.BondType.SINGLE, 2: Chem.BondType.DOUBLE,
                                 3: Chem.BondType.TRIPLE, 1.5: Chem.BondType.AROMATIC}[bond_type]
                    mol.AddBond(atom_map[i], atom_map[j], bond_type)
                else:
                    raise ValueError(f"Molécule: type de liaison invalide {bond_type}")
        
        mol = mol.GetMol()
        try:
            Chem.SanitizeMol(mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL)
        except Exception as e:
            print(f"Erreur de sanitisation pour la molécule: {str(e)}")
            raise
        
        try:
            mol = Chem.AddHs(mol, addCoords=True)
        except Exception as e:
            print(f"Erreur lors de l'ajout des hydrogènes: {str(e)}")
            raise
        
        atoms = []
        for i in range(mol.GetNumAtoms()):
            rd_atom = mol.GetAtomWithIdx(i)
            atoms.append(Atom(idx=i, rd_atom=rd_atom))
        
        conf = Chem.Conformer(mol.GetNumAtoms())
        for i in range(len(positions)):
            conf.SetAtomPosition(i, positions[i])
        for i in range(len(positions), mol.GetNumAtoms()):
            conf.SetAtomPosition(i, [0, 0, 0])
        mol.AddConformer(conf)
        
        try:
            Chem.SanitizeMol(mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL)
        except Exception as e:
            print(f"Erreur de sanitisation après ajout du conformer: {str(e)}")
            raise
        
        bonds = []
        bond_idx = 0
        for rd_bond in mol.GetBonds():
            atom_1_idx = rd_bond.GetBeginAtomIdx()
            atom_2_idx = rd_bond.GetEndAtomIdx()
            if atom_1_idx >= len(atoms) or atom_2_idx >= len(atoms):
                raise ValueError(f"Indice d'atome invalide dans les liaisons: ({atom_1_idx}, {atom_2_idx})")
            bond = Bond(bond_idx, atom_1_idx, atom_2_idx, rd_bond)
            bonds.append(bond)
            atoms[atom_2_idx].add_bond(bond)
            bond_idx += 1
            bond = Bond(bond_idx, atom_2_idx, atom_1_idx, rd_bond)
            bonds.append(bond)
            atoms[atom_1_idx].add_bond(bond)
            bond_idx += 1
        
        return mol, atoms, bonds

    def _compute_shortest_paths(self, mol):
        n_atoms = mol.GetNumAtoms()
        G = nx.Graph()
        rd_bonds = {}
        for bond in mol.GetBonds():
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            rd_bonds[(i, j)] = bond
            rd_bonds[(j, i)] = bond
            G.add_edge(i, j)
        
        paths_dict = {}
        pointer_dict = {}
        ring_dict = {}
        ring_info = mol.GetRingInfo()
        
        for i in range(n_atoms):
            for j in range(i + 1, n_atoms):
                try:
                    path = nx.shortest_path(G, i, j)
                    paths_dict[(i, j)] = path
                    paths_dict[(j, i)] = path[::-1]
                    if len(path) - 1 > self.max_distance:
                        pointer_dict[(i, j)] = path[self.max_distance]
                        pointer_dict[(j, i)] = path[-self.max_distance-1]
                except nx.NetworkXNoPath:
                    pass
                
                rings = [(len(ring), mol.GetBondBetweenAtoms(ring[0], ring[1]).GetIsAromatic())
                         for ring in ring_info.AtomRings() if i in ring and j in ring]
                if rings:
                    ring_dict[ordered_pair(i, j)] = rings
        
        return paths_dict, pointer_dict, ring_dict

    def _get_path_atoms(self, atom_1, atom_2, paths_dict, pointer_dict, max_path_length, truncate=True, self_attn=False):
        path_start, path_end = atom_1, atom_2
        path_greater_max = False
        
        if (atom_1, atom_2) in pointer_dict:
            path_greater_max = True
            if not truncate:
                path_start, path_end = atom_1, pointer_dict[(atom_1, atom_2)]
        
        path_atoms = []
        if (path_start, path_end) in paths_dict:
            path_atoms = paths_dict[(path_start, path_end)]
        elif (path_end, path_start) in paths_dict:
            path_atoms = paths_dict[(path_end, path_start)][::-1]
        
        if len(path_atoms) - 1 > max_path_length:
            path_atoms = [] if truncate else path_atoms[:max_path_length+1]
            path_greater_max = True
        
        mask_ind = 1
        path_length = 0 if path_atoms == [] else len(path_atoms) - 1
        if path_greater_max:
            path_length = max_path_length + 1
            mask_ind = 0
        if not self_attn and atom_1 == atom_2:
            mask_ind = 0
        
        return path_atoms, path_length, mask_ind


# import torch
# import torch.nn as nn
# import torch.optim as optim
# import numpy as np
# import pandas as pd
# import os
# from datetime import datetime
# from torch.utils.data import DataLoader
# # On suppose que PAGTN_Dataset, ModelUtils, StatsTracker, get_num_path_features, et N_ATOM_FEATS sont définis ailleurs

# class MolTransformer(nn.Module):
#     def __init__(self, hidden_size, n_heads, d_k, depth, dropout, max_distance, p_embed, ring_embed, self_attn, no_share, mask_neigh):
#         super(MolTransformer, self).__init__()
#         self.hidden_size = hidden_size
#         self.n_heads = n_heads
#         self.d_k = d_k
#         self.depth = depth
#         self.dropout = dropout
#         self.max_distance = max_distance
#         self.p_embed = p_embed
#         self.ring_embed = ring_embed
#         self.self_attn = self_attn
#         self.no_share = no_share
#         self.mask_neigh = mask_neigh
#         self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#         n_atom_feats = N_ATOM_FEATS
#         n_path_feats = get_num_path_features(self.max_distance, self.p_embed, self.ring_embed)

#         self.W_atom_i = nn.Linear(n_atom_feats, self.n_heads * self.d_k, bias=False)
#         n_score_feats = 2 * self.d_k + n_path_feats
#         if self.no_share:
#             self.W_attn_h = nn.ModuleList([
#                 nn.Linear(n_score_feats, self.d_k) for _ in range(self.depth - 1)])
#             self.W_attn_o = nn.ModuleList([
#                 nn.Linear(self.d_k, 1) for _ in range(self.depth - 1)])
#             self.W_message_h = nn.ModuleList([
#                 nn.Linear(n_score_feats, self.d_k) for _ in range(self.depth - 1)])
#         else:
#             self.W_attn_h = nn.Linear(n_score_feats, self.d_k)
#             self.W_attn_o = nn.Linear(self.d_k, 1)
#             self.W_message_h = nn.Linear(n_score_feats, self.d_k)

#         self.W_atom_o = nn.Linear(n_atom_feats + self.n_heads * self.d_k, self.hidden_size)
#         self.dropout = nn.Dropout(self.dropout)
#         self.output_size = self.hidden_size

#     def get_attn_input(self, atom_h, path_input, max_atoms):
#         atom_h1 = atom_h.unsqueeze(2).expand(-1, -1, max_atoms, -1)
#         atom_h2 = atom_h.unsqueeze(1).expand(-1, max_atoms, -1, -1)
#         atom_pairs_h = torch.cat([atom_h1, atom_h2], dim=3)
#         attn_input = torch.cat([atom_pairs_h, path_input], dim=3)
#         return attn_input

#     def compute_attn_probs(self, attn_input, attn_mask, layer_idx, eps=1e-20):
#         if self.no_share:
#             attn_scores = nn.LeakyReLU(0.2)(self.W_attn_h[layer_idx](attn_input))
#             attn_scores = self.W_attn_o[layer_idx](attn_scores) * attn_mask
#         else:
#             attn_scores = nn.LeakyReLU(0.2)(self.W_attn_h(attn_input))
#             attn_scores = self.W_attn_o(attn_scores) * attn_mask

#         max_scores = torch.max(attn_scores, dim=2, keepdim=True)[0]
#         exp_attn = torch.exp(attn_scores - max_scores) * attn_mask
#         sum_exp = torch.sum(exp_attn, dim=2, keepdim=True) + eps
#         attn_probs = (exp_attn / sum_exp) * attn_mask
#         return attn_probs

#     def compute_nei_score(self, attn_probs, path_mask):
#         nei_probs = attn_probs * path_mask.unsqueeze(3)
#         nei_scores = torch.sum(nei_probs, dim=2)
#         avg_score = torch.sum(nei_scores) / torch.sum(nei_scores != 0).float()
#         return avg_score.item()

#     def avg_attn(self, attn_probs, n_heads, batch_sz, max_atoms):
#         if n_heads > 1:
#             attn_probs = attn_probs.view(n_heads, batch_sz, max_atoms, max_atoms)
#             attn_probs = torch.mean(attn_probs, dim=0)
#         return attn_probs

#     def forward(self, mol_graph, stats_tracker=None):
#         atom_input, scope = mol_graph.get_atom_inputs()
#         max_atoms = ModelUtils.compute_max_atoms(scope)
#         atom_input_3D, atom_mask = ModelUtils.convert_to_3D(
#             atom_input, scope, max_atoms, self.device, self.self_attn)
#         path_input, path_mask = mol_graph.path_input, mol_graph.path_mask

#         batch_sz, _, _ = atom_input_3D.size()
#         n_heads, d_k = self.n_heads, self.d_k

#         path_mask_3D = torch.zeros(batch_sz, max_atoms, max_atoms, device=self.device)
#         for mol_idx, (st, le) in enumerate(scope):
#             path_mask_3D[mol_idx, :le, :le] = path_mask[:le, :le]

#         if self.mask_neigh:
#             attn_mask = path_mask_3D
#         else:
#             attn_mask = atom_mask.float()
#         attn_mask = attn_mask.unsqueeze(3)

#         if n_heads > 1:
#             attn_mask = attn_mask.repeat(n_heads, 1, 1, 1)
#             path_input = path_input.repeat(n_heads, 1, 1, 1)
#             path_mask_3D = path_mask_3D.repeat(n_heads, 1, 1)

#         atom_input_h = self.W_atom_i(atom_input_3D).view(batch_sz, max_atoms, n_heads, d_k)
#         atom_input_h = atom_input_h.permute(2, 0, 1, 3).contiguous().view(-1, max_atoms, d_k)

#         attn_list, nei_scores = [], []
#         atom_h = atom_input_h
#         for layer_idx in range(self.depth - 1):
#             attn_input = self.get_attn_input(atom_h, path_input, max_atoms)
#             attn_probs = self.compute_attn_probs(attn_input, attn_mask, layer_idx)
#             attn_list.append(self.avg_attn(attn_probs, n_heads, batch_sz, max_atoms))
#             nei_scores.append(self.compute_nei_score(attn_probs, path_mask_3D))
#             attn_probs = self.dropout(attn_probs)

#             if self.no_share:
#                 attn_h = self.W_message_h[layer_idx](torch.sum(attn_probs * attn_input, dim=2))
#             else:
#                 attn_h = self.W_message_h(torch.sum(attn_probs * attn_input, dim=2))
#             atom_h = nn.ReLU()(attn_h + atom_input_h)

#         atom_h = atom_h.view(n_heads, batch_sz, max_atoms, -1)
#         atom_h = atom_h.permute(1, 2, 0, 3).contiguous().view(batch_sz, max_atoms, -1)
#         atom_h = ModelUtils.convert_to_2D(atom_h, scope)
#         atom_output = torch.cat([atom_input, atom_h], dim=1)
#         atom_h = nn.ReLU()(self.W_atom_o(atom_output))

#         nei_scores = np.array(nei_scores)
#         if stats_tracker is not None:
#             stats_tracker.add_stat('nei_score', np.mean(nei_scores), 1)

#         return atom_h, attn_list

# class PropPredictor(nn.Module):
#     def __init__(self, hidden_size, agg_func, n_classes=1):
#         super(PropPredictor, self).__init__()
#         self.hidden_size = hidden_size
#         self.agg_func = agg_func

#         # Les hyperparamètres de MolTransformer doivent être passés explicitement
#         self.model = MolTransformer(hidden_size, n_heads, d_k, depth, dropout, max_distance,
#                                    p_embed, ring_embed, self_attn, no_share, mask_neigh)
#         self.W_p_h = nn.Linear(self.model.output_size, self.hidden_size)
#         self.W_p_o = nn.Linear(self.hidden_size, n_classes)

#     def aggregate_atom_h(self, atom_h, scope):
#         mol_h = []
#         for (st, le) in scope:
#             cur_atom_h = atom_h.narrow(0, st, le)
#             if self.agg_func == 'mean':
#                 mol_h.append(cur_atom_h.mean(dim=0))
#             else:
#                 mol_h.append(cur_atom_h.sum(dim=0))
#         mol_h = torch.stack(mol_h, dim=0)
#         return mol_h

#     def forward(self, mol_graph, stats_tracker, output_attn=False):
#         atom_h, attn_list = self.model(mol_graph, stats_tracker)
#         scope = mol_graph.scope
#         mol_h = self.aggregate_atom_h(atom_h, scope)
#         mol_h = nn.ReLU()(self.W_p_h(mol_h))
#         mol_o = self.W_p_o(mol_h)

#         if not output_attn:
#             return mol_o
#         else:
#             return mol_o, attn_list

# def collate_fn(batch):
#     ids, mol_graphs, energies = zip(*batch)
#     return list(ids), list(mol_graphs), torch.stack(energies) if energies[0] is not None else None

# def train_epoch(model, dataloader, criterion, optimizer, device, stats_tracker):
#     model.train()
#     total_loss = 0
#     n_samples = 0

#     for ids, mol_graphs, energies in dataloader:
#         optimizer.zero_grad()
#         energies = energies.to(device)
#         predictions = []
#         for mol_graph in mol_graphs:
#             mol_graph.device = device
#             pred = model(mol_graph, stats_tracker)
#             predictions.append(pred)
#         predictions = torch.cat(predictions, dim=0)
#         loss = criterion(predictions, energies)
#         loss.backward()
#         optimizer.step()
#         total_loss += loss.item() * energies.size(0)
#         n_samples += energies.size(0)
    
#     return total_loss / n_samples

# def evaluate(model, dataloader, criterion, device, stats_tracker):
#     model.eval()
#     total_loss = 0
#     n_samples = 0
#     with torch.no_grad():
#         for ids, mol_graphs, energies in dataloader:
#             energies = energies.to(device)
#             predictions = []
#             for mol_graph in mol_graphs:
#                 mol_graph.device = device
#                 pred = model(mol_graph, stats_tracker)
#                 predictions.append(pred)
#             predictions = torch.cat(predictions, dim=0)
#             loss = criterion(predictions, energies)
#             total_loss += loss.item() * energies.size(0)
#             n_samples += energies.size(0)
    
#     return total_loss / n_samples

# if __name__ == '__main__':
#     # Définir les hyperparamètres ici
#     hidden_size = 128
#     n_heads = 4
#     d_k = 32
#     depth = 3
#     dropout = 0.1
#     max_distance = 5
#     p_embed = True
#     ring_embed = True
#     self_attn = False
#     no_share = False
#     mask_neigh = True
#     agg_func = 'mean'
#     n_classes = 1
#     batch_size = 32
#     epochs = 100
#     lr = 1e-3
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#     # Créer un dataset d'exemple (remplace par ton Data_List réel)
#     sample_data = [
#         {
#             'Id': 'benzene',
#             'Atoms_DataFrame': pd.DataFrame({
#                 'Symbol': ['C', 'C', 'C', 'C', 'C', 'C'],
#                 'x': [0.0, 1.4, 1.4, 0.0, -1.4, -1.4],
#                 'y': [0.0, 0.0, 1.2, 1.2, 1.2, 0.0],
#                 'z': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
#             }),
#             'Energy': -100.0,
#             'Bonds': [(0, 1, 1.5), (1, 2, 1.5), (2, 3, 1.5), (3, 4, 1.5), (4, 5, 1.5), (5, 0, 1.5)]
#         },
#         # Ajoute d'autres molécules ici
#     ]
#     dataset = PAGTN_Dataset(
#         sample_data,
#         max_distance=max_distance,
#         device=device,
#         p_embed=p_embed,
#         ring_embed=ring_embed,
#         self_attn=self_attn
#     )
#     dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

#     # Initialiser le modèle avec les hyperparamètres
#     model = PropPredictor(hidden_size=hidden_size, agg_func=agg_func, n_classes=n_classes).to(device)
#     criterion = nn.MSELoss()
#     optimizer = optim.Adam(model.parameters(), lr=lr)
#     scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

#     # Dossier pour sauvegarder les modèles
#     save_dir = f"models_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
#     os.makedirs(save_dir, exist_ok=True)

#     # Boucle d'entraînement
#     best_val_loss = float('inf')
#     for epoch in range(epochs):
#         train_loss = train_epoch(model, dataloader, criterion, optimizer, device, StatsTracker())
#         val_loss = evaluate(model, dataloader, criterion, device, StatsTracker())
#         scheduler.step(val_loss)

#         print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

#         if val_loss < best_val_loss:
#             best_val_loss = val_loss
#             torch.save(model.state_dict(), os.path.join(save_dir, 'best_model.pth'))
#             print(f"Saved best model with val loss {val_loss:.4f}")

#     print("Entraînement terminé !")