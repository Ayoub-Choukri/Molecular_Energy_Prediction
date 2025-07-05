
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import os
from datetime import datetime
from torch.utils.data import DataLoader
# On suppose que PAGTN_Dataset, ModelUtils, StatsTracker, get_num_path_features, et N_ATOM_FEATS sont définis ailleurs

import sys
MODULES_PATH = "Modules/PAGTN/"
sys.path.append(MODULES_PATH)
from Dataloaders_Preprocessing import PAGTN_Dataset, ModelUtils, get_num_path_features, N_ATOM_FEATS, StatsTracker


class MolTransformer(nn.Module):
    def __init__(self, hidden_size, n_heads, d_k, depth, dropout, max_distance, p_embed, ring_embed, self_attn, no_share, mask_neigh):
        super(MolTransformer, self).__init__()
        self.hidden_size = hidden_size
        self.n_heads = n_heads
        self.d_k = d_k
        self.depth = depth
        self.dropout = dropout
        self.max_distance = max_distance
        self.p_embed = p_embed
        self.ring_embed = ring_embed
        self.self_attn = self_attn
        self.no_share = no_share
        self.mask_neigh = mask_neigh
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        n_atom_feats = N_ATOM_FEATS
        n_path_feats = get_num_path_features(self.max_distance, self.p_embed, self.ring_embed)

        self.W_atom_i = nn.Linear(n_atom_feats, self.n_heads * self.d_k, bias=False)
        n_score_feats = 2 * self.d_k + n_path_feats
        if self.no_share:
            self.W_attn_h = nn.ModuleList([
                nn.Linear(n_score_feats, self.d_k) for _ in range(self.depth - 1)])
            self.W_attn_o = nn.ModuleList([
                nn.Linear(self.d_k, 1) for _ in range(self.depth - 1)])
            self.W_message_h = nn.ModuleList([
                nn.Linear(n_score_feats, self.d_k) for _ in range(self.depth - 1)])
        else:
            self.W_attn_h = nn.Linear(n_score_feats, self.d_k)
            self.W_attn_o = nn.Linear(self.d_k, 1)
            self.W_message_h = nn.Linear(n_score_feats, self.d_k)

        self.W_atom_o = nn.Linear(n_atom_feats + self.n_heads * self.d_k, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout)
        self.output_size = self.hidden_size

    def get_attn_input(self, atom_h, path_input, max_atoms):
        atom_h1 = atom_h.unsqueeze(2).expand(-1, -1, max_atoms, -1)
        atom_h2 = atom_h.unsqueeze(1).expand(-1, max_atoms, -1, -1)
        atom_pairs_h = torch.cat([atom_h1, atom_h2], dim=3)
        attn_input = torch.cat([atom_pairs_h, path_input], dim=3)
        return attn_input

    def compute_attn_probs(self, attn_input, attn_mask, layer_idx, eps=1e-20):
        if self.no_share:
            attn_scores = nn.LeakyReLU(0.2)(self.W_attn_h[layer_idx](attn_input))
            attn_scores = self.W_attn_o[layer_idx](attn_scores) * attn_mask
        else:
            attn_scores = nn.LeakyReLU(0.2)(self.W_attn_h(attn_input))
            attn_scores = self.W_attn_o(attn_scores) * attn_mask

        max_scores = torch.max(attn_scores, dim=2, keepdim=True)[0]
        exp_attn = torch.exp(attn_scores - max_scores) * attn_mask
        sum_exp = torch.sum(exp_attn, dim=2, keepdim=True) + eps
        attn_probs = (exp_attn / sum_exp) * attn_mask
        return attn_probs

    def compute_nei_score(self, attn_probs, path_mask):
        nei_probs = attn_probs * path_mask.unsqueeze(3)
        nei_scores = torch.sum(nei_probs, dim=2)
        avg_score = torch.sum(nei_scores) / torch.sum(nei_scores != 0).float()
        return avg_score.item()

    def avg_attn(self, attn_probs, n_heads, batch_sz, max_atoms):
        if n_heads > 1:
            attn_probs = attn_probs.view(n_heads, batch_sz, max_atoms, max_atoms)
            attn_probs = torch.mean(attn_probs, dim=0)
        return attn_probs

    def forward(self, mol_graph, stats_tracker=None):
        atom_input, scope = mol_graph.get_atom_inputs()
        max_atoms = ModelUtils.compute_max_atoms(scope)
        atom_input_3D, atom_mask = ModelUtils.convert_to_3D(
            atom_input, scope, max_atoms, self.device, self.self_attn)
        path_input, path_mask = mol_graph.path_input, mol_graph.path_mask

        batch_sz, _, _ = atom_input_3D.size()
        n_heads, d_k = self.n_heads, self.d_k

        path_mask_3D = torch.zeros(batch_sz, max_atoms, max_atoms, device=self.device)
        for mol_idx, (st, le) in enumerate(scope):
            path_mask_3D[mol_idx, :le, :le] = path_mask[:le, :le]

        if self.mask_neigh:
            attn_mask = path_mask_3D
        else:
            attn_mask = atom_mask.float()
        attn_mask = attn_mask.unsqueeze(3)

        if n_heads > 1:
            attn_mask = attn_mask.repeat(n_heads, 1, 1, 1)
            path_input = path_input.repeat(n_heads, 1, 1, 1)
            path_mask_3D = path_mask_3D.repeat(n_heads, 1, 1)

        atom_input_h = self.W_atom_i(atom_input_3D).view(batch_sz, max_atoms, n_heads, d_k)
        atom_input_h = atom_input_h.permute(2, 0, 1, 3).contiguous().view(-1, max_atoms, d_k)

        attn_list, nei_scores = [], []
        atom_h = atom_input_h
        for layer_idx in range(self.depth - 1):
            attn_input = self.get_attn_input(atom_h, path_input, max_atoms)
            attn_probs = self.compute_attn_probs(attn_input, attn_mask, layer_idx)
            attn_list.append(self.avg_attn(attn_probs, n_heads, batch_sz, max_atoms))
            nei_scores.append(self.compute_nei_score(attn_probs, path_mask_3D))
            attn_probs = self.dropout(attn_probs)

            if self.no_share:
                attn_h = self.W_message_h[layer_idx](torch.sum(attn_probs * attn_input, dim=2))
            else:
                attn_h = self.W_message_h(torch.sum(attn_probs * attn_input, dim=2))
            atom_h = nn.ReLU()(attn_h + atom_input_h)

        atom_h = atom_h.view(n_heads, batch_sz, max_atoms, -1)
        atom_h = atom_h.permute(1, 2, 0, 3).contiguous().view(batch_sz, max_atoms, -1)
        atom_h = ModelUtils.convert_to_2D(atom_h, scope)
        atom_output = torch.cat([atom_input, atom_h], dim=1)
        atom_h = nn.ReLU()(self.W_atom_o(atom_output))

        nei_scores = np.array(nei_scores)
        if stats_tracker is not None:
            stats_tracker.add_stat('nei_score', np.mean(nei_scores), 1)

        return atom_h, attn_list

class PropPredictor(nn.Module):
    def __init__(self, n_heads, d_k, depth, dropout, max_distance,
                                   p_embed, ring_embed, self_attn, no_share, mask_neigh,hidden_size, agg_func, n_classes=1):
        super(PropPredictor, self).__init__()
        self.hidden_size = hidden_size
        self.agg_func = agg_func

        # Les hyperparamètres de MolTransformer doivent être passés explicitement
        self.model = MolTransformer(hidden_size, n_heads, d_k, depth, dropout, max_distance,
                                   p_embed, ring_embed, self_attn, no_share, mask_neigh)
        self.W_p_h = nn.Linear(self.model.output_size, self.hidden_size)
        self.W_p_o = nn.Linear(self.hidden_size, n_classes)

    def aggregate_atom_h(self, atom_h, scope):
        mol_h = []
        for (st, le) in scope:
            cur_atom_h = atom_h.narrow(0, st, le)
            if self.agg_func == 'mean':
                mol_h.append(cur_atom_h.mean(dim=0))
            else:
                mol_h.append(cur_atom_h.sum(dim=0))
        mol_h = torch.stack(mol_h, dim=0)
        return mol_h

    def forward(self, mol_graph, stats_tracker, output_attn=False):
        atom_h, attn_list = self.model(mol_graph, stats_tracker)
        scope = mol_graph.scope
        mol_h = self.aggregate_atom_h(atom_h, scope)
        mol_h = nn.ReLU()(self.W_p_h(mol_h))
        mol_o = self.W_p_o(mol_h)

        if not output_attn:
            return mol_o
        else:
            return mol_o, attn_list

def collate_fn(batch):
    ids, mol_graphs, energies = zip(*batch)
    return list(ids), list(mol_graphs), torch.stack(energies) if energies[0] is not None else None

def train_epoch(model, dataloader, criterion, optimizer, device, stats_tracker):
    model.train()
    total_loss = 0
    n_samples = 0

    for ids, mol_graphs, energies in dataloader:
        optimizer.zero_grad()
        energies = energies.to(device)
        predictions = []
        for mol_graph in mol_graphs:
            mol_graph.device = device
            pred = model(mol_graph, stats_tracker)
            predictions.append(pred)
        predictions = torch.cat(predictions, dim=0)
        loss = criterion(predictions, energies)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * energies.size(0)
        n_samples += energies.size(0)
    
    return total_loss / n_samples

def evaluate(model, dataloader, criterion, device, stats_tracker):
    model.eval()
    total_loss = 0
    n_samples = 0
    with torch.no_grad():
        for ids, mol_graphs, energies in dataloader:
            energies = energies.to(device)
            predictions = []
            for mol_graph in mol_graphs:
                mol_graph.device = device
                pred = model(mol_graph, stats_tracker)
                predictions.append(pred)
            predictions = torch.cat(predictions, dim=0)
            loss = criterion(predictions, energies)
            total_loss += loss.item() * energies.size(0)
            n_samples += energies.size(0)
    
    return total_loss / n_samples

if __name__ == '__main__':
    # Définir les hyperparamètres ici
    hidden_size = 128
    n_heads = 4
    d_k = 32
    depth = 3
    dropout = 0.1
    max_distance = 5
    p_embed = True
    ring_embed = True
    self_attn = False
    no_share = False
    mask_neigh = True
    agg_func = 'mean'
    n_classes = 1
    batch_size = 32
    epochs = 100
    lr = 1e-3
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Créer un dataset d'exemple (remplace par ton Data_List réel)
    sample_data = [
        {
            'Id': 'benzene',
            'Atoms_DataFrame': pd.DataFrame({
                'Symbol': ['C', 'C', 'C', 'C', 'C', 'C'],
                'x': [0.0, 1.4, 1.4, 0.0, -1.4, -1.4],
                'y': [0.0, 0.0, 1.2, 1.2, 1.2, 0.0],
                'z': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            }),
            'Energy': -100.0,
            'Bonds': [(0, 1, 1.5), (1, 2, 1.5), (2, 3, 1.5), (3, 4, 1.5), (4, 5, 1.5), (5, 0, 1.5)]
        },
        # Ajoute d'autres molécules ici
    ]
    dataset = PAGTN_Dataset(
        sample_data,
        max_distance=max_distance,
        device=device,
        p_embed=p_embed,
        ring_embed=ring_embed,
        self_attn=self_attn
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    # Initialiser le modèle avec les hyperparamètres
    model = PropPredictor(hidden_size=hidden_size, agg_func=agg_func, n_classes=n_classes).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

    # Dossier pour sauvegarder les modèles
    save_dir = f"models_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(save_dir, exist_ok=True)

    # Boucle d'entraînement
    best_val_loss = float('inf')
    for epoch in range(epochs):
        train_loss = train_epoch(model, dataloader, criterion, optimizer, device, StatsTracker())
        val_loss = evaluate(model, dataloader, criterion, device, StatsTracker())
        scheduler.step(val_loss)

        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(save_dir, 'best_model.pth'))
            print(f"Saved best model with val loss {val_loss:.4f}")

    print("Entraînement terminé !")