import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Uni_Model(nn.Module):
    def __init__(self, num_layers=15, embed_dim=512, num_heads=64, ffn_embed_dim=2048,
                 gaussian_channels=128, vocab_size=7, pair_vocab_size=49, max_atoms=256, dropout=0.1):
        super(Uni_Model, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.gaussian_channels = gaussian_channels
        self.max_atoms = max_atoms
        self.head_dim = embed_dim // num_heads
        
        self.atom_embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.gaussian_mu = nn.Parameter(torch.randn(gaussian_channels))
        self.gaussian_sigma = nn.Parameter(torch.ones(gaussian_channels) * 0.1)
        self.pair_type_affine_a = nn.Parameter(torch.randn(pair_vocab_size) * 0.01)
        self.pair_type_affine_b = nn.Parameter(torch.randn(pair_vocab_size) * 0.01)
        
        self.pair_linear = nn.Linear(gaussian_channels, 1)
        self.layers = nn.ModuleList([
            UniMolLayer(embed_dim, num_heads, ffn_embed_dim, dropout)
            for _ in range(num_layers)
        ])
        self.se3_head = SE3EquivariantHead(embed_dim, num_heads)
        self.energy_head = nn.Linear(embed_dim, 1)
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
    
    def check_nan(self, tensor, name):
        if torch.isnan(tensor).any() or torch.isinf(tensor).any():
            print(f"NaN or Inf detected in {name}")
            return True
        return False
    
    def forward(self, atom_types, coords, pair_types, mask=None):
        batch_size, num_atoms = atom_types.shape
        atom_embed = self.atom_embedding(atom_types)
        self.check_nan(atom_embed, "atom_embed after embedding")
        atom_embed = self.dropout(atom_embed)
        
        pair_embed = self.compute_spatial_encoding(coords, pair_types, mask)
        self.check_nan(pair_embed, "pair_embed")
        pair_repr = self.pair_linear(pair_embed).squeeze(-1)
        self.check_nan(pair_repr, "pair_repr after pair_linear")
        
        if mask is not None:
            pair_mask = mask.unsqueeze(2) * mask.unsqueeze(1)
            pair_repr = pair_repr * pair_mask
        self.check_nan(pair_repr, "pair_repr after masking")
        
        
        for i, layer in enumerate(self.layers):
            atom_embed, pair_repr = layer(atom_embed, pair_repr, mask)
            self.check_nan(atom_embed, f"atom_embed after layer {i}")
            self.check_nan(pair_repr, f"pair_repr after layer {i}")
        
        pred_coords = self.se3_head(atom_embed, pair_repr, coords, mask)
        self.check_nan(pred_coords, "pred_coords")
        
        cls_embed = atom_embed[:, 0, :]
        self.check_nan(cls_embed, "cls_embed")
        pred_energy = self.energy_head(cls_embed).squeeze(-1)
        self.check_nan(pred_energy, "pred_energy")
        
        return pred_coords, pred_energy

    def compute_spatial_encoding(self, coords, pair_types, mask):
        batch_size, num_atoms, _ = coords.shape
        dist = torch.norm(coords.unsqueeze(2) - coords.unsqueeze(1), dim=-1)
        dist = dist.masked_fill(~mask.unsqueeze(2), 0.0)
        self.check_nan(dist, "dist")
        
        a = self.pair_type_affine_a[pair_types]
        b = self.pair_type_affine_b[pair_types]
        dist_affine = a * dist + b
        self.check_nan(dist_affine, "dist_affine")
        
        dist_affine = dist_affine.unsqueeze(-1)
        sigma = torch.clamp(self.gaussian_sigma, min=1e-6)
        exponent = -((dist_affine - self.gaussian_mu) ** 2) / (2 * sigma ** 2)
        exponent = torch.clamp(exponent, min=-100, max=0)
        gaussian_term = torch.exp(exponent)
        gaussian_term /= (sigma * math.sqrt(2 * math.pi))
        self.check_nan(gaussian_term, "gaussian_term")
        
        gaussian_term = gaussian_term * mask.unsqueeze(2).unsqueeze(-1)
        return gaussian_term
    

    def compute_spatial_encoding(self, coords, pair_types, mask):
        batch_size, num_atoms, _ = coords.shape
        # Calcul des distances, ignorer les paires paddées
        dist = torch.norm(coords.unsqueeze(2) - coords.unsqueeze(1), dim=-1)
        dist = dist.masked_fill(~mask.unsqueeze(2), 0.0)
        
        a = self.pair_type_affine_a[pair_types]
        b = self.pair_type_affine_b[pair_types]
        dist_affine = a * dist + b
        
        dist_affine = dist_affine.unsqueeze(-1)
        gaussian_term = torch.exp(-((dist_affine - self.gaussian_mu) ** 2)) / (2 * self.gaussian_sigma ** 2)
        gaussian_term /= (self.gaussian_sigma * math.sqrt(2 * math.pi))
        
        # Masquer les termes gaussiens pour les paires paddées
        gaussian_term = gaussian_term * mask.unsqueeze(2).unsqueeze(-1)
        return gaussian_term

class UniMolLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, ffn_dim, dropout):
        super(UniMolLayer, self).__init__()
        self.self_attention = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, embed_dim),
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, atom_embed, pair_repr, mask):
        atom_embed_residual = atom_embed
        atom_embed = self.norm1(atom_embed)
        atom_embed = self.self_attention(atom_embed, pair_repr, mask)
        atom_embed = self.dropout(atom_embed)
        atom_embed += atom_embed_residual
        
        atom_embed_residual = atom_embed
        atom_embed = self.norm2(atom_embed)
        atom_embed = self.ffn(atom_embed)
        atom_embed = self.dropout(atom_embed)
        atom_embed += atom_embed_residual
        
        pair_repr_updated = pair_repr  # Simplifié
        return atom_embed, pair_repr_updated
    
class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout):
        super(MultiHeadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.out = nn.Linear(embed_dim, embed_dim)
        self.pair_proj = nn.Linear(1, num_heads)
        self.dropout = nn.Dropout(dropout)
    
    def check_nan(self, tensor, name):
        if torch.isnan(tensor).any() or torch.isinf(tensor).any():
            print(f"NaN or Inf detected in {name}")
            return True
        return False
    
    def forward(self, x, pair_repr, mask):
        batch_size, num_atoms, _ = x.shape
        qkv = self.qkv(x).view(batch_size, num_atoms, 3, self.num_heads, self.head_dim)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
        self.check_nan(q, "q")
        self.check_nan(k, "k")
        self.check_nan(v, "v")
        
        attn_scores = torch.einsum("bnhd,bmhd->bhnm", q, k) / math.sqrt(self.head_dim)
        self.check_nan(attn_scores, "attn_scores before pair_repr")
        
        pair_repr_projected = self.pair_proj(pair_repr.unsqueeze(-1)).permute(0, 3, 1, 2)
        self.check_nan(pair_repr_projected, "pair_repr_projected")
        attn_scores += pair_repr_projected
        self.check_nan(attn_scores, "attn_scores after pair_repr")
        
        # print(f"attn_scores shape: {attn_scores.shape}")
        # print(f"pair_repr shape in MultiHeadAttention: {pair_repr.shape}")
        
        if mask is not None:
            attn_scores = attn_scores.masked_fill(~mask.unsqueeze(1).unsqueeze(-1), float('-inf'))
            attn_scores = torch.where(torch.isinf(attn_scores), torch.full_like(attn_scores, -1e9), attn_scores)
        self.check_nan(attn_scores, "attn_scores after masking")
        
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = torch.where(torch.isnan(attn_probs), torch.zeros_like(attn_probs), attn_probs)
        self.check_nan(attn_probs, "attn_probs")
        
        out = torch.einsum("bhnm,bmhd->bnhd", attn_probs, v)
        self.check_nan(out, "out before view")
        out = out.contiguous().view(batch_size, num_atoms, self.embed_dim)
        out = self.out(out)
        self.check_nan(out, "out final")
        
        return out
    
class SE3EquivariantHead(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(SE3EquivariantHead, self).__init__()
        self.proj_U = nn.Linear(1, 1)  # Simplifié pour pair_repr scalaire
        self.proj_W = nn.Linear(1, 1)
        
    def forward(self, atom_embed, pair_repr, coords, mask):
        batch_size, num_atoms, _ = coords.shape
        c_ij = F.relu(pair_repr.unsqueeze(-1))  # [batch_size, num_atoms, num_atoms, 1]
        c_ij = self.proj_U(c_ij)  # [batch_size, num_atoms, num_atoms, 1]
        c_ij = self.proj_W(c_ij)  # [batch_size, num_atoms, num_atoms, 1]
        
        # Masquer les contributions des atomes paddés
        if mask is not None:
            c_ij = c_ij * mask.unsqueeze(2).unsqueeze(-1)  # [batch_size, num_atoms, 1, 1]
        
        delta_coords = coords.unsqueeze(2) - coords.unsqueeze(1)  # [batch_size, num_atoms, num_atoms, 3]
        update = (delta_coords * c_ij).sum(dim=2)  # [batch_size, num_atoms, 3]
        
        # Normalize by number of valid atoms, ensure broadcasting
        if mask is not None:
            num_valid = mask.sum(dim=-1, keepdim=True).unsqueeze(-1) + 1e-6  # [batch_size, 1, 1]
            update = update / num_valid  # Broadcasting: [2, 10, 3] / [2, 1, 1]
        
        pred_coords = coords + update
        
        return pred_coords
    
if __name__ == "__main__":
    batch_size = 2
    max_atoms = 10
    vocab_size = 7
    pair_vocab_size = 49

    model = Uni_Model(
        num_layers=15,
        embed_dim=512,
        num_heads=64,
        ffn_embed_dim=2048,
        gaussian_channels=128,
        vocab_size=vocab_size,
        pair_vocab_size=pair_vocab_size,
        max_atoms=max_atoms,
        dropout=0.1
    ).cuda()

    atom_types = torch.randint(0, vocab_size, (batch_size, max_atoms)).cuda()
    atom_types[:, 0] = vocab_size - 1  # Set [CLS] token
    coords = torch.randn(batch_size, max_atoms, 3).cuda()
    coords = coords / (coords.norm(dim=-1, keepdim=True) + 1e-6) * 10.0
    pair_types = torch.randint(0, pair_vocab_size, (batch_size, max_atoms, max_atoms)).cuda()
    mask = torch.ones(batch_size, max_atoms, dtype=torch.bool).cuda()
    
    mask[:, -1] = False
    atom_types[:, -1] = 0
    coords[:, -1] = 0.0
    pair_types[:, -1, :] = 0
    pair_types[:, :, -1] = 0

    model.eval()
    with torch.no_grad():
        pred_coords, pred_energy = model(atom_types, coords, pair_types, mask)
    
    print(f"Predicted Coordinates Shape: {pred_coords.shape}")
    print(f"Predicted Energy Shape: {pred_energy.shape}")
    print(f"Sample Predicted Energy: {pred_energy[0].item()}")