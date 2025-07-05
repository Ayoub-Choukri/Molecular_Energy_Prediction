import torch
import torch.nn as nn

import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, Input_Dim, Nb_Hidden_Layers, Hidden_Layers_Size_List, Output_Dim, Activation_Name):
        super().__init__()
        assert len(Hidden_Layers_Size_List) == Nb_Hidden_Layers, \
            f"Number of hidden layers ({Nb_Hidden_Layers}) must match length of Hidden_Layers_Size_List ({len(Hidden_Layers_Size_List)})"
        
        layers = []
        current_dim = Input_Dim
        for i in range(Nb_Hidden_Layers):
            layers.append(nn.Linear(current_dim, Hidden_Layers_Size_List[i]))
            layers.append(self.get_activation_function(Activation_Name))
            current_dim = Hidden_Layers_Size_List[i]
        layers.append(nn.Linear(current_dim, Output_Dim))
        self.model = nn.Sequential(*layers)

    def get_activation_function(self, activation_name):
        if activation_name == 'ReLU':
            return nn.ReLU()
        elif activation_name == 'Sigmoid':
            return nn.Sigmoid()
        elif activation_name == 'Tanh':
            return nn.Tanh()
        elif activation_name == 'LeakyReLU':
            return nn.LeakyReLU()
        elif activation_name == 'PReLU':
            return nn.PReLU()
        elif activation_name == 'ELU':
            return nn.ELU()
        elif activation_name == 'Softmax':
            return nn.Softmax(dim=1)
        elif activation_name == 'GELU':
            return nn.GELU()
        else:
            raise ValueError(f"Activation function '{activation_name}' is not supported.")
        
    def forward(self, x):
        return self.model(x)
import torch
import torch.nn as nn
import torch
import torch.nn as nn

import torch
import torch.nn as nn

class Multi_Head_Attention(nn.Module):
    def __init__(self, Nb_Heads, Embeddings_Size, Query_Size, Key_Size, Value_Size,
                 Nb_Hidden_Layers_MLP_Attention_WQ_WK_WV, Hidden_Layers_Size_List_MLP_Attention_WQ_WK_WV, 
                 Activation_Name_MLP_Attention_WQ_WK_WV, Nb_Hidden_Layers_MLP_Attention_WO,
                 Hidden_Layers_Size_List_MLP_Attention_WO, Activation_Name_MLP_Attention_WO):
        super().__init__()
        self.Nb_Heads = Nb_Heads
        self.Embeddings_Size = Embeddings_Size
        self.Query_Size = Query_Size
        self.Key_Size = Key_Size
        self.Value_Size = Value_Size
        self.Nb_Hidden_Layers_MLP_Attention_WQ_WK_WV = Nb_Hidden_Layers_MLP_Attention_WQ_WK_WV
        self.Hidden_Layers_Size_List_MLP_Attention_WQ_WK_WV = Hidden_Layers_Size_List_MLP_Attention_WQ_WK_WV
        self.Activation_Name_MLP_Attention_WQ_WK_WV = Activation_Name_MLP_Attention_WQ_WK_WV
        self.Nb_Hidden_Layers_MLP_Attention_WO = Nb_Hidden_Layers_MLP_Attention_WO
        self.Hidden_Layers_Size_List_MLP_Attention_WO = Hidden_Layers_Size_List_MLP_Attention_WO
        self.Activation_Name_MLP_Attention_WO = Activation_Name_MLP_Attention_WO

        # Create MLPs for each head for WQ, WK, WV
        self.WQ = nn.ModuleList([
            MLP(
                Input_Dim=Embeddings_Size,
                Nb_Hidden_Layers=Nb_Hidden_Layers_MLP_Attention_WQ_WK_WV,
                Hidden_Layers_Size_List=Hidden_Layers_Size_List_MLP_Attention_WQ_WK_WV,
                Output_Dim=Query_Size,
                Activation_Name=Activation_Name_MLP_Attention_WQ_WK_WV
            ) for _ in range(Nb_Heads)
        ])
        self.WK = nn.ModuleList([
            MLP(
                Input_Dim=Embeddings_Size,
                Nb_Hidden_Layers=Nb_Hidden_Layers_MLP_Attention_WQ_WK_WV,
                Hidden_Layers_Size_List=Hidden_Layers_Size_List_MLP_Attention_WQ_WK_WV,
                Output_Dim=Key_Size,
                Activation_Name=Activation_Name_MLP_Attention_WQ_WK_WV
            ) for _ in range(Nb_Heads)
        ])
        self.WV = nn.ModuleList([
            MLP(
                Input_Dim=Embeddings_Size,
                Nb_Hidden_Layers=Nb_Hidden_Layers_MLP_Attention_WQ_WK_WV,
                Hidden_Layers_Size_List=Hidden_Layers_Size_List_MLP_Attention_WQ_WK_WV,
                Output_Dim=Value_Size,
                Activation_Name=Activation_Name_MLP_Attention_WQ_WK_WV
            ) for _ in range(Nb_Heads)
        ])
        # Create MLPs for WO per head, outputting full Embeddings_Size
        self.WO = nn.ModuleList([
            MLP(
                Input_Dim=Value_Size,
                Nb_Hidden_Layers=Nb_Hidden_Layers_MLP_Attention_WO,
                Hidden_Layers_Size_List=Hidden_Layers_Size_List_MLP_Attention_WO,
                Output_Dim=Embeddings_Size,  # Each WO outputs Embeddings_Size
                Activation_Name=Activation_Name_MLP_Attention_WO
            ) for _ in range(Nb_Heads)
        ])
        # Projection layer to map concatenated heads back to Embeddings_Size
        self.projection = nn.Linear(Nb_Heads * Embeddings_Size, Embeddings_Size)

    def forward(self, x, mask=None,return_cls_token=False):
        batch_size, nb_atomes, _ = x.size()
        
        # Reshape for MLP: (batch_size, nb_atomes, Embeddings_Size) -> (batch_size * nb_atomes, Embeddings_Size)
        x_flat = x.reshape(-1, self.Embeddings_Size)
        
        # Calculate Q, K, V for each head
        Q_list, K_list, V_list = [], [], []
        for h in range(self.Nb_Heads):
            Q_h = self.WQ[h](x_flat).reshape(batch_size, nb_atomes, self.Query_Size)
            K_h = self.WK[h](x_flat).reshape(batch_size, nb_atomes, self.Key_Size)
            V_h = self.WV[h](x_flat).reshape(batch_size, nb_atomes, self.Value_Size)
            Q_list.append(Q_h.unsqueeze(1))
            K_list.append(K_h.unsqueeze(1))
            V_list.append(V_h.unsqueeze(1))

        # Concatenate Q, K, V for all heads
        Q = torch.cat(Q_list, dim=1)  # (batch_size, Nb_Heads, nb_atomes, Query_Size)
        K = torch.cat(K_list, dim=1)  # (batch_size, Nb_Heads, nb_atomes, Key_Size)
        V = torch.cat(V_list, dim=1)  # (batch_size, Nb_Heads, nb_atomes, Value_Size)

        # Calculate attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.Key_Size ** 0.5)
        attn_weights = torch.softmax(scores, dim=-1)
        
        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(1).unsqueeze(2)
            mask = mask.expand(-1, self.Nb_Heads, nb_atomes, -1)
            attn_weights = attn_weights * mask
            attn_weights = attn_weights / (attn_weights.sum(dim=-1, keepdim=True) + 1e-9)
        
        # Calculate attention output
        output = torch.matmul(attn_weights, V)  # (batch_size, Nb_Heads, nb_atomes, Value_Size)
        
        # Apply WO for each head
        output_list = []
        for h in range(self.Nb_Heads):
            output_h = output[:, h, :, :].reshape(-1, self.Value_Size)
            output_h = self.WO[h](output_h).reshape(batch_size, nb_atomes, self.Embeddings_Size)
            output_list.append(output_h)  # (batch_size, nb_atomes, Embeddings_Size)

        # Concatenate outputs from all heads along a new dimension
        output = torch.stack(output_list, dim=1)  # (batch_size, Nb_Heads, nb_atomes, Embeddings_Size)
        
        # Reshape for projection: (batch_size, nb_atomes, Nb_Heads * Embeddings_Size)
        output = output.reshape(batch_size, nb_atomes, self.Nb_Heads * self.Embeddings_Size)
        
        # Apply projection layer
        output = self.projection(output)  # (batch_size, nb_atomes, Embeddings_Size)

        if return_cls_token:
            return output, self.cls_token.expand(batch_size, -1, -1)
        
        return output
    

import torch
import torch.nn as nn
import torch
import torch.nn as nn
import torch
import torch.nn as nn

class Transformer(nn.Module):
    def __init__(self, Embeddings_Size, Nb_Heads, Query_Size, Key_Size, Value_Size,
                 Nb_Hidden_Layers_MLP_Output, Hidden_Layers_Size_List_MLP_Output, Activation_Name_Output,
                 Output_Size,
                 Nb_Hidden_Layers_MLP_Attention_WQ_WK_WV, Hidden_Layers_Size_List_MLP_Attention_WQ_WK_WV,
                 Activation_Name_MLP_Attention_WQ_WK_WV, Nb_Hidden_Layers_MLP_Attention_WO,
                 Hidden_Layers_Size_List_MLP_Attention_WO, Activation_Name_MLP_Attention_WO,
                 Nb_Attention_Blocks):
        super().__init__()  # Syntaxe simplifiée pour Python 3
        self.Embeddings_Size = Embeddings_Size
        self.Nb_Heads = Nb_Heads
        self.Query_Size = Query_Size
        self.Key_Size = Key_Size
        self.Value_Size = Value_Size
        self.Nb_Hidden_Layers_MLP_Output = Nb_Hidden_Layers_MLP_Output
        self.Hidden_Layers_Size_List_MLP_Output = Hidden_Layers_Size_List_MLP_Output
        self.Activation_Name_Output = Activation_Name_Output
        self.Nb_Hidden_Layers_MLP_Attention_WQ_WK_WV = Nb_Hidden_Layers_MLP_Attention_WQ_WK_WV
        self.Hidden_Layers_Size_List_MLP_Attention_WQ_WK_WV = Hidden_Layers_Size_List_MLP_Attention_WQ_WK_WV
        self.Activation_Name_MLP_Attention_WQ_WK_WV = Activation_Name_MLP_Attention_WQ_WK_WV
        self.Nb_Hidden_Layers_MLP_Attention_WO = Nb_Hidden_Layers_MLP_Attention_WO
        self.Hidden_Layers_Size_List_MLP_Attention_WO = Hidden_Layers_Size_List_MLP_Attention_WO
        self.Activation_Name_MLP_Attention_WO = Activation_Name_MLP_Attention_WO
        self.Output_Size = Output_Size
        self.Nb_Attention_Blocks = Nb_Attention_Blocks

        # Embedding pour les types d'atomes (6 types : H, C, N, O, S, Cl)
        self.atom_embedding = nn.Embedding(num_embeddings=6, embedding_dim=Embeddings_Size // 2)
        # Projection linéaire pour les positions (3D -> Embeddings_Size // 2)
        self.position_projection = nn.Linear(3, Embeddings_Size // 2)
        # CLS token apprenable (1, Embeddings_Size)
        self.cls_token = nn.Parameter(torch.randn(1, 1, Embeddings_Size))

        # Créer une liste de blocs d'attention
        self.attention_blocks = nn.ModuleList([
            nn.ModuleDict({
                'attention': Multi_Head_Attention(
                    Nb_Heads=Nb_Heads,
                    Embeddings_Size=Embeddings_Size,
                    Query_Size=Query_Size,
                    Key_Size=Key_Size,
                    Value_Size=Value_Size,
                    Nb_Hidden_Layers_MLP_Attention_WQ_WK_WV=Nb_Hidden_Layers_MLP_Attention_WQ_WK_WV,
                    Hidden_Layers_Size_List_MLP_Attention_WQ_WK_WV=Hidden_Layers_Size_List_MLP_Attention_WQ_WK_WV,
                    Activation_Name_MLP_Attention_WQ_WK_WV=Activation_Name_MLP_Attention_WQ_WK_WV,
                    Nb_Hidden_Layers_MLP_Attention_WO=Nb_Hidden_Layers_MLP_Attention_WO,
                    Hidden_Layers_Size_List_MLP_Attention_WO=Hidden_Layers_Size_List_MLP_Attention_WO,
                    Activation_Name_MLP_Attention_WO=Activation_Name_MLP_Attention_WO
                ),
                'norm': nn.LayerNorm(Embeddings_Size)
            }) for _ in range(Nb_Attention_Blocks)
        ])

        self.MLP_Output = MLP(
            Input_Dim=Embeddings_Size,
            Nb_Hidden_Layers=Nb_Hidden_Layers_MLP_Output,
            Hidden_Layers_Size_List=Hidden_Layers_Size_List_MLP_Output,
            Output_Dim=Output_Size,
            Activation_Name=Activation_Name_Output
        )

    def __Add_CLS_Token(self, x):
        batch_size = x.size(0)
        # Répéter le CLS token pour chaque échantillon dans le batch
        cls_token = self.cls_token.expand(batch_size, -1, -1)  # (batch_size, 1, Embeddings_Size)
        return torch.cat((cls_token, x), dim=1)

    def __get_activation__(self, activation_name):
        if activation_name == "GELU":
            return nn.GELU()
        elif activation_name == "ReLU":
            return nn.ReLU()
        elif activation_name == "LeakyReLU":
            return nn.LeakyReLU()
        elif activation_name == "Sigmoid":
            return nn.Sigmoid()
        elif activation_name == "Tanh":
            return nn.Tanh()
        elif activation_name == "Softmax":
            return nn.Softmax(dim=1)
        elif activation_name == 'PReLU' or activation_name == 'prelu':
            return nn.PReLU()
        elif activation_name == 'SELU' or activation_name == 'selu':
            return nn.SELU()
        elif activation_name == 'CELU' or activation_name == 'celu':
            return nn.CELU()
        elif activation_name == 'GLU' or activation_name == 'glu':
            return nn.GLU()
        else:
            assert False, "Activation function not recognized"

    def forward(self, symbols, positions, mask=None, return_cls_token=False):
        batch_size, nb_atomes, _ = positions.size()

        # Créer les embeddings
        atom_emb = self.atom_embedding(symbols)  # (batch_size, nb_atomes, Embeddings_Size // 2)
        pos_emb = self.position_projection(positions)  # (batch_size, nb_atomes, Embeddings_Size // 2)
        embeddings = torch.cat([atom_emb, pos_emb], dim=-1)  # (batch_size, nb_atomes, Embeddings_Size)

        # Ajouter le token CLS
        embeddings = self.__Add_CLS_Token(embeddings)  # (batch_size, nb_atomes + 1, Embeddings_Size)

        # Ajuster le masque pour inclure CLS
        if mask is not None:
            cls_mask = torch.ones(batch_size, 1, device=symbols.device)
            mask = torch.cat((cls_mask, mask), dim=1)  # (batch_size, nb_atomes + 1)

        # Appliquer les blocs d'attention séquentiellement
        for block in self.attention_blocks:
            corrections = block['attention'](embeddings, mask=mask)
            embeddings = embeddings + corrections  # Connexion résiduelle
            embeddings = block['norm'](embeddings)  # Normalisation

        # Extraire le token CLS
        cls_token = embeddings[:, 0, :]  # (batch_size, Embeddings_Size)

        # Appliquer le MLP uniquement au token CLS
        output = self.MLP_Output(cls_token)  # (batch_size, Output_Size)

        if return_cls_token:
            return output, cls_token
        return output

# Exemple de test pour la classe Transformer
if __name__ == "__main__":
    # Paramètres du modèle
    batch_size = 32
    nb_atomes = 23
    embeddings_size = 40  # Taille des embeddings (doit être divisible par 2)
    nb_heads = 4
    query_size = key_size = value_size = 20
    nb_hidden_layers_mlp_output = 2
    hidden_layers_size_list_mlp_output = [64, 32]
    output_size = 1  # Pour prédire l'énergie
    activation_name = 'GELU'
    nb_hidden_layers_mlp_attention = 2
    hidden_layers_size_list_mlp_attention = [64, 32]
    activation_name_mlp_attention = 'GELU'
    Nb_Attention_Blocks = 3

    # Créer le modèle Transformer
    model = Transformer(
        Embeddings_Size=embeddings_size,
        Nb_Heads=nb_heads,
        Query_Size=query_size,
        Key_Size=key_size,
        Value_Size=value_size,
        Nb_Hidden_Layers_MLP_Output=nb_hidden_layers_mlp_output,
        Hidden_Layers_Size_List_MLP_Output=hidden_layers_size_list_mlp_output,
        Output_Size=output_size,
        Nb_Hidden_Layers_MLP_Attention=nb_hidden_layers_mlp_attention,
        Hidden_Layers_Size_List_MLP_Attention=hidden_layers_size_list_mlp_attention,
        Activation_Name=activation_name,
        Activation_Name_MLP_Attention=activation_name_mlp_attention,
        Nb_Attention_Blocks=Nb_Attention_Blocks
    )

    # Générer des entrées fictives
    symbols = torch.randint(0, 6, (batch_size, nb_atomes))  # Indices aléatoires pour H, C, N, O, S, Cl
    positions = torch.randn(batch_size, nb_atomes, 3)  # Positions 3D aléatoires

    # Créer un masque exemple
    mask = torch.ones(batch_size, nb_atomes)
    mask[:, 20:] = 0  # Les 3 derniers atomes sont du padding
    mask[0, 19:] = 0  # Les 4 derniers atomes du premier échantillon sont du padding

    # Mettre le modèle en mode évaluation
    model.eval()

    # Propagation avant
    with torch.no_grad():
        output = model(symbols, positions, mask=mask)

    # Vérifier la forme de la sortie
    print(f"Shape de la sortie : {output.shape}")  # Devrait être (32, 1)

    # Vérifier que la sortie est cohérente
    assert output.shape == (batch_size, output_size), \
        f"Forme de sortie inattendue : {output.shape}, attendu : {(batch_size, output_size)}"

    # Test supplémentaire : Vérifier que le padding n'a pas d'impact
    mask_all_padded = torch.zeros(batch_size, nb_atomes)
    output_padded = model(symbols, positions, mask=mask_all_padded)

    print(f"Shape de la sortie avec tout masqué sauf CLS : {output_padded.shape}")  # Devrait être (32, 1)

    # Vérifier que les sorties sont différentes
    print(f"Différence moyenne entre les sorties : {torch.mean(torch.abs(output - output_padded))}")

    print("Test du Transformer réussi !")