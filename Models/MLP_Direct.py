
import torch
from torch import nn




class MLP(nn.Module):
    def __init__(self, Input_Dim,Nb_Hidden_Layers, Hidden_Layers_Size_List, Output_Dim, Activation_Name):
        super(MLP, self).__init__()
        
        self.Input_Dim = Input_Dim
        self.Nb_Hidden_Layers = Nb_Hidden_Layers
        self.Hidden_Layers_Size_List = Hidden_Layers_Size_List
        self.Output_Dim = Output_Dim
        self.Activation_Name = Activation_Name
        
        # Define the layers of the MLP
        layers = []
        
        # Input layer
        layers.append(nn.Linear(Input_Dim, Hidden_Layers_Size_List[0]))
        
        # Hidden layers
        for i in range(Nb_Hidden_Layers - 1):
            layers.append(self.get_activation_function(Activation_Name))
            layers.append(nn.Linear(Hidden_Layers_Size_List[i], Hidden_Layers_Size_List[i + 1]))
        
        # Output layer
        layers.append(self.get_activation_function(Activation_Name))
        layers.append(nn.Linear(Hidden_Layers_Size_List[-1], Output_Dim))
        
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
        """
        Forward pass through the MLP.
        
        :param x: Input tensor of shape (batch_size, Input_Dim).
        :return: Output tensor of shape (batch_size, Output_Dim).
        """
        return self.model(x)
    




import torch
import torch.nn as nn

class Model_MLP_Direct(nn.Module):
    def __init__(self, Atom_Vocab_Size, Embedding_Size, Nb_Hidden_Layers, Hidden_Layers_Size_List,
                 Output_Dim, Activation_Name, Nb_Atomes_Max_In_Molecule=23):
        super(Model_MLP_Direct, self).__init__()
        self.Nb_Atomes_Max_In_Molecule = Nb_Atomes_Max_In_Molecule
        self.Embedding_Size = Embedding_Size

        self.Embedding = nn.Embedding(Atom_Vocab_Size, Embedding_Size)

        self.True_Input_Dim = (1 + Embedding_Size + 3) * Nb_Atomes_Max_In_Molecule

        self.MLP_Model = MLP(Input_Dim=self.True_Input_Dim,
                             Nb_Hidden_Layers=Nb_Hidden_Layers,
                             Hidden_Layers_Size_List=Hidden_Layers_Size_List,
                             Output_Dim=Output_Dim,
                             Activation_Name=Activation_Name)

    def forward(self, x):
        """
        :param x: Tensor de forme (B, Nb_Atomes_Max * 5)
        """
        B = x.shape[0]
        N = self.Nb_Atomes_Max_In_Molecule

        # Reshape en (B, N, 5)
        x = x.view(B, N, 5)

        # Extraire les champs
        presence = x[:, :, 0].unsqueeze(-1)                     # (B, N, 1)
        symbols = x[:, :, 1].long()                             # (B, N)
        positions = x[:, :, 2:]                                 # (B, N, 3)

        # Embedding des symboles
        embeddings = self.Embedding(symbols)                    # (B, N, Embedding_Size)

        # Concaténation : [presence | embeddings | positions]
        features = torch.cat([presence, embeddings, positions], dim=-1)  # (B, N, 1+E+3)

        # Mise à plat pour MLP
        features_flat = features.view(B, -1)  # (B, N*(1+E+3))

        return self.MLP_Model(features_flat)


