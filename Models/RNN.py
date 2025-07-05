import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils


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

    def get_activation_function(self, Activation_Name):
        if Activation_Name == 'ReLU':
            return nn.ReLU()
        elif Activation_Name == 'Sigmoid':
            return nn.Sigmoid()
        elif Activation_Name == 'Tanh':
            return nn.Tanh()
        elif Activation_Name == 'LeakyReLU':
            return nn.LeakyReLU()
        elif Activation_Name == 'PReLU':
            return nn.PReLU()
        elif Activation_Name == 'ELU':
            return nn.ELU()
        elif Activation_Name == 'Softmax':
            return nn.Softmax(dim=1)
        elif Activation_Name == 'GELU':
            return nn.GELU()
        else:
            raise ValueError(f"Activation function '{Activation_Name}' is not supported.")
        
    def forward(self, input_tensor):
        return self.model(input_tensor)


class MoleculeLSTM(nn.Module):
    def __init__(self, Num_Atom_Types=6, Embedding_Dim=64, LSTM_Hidden_Dim=128, LSTM_Num_Layers=2,
                 Nb_Hidden_Layers_MLP=2, Hidden_Layers_Size_List_MLP=[64, 32], Output_Dim=1, Activation_Name_MLP='GELU'):
        """
        Modèle LSTM pour prédire l'énergie des molécules.

        :param Num_Atom_Types: Nombre de types d'atomes (par exemple, 6 pour H, C, N, O, S, Cl).
        :param Embedding_Dim: Dimension des embeddings pour les atomes.
        :param LSTM_Hidden_Dim: Dimension des couches cachées du LSTM.
        :param LSTM_Num_Layers: Nombre de couches LSTM.
        :param Nb_Hidden_Layers_MLP: Nombre de couches cachées dans le MLP de sortie.
        :param Hidden_Layers_Size_List_MLP: Liste des tailles des couches cachées du MLP.
        :param Output_Dim: Dimension de la sortie (par exemple, 1 pour l'énergie).
        :param Activation_Name_MLP: Nom de la fonction d'activation pour le MLP.
        """
        super().__init__()
        self.embedding_dim = Embedding_Dim
        self.lstm_hidden_dim = LSTM_Hidden_Dim
        self.lstm_num_layers = LSTM_Num_Layers

        # Embedding pour les types d'atomes
        self.atom_embedding = nn.Embedding(num_embeddings=Num_Atom_Types, embedding_dim=Embedding_Dim)
        
        # LSTM
        self.lstm = nn.LSTM(
            input_size=Embedding_Dim + 3,  # Embedding + 3D positions
            hidden_size=LSTM_Hidden_Dim,
            num_layers=LSTM_Num_Layers,
            batch_first=True
        )
        
        # MLP pour la sortie
        self.mlp = MLP(
            Input_Dim=LSTM_Hidden_Dim,
            Nb_Hidden_Layers=Nb_Hidden_Layers_MLP,
            Hidden_Layers_Size_List=Hidden_Layers_Size_List_MLP,
            Output_Dim=Output_Dim,
            Activation_Name=Activation_Name_MLP
        )

    def forward(self, atom_indices, atom_positions, sequence_lengths):
        """
        Propagation avant du modèle LSTM.

        :param atom_indices: Tenseur d'indices des atomes (batch_size, max_length).
        :param atom_positions: Tenseur des positions 3D (batch_size, max_length, 3).
        :param sequence_lengths: Tenseur des longueurs des séquences (batch_size).
        :return: Prédiction de l'énergie (batch_size, Output_Dim).
        """
        batch_size = atom_indices.size(0)
        
        # Obtenir les embeddings des atomes
        atom_emb = self.atom_embedding(atom_indices)  # (batch_size, max_length, Embedding_Dim)
        
        # Concaténer avec les positions
        input_sequence = torch.cat([atom_emb, atom_positions], dim=-1)  # (batch_size, max_length, Embedding_Dim + 3)
        
        # Packer les séquences pour gérer les longueurs variables
        packed_sequence = rnn_utils.pack_padded_sequence(input_sequence, sequence_lengths, batch_first=True, enforce_sorted=False)
        
        # Passer par le LSTM
        packed_output, (hidden_state, _) = self.lstm(packed_sequence)  # packed_output: PackedSequence
        
        # Désempaqueter les sorties pour obtenir un tenseur (batch_size, max_length, LSTM_Hidden_Dim)
        output, _ = rnn_utils.pad_packed_sequence(packed_output, batch_first=True)  # output: (batch_size, max_length, LSTM_Hidden_Dim)
        
        # Sommer les sorties sur la dimension temporelle (max_length)
        sum_output = torch.sum(output, dim=1)  # (batch_size, LSTM_Hidden_Dim)
        
        # Appliquer le MLP
        output = self.mlp(sum_output)  # (batch_size, Output_Dim)
        
        return output


# Exemple de test pour la classe MoleculeLSTM
if __name__ == "__main__":
    # Paramètres du modèle
    batch_size = 32
    max_sequence_length = 23
    num_atom_types = 6
    embedding_dim = 64
    lstm_hidden_dim = 128
    lstm_num_layers = 2
    nb_hidden_layers_mlp = 2
    hidden_layers_sizes_mlp = [64, 32]
    output_dim = 1
    activation_name_mlp = 'GELU'

    # Créer le modèle
    model = MoleculeLSTM(
        Num_Atom_Types=num_atom_types,
        Embedding_Dim=embedding_dim,
        LSTM_Hidden_Dim=lstm_hidden_dim,
        LSTM_Num_Layers=lstm_num_layers,
        Nb_Hidden_Layers_MLP=nb_hidden_layers_mlp,
        Hidden_Layers_Size_List_MLP=hidden_layers_sizes_mlp,
        Output_Dim=output_dim,
        Activation_Name_MLP=activation_name_mlp
    )

    # Générer des entrées fictives
    atom_indices = torch.randint(0, num_atom_types, (batch_size, max_sequence_length))  # Indices aléatoires
    atom_positions = torch.randn(batch_size, max_sequence_length, 3)  # Positions 3D aléatoires
    sequence_lengths = torch.randint(10, max_sequence_length + 1, (batch_size,))  # Longueurs aléatoires

    # Mettre le modèle en mode évaluation
    model.eval()

    # Propagation avant
    with torch.no_grad():
        output = model(atom_indices, atom_positions, sequence_lengths)

    # Vérifier la forme de la sortie
    print(f"Shape de la sortie : {output.shape}")  # Devrait être (32, 1)

    # Vérifier que la sortie est cohérente
    assert output.shape == (batch_size, output_dim), \
        f"Forme de sortie inattendue : {output.shape}, attendu : {(batch_size, output_dim)}"

    print("Test du MoleculeLSTM réussi !")