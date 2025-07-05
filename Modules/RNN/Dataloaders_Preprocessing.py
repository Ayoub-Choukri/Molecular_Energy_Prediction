import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np


class MoleculeDataset(Dataset):
    def __init__(self, Data_List, Return_Energies=True):
        """
        Initialisation du dataset avec une liste de données de molécules.
        
        :param Data_List: Liste de dictionnaires contenant les informations des molécules.
        :param Return_Energies: Si True, retourne les énergies.
        """
        self.Data_List = Data_List
        self.N = len(Data_List)
        if self.N == 0:
            raise ValueError("La liste de données est vide. Veuillez fournir des données valides.")

        self.Return_Energies = Return_Energies
        self.Dict_Encoding_Atoms = {
            'H': 0,
            'C': 1,
            'N': 2,
            'O': 3,
            'S': 4,
            'Cl': 5,
        }

    def __len__(self):
        """
        Retourne la taille du dataset.
        
        :return: Nombre d'éléments dans le dataset.
        """
        return self.N
    
    def __getitem__(self, idx):
        if idx < 0 or idx >= self.N:
            raise IndexError(f"Index {idx} hors des limites du dataset. Taille du dataset: {self.N}.")

        Atoms_DataFrame = self.Data_List[idx]['Atoms_DataFrame']
        Energy = self.Data_List[idx]['Energy'] if self.Return_Energies else 0

        if Atoms_DataFrame is None or Energy is None:
            raise ValueError(f"Données incomplètes à l'index {idx}.")

        Id = self.Data_List[idx]['Id'] 
        Symbols = Atoms_DataFrame.iloc[:,0].values
        Positions = Atoms_DataFrame.iloc[:,1:4].values

        # Calcul du centre de la molécule
        center = Positions.mean(axis=0)

        # Calcul des distances de chaque atome au centre
        distances = np.linalg.norm(Positions - center, axis=1)

        # Tri des indices selon distance croissante
        sorted_indices = distances.argsort()

        # Tri des Symbols et Positions selon ces indices
        Symbols_sorted = Symbols[sorted_indices]
        Positions_sorted = Positions[sorted_indices]

        Symbols_Encoded = np.array([self.Dict_Encoding_Atoms[symbol] for symbol in Symbols_sorted])

        return Id, Symbols_sorted, Symbols_Encoded, Positions_sorted, Energy


class MoleculeDataset_LSTM(Dataset):
    def __init__(self, Data_List, Return_Energies=True, transform=None):
        """
        Dataset pour des molécules compatible avec un RNN LSTM, avec indices pour nn.Embedding.

        :param Data_List: Liste de dictionnaires contenant les molécules.
        :param Return_Energies: Si True, retourne les énergies.
        :param transform: Fonction de transformation à appliquer aux positions.
        """
        self.Data_List = Data_List
        self.N = len(Data_List)
        self.Return_Energies = Return_Energies
        self.transform = transform

        if self.N == 0:
            raise ValueError("La liste de données est vide.")

        self.Molecule_Dataset = MoleculeDataset(Data_List, Return_Energies=Return_Energies)

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        # Obtenir les données de MoleculeDataset
        Id, Symbols_sorted, Symbols_Encoded, Positions_sorted, Energy = self.Molecule_Dataset[idx]

        # Appliquer une transformation aux positions si spécifiée
        if self.transform is not None:
            Positions_sorted = self.transform(Positions_sorted)

        # Convertir en tenseurs
        symbols_encoded = torch.tensor(Symbols_Encoded, dtype=torch.long)  # Indices pour nn.Embedding
        positions = torch.tensor(Positions_sorted, dtype=torch.float32)

        # Retourner les données
        if self.Return_Energies:
            return Id, symbols_encoded, positions, torch.tensor(Energy, dtype=torch.float32)
        else:
            return Id, symbols_encoded, positions


def Create_DataLoader_LSTM(Data_List, Batch_Size=32, Test_Size=0.3, Shuffle=True, Num_Workers=0, 
                          Return_Energies=True, Transform=None):
    """
    Crée un DataLoader pour les molécules compatible avec un RNN LSTM.

    :param Data_List: Liste de dictionnaires contenant les informations des molécules.
    :param Batch_Size: Taille du batch pour le DataLoader.
    :param Test_Size: Proportion des données à utiliser pour le test.
    :param Shuffle: Indique si les données doivent être mélangées.
    :param Num_Workers: Nombre de workers pour le DataLoader.
    :param Return_Energies: Si True, retourne les énergies.
    :param Transform: Fonction de transformation à appliquer aux positions.

    :return: DataLoader(s) configuré(s) pour itérer sur les données de molécules.
    """
    dataset = MoleculeDataset_LSTM(
        Data_List,
        Return_Energies=Return_Energies,
        transform=Transform
    )

    if Test_Size > 0:
        total_size = len(dataset)
        size_train = int(total_size * (1 - Test_Size))
        size_test = total_size - size_train

        train_dataset, test_dataset = torch.utils.data.random_split(
            dataset, 
            [size_train, size_test],
        )
        train_loader = DataLoader(
            train_dataset, 
            batch_size=Batch_Size, 
            shuffle=Shuffle, 
            num_workers=Num_Workers,
            collate_fn=lambda batch: collate_fn_lstm(batch)
        )
        test_loader = DataLoader(
            test_dataset, 
            batch_size=Batch_Size, 
            shuffle=False, 
            num_workers=Num_Workers,
            collate_fn=lambda batch: collate_fn_lstm(batch)
        )

        return train_loader, test_loader
    else:
        return DataLoader(
            dataset, 
            batch_size=Batch_Size, 
            shuffle=Shuffle, 
            num_workers=Num_Workers,
            collate_fn=lambda batch: collate_fn_lstm(batch)
        )


def collate_fn_lstm(batch):
    """
    Fonction pour collationner les batches pour un RNN LSTM.

    :param batch: Liste de tuples (Id, symbols_encoded, positions, Energy) ou (Id, symbols_encoded, positions).
    :return: Tuple contenant les tenseurs batchés.
    """
    ids = [item[0] for item in batch]
    symbols_encoded = [item[1] for item in batch]
    positions = [item[2] for item in batch]
    energies = [item[3] for item in batch] if len(batch[0]) == 4 else None

    # Obtenir les longueurs des séquences
    lengths = [s.shape[0] for s in symbols_encoded]

    # Padding des symboles encodés et des positions
    max_length = max(lengths)
    padded_symbols = torch.zeros(len(batch), max_length, dtype=torch.long)
    padded_positions = torch.zeros(len(batch), max_length, 3, dtype=torch.float32)

    for i, (s, p) in enumerate(zip(symbols_encoded, positions)):
        padded_symbols[i, :lengths[i]] = s
        padded_positions[i, :lengths[i], :] = p  # Ligne correcte !

    # Convertir les longueurs en tenseur
    lengths = torch.tensor(lengths, dtype=torch.long)

    if energies is not None:
        energies = torch.stack(energies)
        return ids, padded_symbols, padded_positions, lengths, energies
    else:
        return ids, padded_symbols, padded_positions, lengths