import torch 
from torch.utils.data import Dataset, DataLoader
import numpy as np





import torch
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import random_split

class MoleculeDataset_Uni_Model(Dataset):
    def __init__(self, Data_List, Return_Energies=True):
        """
        Initialisation du dataset avec une liste de données de molécules.
        
        :param Data_List: Liste de dictionnaires contenant les informations des molécules.
        """
        self.Data_List = Data_List
        self.N = len(Data_List)
        if self.N == 0:
            raise ValueError("La liste de données est vide. Veuillez fournir des données valides.")

        self.Return_Energies = Return_Energies
        self.Dict_Encoding_Atoms = {
            'H': 0, 'C': 1, 'N': 2, 'O': 3, 'S': 4, 'Cl': 5,
            '[CLS]': 6  # Ajout du token [CLS]
        }
        
        # Créer un mapping pour les types de paires
        self.pair_type_mapping = {}
        idx = 0
        for at1 in ['H', 'C', 'N', 'O', 'S', 'Cl', '[CLS]']:
            for at2 in ['H', 'C', 'N', 'O', 'S', 'Cl', '[CLS]']:
                self.pair_type_mapping[(at1, at2)] = idx
                idx += 1

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
        Id = self.Data_List[idx]['Id']

        if Atoms_DataFrame is None or (self.Return_Energies and Energy is None):
            raise ValueError(f"Données incomplètes à l'index {idx}.")

        Symbols = Atoms_DataFrame.iloc[:, 0].values
        Positions = Atoms_DataFrame.iloc[:, 1:4].values

        # Calcul du centre de la molécule
        center = Positions.mean(axis=0)

        # Calcul des distances de chaque atome au centre
        distances = np.linalg.norm(Positions - center, axis=1)

        # Tri des indices selon distance croissante
        sorted_indices = distances.argsort()

        # Tri des Symbols et Positions
        Symbols_sorted = Symbols[sorted_indices]
        Positions_sorted = Positions[sorted_indices]

        # Ajouter le token [CLS]
        Symbols_sorted = np.concatenate([['[CLS]'], Symbols_sorted])
        Positions_sorted = np.concatenate([[center], Positions_sorted])  # [CLS] au centre
        Symbols_Encoded = np.array([self.Dict_Encoding_Atoms[symbol] for symbol in Symbols_sorted])

        # Créer la matrice des types de paires
        num_atoms = len(Symbols_sorted)
        pair_types = np.zeros((num_atoms, num_atoms), dtype=np.int64)
        for i in range(num_atoms):
            for j in range(num_atoms):
                pair_types[i, j] = self.pair_type_mapping[(Symbols_sorted[i], Symbols_sorted[j])]

        # Convertir en tenseurs PyTorch
        Symbols_Encoded = torch.tensor(Symbols_Encoded, dtype=torch.long)
        Positions_sorted = torch.tensor(Positions_sorted, dtype=torch.float32)
        pair_types = torch.tensor(pair_types, dtype=torch.long)
        Energy = torch.tensor(Energy, dtype=torch.float32) if self.Return_Energies else torch.tensor(0.0)

        return Id, Symbols_sorted, Symbols_Encoded, Positions_sorted, pair_types, Energy
    


def Create_DataLoader_Uni_Model(
    Data_List,
    Batch_Size=16,
    Shuffle=True,
    Num_Workers=0,
    Test_Size=0.2,
    Return_Energies=True,
    Transform=None,
    Max_Atoms=256):
    """
    Crée des DataLoader pour l'entraînement, la validation, et éventuellement le test pour Uni_Model.

    Args:
        Data_List: Liste de dictionnaires contenant les données des molécules.
        Batch_Size: Taille des batches pour les DataLoader.
        Shuffle: Si True, mélange les données (pour train/val, pas pour test).
        Num_Workers: Nombre de travailleurs pour le chargement des données.
        Test_Size: Proportion des données pour la validation (0 pour le test).
        Return_Energies: Si True, inclut les énergies dans les sorties du dataset.
        Transform: Transformation optionnelle à appliquer aux données (non utilisé ici).
        Max_Atoms: Nombre maximum d'atomes pour le padding (par défaut 256, comme dans Uni_Model).

    Returns:
        Tuple contenant le DataLoader d'entraînement et, selon Test_Size, le DataLoader de validation
        ou None si Test_Size=0 (pour le mode test).
    """
    dataset = MoleculeDataset_Uni_Model(Data_List=Data_List, Return_Energies=Return_Energies)

    if len(dataset) == 0:
        raise ValueError("Le dataset est vide. Veuillez fournir des données valides.")

    if Test_Size > 0:

        val_size = int(len(dataset) * Test_Size)
        train_size = len(dataset) - val_size
        if train_size <= 0 or val_size <= 0:
            raise ValueError(f"Test_Size={Test_Size} conduit à des tailles de train ({train_size}) ou val ({val_size}) invalides.")

        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

        def collate_fn(batch):
            ids, symbols, atom_types, coords, pair_types, energies = zip(*batch)
            max_atoms_batch = min(max(len(s) for s in symbols), Max_Atoms)
            batch_atom_types = torch.zeros(len(batch), max_atoms_batch, dtype=torch.long)
            batch_coords = torch.zeros(len(batch), max_atoms_batch, 3, dtype=torch.float32)
            batch_pair_types = torch.zeros(len(batch), max_atoms_batch, max_atoms_batch, dtype=torch.long)
            batch_mask = torch.zeros(len(batch), max_atoms_batch, dtype=torch.bool)
            batch_energies = torch.tensor(energies, dtype=torch.float32) if Return_Energies else None

            for i, (id_, sym, at, coord, pt, e) in enumerate(batch):
                num_atoms = min(len(sym), Max_Atoms)
                batch_atom_types[i, :num_atoms] = at[:num_atoms]
                batch_coords[i, :num_atoms] = coord[:num_atoms]
                batch_pair_types[i, :num_atoms, :num_atoms] = pt[:num_atoms, :num_atoms]
                batch_mask[i, :num_atoms] = True

            return {
                'ids': list(ids),
                'atom_types': batch_atom_types,
                'coords': batch_coords,
                'pair_types': batch_pair_types,
                'mask': batch_mask,
                'energies': batch_energies if Return_Energies else None
            }

        train_loader = DataLoader(
            train_dataset,
            batch_size=Batch_Size,
            shuffle=Shuffle,
            num_workers=Num_Workers,
            collate_fn=collate_fn,
            pin_memory=True if torch.cuda.is_available() else False
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=Batch_Size,
            shuffle=False,
            num_workers=Num_Workers,
            collate_fn=collate_fn,
            pin_memory=True if torch.cuda.is_available() else False
        )

        return train_loader, val_loader
    else:
        def collate_fn(batch):
            ids, symbols, atom_types, coords, pair_types, energies = zip(*batch)
            max_atoms_batch = min(max(len(s) for s in symbols), Max_Atoms)
            batch_atom_types = torch.zeros(len(batch), max_atoms_batch, dtype=torch.long)
            batch_coords = torch.zeros(len(batch), max_atoms_batch, 3, dtype=torch.float32)
            batch_pair_types = torch.zeros(len(batch), max_atoms_batch, max_atoms_batch, dtype=torch.long)
            batch_mask = torch.zeros(len(batch), max_atoms_batch, dtype=torch.bool)
            batch_energies = torch.tensor(energies, dtype=torch.float32) if Return_Energies else None

            for i, (id_, sym, at, coord, pt, e) in enumerate(batch):
                num_atoms = min(len(sym), Max_Atoms)
                batch_atom_types[i, :num_atoms] = at[:num_atoms]
                batch_coords[i, :num_atoms] = coord[:num_atoms]
                batch_pair_types[i, :num_atoms, :num_atoms] = pt[:num_atoms, :num_atoms]
                batch_mask[i, :num_atoms] = True

            return {
                'ids': list(ids),
                'atom_types': batch_atom_types,
                'coords': batch_coords,
                'pair_types': batch_pair_types,
                'mask': batch_mask,
                'energies': batch_energies if Return_Energies else None
            }

        train_loader = DataLoader(
            dataset,
            batch_size=Batch_Size,
            shuffle=Shuffle,
            num_workers=Num_Workers,
            collate_fn=collate_fn,
            pin_memory=True if torch.cuda.is_available() else False
        )

        return train_loader, None

