import torch 
from torch.utils.data import Dataset, DataLoader
import numpy as np





class MoleculeDataset(Dataset):
    def __init__(self, Data_List,Return_Energies = True):
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

        return Id,Symbols_sorted, Symbols_Encoded, Positions_sorted, Energy


import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import torch.nn.functional as F

class XYZDataset(Dataset):
    def __init__(self, data_list):
        self.data_list = data_list
        self.mapping = {
            'H': 1,
            'C': 6,
            'N': 7,
            'O': 8,
            'S': 16,
            'Cl': 17
        }

    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        overlapping_precision = 1e-1
        sigma = 2.0
        min_dist = np.inf
        data = self.data_list[idx]
        label_id = data['Id']
        atoms_list = data['Atoms_List']
        energy = data['Energy']

        # Extract atom types and positions
        atom_types = [atom['Symbol'] for atom in atoms_list]
        positions = [[atom['X'], atom['Y'], atom['Z']] for atom in atoms_list]

        # Convert to tensors and map atom types
        atom_types = torch.tensor([self.mapping[atom] for atom in atom_types])

        # Calculate valence charges
        valence_charges = torch.zeros_like(atom_types)
        mask = atom_types <= 2
        valence_charges[mask] = atom_types[mask]

        mask = (atom_types > 2) & (atom_types <= 10)
        valence_charges[mask] = atom_types[mask] - 2

        mask = (atom_types > 10) & (atom_types <= 18)
        valence_charges[mask] = atom_types[mask] - 10

        positions = torch.tensor(positions, dtype=torch.float32)

        # Scale positions
        min_dist = torch.pdist(positions).min()
        if min_dist > 0:  # Avoid division by zero
            delta = sigma * np.sqrt(-8 * np.log(overlapping_precision))
            positions = positions * delta / min_dist

        # Padding for fixed size
        max_atoms = 23  # Same as original
        full_charge = F.pad(atom_types, (0, max_atoms - len(atom_types)), value=0)
        valence_charges = F.pad(valence_charges, (0, max_atoms - len(valence_charges)), value=0)
        positions = F.pad(positions, (0, 0, 0, max_atoms - positions.shape[0]), value=0)

        # Convert energy to tensor
        energy = torch.tensor(energy, dtype=torch.float32)

        return full_charge, valence_charges, positions, energy
    


import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import torch.nn.functional as F

class XYZDataset_V2(Dataset):
    def __init__(self, data_list, return_energy=True):
        self.data_list = data_list
        self.return_energy = return_energy
        self.mapping = {
            'H': 1,
            'C': 6,
            'N': 7,
            'O': 8,
            'S': 16,
            'Cl': 17
        }

    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        overlapping_precision = 1e-1
        sigma = 2.0
        min_dist = np.inf
        data = self.data_list[idx]
        label_id = data['Id']
        atoms_list = data['Atoms_List']

        # Extract atom types and positions
        atom_types = [atom['Symbol'] for atom in atoms_list]
        positions = [[atom['X'], atom['Y'], atom['Z']] for atom in atoms_list]

        # Convert to tensors and map atom types
        atom_types = torch.tensor([self.mapping[atom] for atom in atom_types])

        # Calculate valence charges
        valence_charges = torch.zeros_like(atom_types)
        mask = atom_types <= 2
        valence_charges[mask] = atom_types[mask]

        mask = (atom_types > 2) & (atom_types <= 10)
        valence_charges[mask] = atom_types[mask] - 2

        mask = (atom_types > 10) & (atom_types <= 18)
        valence_charges[mask] = atom_types[mask] - 10

        positions = torch.tensor(positions, dtype=torch.float32)

        # Scale positions
        min_dist = torch.pdist(positions).min()
        if min_dist > 0:  # Avoid division by zero
            delta = sigma * np.sqrt(-8 * np.log(overlapping_precision))
            positions = positions * delta / min_dist

        # Padding for fixed size
        max_atoms = 23  # Same as original
        full_charge = F.pad(atom_types, (0, max_atoms - len(atom_types)), value=0)
        valence_charges = F.pad(valence_charges, (0, max_atoms - len(valence_charges)), value=0)
        positions = F.pad(positions, (0, 0, 0, max_atoms - positions.shape[0]), value=0)

        # Prepare return values
        return_values = (label_id,full_charge, valence_charges, positions)

        # Add energy if required and available
        if self.return_energy:
            energy = data.get('Energy', 0.0)  # Default to 0.0 if energy is not present
            energy = torch.tensor(energy, dtype=torch.float32)
            return_values += (energy,)

        return return_values