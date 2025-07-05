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
from torch.utils.data import Dataset
import numpy as np


def random_translate_and_rotate(positions, max_translation=0.5):
    positions = np.array(positions)

    # Translation aléatoire
    translation = np.random.uniform(-max_translation, max_translation, size=(1, 3))
    positions = positions + translation

    # Rotation aléatoire
    theta = np.random.uniform(0, 2 * np.pi)
    axis = np.random.randn(3)
    axis = axis / np.linalg.norm(axis)

    K = np.array([[0, -axis[2], axis[1]],
                  [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]])
    R = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)

    return (positions @ R.T).astype(np.float32)


class MoleculeDataset_MLP(Dataset):
    def __init__(self, Data_List, Nb_Atoms_Max_In_Molecule=23, Return_Energies=True, transform=None):
        """
        Dataset pour des molécules avec padding et transformation optionnelle.

        :param Data_List: Liste de dictionnaires contenant les molécules.
        :param Nb_Atoms_Max_In_Molecule: Nombre max d'atomes dans une molécule (padding).
        :param Return_Energies: Si True, retourne les énergies.
        :param transform: Fonction de transformation à appliquer aux positions.
        """
        self.Data_List = Data_List
        self.N = len(Data_List)
        self.Nb_Atoms_Max = Nb_Atoms_Max_In_Molecule
        if self.N == 0:
            raise ValueError("La liste de données est vide.")
        self.Return_Energies = Return_Energies
        self.Molecule_Dataset = MoleculeDataset(Data_List, Return_Energies=Return_Energies)
        self.transform = transform

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        Ids, Symbols_sorted, Symbols_Encoded, Positions_sorted, Energy = self.Molecule_Dataset[idx]

        n_atoms = len(Symbols_Encoded)
        max_atoms = self.Nb_Atoms_Max

        # Optionnel : appliquer une transformation aux positions
        if self.transform is not None:
            Positions_sorted = self.transform(Positions_sorted)

        features = np.zeros((max_atoms, 5), dtype=np.float32)
        features[:n_atoms, 0] = 1.0  # présence
        features[:n_atoms, 1] = Symbols_Encoded
        features[:n_atoms, 2:] = Positions_sorted

        flat_vector = features.flatten()

        return Ids, torch.tensor(flat_vector, dtype=torch.float32), torch.tensor(Energy, dtype=torch.float32)
