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




##################
    #   Invariance Dataset V2
##################




class MoleculeDataset_Invariance_V3(Dataset):
    def __init__(self, Data_List, Nb_Atoms_Max_In_Molecule=23, Return_Energies=True, transform=None):
        """
        Dataset pour des molécules avec positions représentées par [distance_CM, angle_signed_CMA, angle_signed_CMB] en 3D.

        :param Data_List: Liste de dictionnaires contenant les molécules.
        :param Nb_Atoms_Max_In_Molecule: Nombre max d'atomes dans une molécule (padding).
        :param Return_Energies: Si True, retourne les énergies.
        :param transform: Fonction de transformation à appliquer aux positions cartésiennes.
        """
        self.Data_List = Data_List
        self.N = len(Data_List)
        self.Nb_Atoms_Max = Nb_Atoms_Max_In_Molecule
        self.Return_Energies = Return_Energies
        self.transform = transform

        if self.N == 0:
            raise ValueError("La liste de données est vide.")

        self.Molecule_Dataset = MoleculeDataset(Data_List, Return_Energies=Return_Energies)
        self.Dict_Encoding_Atoms = {
            'H': 0,
            'C': 1,
            'N': 2,
            'O': 3,
            'S': 4,
            'Cl': 5,
        }

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        # Obtenir les données de MoleculeDataset
        Id, Symbols_sorted, Symbols_Encoded, Positions_sorted, Energy = self.Molecule_Dataset[idx]

        n_atoms = len(Symbols_Encoded)
        max_atoms = self.Nb_Atoms_Max

        # Appliquer une transformation aux positions si spécifiée
        if self.transform is not None:
            Positions_sorted = self.transform(Positions_sorted)

        # Calculer le centre de la molécule (C)
        center = np.mean(Positions_sorted, axis=0)

        # Calculer les distances au centre
        distances = np.linalg.norm(Positions_sorted - center, axis=1)

        # Identifier l'atome le plus proche (A) et le plus éloigné (B)
        idx_A = np.argmin(distances)
        idx_B = np.argmax(distances)
        pos_A = Positions_sorted[idx_A]
        pos_B = Positions_sorted[idx_B]

        # Calculer les nouvelles caractéristiques pour chaque atome M
        new_positions = np.zeros((n_atoms, 3), dtype=np.float32)  # [distance_CM, angle_signed_CMA, angle_signed_CMB]
        vec_CA = pos_A - center
        vec_CB = pos_B - center
        norm_CA = np.linalg.norm(vec_CA) + 1e-9  # Éviter division par zéro
        norm_CB = np.linalg.norm(vec_CB) + 1e-9

        # Calculer le vecteur normal au plan défini par CA et CB
        normal = np.cross(vec_CA, vec_CB)
        norm_normal = np.linalg.norm(normal) + 1e-9
        normal = normal / norm_normal

        for i in range(n_atoms):
            pos_M = Positions_sorted[i]
            vec_CM = pos_M - center
            norm_CM = np.linalg.norm(vec_CM) + 1e-9

            # Distance CM
            new_positions[i, 0] = norm_CM

            # Angle signé CMA (angle entre CM et CA, signé par rapport à CB)
            cos_CMA = np.dot(vec_CM, vec_CA) / (norm_CM * norm_CA)
            cos_CMA = np.clip(cos_CMA, -1.0, 1.0)
            angle_CMA = np.arccos(cos_CMA)

            # Déterminer le signe via le produit vectoriel
            cross_CM_CA = np.cross(vec_CM, vec_CA)
            sign_CMA = np.sign(np.dot(cross_CM_CA, normal))
            angle_signed_CMA = sign_CMA * angle_CMA
            new_positions[i, 1] = angle_signed_CMA

            # Angle signé CMB (angle entre CM et CB, signé par rapport à CA)
            cos_CMB = np.dot(vec_CM, vec_CB) / (norm_CM * norm_CB)
            cos_CMB = np.clip(cos_CMB, -1.0, 1.0)
            angle_CMB = np.arccos(cos_CMB)

            # Déterminer le signe via le produit vectoriel
            cross_CM_CB = np.cross(vec_CM, vec_CB)
            sign_CMB = np.sign(np.dot(cross_CM_CB, normal))
            angle_signed_CMB = sign_CMB * angle_CMB
            new_positions[i, 2] = angle_signed_CMB

        # Convertir en tenseurs
        symbols_encoded = torch.tensor(Symbols_Encoded, dtype=torch.long)
        positions = torch.tensor(new_positions, dtype=torch.float32)

        # Appliquer le padding
        padded_symbols = torch.zeros(max_atoms, dtype=torch.long)
        padded_symbols[:n_atoms] = symbols_encoded

        padded_positions = torch.zeros(max_atoms, 3, dtype=torch.float32)
        padded_positions[:n_atoms] = positions

        # Créer le masque
        mask = torch.zeros(max_atoms, dtype=torch.float32)
        mask[:n_atoms] = 1.0

        # Retourner les données
        if self.Return_Energies:
            return Id, padded_symbols, padded_positions, mask, torch.tensor(Energy, dtype=torch.float32)
        else:
            return Id, padded_symbols, padded_positions, mask

def Create_DataLoader_Invariance_V3(Data_List, Batch_Size=32, Test_Size=0.3, Shuffle=True, Num_Workers=0,
                                    Return_Energies=True, Transform=None, Nb_Atoms_Max_In_Molecule=23):
    """
    Crée un DataLoader pour MoleculeDataset_Invariance_V2.

    :param Data_List: Liste de dictionnaires contenant les molécules.
    :param Batch_Size: Taille du batch.
    :param Test_Size: Proportion des données pour le test.
    :param Shuffle: Si True, mélange les données.
    :param Num_Workers: Nombre de workers pour le DataLoader.
    :param Return_Energies: Si True, retourne les énergies.
    :param Transform: Fonction de transformation des positions.
    :param Nb_Atoms_Max_In_Molecule: Nombre max d'atomes (padding).

    :return: DataLoader(s) pour train et test (si Test_Size > 0).
    """
    dataset = MoleculeDataset_Invariance_V3 (
        Data_List,
        Nb_Atoms_Max_In_Molecule=Nb_Atoms_Max_In_Molecule,
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
            num_workers=Num_Workers
        )
        test_loader = DataLoader(
            test_dataset, 
            batch_size=Batch_Size, 
            shuffle=False, 
            num_workers=Num_Workers
        )

        return train_loader, test_loader
    else:
        return DataLoader(
            dataset, 
            batch_size=Batch_Size, 
            shuffle=Shuffle, 
            num_workers=Num_Workers
        )
    



