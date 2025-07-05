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


class MoleculeDataset_Invariance_V1(Dataset):
    def __init__(self, Data_List, Nb_Atoms_Max_In_Molecule=23, Return_Energies=True, transform=None):
        """
        Dataset pour des molécules avec positions représentées par [distance_CM, angle_CMA, angle_CMB].

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
        new_positions = np.zeros((n_atoms, 3), dtype=np.float32)  # [distance_CM, angle_CMA, angle_CMB]
        vec_CA = pos_A - center
        vec_CB = pos_B - center
        norm_CA = np.linalg.norm(vec_CA) + 1e-9  # Éviter division par zéro
        norm_CB = np.linalg.norm(vec_CB) + 1e-9

        for i in range(n_atoms):
            pos_M = Positions_sorted[i]
            vec_CM = pos_M - center
            norm_CM = np.linalg.norm(vec_CM) + 1e-9

            # Distance CM
            new_positions[i, 0] = norm_CM

            # Angle CMA = arccos((CM·CA)/(|CM||CA|))
            cos_CMA = np.dot(vec_CM, vec_CA) / (norm_CM * norm_CA)
            cos_CMA = np.clip(cos_CMA, -1.0, 1.0)  # Éviter erreurs numériques
            angle_CMA = np.arccos(cos_CMA)
            new_positions[i, 1] = angle_CMA

            # Angle CMB = arccos((CM·CB)/(|CM||CB|))
            cos_CMB = np.dot(vec_CM, vec_CB) / (norm_CM * norm_CB)
            cos_CMB = np.clip(cos_CMB, -1.0, 1.0)
            angle_CMB = np.arccos(cos_CMB)
            new_positions[i, 2] = angle_CMB

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















##################
    #   Invariance Dataset V2
##################



import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class MoleculeDataset_Invariance_V2(Dataset):
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


##########################
# XGBOOST
###########################



import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.stats import skew, kurtosis
from scipy.stats import skew, kurtosis
import numpy as np
import torch
from torch.utils.data import Dataset

class MoleculeDataset_XGBoost_V2(Dataset):
    def __init__(self, Data_List, Nb_Atoms_Max_In_Molecule=23, Return_Energies=True, transform=None):
        """
        Dataset pour des molécules avec un ensemble fixe de caractéristiques pour XGBoost.

        :param Data_List: Liste de dictionnaires contenant les informations des molécules.
        :param Nb_Atoms_Max_In_Molecule: Nombre max d'atomes dans une molécule (padding).
        :param Return_Energies: Si True, retourne les énergies avec les caractéristiques.
        :param transform: Fonction de transformation à appliquer aux positions cartésiennes.
        """
        # Instancier MoleculeDataset_Invariance_V1 (supposé défini ailleurs)
        self.dataset_invariance = MoleculeDataset_Invariance_V1(
            Data_List, Nb_Atoms_Max_In_Molecule, Return_Energies, transform
        )
        self.N = len(self.dataset_invariance)
        self.Return_Energies = Return_Energies

        # Propriétés atomiques pour H, C, N, O, S, Cl avec nouvelles données ajoutées
        self.atomic_properties = {
            0: {'protons': 1, 'electrons': 1, 'mass': 1.008, 'electronegativity': 2.20, 'atomic_radius': 53, 'ionization_energy': 13.6, 'electron_affinity': 0.754, 'valence': 1},
            1: {'protons': 6, 'electrons': 6, 'mass': 12.011, 'electronegativity': 2.55, 'atomic_radius': 67, 'ionization_energy': 11.3, 'electron_affinity': 1.26, 'valence': 4},
            2: {'protons': 7, 'electrons': 7, 'mass': 14.007, 'electronegativity': 3.04, 'atomic_radius': 56, 'ionization_energy': 14.5, 'electron_affinity': -0.07, 'valence': 5},
            3: {'protons': 8, 'electrons': 8, 'mass': 15.999, 'electronegativity': 3.44, 'atomic_radius': 48, 'ionization_energy': 13.6, 'electron_affinity': 1.46, 'valence': 6},
            4: {'protons': 16, 'electrons': 16, 'mass': 32.06, 'electronegativity': 2.58, 'atomic_radius': 88, 'ionization_energy': 10.4, 'electron_affinity': 2.07, 'valence': 6},
            5: {'protons': 17, 'electrons': 17, 'mass': 35.45, 'electronegativity': 3.16, 'atomic_radius': 79, 'ionization_energy': 13.0, 'electron_affinity': 3.61, 'valence': 7}
        }

    def __len__(self):
        """Retourne le nombre total de molécules dans le dataset."""
        return self.N

    def __getitem__(self, idx):
        """
        Retourne les caractéristiques calculées pour la molécule à l'index donné.
Je veux que tu m'ajoute une p
        :param idx: Index de la molécule.
        :return: Tuple (Id, features, Energy) si Return_Energies est True, sinon (Id, features).
        """
        # Obtenir les données de MoleculeDataset_Invariance_V1
        data = self.dataset_invariance[idx]
        if self.Return_Energies:
            Id, padded_symbols, padded_positions, mask, Energy = data
        else:
            Id, padded_symbols, padded_positions, mask = data

        # Extraire les données réelles (non paddées)
        n_atoms = int(mask.sum().item())
        symbols = padded_symbols[:n_atoms].numpy()
        positions = padded_positions[:n_atoms].numpy()  # [distance_CM, angle_CMA, angle_CMB]

        # Calculer les caractéristiques fixes étendues
        features = self.compute_features(symbols, positions, n_atoms)

        if self.Return_Energies:
            return Id, features, Energy
        else:
            return Id, features

    def compute_features(self, symbols, positions, n_atoms):
        """
        Calcule un vecteur de caractéristiques étendu pour la molécule.
        """
        atom_counts = np.bincount(symbols, minlength=6)
        atom_props = {
            'protons': np.array([self.atomic_properties[s]['protons'] for s in symbols], dtype=np.float32),
            'electrons': np.array([self.atomic_properties[s]['electrons'] for s in symbols], dtype=np.float32),
            'mass': np.array([self.atomic_properties[s]['mass'] for s in symbols], dtype=np.float32),
            'electronegativity': np.array([self.atomic_properties[s]['electronegativity'] for s in symbols], dtype=np.float32),
            'atomic_radius': np.array([self.atomic_properties[s]['atomic_radius'] for s in symbols], dtype=np.float32),
            'ionization_energy': np.array([self.atomic_properties[s]['ionization_energy'] for s in symbols], dtype=np.float32),
            'electron_affinity': np.array([self.atomic_properties[s]['electron_affinity'] for s in symbols], dtype=np.float32),
            'valence': np.array([self.atomic_properties[s]['valence'] for s in symbols], dtype=np.float32),
        }

        distances = positions[:, 0]
        angles_CMA = positions[:, 1]
        angles_CMB = positions[:, 2]

        # Statistiques basiques + avancées pour distances
        dist_stats = [
            np.mean(distances),
            np.std(distances),
            np.min(distances),
            np.max(distances),
            np.median(distances),
            skew(distances),
            kurtosis(distances)
        ]

        # Statistiques pour angles CMA
        angle_CMA_stats = [
            np.mean(angles_CMA),
            np.std(angles_CMA),
            np.min(angles_CMA),
            np.max(angles_CMA),
            np.median(angles_CMA),
            skew(angles_CMA),
            kurtosis(angles_CMA)
        ]

        # Statistiques pour angles CMB
        angle_CMB_stats = [
            np.mean(angles_CMB),
            np.std(angles_CMB),
            np.min(angles_CMB),
            np.max(angles_CMB),
            np.median(angles_CMB),
            skew(angles_CMB),
            kurtosis(angles_CMB)
        ]

        # Types A et B (plus proche et plus éloigné du centre)
        idx_A = np.argmin(distances)
        idx_B = np.argmax(distances)
        type_A = symbols[idx_A]
        type_B = symbols[idx_B]

        # Statistiques avancées sur propriétés atomiques (incluant nouvelles)
        prop_stats = {}
        for prop_name, values in atom_props.items():
            prop_stats[prop_name] = [
                np.sum(values),
                np.mean(values),
                np.std(values),
                np.min(values),
                np.max(values),
                np.median(values),
                skew(values),
                kurtosis(values)
            ]

        # Proportions des types d'atomes
        atom_props_ratios = atom_counts / n_atoms

        # Distances inter-atomes (matrice symétrique)
        if positions.shape[1] == 3:
            coord = positions
            dists_pairwise = np.sqrt(((coord[:, None, :] - coord[None, :, :]) ** 2).sum(axis=-1))
            mask = ~np.eye(n_atoms, dtype=bool)
            pairwise_distances = dists_pairwise[mask]
            pair_dist_stats = [
                np.mean(pairwise_distances),
                np.std(pairwise_distances),
                np.min(pairwise_distances),
                np.max(pairwise_distances),
                np.median(pairwise_distances),
                skew(pairwise_distances),
                kurtosis(pairwise_distances)
            ]
        else:
            pair_dist_stats = [0.] * 7

        # Rayon de giration simplifié (en 3D)
        if positions.shape[1] == 3:
            mass = atom_props['mass']
            coords = positions
            centroid = np.average(coords, axis=0, weights=mass)
            diff = coords - centroid
            rgyr = np.sqrt(np.average(np.sum(diff**2, axis=1), weights=mass))
        else:
            rgyr = 0.

        # Nombre d'atomes
        n_atoms_feature = n_atoms

        # Assemblage final des features
        features = np.concatenate([
            atom_counts.astype(np.float32),                  # 6
            atom_props_ratios.astype(np.float32),            # 6
            np.array([type_A, type_B], dtype=np.float32),   # 2
            np.array(dist_stats, dtype=np.float32),          # 7
            np.array(angle_CMA_stats, dtype=np.float32),     # 7
            np.array(angle_CMB_stats, dtype=np.float32),     # 7
            np.array([n_atoms_feature], dtype=np.float32),   # 1
            np.array(prop_stats['protons'], dtype=np.float32),         # 8
            np.array(prop_stats['electrons'], dtype=np.float32),       # 8
            np.array(prop_stats['mass'], dtype=np.float32),            # 8
            np.array(prop_stats['electronegativity'], dtype=np.float32), # 8
            np.array(prop_stats['atomic_radius'], dtype=np.float32),    # 8
            np.array(prop_stats['ionization_energy'], dtype=np.float32),# 8
            np.array(prop_stats['electron_affinity'], dtype=np.float32),# 8
            np.array(prop_stats['valence'], dtype=np.float32),          # 8
            np.array(pair_dist_stats, dtype=np.float32),                 # 7
            np.array([rgyr], dtype=np.float32)                           # 1
        ])

        return features

def extract_data_for_xgboost(dataset):
    """
    Extrait les données sous forme de tableaux NumPy pour XGBoost.

    :param dataset: Instance de MoleculeDataset_XGBoost_V2.
    :return: Tuple (ids, features, energies) si énergies disponibles, sinon (ids, features).
    """
    ids_list = []
    features_list = []
    energies_list = []
    for item in dataset:
        Id, features = item[:2]
        ids_list.append(Id)
        features_list.append(features)
        if len(item) == 3:
            energies_list.append(item[2].numpy() if isinstance(item[2], torch.Tensor) else item[2])
    if energies_list:
        return np.array(ids_list), np.array(features_list), np.array(energies_list)
    return np.array(ids_list), np.array(features_list)


class MoleculeDataset_XGBoost(Dataset):
    def __init__(self, Data_List, Return_Energies=True):
        """
        Dataset pour des molécules avec un ensemble fixe de caractéristiques pour XGBoost.
        :param Data_List: Liste de dictionnaires contenant les informations des molécules.
        :param Return_Energies: Si True, retourne les énergies avec les caractéristiques.
        """
        self.dataset = MoleculeDataset(Data_List, Return_Energies=Return_Energies)
        self.N = len(self.dataset)
        self.Return_Energies = Return_Energies

        # Propriétés atomiques pour H, C, N, O, S, Cl
        self.atomic_properties = {
            0: {'protons': 1, 'electrons': 1, 'mass': 1.008, 'electronegativity': 2.20, 'atomic_radius': 53, 'ionization_energy': 13.6, 'electron_affinity': 0.754, 'valence': 1},
            1: {'protons': 6, 'electrons': 6, 'mass': 12.011, 'electronegativity': 2.55, 'atomic_radius': 67, 'ionization_energy': 11.3, 'electron_affinity': 1.26, 'valence': 4},
            2: {'protons': 7, 'electrons': 7, 'mass': 14.007, 'electronegativity': 3.04, 'atomic_radius': 56, 'ionization_energy': 14.5, 'electron_affinity': -0.07, 'valence': 5},
            3: {'protons': 8, 'electrons': 8, 'mass': 15.999, 'electronegativity': 3.44, 'atomic_radius': 48, 'ionization_energy': 13.6, 'electron_affinity': 1.46, 'valence': 6},
            4: {'protons': 16, 'electrons': 16, 'mass': 32.06, 'electronegativity': 2.58, 'atomic_radius': 88, 'ionization_energy': 10.4, 'electron_affinity': 2.07, 'valence': 6},
            5: {'protons': 17, 'electrons': 17, 'mass': 35.45, 'electronegativity': 3.16, 'atomic_radius': 79, 'ionization_energy': 13.0, 'electron_affinity': 3.61, 'valence': 7}
        }

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        data = self.dataset[idx]

        if self.Return_Energies:
            Id, Symbols_sorted, Symbols_Encoded, Positions_sorted, Energy = data
        else:
            Id, Symbols_sorted, Symbols_Encoded, Positions_sorted, _ = data

        symbols = Symbols_Encoded  # Numérique [0–5]
        positions = Positions_sorted  # Nx3

        features = self.compute_features(symbols, positions)

        if self.Return_Energies:
            return Id, features, Energy
        else:
            return Id, features
    def compute_features(self, symbols, positions):
        n_atoms = len(symbols)
        atom_counts = np.bincount(symbols, minlength=6)

        atom_props = {prop: np.array([self.atomic_properties[s][prop] for s in symbols], dtype=np.float32)
                    for prop in ['protons', 'electrons', 'mass', 'electronegativity', 'atomic_radius',
                                    'ionization_energy', 'electron_affinity', 'valence']}

        Z = atom_props['protons']  # Numéro atomique

        # --- Matrice de Coulomb ---
        dists = np.linalg.norm(positions[:, None, :] - positions[None, :, :], axis=-1)
        with np.errstate(divide='ignore'):
            C = np.where(dists == 0,
                        0.5 * Z[:, None] ** 2.4,
                        Z[:, None] * Z[None, :] / dists)
        # Remplace NaN possibles sur la diagonale
        np.fill_diagonal(C, 0.5 * Z ** 2.4)

        # Vectoriser : éléments hors diagonale triés par valeur absolue
        triu_indices = np.triu_indices(n_atoms, k=1)
        C_upper = np.abs(C[triu_indices])
        coulomb_vector = np.sort(C_upper)[::-1]  # tri décroissant

        # Pour garder une dimension fixe : padding ou troncature
        max_coulomb_len = 23  # par exemple
        if len(coulomb_vector) >= max_coulomb_len:
            coulomb_vector = coulomb_vector[:max_coulomb_len]
        else:
            coulomb_vector = np.pad(coulomb_vector, (0, max_coulomb_len - len(coulomb_vector)))

        # --- Reste inchangé ---
        center = positions.mean(axis=0)
        distances = np.linalg.norm(positions - center, axis=1)
        idx_A = np.argmin(distances)
        idx_B = np.argmax(distances)
        type_A = symbols[idx_A]
        type_B = symbols[idx_B]

        dists_pairwise = dists
        mask = ~np.eye(n_atoms, dtype=bool)
        pairwise_distances = dists_pairwise[mask]

        mass = atom_props['mass']
        centroid = np.average(positions, axis=0, weights=mass)
        diff = positions - centroid
        rgyr = np.sqrt(np.average(np.sum(diff**2, axis=1), weights=mass))

        I = np.sum(mass[:, None] * diff**2, axis=0)
        moment_inertia_stats = [I.mean(), I.std(), I.min(), I.max()]

        total_mass = np.sum(mass)
        volume = (2 * rgyr)**3
        density = total_mass / volume if volume > 0 else 0.
        sphericity = rgyr / distances.max() if distances.max() > 0 else 0.
        valence = atom_props['valence']
        electroneg = atom_props['electronegativity']
        weighted_electronegativity = np.sum(valence * electroneg)
        n_heavy_atoms = np.sum(symbols != 0)
        electron_mass_ratio = np.sum(atom_props['electrons']) / total_mass if total_mass > 0 else 0.

        prop_stats = []
        for prop_array in atom_props.values():
            prop_stats.extend([
                np.sum(prop_array), np.mean(prop_array), np.std(prop_array),
                np.min(prop_array), np.max(prop_array),
                np.median(prop_array), skew(prop_array), kurtosis(prop_array)
            ])

        atom_props_ratios = atom_counts / n_atoms

        def stats(arr): return [
            np.mean(arr), np.std(arr), np.min(arr), np.max(arr),
            np.median(arr), skew(arr), kurtosis(arr)
        ]
        dist_stats = stats(distances)
        pair_dist_stats = stats(pairwise_distances)
        angle_CMA_stats = [0.] * 7
        angle_CMB_stats = [0.] * 7

        # --- Assemblage final ---
        features = np.concatenate([
            atom_counts.astype(np.float32),
            atom_props_ratios.astype(np.float32),
            np.array([type_A, type_B], dtype=np.float32),
            np.array([n_atoms], dtype=np.float32),
            np.array([n_heavy_atoms], dtype=np.float32),
            np.array([total_mass], dtype=np.float32),
            np.array([density], dtype=np.float32),
            np.array([rgyr], dtype=np.float32),
            np.array(moment_inertia_stats, dtype=np.float32),
            np.array([sphericity], dtype=np.float32),
            np.array([electron_mass_ratio], dtype=np.float32),
            np.array([weighted_electronegativity], dtype=np.float32),
            np.array(dist_stats, dtype=np.float32),
            np.array(pair_dist_stats, dtype=np.float32),
            np.array(angle_CMA_stats, dtype=np.float32),
            np.array(angle_CMB_stats, dtype=np.float32),
            np.array(prop_stats, dtype=np.float32),
            coulomb_vector.astype(np.float32)  # <<=== matrice de Coulomb vectorisée
        ])
        return features
