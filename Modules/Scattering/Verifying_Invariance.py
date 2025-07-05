import torch
import pandas as pd
import numpy as np
from tabulate import tabulate
from termcolor import colored
import datetime
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from torch.utils.data import Dataset, DataLoader
import os

from Dataloaders_Preprocessing import XYZDataset_V2
from sklearn import linear_model, model_selection, preprocessing, pipeline
from scipy.spatial.distance import pdist
from kymatio.torch import HarmonicScattering3D
from kymatio.scattering3d.utils import generate_weighted_sum_of_gaussians
from kymatio.datasets import fetch_qm7
from kymatio.caching import get_cache_dir
import torch.nn.functional as F
from kymatio.scattering3d.backend.torch_backend import TorchBackend3D
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, train_test_split
from tqdm.auto import tqdm
from sklearn.preprocessing import StandardScaler


# Fonction pour translater une molécule (inchangée)
def translate_molecule(data, translation_vector):
    if hasattr(translation_vector, 'numpy'):
        translation_vector = translation_vector.cpu().numpy()
    
    atoms_list = data['Atoms_List']
    precision = 6
    
    translated_atoms = []
    for atom in atoms_list:
        new_atom = atom.copy()
        new_atom['X'] = round(float(new_atom['X']) + float(translation_vector[0]), precision)
        new_atom['Y'] = round(float(new_atom['Y']) + float(translation_vector[1]), precision)
        new_atom['Z'] = round(float(new_atom['Z']) + float(translation_vector[2]), precision)
        translated_atoms.append(new_atom)
    
    df_translated = data['Atoms_DataFrame'].copy()
    df_translated[['X', 'Y', 'Z']] = np.round(df_translated[['X', 'Y', 'Z']] + translation_vector, decimals=precision)
    
    translated_data = data.copy()
    translated_data['Atoms_List'] = translated_atoms
    translated_data['Atoms_DataFrame'] = df_translated
    
    return translated_data

# Fonction pour calculer les coefficients de scattering (inchangée)
def compute_scattering_coefficients(dataset, grid, scattering, sigma, integral_powers, device):
    order_0, orders_1_and_2 = [], []
    
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    for batch in data_loader:
        Id_batch, full_charge_batch, valence_batch, pos_batch, energie = batch
        
        full_density_batch = generate_weighted_sum_of_gaussians(grid, pos_batch, full_charge_batch, sigma)
        full_density_batch = torch.from_numpy(full_density_batch).to(device).float()
        full_order_0 = TorchBackend3D.compute_integrals(full_density_batch, integral_powers)
        full_scattering = scattering(full_density_batch)
        
        val_density_batch = generate_weighted_sum_of_gaussians(grid, pos_batch, valence_batch, sigma)
        val_density_batch = torch.from_numpy(val_density_batch).to(device).float()
        val_order_0 = TorchBackend3D.compute_integrals(val_density_batch, integral_powers)
        val_scattering = scattering(val_density_batch)
        
        core_density_batch = full_density_batch - val_density_batch
        core_order_0 = TorchBackend3D.compute_integrals(core_density_batch, integral_powers)
        core_scattering = scattering(core_density_batch)
        
        batch_order_0 = torch.stack((full_order_0, val_order_0, core_order_0), dim=-1)
        batch_orders_1_and_2 = torch.stack((full_scattering, val_scattering, core_scattering), dim=-1)
        
        order_0.append(batch_order_0)
        orders_1_and_2.append(batch_orders_1_and_2)
    
    order_0 = torch.cat(order_0, dim=0)
    orders_1_and_2 = torch.cat(orders_1_and_2, dim=0)
    
    return order_0, orders_1_and_2

# Fonction pour calculer l'erreur relative maximale
def compute_max_relative_error(tensor1, tensor2):
    abs_diff = torch.abs(tensor1 - tensor2)
    max_magnitude = torch.max(torch.abs(tensor1), torch.abs(tensor2))
    # Éviter la division par zéro : utiliser un petit epsilon là où max_magnitude est proche de 0
    epsilon = 1e-10
    relative_error = abs_diff / (max_magnitude + epsilon)
    return relative_error.max().item()

# Fonction pour tester l'invariance par translation avec erreur relative
def test_scattering_translation_invariance(data_list, index_to_test, device, grid, scattering, sigma, integral_powers, plot_positions=False, relative_error_threshold=1e-3):
    """
    Teste l'invariance par translation des coefficients de scattering pour une molécule donnée en utilisant l'erreur relative.

    Args:
        data_list: Liste des dictionnaires de molécules.
        index_to_test: Index de la molécule à tester.
        device: torch.device.
        grid, scattering, sigma, integral_powers: Paramètres pour le calcul des coefficients.
        plot_positions: Boolean pour activer/désactiver le tracé 3D des positions (défaut False).
        relative_error_threshold: Seuil pour l'erreur relative (défaut 1e-4).
    """
    # Extraire la molécule
    molecule = data_list[index_to_test:index_to_test+1]
    
    # Créer le dataset non translaté
    dataset_not_translated = XYZDataset_V2(
        data_list=molecule,
        return_energy=True
    )
    
    # Appliquer une translation aléatoire
    translation_vector = torch.Tensor(np.random.uniform(-0.5, 0.5, size=(3,))).to(device)
    molecule_translated = translate_molecule(molecule[0], translation_vector)
    
    # Créer le dataset translaté
    dataset_translated = XYZDataset_V2(
        data_list=[molecule_translated],
        return_energy=True
    )
    
    # Calculer les coefficients de scattering pour les deux datasets
    order_0_not_trans, orders_1_and_2_not_trans = compute_scattering_coefficients(
        dataset_not_translated, grid, scattering, sigma, integral_powers, device
    )
    order_0_trans, orders_1_and_2_trans = compute_scattering_coefficients(
        dataset_translated, grid, scattering, sigma, integral_powers, device
    )
    
    # Calculer les erreurs relatives maximales
    order_0_rel_error = compute_max_relative_error(order_0_not_trans, order_0_trans)
    orders_1_and_2_rel_error = compute_max_relative_error(orders_1_and_2_not_trans, orders_1_and_2_trans)
    
    # Vérifier l'invariance
    invariance_check = "PASS" if max(order_0_rel_error, orders_1_and_2_rel_error) < relative_error_threshold else "FAIL"
    
    # Horodatage
    timestamp = datetime.datetime.now().strftime("%H:%M:%S %Y-%m-%d")
    
    # Afficher le rapport
    print(colored(f"\n=== Scattering Translation Invariance Test Report ({timestamp}) ===", 'blue', attrs=['bold']))
    print(colored(f"Molecule ID: {molecule[0]['Id']}", 'cyan'))
    print("\n" + tabulate(
        [["Max Relative Error (Order 0)", f"{order_0_rel_error:.6e}"],
         ["Max Relative Error (Orders 1 and 2)", f"{orders_1_and_2_rel_error:.6e}"],
         ["Invariance Check", colored(invariance_check, 'green' if invariance_check == "PASS" else 'red')]],
        headers=["Metric", "Value"], tablefmt="pretty"
    ))
    
    print(colored("\nTranslation Vector:", 'cyan'))
    print(tabulate([["X", f"{translation_vector[0].cpu().item():.6f}"],
                    ["Y", f"{translation_vector[1].cpu().item():.6f}"],
                    ["Z", f"{translation_vector[2].cpu().item():.6f}"]],
                   headers=["Axis", "Value"], tablefmt="pretty"))
    
    # Comparaison des positions
    old_positions = molecule[0]['Atoms_DataFrame'][['X', 'Y', 'Z']].values
    new_positions = molecule_translated['Atoms_DataFrame'][['X', 'Y', 'Z']].values
    diff_positions = np.round(new_positions - old_positions, decimals=6)
    df_positions = pd.DataFrame({
        'X_old': old_positions[:, 0],
        'Y_old': old_positions[:, 1],
        'Z_old': old_positions[:, 2],
        'X_new': new_positions[:, 0],
        'Y_new': new_positions[:, 1],
        'Z_new': new_positions[:, 2],
        'ΔX': diff_positions[:, 0],
        'ΔY': diff_positions[:, 1],
        'ΔZ': diff_positions[:, 2]
    })
    
    print(colored("\nAtom Position Comparison (Part 1: Original Positions):", 'cyan'))
    print(tabulate(df_positions[['X_old', 'Y_old', 'Z_old']], 
                   headers=['X_old', 'Y_old', 'Z_old'], 
                   tablefmt="pretty", 
                   showindex=range(1, len(df_positions) + 1)))
    
    print(colored("\nAtom Position Comparison (Part 2: Translated Positions):", 'cyan'))
    print(tabulate(df_positions[['X_new', 'Y_new', 'Z_new']], 
                   headers=['X_new', 'Y_new', 'Z_new'], 
                   tablefmt="pretty", 
                   showindex=range(1, len(df_positions) + 1)))
    
    print(colored("\nAtom Position Comparison (Part 3: Position Differences):", 'cyan'))
    print(tabulate(df_positions[['ΔX', 'ΔY', 'ΔZ']], 
                   headers=['ΔX', 'ΔY', 'ΔZ'], 
                   tablefmt="pretty", 
                   showindex=range(1, len(df_positions) + 1)))
    
    # Tracé des positions si activé
    if plot_positions:
        fig = plt.figure(figsize=(12, 6))
        
        ax1 = fig.add_subplot(121, projection='3d')
        ax1.scatter(old_positions[:, 0], old_positions[:, 1], old_positions[:, 2], c='blue', label='Original')
        ax1.set_title('Original Positions')
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')
        ax1.legend()
        
        ax2 = fig.add_subplot(122, projection='3d')
        ax2.scatter(new_positions[:, 0], new_positions[:, 1], new_positions[:, 2], c='green', label='Translated')
        ax2.set_title('Translated Positions')
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.set_zlabel('Z')
        ax2.legend()
        
        plt.tight_layout()
        plt.show()
    
    print(colored("\n=== End of Report ===", 'blue', attrs=['bold']))
    
    return invariance_check, order_0_rel_error, orders_1_and_2_rel_error



import torch
import pandas as pd
import numpy as np
from tabulate import tabulate
from termcolor import colored
import datetime
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from torch.utils.data import Dataset, DataLoader
import os

# Fonction pour calculer la matrice de rotation 3D autour d'un axe
def rotation_matrix_3d(axis, theta):
    """
    Calcule la matrice de rotation 3D autour d'un axe donné avec un angle theta (en radians).
    
    Args:
        axis: Vecteur 3D définissant l'axe de rotation (normalisé).
        theta: Angle de rotation en radians.
    
    Returns:
        Matrice de rotation 3x3 (numpy array).
    """
    axis = axis / np.linalg.norm(axis)  # Normaliser l'axe
    ux, uy, uz = axis
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    one_minus_cos = 1 - cos_theta
    
    R = np.array([
        [cos_theta + ux**2 * one_minus_cos, ux * uy * one_minus_cos - uz * sin_theta, ux * uz * one_minus_cos + uy * sin_theta],
        [uy * ux * one_minus_cos + uz * sin_theta, cos_theta + uy**2 * one_minus_cos, uy * uz * one_minus_cos - ux * sin_theta],
        [uz * ux * one_minus_cos - uy * sin_theta, uz * uy * one_minus_cos + ux * sin_theta, cos_theta + uz**2 * one_minus_cos]
    ])
    return R

# Fonction pour appliquer une rotation à une molécule
def rotate_molecule(data, axis_point_a, axis_point_b, angle_degrees):
    """
    Applique une rotation 3D à une molécule autour de l'axe défini par les points A et B.
    
    Args:
        data: Dictionnaire contenant 'Atoms_List' et 'Atoms_DataFrame'.
        axis_point_a: Point A définissant l'axe (numpy array [x, y, z]).
        axis_point_b: Point B définissant l'axe (numpy array [x, y, z]).
        angle_degrees: Angle de rotation en degrés.
    
    Returns:
        Dictionnaire avec les positions des atomes rotatées.
    """
    if hasattr(axis_point_a, 'numpy'):
        axis_point_a = axis_point_a.cpu().numpy()
    if hasattr(axis_point_b, 'numpy'):
        axis_point_b = axis_point_b.cpu().numpy()
    
    atoms_list = data['Atoms_List']
    precision = 6
    theta = np.radians(angle_degrees)  # Convertir l'angle en radians
    axis = axis_point_b - axis_point_a  # Vecteur de l'axe de rotation
    
    # Calculer la matrice de rotation
    R = rotation_matrix_3d(axis, theta)
    
    # Créer une nouvelle liste d'atomes rotatés
    rotated_atoms = []
    for atom in atoms_list:
        new_atom = atom.copy()
        # Coordonnées de l'atome
        pos = np.array([float(new_atom['X']), float(new_atom['Y']), float(new_atom['Z'])])
        # Translater pour que le point A soit à l'origine
        pos_translated = pos - axis_point_a
        # Appliquer la rotation
        pos_rotated = R @ pos_translated
        # Retranslater vers la position originale
        pos_final = pos_rotated + axis_point_a
        # Mettre à jour les coordonnées
        new_atom['X'] = round(float(pos_final[0]), precision)
        new_atom['Y'] = round(float(pos_final[1]), precision)
        new_atom['Z'] = round(float(pos_final[2]), precision)
        rotated_atoms.append(new_atom)
    
    # Créer une nouvelle DataFrame pour les positions
    df_rotated = data['Atoms_DataFrame'].copy()
    positions = df_rotated[['X', 'Y', 'Z']].values
    # Appliquer la rotation à toutes les positions
    positions_translated = positions - axis_point_a
    positions_rotated = (R @ positions_translated.T).T
    positions_final = positions_rotated + axis_point_a
    df_rotated[['X', 'Y', 'Z']] = np.round(positions_final, decimals=precision)
    
    # Créer une copie des données avec les atomes rotatés
    rotated_data = data.copy()
    rotated_data['Atoms_List'] = rotated_atoms
    rotated_data['Atoms_DataFrame'] = df_rotated
    
    return rotated_data

# Fonction pour calculer les coefficients de scattering
def compute_scattering_coefficients(dataset, grid, scattering, sigma, integral_powers, device):
    order_0, orders_1_and_2 = [], []
    
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    for batch in data_loader:
        Id_batch, full_charge_batch, valence_batch, pos_batch, energie = batch
        
        full_density_batch = generate_weighted_sum_of_gaussians(grid, pos_batch, full_charge_batch, sigma)
        full_density_batch = torch.from_numpy(full_density_batch).to(device).float()
        full_order_0 = TorchBackend3D.compute_integrals(full_density_batch, integral_powers)
        full_scattering = scattering(full_density_batch)
        
        val_density_batch = generate_weighted_sum_of_gaussians(grid, pos_batch, valence_batch, sigma)
        val_density_batch = torch.from_numpy(val_density_batch).to(device).float()
        val_order_0 = TorchBackend3D.compute_integrals(val_density_batch, integral_powers)
        val_scattering = scattering(val_density_batch)
        
        core_density_batch = full_density_batch - val_density_batch
        core_order_0 = TorchBackend3D.compute_integrals(core_density_batch, integral_powers)
        core_scattering = scattering(core_density_batch)
        
        batch_order_0 = torch.stack((full_order_0, val_order_0, core_order_0), dim=-1)
        batch_orders_1_and_2 = torch.stack((full_scattering, val_scattering, core_scattering), dim=-1)
        
        order_0.append(batch_order_0)
        orders_1_and_2.append(batch_orders_1_and_2)
    
    order_0 = torch.cat(order_0, dim=0)
    orders_1_and_2 = torch.cat(orders_1_and_2, dim=0)
    
    return order_0, orders_1_and_2
import torch

def compute_max_relative_error(v1, v2, ref='v1', epsilon=1e-10):
    """
    Calcule l'erreur relative maximale entre v1 et v2.

    Args:
        v1 (torch.Tensor): Premier vecteur (référence ou à comparer).
        v2 (torch.Tensor): Deuxième vecteur.
        ref (str): Référence pour le dénominateur : 'v1' ou 'v2'.
        epsilon (float): Petite valeur pour éviter la division par zéro.

    Returns:
        float: Erreur relative maximale.
    """
    abs_diff = torch.abs(v1 - v2)
    
    if ref == 'v1':
        denominator = torch.abs(v1)
    elif ref == 'v2':
        denominator = torch.abs(v2)
    else:
        raise ValueError("ref doit être 'v1' ou 'v2'")
    
    relative_error = abs_diff / (denominator + epsilon)
    return relative_error.max().item()


# Fonction pour tester l'invariance par rotation
def test_scattering_rotation_invariance(data_list, index_to_test, device, grid, scattering, sigma, integral_powers, plot_positions=False, relative_error_threshold=1e-3):
    """
    Teste l'invariance par rotation des coefficients de scattering pour une molécule donnée en utilisant l'erreur relative.

    Args:
        data_list: Liste des dictionnaires de molécules.
        index_to_test: Index de la molécule à tester.
        device: torch.device.
        grid, scattering,, sigma, integral_powers: Paramètres pour le calcul des coefficients.
        plot_positions: Boolean pour activer/désactiver le tracé 3D des positions (défaut False).
        relative_error_threshold: Seuil pour l'erreur relative (défaut 1e-3).
    """
    # Extraire la molécule
    molecule = data_list[index_to_test:index_to_test+1]
    
    # Créer le dataset non rotaté
    dataset_not_rotated = XYZDataset_V2(
        data_list=molecule,
        return_energy=True
    )
    
    # Générer un axe de rotation (points A et B) et un angle aléatoire
    axis_point_a = torch.Tensor(np.random.uniform(-0.5, 0.5, size=(3,))).to(device)
    axis_point_b = torch.Tensor(np.random.uniform(-0.5, 0.5, size=(3,))).to(device)
    angle_degrees = 90  # Angle entre 0 et 360 degrés

    # Appliquer la rotation
    molecule_rotated = rotate_molecule(molecule[0], axis_point_a, axis_point_b, angle_degrees)
    
    # Créer le dataset rotaté
    dataset_rotated = XYZDataset_V2(
        data_list=[molecule_rotated],
        return_energy=True
    )
    
    # Calculer les coefficients de scattering pour les deux datasets
    order_0_not_rot, orders_1_and_2_not_rot = compute_scattering_coefficients(
        dataset_not_rotated, grid, scattering, sigma, integral_powers, device
    )
    order_0_rot, orders_1_and_2_rot = compute_scattering_coefficients(
        dataset_rotated, grid, scattering, sigma, integral_powers, device
    )
    
    # Calculer les erreurs relatives maximales
    order_0_rel_error = compute_max_relative_error(order_0_not_rot, order_0_rot, ref='v1')
    orders_1_and_2_rel_error = compute_max_relative_error(orders_1_and_2_not_rot, orders_1_and_2_rot, ref='v1')

    # Vérifier l'invariance
    invariance_check = "PASS" if max(order_0_rel_error, orders_1_and_2_rel_error) < relative_error_threshold else "FAIL"
    
    # Horodatage
    timestamp = datetime.datetime.now().strftime("%H:%M:%S %Y-%m-%d")
    
    # Afficher le rapport
    print(colored(f"\n=== Scattering Rotation Invariance Test Report ({timestamp}) ===", 'blue', attrs=['bold']))
    print(colored(f"Molecule ID: {molecule[0]['Id']}", 'cyan'))
    print("\n" + tabulate(
        [["Max Relative Error (Order 0)", f"{order_0_rel_error:.6e}"],
         ["Max Relative Error (Orders 1 and 2)", f"{orders_1_and_2_rel_error:.6e}"],
         ["Invariance Check", colored(invariance_check, 'green' if invariance_check == "PASS" else 'red')]],
        headers=["Metric", "Value"], tablefmt="pretty"
    ))
    
    print(colored("\nRotation Details:", 'cyan'))
    print(tabulate([
        ["Point A (X, Y, Z)", f"({axis_point_a[0].cpu().item():.6f}, {axis_point_a[1].cpu().item():.6f}, {axis_point_a[2].cpu().item():.6f})"],
        ["Point B (X, Y, Z)", f"({axis_point_b[0].cpu().item():.6f}, {axis_point_b[1].cpu().item():.6f}, {axis_point_b[2].cpu().item():.6f})"],
        ["Angle (degrees)", f"{angle_degrees:.6f}"]
    ], headers=["Parameter", "Value"], tablefmt="pretty"))
    
    # Comparaison des positions
    old_positions = molecule[0]['Atoms_DataFrame'][['X', 'Y', 'Z']].values
    new_positions = molecule_rotated['Atoms_DataFrame'][['X', 'Y', 'Z']].values
    diff_positions = np.round(new_positions - old_positions, decimals=6)
    df_positions = pd.DataFrame({
        'X_old': old_positions[:, 0],
        'Y_old': old_positions[:, 1],
        'Z_old': old_positions[:, 2],
        'X_new': new_positions[:, 0],
        'Y_new': new_positions[:, 1],
        'Z_new': new_positions[:, 2],
        'ΔX': diff_positions[:, 0],
        'ΔY': diff_positions[:, 1],
        'ΔZ': diff_positions[:, 2]
    })
    
    
    # Affichage du tableau des distances paires d'atomes
    n_atoms = old_positions.shape[0]
    distance_rows = []
    for i in range(n_atoms):
        for j in range(i+1, n_atoms):
            d_old = np.linalg.norm(old_positions[i] - old_positions[j])
            d_new = np.linalg.norm(new_positions[i] - new_positions[j])
            diff = d_new - d_old
            distance_rows.append([
                f"{i+1}-{j+1}",
                f"{d_old:.6f}",
                f"{d_new:.6f}",
                f"{diff:.6f}"
            ])
    print(colored("\nAtom Pairwise Distances (Original, Rotated, Difference):", 'cyan'))
    print(tabulate(distance_rows, headers=["Atom Pair", "Old Distance", "New Distance", "Diff"], tablefmt="pretty"))

    # Tracé des positions si activé
    if plot_positions:
        fig = plt.figure(figsize=(12, 6))
        
        ax1 = fig.add_subplot(121, projection='3d')
        ax1.scatter(old_positions[:, 0], old_positions[:, 1], old_positions[:, 2], c='blue', label='Original')
        ax1.set_title('Original Positions')
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')
        ax1.legend()
        
        ax2 = fig.add_subplot(122, projection='3d')
        ax2.scatter(new_positions[:, 0], new_positions[:, 1], new_positions[:, 2], c='green', label='Rotated')
        ax2.set_title('Rotated Positions')
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.set_zlabel('Z')
        ax2.legend()
        
        plt.tight_layout()
        plt.show()
    
    print(colored("\n=== End of Report ===", 'blue', attrs=['bold']))
    
    return invariance_check, order_0_rel_error, orders_1_and_2_rel_error
