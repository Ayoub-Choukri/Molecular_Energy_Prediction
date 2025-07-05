import torch
import pandas as pd
import numpy as np
from tabulate import tabulate
from termcolor import colored
import datetime
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from Dataloaders_Preprocessing import MoleculeDataset_Transformer


def translate_molecule(data, translation_vector):
    if hasattr(translation_vector, 'numpy'):
        translation_vector = translation_vector.cpu().numpy()
    
    df = data['Atoms_DataFrame']
    precision = max((len(str(abs(float(val))).split('.')[-1]) 
                    for val in df[['X', 'Y', 'Z']].values.flatten() 
                    if abs(float(val)) > 0), default=6)
    
    translation_vector = np.round(translation_vector, decimals=precision)
    df_translated = data['Atoms_DataFrame'].copy()
    df_translated[['X', 'Y', 'Z']] = np.round(df_translated[['X', 'Y', 'Z']] + translation_vector, decimals=precision)
    
    translated_atoms = []
    for atom in data['Atoms_List']:
        new_atom = atom.copy()
        new_atom['X'] = round(float(new_atom['X']) + float(translation_vector[0]), precision)
        new_atom['Y'] = round(float(new_atom['Y']) + float(translation_vector[1]), precision)
        new_atom['Z'] = round(float(new_atom['Z']) + float(translation_vector[2]), precision)
        translated_atoms.append(new_atom)
    
    translated_data = data.copy()
    translated_data['Atoms_DataFrame'] = df_translated
    translated_data['Atoms_List'] = translated_atoms
    
    return translated_data

def test_translation_invariance(model, data_list, index_to_test, device, plot_positions=False):
    """
    Test the translation invariance of the model on a given molecule with enhanced output.

    Args:
        model: The Transformer model.
        data_list: List of molecule dictionaries containing 'Atoms_DataFrame' and 'Energy'.
        index_to_test: Index of the molecule to test.
        device: torch.device.
        plot_positions: Boolean to enable/disable 3D plotting of atom positions (default False).
    """
    # Extract the molecule
    molecule = data_list[index_to_test:index_to_test+1]

    # Non-translated dataset
    dataset_not_translated = MoleculeDataset_Transformer(
        Data_List=molecule,
        Nb_Atoms_Max_In_Molecule=23,
        Return_Energies=True,
        transform=None
    )

    # Translate the molecule
    translation_vector = torch.Tensor(np.random.uniform(-50, 50, size=(3,))).to(device)
    molecule_translated = translate_molecule(molecule[0], translation_vector)

    dataset_translated = MoleculeDataset_Transformer(
        Data_List=[molecule_translated],
        Nb_Atoms_Max_In_Molecule=23,
        Return_Energies=True,
        transform=None
    )

    # Predictions
    model.eval()
    model.to(device)
    with torch.no_grad():
        # Non-translated
        ids_not_translated, symbols_not_translated, positions_not_translated, mask_not_translated, energies_not_translated = dataset_not_translated[0]
        symbols_not_translated = symbols_not_translated.unsqueeze(0).to(device)
        positions_not_translated = positions_not_translated.unsqueeze(0).to(device)
        mask_not_translated = mask_not_translated.unsqueeze(0).to(device)
        predicted_energy_not_translated = model(symbols_not_translated, positions_not_translated, mask=mask_not_translated).item()

        # Translated
        ids_translated, symbols_translated, positions_translated, mask_translated, energies_translated = dataset_translated[0]
        symbols_translated = symbols_translated.unsqueeze(0).to(device)
        positions_translated = positions_translated.unsqueeze(0).to(device)
        mask_translated = mask_translated.unsqueeze(0).to(device)
        predicted_energy_translated = model(symbols_translated, positions_translated, mask=mask_translated).item()

    # Calculate difference
    energy_diff = abs(predicted_energy_not_translated - predicted_energy_translated)
    invariance_check = "PASS" if energy_diff < 1e-4 else "FAIL"

    # Current timestamp
    timestamp = datetime.datetime.now().strftime("%H:%M:%S %Y-%m-%d")

    # Print enhanced output
    print(colored(f"\n=== Translation Invariance Test Report ({timestamp}) ===", 'blue', attrs=['bold']))
    print(colored(f"Molecule ID: {molecule[0]['Id']}", 'cyan'))
    print("\n" + tabulate(
        [["Predicted Energy (Not Translated)", f"{predicted_energy_not_translated:.6f} eV"],
         ["Predicted Energy (Translated)", f"{predicted_energy_translated:.6f} eV"],
         ["True Energy", f"{energies_translated.item():.6f} eV"],
         ["Energy Difference", f"{energy_diff:.6f} eV"],
         ["Invariance Check", colored(invariance_check, 'green' if invariance_check == "PASS" else 'red')]],
        headers=["Metric", "Value"], tablefmt="pretty"
    ))

    print(colored("\nTranslation Vector:", 'cyan'))
    print(tabulate([["X", f"{translation_vector[0].cpu().item():.6f}"],
                    ["Y", f"{translation_vector[1].cpu().item():.6f}"],
                    ["Z", f"{translation_vector[2].cpu().item():.6f}"]],
                   headers=["Axis", "Value"], tablefmt="pretty"))

    # Position comparison split into three parts
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

    # Plotting positions if enabled
    if plot_positions:
        fig = plt.figure(figsize=(12, 6))
        
        # Original positions
        ax1 = fig.add_subplot(121, projection='3d')
        ax1.scatter(old_positions[:, 0], old_positions[:, 1], old_positions[:, 2], c='blue', label='Original')
        ax1.set_title('Original Positions')
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')
        ax1.legend()

        # Translated positions
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





def rotate_molecule(data, rotation_angle, axis=[0, 0, 1]):
    """
    Rotate the molecule coordinates around the specified axis by the given angle (in radians).

    Args:
        data: Dictionary containing 'Atoms_DataFrame' and 'Atoms_List'.
        rotation_angle: Angle of rotation in radians.
        axis: Rotation axis as a list or array [A, B, C] (default [0, 0, 1] for Z-axis).

    Returns:
        Dictionary with rotated coordinates in 'Atoms_DataFrame' and 'Atoms_List'.
    """
    df = data['Atoms_DataFrame'].copy()
    precision = max((len(str(abs(float(val))).split('.')[-1]) 
                    for val in df[['X', 'Y', 'Z']].values.flatten() 
                    if abs(float(val)) > 0), default=6)
    
    # Normalize the axis
    axis = np.array(axis) / np.linalg.norm(axis)
    u, v, w = axis
    
    # Rotation matrix for angle theta around axis (u, v, w)
    theta = rotation_angle
    c = np.cos(theta)
    s = np.sin(theta)
    t = 1 - c
    rotation_matrix = np.array([
        [c + u*u*t, u*v*t - w*s, u*w*t + v*s],
        [v*u*t + w*s, c + v*v*t, v*w*t - u*s],
        [w*u*t - v*s, w*v*t + u*s, c + w*w*t]
    ])
    
    # Extract and rotate coordinates
    coordinates = df[['X', 'Y', 'Z']].values
    rotated_coords = np.dot(coordinates, rotation_matrix.T)
    df_rotated = df.copy()
    df_rotated[['X', 'Y', 'Z']] = np.round(rotated_coords, decimals=precision)
    
    # Update Atoms_List with rotated coordinates
    rotated_atoms = []
    for atom in data['Atoms_List']:
        new_atom = atom.copy()
        coord = np.array([float(new_atom['X']), float(new_atom['Y']), float(new_atom['Z'])])
        rotated_coord = np.dot(coord, rotation_matrix.T)
        new_atom['X'] = round(float(rotated_coord[0]), precision)
        new_atom['Y'] = round(float(rotated_coord[1]), precision)
        new_atom['Z'] = round(float(rotated_coord[2]), precision)
        rotated_atoms.append(new_atom)
    
    # Create new dictionary with rotated data
    rotated_data = data.copy()
    rotated_data['Atoms_DataFrame'] = df_rotated
    rotated_data['Atoms_List'] = rotated_atoms
    
    return rotated_data
def rotation_matrix_3d(axis, theta):
    """
    Compute the 3D rotation matrix for a given axis and angle (in radians).
    """
    axis = axis / np.linalg.norm(axis)
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

def rotate_molecule_axis_points(data, axis_point_a, axis_point_b, angle_radians):
    """
    Rotate the molecule coordinates around the axis defined by two points (A and B) by the given angle (in radians).

    Args:
        data: Dictionary containing 'Atoms_DataFrame' and 'Atoms_List'.
        axis_point_a: numpy array or tensor, point A on the axis.
        axis_point_b: numpy array or tensor, point B on the axis.
        angle_radians: Angle of rotation in radians.

    Returns:
        Dictionary with rotated coordinates in 'Atoms_DataFrame' and 'Atoms_List'.
    """
    if hasattr(axis_point_a, 'numpy'):
        axis_point_a = axis_point_a.cpu().numpy()
    if hasattr(axis_point_b, 'numpy'):
        axis_point_b = axis_point_b.cpu().numpy()

    axis = axis_point_b - axis_point_a
    R = rotation_matrix_3d(axis, angle_radians)
    df = data['Atoms_DataFrame'].copy()
    precision = max((len(str(abs(float(val))).split('.')[-1]) 
                    for val in df[['X', 'Y', 'Z']].values.flatten() 
                    if abs(float(val)) > 0), default=6)

    # Rotate DataFrame
    positions = df[['X', 'Y', 'Z']].values
    positions_translated = positions - axis_point_a
    positions_rotated = (R @ positions_translated.T).T
    positions_final = positions_rotated + axis_point_a
    df_rotated = df.copy()
    df_rotated[['X', 'Y', 'Z']] = np.round(positions_final, decimals=precision)

    # Rotate Atoms_List
    rotated_atoms = []
    for atom in data['Atoms_List']:
        new_atom = atom.copy()
        pos = np.array([float(new_atom['X']), float(new_atom['Y']), float(new_atom['Z'])])
        pos_translated = pos - axis_point_a
        pos_rotated = R @ pos_translated
        pos_final = pos_rotated + axis_point_a
        new_atom['X'] = round(float(pos_final[0]), precision)
        new_atom['Y'] = round(float(pos_final[1]), precision)
        new_atom['Z'] = round(float(pos_final[2]), precision)
        rotated_atoms.append(new_atom)

    rotated_data = data.copy()
    rotated_data['Atoms_DataFrame'] = df_rotated
    rotated_data['Atoms_List'] = rotated_atoms
    return rotated_data

def test_rotation_invariance(model, data_list, index_to_test, device, plot_positions=False, error_threshold=1e-4):
    """
    Test the rotation invariance of the model on a given molecule, using a random axis defined by two points.

    Args:
        model: The Transformer model.
        data_list: List of molecule dictionaries containing 'Atoms_DataFrame' and 'Energy'.
        index_to_test: Index of the molecule to test.
        device: torch.device.
        plot_positions: Boolean to enable/disable 3D plotting of atom positions (default False).
        error_threshold: Threshold for absolute energy difference to pass invariance test (default 1e-4).
    """
    molecule = data_list[index_to_test:index_to_test+1]

    dataset_not_rotated = MoleculeDataset_Transformer(
        Data_List=molecule,
        Nb_Atoms_Max_In_Molecule=23,
        Return_Energies=True,
        transform=None
    )

    # Generate random axis points and angle
    axis_point_a = np.random.uniform(-1, 1, size=(3,))
    axis_point_b = np.random.uniform(-1, 1, size=(3,))
    angle_radians = np.random.uniform(0, 2 * np.pi)

    molecule_rotated = rotate_molecule_axis_points(molecule[0], axis_point_a, axis_point_b, angle_radians)

    dataset_rotated = MoleculeDataset_Transformer(
        Data_List=[molecule_rotated],
        Nb_Atoms_Max_In_Molecule=23,
        Return_Energies=True,
        transform=None
    )

    model.eval()
    model.to(device)
    with torch.no_grad():
        _, symbols_not_rotated, positions_not_rotated, mask_not_rotated, _ = dataset_not_rotated[0]
        symbols_not_rotated = symbols_not_rotated.unsqueeze(0).to(device)
        positions_not_rotated = positions_not_rotated.unsqueeze(0).to(device)
        mask_not_rotated = mask_not_rotated.unsqueeze(0).to(device)
        predicted_energy_not_rotated = model(symbols_not_rotated, positions_not_rotated, mask=mask_not_rotated).item()

        _, symbols_rotated, positions_rotated, mask_rotated, energies_rotated = dataset_rotated[0]
        symbols_rotated = symbols_rotated.unsqueeze(0).to(device)
        positions_rotated = positions_rotated.unsqueeze(0).to(device)
        mask_rotated = mask_rotated.unsqueeze(0).to(device)
        predicted_energy_rotated = model(symbols_rotated, positions_rotated, mask=mask_rotated).item()

    energy_diff = abs(predicted_energy_not_rotated - predicted_energy_rotated)
    invariance_check = "PASS" if energy_diff < error_threshold else "FAIL"

    print(colored(f"\n=== Rotation Invariance Test ===", 'blue', attrs=['bold']))
    print(colored(f"Molecule ID: {molecule[0]['Id']}", 'cyan'))
    print(tabulate(
        [["Predicted Energy (Not Rotated)", f"{predicted_energy_not_rotated:.6f} eV"],
         ["Predicted Energy (Rotated)", f"{predicted_energy_rotated:.6f} eV"],
         ["True Energy", f"{energies_rotated.item():.6f} eV"],
         ["Energy Difference", f"{energy_diff:.6f} eV"],
         ["Invariance Check", colored(invariance_check, 'green' if invariance_check == "PASS" else 'red')]],
        headers=["Metric", "Value"], tablefmt="pretty"
    ))

    print(colored("\nRotation Applied:", 'cyan'))
    print(tabulate([
        ["Point A (X, Y, Z)", f"({axis_point_a[0]:.6f}, {axis_point_a[1]:.6f}, {axis_point_a[2]:.6f})"],
        ["Point B (X, Y, Z)", f"({axis_point_b[0]:.6f}, {axis_point_b[1]:.6f}, {axis_point_b[2]:.6f})"],
        ["Axis (u, v, w)", f"[{(axis_point_b-axis_point_a)[0]:.6f}, {(axis_point_b-axis_point_a)[1]:.6f}, {(axis_point_b-axis_point_a)[2]:.6f}]"],
        ["Angle (radians)", f"{angle_radians:.6f}"],
        ["Angle (degrees)", f"{np.degrees(angle_radians):.2f}"]
    ], headers=["Parameter", "Value"], tablefmt="pretty"))

    old_positions = molecule[0]['Atoms_DataFrame'][['X', 'Y', 'Z']].values
    new_positions = molecule_rotated['Atoms_DataFrame'][['X', 'Y', 'Z']].values

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

    print(colored("\n=== End of Test ===", 'blue', attrs=['bold']))

import torch
import pandas as pd
import numpy as np
from tabulate import tabulate
from termcolor import colored
import datetime
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def permute_molecule(data):
    """
    Permute the order of atoms in the molecule coordinates randomly.
    """
    df = data['Atoms_DataFrame']
    precision = max((len(str(abs(float(val))).split('.')[-1]) 
                    for val in df[['X', 'Y', 'Z']].values.flatten() 
                    if abs(float(val)) > 0), default=6)
    
    # Generate a random permutation of indices
    indices = np.random.permutation(len(df))
    
    # Permute Atoms_DataFrame
    df_permuted = df.iloc[indices].copy()
    df_permuted.reset_index(drop=True, inplace=True)
    
    # Permute Atoms_List
    permuted_atoms = [data['Atoms_List'][i] for i in indices]
    
    # Create new dictionary with permuted data
    permuted_data = data.copy()
    permuted_data['Atoms_DataFrame'] = df_permuted
    permuted_data['Atoms_List'] = permuted_atoms
    
    return permuted_data

def test_permutation_invariance(model, data_list, index_to_test, device, plot_positions=False):
    """
    Test the permutation invariance of the model on a given molecule with enhanced output.

    Args:
        model: The Transformer model.
        data_list: List of molecule dictionaries containing 'Atoms_DataFrame' and 'Energy'.
        index_to_test: Index of the molecule to test.
        device: torch.device.
        plot_positions: Boolean to enable/disable 3D plotting of atom positions (default False).
    """
    # Extract the molecule
    molecule = data_list[index_to_test:index_to_test+1]

    # Non-permuted dataset
    dataset_not_permuted = MoleculeDataset_Transformer(
        Data_List=molecule,
        Nb_Atoms_Max_In_Molecule=23,
        Return_Energies=True,
        transform=None
    )

    # Permute the molecule
    molecule_permuted = permute_molecule(molecule[0])

    dataset_permuted = MoleculeDataset_Transformer(
        Data_List=[molecule_permuted],
        Nb_Atoms_Max_In_Molecule=23,
        Return_Energies=True,
        transform=None
    )

    # Predictions
    model.eval()
    model.to(device)
    with torch.no_grad():
        # Non-permuted
        ids_not_permuted, symbols_not_permuted, positions_not_permuted, mask_not_permuted, energies_not_permuted = dataset_not_permuted[0]
        symbols_not_permuted = symbols_not_permuted.unsqueeze(0).to(device)
        positions_not_permuted = positions_not_permuted.unsqueeze(0).to(device)
        mask_not_permuted = mask_not_permuted.unsqueeze(0).to(device)
        predicted_energy_not_permuted = model(symbols_not_permuted, positions_not_permuted, mask=mask_not_permuted).item()

        # Permuted
        ids_permuted, symbols_permuted, positions_permuted, mask_permuted, energies_permuted = dataset_permuted[0]
        symbols_permuted = symbols_permuted.unsqueeze(0).to(device)
        positions_permuted = positions_permuted.unsqueeze(0).to(device)
        mask_permuted = mask_permuted.unsqueeze(0).to(device)
        predicted_energy_permuted = model(symbols_permuted, positions_permuted, mask=mask_permuted).item()

    # Calculate difference
    energy_diff = abs(predicted_energy_not_permuted - predicted_energy_permuted)
    invariance_check = "PASS" if energy_diff < 1e-4 else "FAIL"

    # Current timestamp
    timestamp = datetime.datetime.now().strftime("%H:%M:%S %Y-%m-%d")

    # Print enhanced output
    print(colored(f"\n=== Permutation Invariance Test Report ({timestamp}) ===", 'blue', attrs=['bold']))
    print(colored(f"Molecule ID: {molecule[0]['Id']}", 'cyan'))
    print("\n" + tabulate(
        [["Predicted Energy (Not Permuted)", f"{predicted_energy_not_permuted:.6f} eV"],
         ["Predicted Energy (Permuted)", f"{predicted_energy_permuted:.6f} eV"],
         ["True Energy", f"{energies_permuted.item():.6f} eV"],
         ["Energy Difference", f"{energy_diff:.6f} eV"],
         ["Invariance Check", colored(invariance_check, 'green' if invariance_check == "PASS" else 'red')]],
        headers=["Metric", "Value"], tablefmt="pretty"
    ))

    # Display atom order before and after permutation
    old_order = [(i, atom['Symbol']) for i, atom in enumerate(molecule[0]['Atoms_List'])]
    new_order = [(i, atom['Symbol']) for i, atom in enumerate(molecule_permuted['Atoms_List'])]
    print(colored("\nAtom Order Before Permutation:", 'cyan'))
    print(tabulate(old_order, headers=["Index", "Symbol"], tablefmt="pretty", showindex=False))
    print(colored("\nAtom Order After Permutation:", 'cyan'))
    print(tabulate(new_order, headers=["Index", "Symbol"], tablefmt="pretty", showindex=False))

    # Position comparison
    old_positions = molecule[0]['Atoms_DataFrame'][['X', 'Y', 'Z']].values
    new_positions = molecule_permuted['Atoms_DataFrame'][['X', 'Y', 'Z']].values
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
    print(colored("\nAtom Position Comparison:", 'cyan'))
    # print(tabulate(df_positions, headers="keys", tablefmt="pretty", showindex=range(1, len(df_positions) + 1)))

    # Plotting positions if enabled
    if plot_positions:
        fig = plt.figure(figsize=(12, 6))
        
        # Original positions
        ax1 = fig.add_subplot(121, projection='3d')
        ax1.scatter(old_positions[:, 0], old_positions[:, 1], old_positions[:, 2], c='blue', label='Original')
        ax1.set_title('Original Positions')
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')
        ax1.legend()

        # Permuted positions (same coordinates, different order)
        ax2 = fig.add_subplot(122, projection='3d')
        ax2.scatter(new_positions[:, 0], new_positions[:, 1], new_positions[:, 2], c='green', label='Permuted')
        ax2.set_title('Permuted Positions')
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.set_zlabel('Z')
        ax2.legend()

        plt.tight_layout()
        plt.show()

    print(colored("\n=== End of Report ===", 'blue', attrs=['bold']))



