import numpy as np
import pandas as pd

# Liste fixe
ALL_SYMBOLS = ['H', 'C', 'O', 'N', 'S', 'Cl']

# Données atomiques étendues avec plus de propriétés
atomic_properties = {
    'H': {'Z': 1, 'mass': 1.0079, 'electronegativity': 2.20, 'radius': 0.53, 'ionization_energy': 13.5984},
    'C': {'Z': 6, 'mass': 12.0107, 'electronegativity': 2.55, 'radius': 0.77, 'ionization_energy': 11.2603},
    'O': {'Z': 8, 'mass': 15.999, 'electronegativity': 3.44, 'radius': 0.73, 'ionization_energy': 13.6181},
    'N': {'Z': 7, 'mass': 14.0067, 'electronegativity': 3.04, 'radius': 0.70, 'ionization_energy': 14.5341},
    'S': {'Z': 16, 'mass': 32.065, 'electronegativity': 2.58, 'radius': 1.04, 'ionization_energy': 10.3600},
    'Cl': {'Z': 17, 'mass': 35.453, 'electronegativity': 3.16, 'radius': 0.99, 'ionization_energy': 12.9676}
}

MAX_ATOMS = 23

def compute_coulomb_matrix(coords, charges):
    n = len(coords)
    M = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                M[i, j] = 0.5 * charges[i] ** 2.4
            else:
                dist = np.linalg.norm(coords[i] - coords[j])
                M[i, j] = charges[i] * charges[j] / dist if dist > 1e-8 else 0.0
    return M






def extract_features_and_ids(List_Data, all_symbols=ALL_SYMBOLS, max_atoms=MAX_ATOMS):
    ids = []
    feature_list = []
    coulomb_matrices = []

    for molecule in List_Data:
        ids.append(molecule['Id'])
        atoms = molecule['Atoms_List']

        symbols = [atom['Symbol'] for atom in atoms]
        coords = np.array([[atom['X'], atom['Y'], atom['Z']] for atom in atoms])

        Zs = np.array([atomic_properties[sym]['Z'] for sym in symbols])
        n_atoms = len(atoms)

        # Matrice de Coulomb
        CM = compute_coulomb_matrix(coords, Zs)
        CM_padded = np.zeros((max_atoms, max_atoms))
        CM_padded[:n_atoms, :n_atoms] = CM
        coulomb_matrices.append(CM_padded)

        # Features dérivées de la matrice de Coulomb
        # 1. Eigenvalues et vecteurs propres (triés par valeur propre décroissante)
        eigenvalues, eigenvectors = np.linalg.eig(CM)
        eigenvalues = eigenvalues.real
        eigenvectors = eigenvectors.real
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        eigenvalues_padded = np.zeros(max_atoms)
        eigenvalues_padded[:len(eigenvalues)] = eigenvalues

        # Tri des coefficients de chaque vecteur propre (ordre croissant)
        sorted_eigenvectors = []
        for i in range(min(max_atoms, eigenvectors.shape[1])):
            vec = eigenvectors[:, i]
            sorted_vec = np.sort(vec)
            padded_vec = np.zeros(max_atoms)
            padded_vec[:len(sorted_vec)] = sorted_vec
            sorted_eigenvectors.append(padded_vec)
        while len(sorted_eigenvectors) < max_atoms:
            sorted_eigenvectors.append(np.zeros(max_atoms))

        # 2. Trace (somme des éléments diagonaux)
        trace = np.trace(CM)

        # 3. Norme de Frobenius
        frobenius_norm = np.linalg.norm(CM, 'fro')

        # 4. Moyenne, max et min des éléments hors diagonale (valeurs absolues)
        off_diagonal = np.abs(CM[np.triu_indices(n_atoms, k=1)])
        mean_off_diagonal = np.mean(off_diagonal) if len(off_diagonal) > 0 else 0.0
        max_off_diagonal = np.max(off_diagonal) if len(off_diagonal) > 0 else 0.0
        min_off_diagonal = np.min(off_diagonal) if len(off_diagonal) > 0 else 0.0

        # Dictionnaire des features (nombre fixe : MAX_ATOMS + 4 + MAX_ATOMS*MAX_ATOMS)
        feature_dict = {
            'Num_Atoms': n_atoms,
            'Trace': trace,
            'Frobenius_Norm': frobenius_norm,
            'Mean_Off_Diagonal': mean_off_diagonal,
            'Max_Off_Diagonal': max_off_diagonal,
            'Min_Off_Diagonal': min_off_diagonal
        }
        # Ajout des eigenvalues comme features individuelles
        for i in range(max_atoms):
            feature_dict[f'Eigenvalue_{i+1}'] = eigenvalues_padded[i]
        # Ajout des coefficients triés de chaque vecteur propre comme features individuelles
        for i in range(max_atoms):
            for j in range(max_atoms):
                feature_dict[f'Sorted_Eigenvector_{i+1}_Coeff_{j+1}'] = sorted_eigenvectors[i][j]

        feature_list.append(feature_dict)

    sorted_ids = sorted(ids)
    features_sorted = [feature_list[ids.index(id)] for id in sorted_ids]
    features_df = pd.DataFrame(features_sorted, index=sorted_ids).fillna(0).sort_index()

    features_array = features_df.values

    return sorted_ids, features_array


















def extract_extra_features_and_ids(List_Data, all_symbols=ALL_SYMBOLS, max_atoms=MAX_ATOMS):
    ids = []
    feature_list = []
    coulomb_matrices = []

    for molecule in List_Data:
        ids.append(molecule['Id'])
        atoms = molecule['Atoms_List']

        symbols = [atom['Symbol'] for atom in atoms]
        coords = np.array([[atom['X'], atom['Y'], atom['Z']] for atom in atoms])

        Zs = np.array([atomic_properties[sym]['Z'] for sym in symbols])
        masses = np.array([atomic_properties[sym]['mass'] for sym in symbols])
        electronegativities = np.array([atomic_properties[sym]['electronegativity'] for sym in symbols])
        radii = np.array([atomic_properties[sym]['radius'] for sym in symbols])
        ionization_energies = np.array([atomic_properties[sym]['ionization_energy'] for sym in symbols])

        # Centre de masse (barycentre)
        barycentre = np.average(coords, axis=0, weights=masses)
        coords_centered = coords - barycentre
        distances = np.linalg.norm(coords_centered, axis=1)

        # Distances interatomiques
        pairwise_dists = []
        for i in range(len(coords_centered)):
            for j in range(i+1, len(coords_centered)):
                pairwise_dists.append(np.linalg.norm(coords_centered[i] - coords_centered[j]))
        pairwise_dists = np.array(pairwise_dists) if pairwise_dists else np.array([0.0])

        # Matrice de Coulomb
        CM = compute_coulomb_matrix(coords, Zs)
        CM_padded = np.zeros((max_atoms, max_atoms))
        n_atoms = len(atoms)
        CM_padded[:n_atoms, :n_atoms] = CM
        coulomb_matrices.append(CM_padded)

        # Features dérivées de la matrice de Coulomb
        # 1. Eigenvalues (triées et remplies avec des zéros jusqu'à MAX_ATOMS)
        eigenvalues, eigenvectors = np.linalg.eig(CM)
        eigenvalues = eigenvalues.real
        eigenvectors = eigenvectors.real
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        eigenvalues_padded = np.zeros(max_atoms)
        eigenvalues_padded[:len(eigenvalues)] = eigenvalues

        # 1b. Sorted eigenvector coefficients (for each eigenvector, sort its coefficients)
        sorted_eigenvectors = []
        for i in range(min(max_atoms, eigenvectors.shape[1])):
            vec = eigenvectors[:, i]
            sorted_vec = np.sort(vec)  # ascending sort
            padded_vec = np.zeros(max_atoms)
            padded_vec[:len(sorted_vec)] = sorted_vec
            sorted_eigenvectors.append(padded_vec)
        # If less than max_atoms eigenvectors, pad with zeros
        while len(sorted_eigenvectors) < max_atoms:
            sorted_eigenvectors.append(np.zeros(max_atoms))

        # 2. Trace (somme des éléments diagonaux)
        trace = np.trace(CM)

        # 3. Norme de Frobenius
        frobenius_norm = np.linalg.norm(CM, 'fro')

        # 4. Moyenne, max et min des éléments hors diagonale (valeurs absolues)
        off_diagonal = np.abs(CM[np.triu_indices(n_atoms, k=1)])
        mean_off_diagonal = np.mean(off_diagonal) if len(off_diagonal) > 0 else 0.0
        max_off_diagonal = np.max(off_diagonal) if len(off_diagonal) > 0 else 0.0
        min_off_diagonal = np.min(off_diagonal) if len(off_diagonal) > 0 else 0.0

        # Moment d'inertie
        inertia_tensor = np.zeros((3, 3))
        for i, coord in enumerate(coords_centered):
            x, y, z = coord
            mass = masses[i]
            inertia_tensor[0, 0] += mass * (y**2 + z**2)
            inertia_tensor[1, 1] += mass * (x**2 + z**2)
            inertia_tensor[2, 2] += mass * (x**2 + y**2)
            inertia_tensor[0, 1] -= mass * x * y
            inertia_tensor[0, 2] -= mass * x * z
            inertia_tensor[1, 2] -= mass * y * z
        inertia_tensor[1, 0] = inertia_tensor[0, 1]
        inertia_tensor[2, 0] = inertia_tensor[0, 2]
        inertia_tensor[2, 1] = inertia_tensor[1, 2]
        moments_of_inertia = np.linalg.eigvals(inertia_tensor)

        # Angles entre atomes (approximation via triplets)
        angles = []
        for i in range(len(coords)):
            for j in range(i+1, len(coords)):
                for k in range(j+1, len(coords)):
                    v1 = coords[j] - coords[i]
                    v2 = coords[k] - coords[i]
                    norm_v1 = np.linalg.norm(v1)
                    norm_v2 = np.linalg.norm(v2)
                    if norm_v1 > 1e-8 and norm_v2 > 1e-8:
                        cos_angle = np.dot(v1, v2) / (norm_v1 * norm_v2)
                        cos_angle = np.clip(cos_angle, -1.0, 1.0)
                        angle = np.degrees(np.arccos(cos_angle))
                        angles.append(angle)
        angles = np.array(angles) if angles else np.array([0.0])

        # Features électrostatiques
        dipole_moment = np.sum([Zs[i] * coords_centered[i] for i in range(n_atoms)], axis=0)
        dipole_magnitude = np.linalg.norm(dipole_moment)

        # Initialisation du dictionnaire avec toutes les features
        feature_dict = {
            # 50 features générales
            'Num_Atoms': n_atoms,
            'Total_Mass': np.sum(masses),
            'Mean_Mass': np.mean(masses),
            'Var_Mass': np.var(masses),
            'Min_Mass': np.min(masses),
            'Max_Mass': np.max(masses),
            'Skew_Mass': pd.Series(masses).skew() if n_atoms > 2 else 0.0,
            'Kurtosis_Mass': pd.Series(masses).kurtosis() if n_atoms > 3 else 0.0,

            'Total_Z': np.sum(Zs),
            'Mean_Z': np.mean(Zs),
            'Var_Z': np.var(Zs),
            'Min_Z': np.min(Zs),
            'Max_Z': np.max(Zs),
            'Skew_Z': pd.Series(Zs).skew() if n_atoms > 2 else 0.0,
            'Kurtosis_Z': pd.Series(Zs).kurtosis() if n_atoms > 3 else 0.0,

            'Total_Electronegativity': np.sum(electronegativities),
            'Mean_Electronegativity': np.mean(electronegativities),
            'Var_Electronegativity': np.var(electronegativities),
            'Min_Electronegativity': np.min(electronegativities),
            'Max_Electronegativity': np.max(electronegativities),
            'Skew_Electronegativity': pd.Series(electronegativities).skew() if n_atoms > 2 else 0.0,
            'Kurtosis_Electronegativity': pd.Series(electronegativities).kurtosis() if n_atoms > 3 else 0.0,

            'Total_Radius': np.sum(radii),
            'Mean_Radius': np.mean(radii),
            'Var_Radius': np.var(radii),
            'Min_Radius': np.min(radii),
            'Max_Radius': np.max(radii),
            'Skew_Radius': pd.Series(radii).skew() if n_atoms > 2 else 0.0,
            'Kurtosis_Radius': pd.Series(radii).kurtosis() if n_atoms > 3 else 0.0,

            'Total_Ionization_Energy': np.sum(ionization_energies),
            'Mean_Ionization_Energy': np.mean(ionization_energies),
            'Var_Ionization_Energy': np.var(ionization_energies),
            'Min_Ionization_Energy': np.min(ionization_energies),
            'Max_Ionization_Energy': np.max(ionization_energies),
            'Skew_Ionization_Energy': pd.Series(ionization_energies).skew() if n_atoms > 2 else 0.0,
            'Kurtosis_Ionization_Energy': pd.Series(ionization_energies).kurtosis() if n_atoms > 3 else 0.0,

            'Max_Distance_Barycentre': np.max(distances),
            'Mean_Distance_Barycentre': np.mean(distances),
            'Std_Distance_Barycentre': np.std(distances),
            'Min_Distance_Barycentre': np.min(distances),
            'Skew_Distance_Barycentre': pd.Series(distances).skew() if n_atoms > 2 else 0.0,

            'Mean_Pairwise_Distance': np.mean(pairwise_dists),
            'Var_Pairwise_Distance': np.var(pairwise_dists),
            'Min_Pairwise_Distance': np.min(pairwise_dists),
            'Max_Pairwise_Distance': np.max(pairwise_dists),
            'Sum_Pairwise_Distance': np.sum(pairwise_dists),
            'Std_Pairwise_Distance': np.std(pairwise_dists),
            'Skew_Pairwise_Distance': pd.Series(pairwise_dists).skew() if len(pairwise_dists) > 2 else 0.0,
            'Kurtosis_Pairwise_Distance': pd.Series(pairwise_dists).kurtosis() if len(pairwise_dists) > 3 else 0.0,

            'Dipole_Magnitude': dipole_magnitude,
            'Moment_Inertia_X': moments_of_inertia[0] if n_atoms > 1 else 0.0,
            'Moment_Inertia_Y': moments_of_inertia[1] if n_atoms > 1 else 0.0,
            'Moment_Inertia_Z': moments_of_inertia[2] if n_atoms > 1 else 0.0,
            'Mean_Angle': np.mean(angles),
            'Var_Angle': np.var(angles),
            'Min_Angle': np.min(angles),
            'Max_Angle': np.max(angles),
            'Skew_Angle': pd.Series(angles).skew() if len(angles) > 2 else 0.0,
            'Kurtosis_Angle': pd.Series(angles).kurtosis() if len(angles) > 3 else 0.0,

            # Features de la matrice de Coulomb (4 + MAX_ATOMS)
            'Trace': trace,
            'Frobenius_Norm': frobenius_norm,
            'Mean_Off_Diagonal': mean_off_diagonal,
            'Max_Off_Diagonal': max_off_diagonal,
            'Min_Off_Diagonal': min_off_diagonal
        }

        # Ajout des eigenvalues comme features individuelles
        for i in range(max_atoms):
            feature_dict[f'Eigenvalue_{i+1}'] = eigenvalues_padded[i]

        # Ajout des coefficients triés de chaque vecteur propre comme features individuelles
        for i in range(max_atoms):
            for j in range(max_atoms):
                feature_dict[f'Sorted_Eigenvector_{i+1}_Coeff_{j+1}'] = sorted_eigenvectors[i][j]

        # Compte par symbole (6 features: Num_H, Num_C, ..., Num_Cl)
        for sym in all_symbols:
            feature_dict[f'Num_{sym}'] = symbols.count(sym)

        # Features par type d'atome (6 symbols × 5 features = 30 features)
        for sym in all_symbols:
            mask = np.array(symbols) == sym
            feature_dict[f'Mean_Dist_{sym}'] = np.mean(distances[mask]) if np.sum(mask) > 0 else 0.0
            feature_dict[f'Std_Dist_{sym}'] = np.std(distances[mask]) if np.sum(mask) > 1 else 0.0
            feature_dict[f'Mean_Electroneg_{sym}'] = np.mean(electronegativities[mask]) if np.sum(mask) > 0 else 0.0
            feature_dict[f'Mean_Radius_{sym}'] = np.mean(radii[mask]) if np.sum(mask) > 0 else 0.0
            feature_dict[f'Mean_Ionization_{sym}'] = np.mean(ionization_energies[mask]) if np.sum(mask) > 0 else 0.0

        feature_list.append(feature_dict)

    sorted_ids = sorted(ids)
    features_sorted = [feature_list[ids.index(id)] for id in sorted_ids]
    features_df = pd.DataFrame(features_sorted, index=sorted_ids).fillna(0).sort_index()

    features_array = features_df.values

    return sorted_ids, features_array


