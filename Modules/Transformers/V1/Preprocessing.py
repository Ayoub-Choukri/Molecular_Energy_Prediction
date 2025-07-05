import os 
import pandas as pd
def Parse_Xyz_File(Path):
    Atoms_List = []
    with open(Path, 'r') as File:
        Lines_List = File.readlines()

    Natoms = int(Lines_List[0].strip())
    # Ignorer la ligne des propriétés (ligne 2)
    for I in range(2, 2 + Natoms):
        Parts = Lines_List[I].split()
        Atom_Dict = {
            "Symbol": Parts[0],
            "X": float(Parts[1]),
            "Y": float(Parts[2]),
            "Z": float(Parts[3])
        }
        Atoms_List.append(Atom_Dict)
    return Atoms_List


import os
import csv


def Load_Data(Data_XYZ_Folder, Energy_CSV_Path):
    # Lire le CSV d'énergie pour faire un mapping Id -> Energy
    Id_To_Energy = {}
    with open(Energy_CSV_Path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            Id_To_Energy[int(row['id'])] = float(row['energy'])

    List_Data = []
    for file in os.listdir(Data_XYZ_Folder):
        if file.endswith('.xyz'):
            # Extraire l'id à partir du nom : "id_{id}.xyz"
            base_name = os.path.splitext(file)[0]  # id_{id}
            try:
                Id_Str = base_name.split('_')[1]   # partie après 'id_'
                Id = int(Id_Str)
            except (IndexError, ValueError):
                print(f"Warning: Impossible d'extraire l'id depuis le fichier {file}")
                continue

            File_Path = os.path.join(Data_XYZ_Folder, file)
            Atoms_List = Parse_Xyz_File(File_Path)
            Energy = Id_To_Energy.get(Id, None)  # None si absent

            # Création du DataFrame Atoms_DataFrame
            Atoms_DataFrame = pd.DataFrame(Atoms_List)

            List_Data.append({
                'Id': Id,
                'Atoms_List': Atoms_List,
                'Atoms_DataFrame': Atoms_DataFrame,
                'Energy': Energy
            })

    return List_Data





# Same as Load_Data but without the energy mapping
def Load_Test_Data(Data_XYZ_Folder):
    List_Data = []
    for file in os.listdir(Data_XYZ_Folder):
        if file.endswith('.xyz'):
            # Extraire l'id à partir du nom : "id_{id}.xyz"
            base_name = os.path.splitext(file)[0]  # id_{id}
            try:
                Id_Str = base_name.split('_')[1]   # prendre la partie après 'id_'
                Id = int(Id_Str)
            except (IndexError, ValueError):
                print(f"Warning: Impossible d'extraire l'id depuis le fichier {file}")
                continue

            File_Path = os.path.join(Data_XYZ_Folder, file)
            Atoms_List = Parse_Xyz_File(File_Path)  

            Atoms_DataFrame = pd.DataFrame(Atoms_List)

            List_Data.append({
                'Id': Id,
                'Atoms_List': Atoms_List,
                'Atoms_DataFrame': Atoms_DataFrame
            })

    return List_Data




import py3Dmol
from collections import Counter

def get_chemical_formula(atoms_list):
    """
    Génère la formule chimique à partir de la liste des atomes.
    Args:
        atoms_list: Liste de dictionnaires contenant les symboles des atomes.
    Returns:
        str: Formule chimique (ex. 'H2O', 'CH4').
    """
    # Compter les occurrences de chaque symbole
    symbol_counts = Counter(atom['Symbol'] for atom in atoms_list)
    # Construire la formule en suivant l'ordre conventionnel (C, H, autres par ordre alphabétique)
    elements = sorted(symbol_counts.keys(), key=lambda x: (x != 'C', x != 'H', x))
    formula = ''
    for element in elements:
        count = symbol_counts[element]
        formula += element
        if count > 1:
            formula += str(count)
    return formula

def Display_Molecule_From_Atom_List(Atoms_List, Energy=None, Width=600, Height=600, Background_Color='yellow'):
    """
    Affiche une molécule en 3D avec son nom chimique et son énergie.
    
    Args:
        Atoms_List: Liste de dictionnaires [{'Symbol': str, 'X': float, 'Y': float, 'Z': float}, ...]
        Energy: Énergie de la molécule (float, optionnel)
        Width: Largeur de la visualisation (int, défaut=600)
        Height: Hauteur de la visualisation (int, défaut=600)
        Background_Color: Couleur de fond (str, défaut='yellow')
    """
    # Nombre d'atomes
    Natoms = len(Atoms_List)
    
    # Construire le contenu XYZ sous forme de chaîne
    header = f"{Natoms}\nProperties=species:S:1:pos:R:3 pbc=\"F F F\"\n"
    lines = [
        f"{atom['Symbol']} {atom['X']:.6f} {atom['Y']:.6f} {atom['Z']:.6f}"
        for atom in Atoms_List
    ]
    xyz_data = header + "\n".join(lines)
    
    # Création du viewer
    view = py3Dmol.view(width=Width, height=Height)
    view.addModel(xyz_data, 'xyz')
    
    # Style de base : sphères + sticks
    view.setStyle({}, {
        "stick": {"radius": 0.15},
        "sphere": {"scale": 0.3}
    })
    
    # Ajouter des labels pour chaque atome
    for atom in Atoms_List:
        symbol = atom['Symbol']
        x, y, z = atom['X'], atom['Y'], atom['Z']
        view.addLabel(symbol, {
            'fontSize': 12,
            'fontColor': 'black',
            'backgroundColor': 'white',
            'backgroundOpacity': 0.7,
            'borderRadius': 3,
            'position': {'x': x, 'y': y, 'z': z},
            'alignment': 'center'
        })
    
    # Obtenir la formule chimique
    chemical_formula = get_chemical_formula(Atoms_List)
    
    # Calculer la position pour le label de la formule chimique (au-dessus de la molécule)
    if Atoms_List:
        max_z = max(atom['Z'] for atom in Atoms_List)
        center_x = sum(atom['X'] for atom in Atoms_List) / Natoms
        center_y = sum(atom['Y'] for atom in Atoms_List) / Natoms
        formula_position = {'x': center_x, 'y': center_y, 'z': max_z + 2.0}
        
        # Ajouter le label pour la formule chimique
        view.addLabel(f"Formula: {chemical_formula}", {
            'fontSize': 14,
            'fontColor': 'blue',
            'backgroundColor': 'white',
            'backgroundOpacity': 0.7,
            'borderRadius': 3,
            'position': formula_position,
            'alignment': 'center'
        })
    
    # Ajouter le label pour l'énergie si fournie
    if Energy is not None:
        energy_position = {'x': center_x, 'y': center_y, 'z': max_z + 3.5}
        view.addLabel(f"Energy: {Energy:.2f} eV", {
            'fontSize': 14,
            'fontColor': 'red',
            'backgroundColor': 'white',
            'backgroundOpacity': 0.7,
            'borderRadius': 3,
            'position': energy_position,
            'alignment': 'center'
        })
    
    # Finalisation
    view.setBackgroundColor(Background_Color)
    view.zoomTo()
    view.show()
    




