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

def Display_Molecule_From_Atom_List(Atoms_List, Width=600, Height=600, Background_Color='yellow'):
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
    
    # Ajouter des labels
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
    
    # Finalisation
    view.setBackgroundColor(Background_Color)
    view.zoomTo()
    view.show()




