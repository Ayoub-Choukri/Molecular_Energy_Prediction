import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import pandas as pd
def Compute_Predictions(Model, Dataset_Test, Device, batch_size=32):
    """
    Computes predictions on the test dataset using the Transformer model.

    Args:
        Model: The Transformer model.
        Dataset_Test: MoleculeDataset_Transformer instance for the test set.
        Device: torch.device (e.g., 'cuda:0' or 'cpu').
        batch_size: Batch size for processing (default: 32).

    Returns:
        Ids: List of molecule identifiers.
        Predictions: List of predicted energy values (scalars).
    """
    Model.eval()  # Set the model to evaluation mode
    Model.to(Device)

    # Create DataLoader for efficient batch processing
    test_loader = DataLoader(Dataset_Test, batch_size=batch_size, shuffle=False)

    Predictions = []
    Ids = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Computing Predictions"):
            Ids_batch, symbols, positions, mask = batch  # Ignore energy
            symbols = symbols.to(Device)
            positions = positions.to(Device)
            mask = mask.to(Device)

            # Forward pass
            outputs = Model(symbols, positions, mask=mask)  # Shape: (batch_size, 1)
            outputs = outputs.squeeze().cpu().tolist()  # Convert to list of scalars

            # Extend lists
            Ids.extend(Ids_batch)
            if isinstance(outputs, float):  # Single prediction case
                Predictions.append(outputs)
            else:  # Batch of predictions
                Predictions.extend(outputs)


    # transform ids to list of ints
    Ids = [int(id) for id in Ids]

    return Ids, Predictions



# create a csv file with the predictions
def Save_Predictions_To_CSV(Ids, Predictions, Output_CSV_Path):
    df = pd.DataFrame({
        'id': Ids,
        'energy': Predictions
    })
        # sort the dataframe by id
    df.sort_values(by='id', inplace=True)
    df.to_csv(Output_CSV_Path, index=False)


    print(f"Predictions saved to {Output_CSV_Path}")

    return df





# Extracting the CLS Tokens for visualisation 


import torch
from collections import Counter

# Ton dictionnaire de base
Dict_Encoding_Atoms = {
    'H': 0,
    'C': 1,
    'N': 2,
    'O': 3,
    'S': 4,
    'Cl': 5,
}

# Inverser l'encodage pour retrouver les symboles
encoding_to_atom = {v: k for k, v in Dict_Encoding_Atoms.items()}

# Définir l'ordre chimique
atom_order = ['C', 'H', 'N', 'O', 'S', 'Cl']

def symbols_to_formula(symbols_tensor):
    """
    Crée une formule moléculaire avec les atomes triés dans l'ordre chimique.
    """
    if symbols_tensor.dim() == 2:
        symbols_tensor = symbols_tensor[0]

    symbols_list = symbols_tensor.cpu().tolist()
    counts = Counter(symbols_list)

    # Construire un dict {atom: count}
    atom_counts = {}
    for enc_id, count in counts.items():
        if enc_id in encoding_to_atom:
            atom = encoding_to_atom[enc_id]
            atom_counts[atom] = count

    # Générer la formule dans l'ordre spécifié
    formula = ''
    for atom in atom_order:
        if atom in atom_counts:
            n = atom_counts[atom]
            formula += f"{atom}{n if n > 1 else ''}"

    return formula







def Extract_CLS_Tokens_and_Energies(Model,Dataset, Device, batch_size=32):
    """
    Extracts CLS tokens from the Transformer model for each molecule in the dataset.

    Args:
        Model: The Transformer model.
        Dataset: MoleculeDataset_Transformer instance.
        Device: torch.device (e.g., 'cuda:0' or 'cpu').
        batch_size: Batch size for processing (default: 32).

    Returns:
        List of CLS tokens for each molecule.
    """
    Model.eval()  # Set the model to evaluation mode
    Model.to(Device)

    # Create DataLoader for efficient batch processing
    data_loader = DataLoader(Dataset, batch_size=batch_size, shuffle=False)

    CLS_Tokens = []
    Energies = []

    Formulas = []

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Extracting CLS Tokens"):
            Id, symbols, positions, mask, _ = batch
            symbols = symbols.to(Device)
            positions = positions.to(Device)
            mask = mask.to(Device)

            Predicted_Energy, cls_token = Model.forward(symbols, positions, mask=mask, return_cls_token=True)
            cls_token = cls_token.squeeze().cpu().tolist()
            Energies.extend(Predicted_Energy.squeeze().cpu().tolist())
            CLS_Tokens.extend(cls_token)

            # Construire la formule pour chaque molécule du batch
            for s in symbols.cpu():
                Formulas.append(symbols_to_formula(s))

    return CLS_Tokens, Energies, Formulas


def plot_tsne_cls_tokens(CLS_Tokens, Energies, Formulas, num_points=180, random_state=42):
    """
    Affiche un t-SNE des CLS tokens, coloré par l'énergie et annoté par la formule brute.

    Args:
        CLS_Tokens (array-like): Liste ou array des CLS tokens (shape: [N, D]).
        Energies (array-like): Liste ou array des énergies associées (shape: [N]).
        Formulas (array-like): Liste ou array des formules brutes (shape: [N]).
        num_points (int): Nombre de points à afficher (par défaut 180).
        random_state (int): Graine pour la reproductibilité.
    """
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE
    from adjustText import adjust_text
    import numpy as np

    CLS_array = np.array(CLS_Tokens)
    energies = np.array(Energies)
    formulas = np.array(Formulas)

    np.random.seed(random_state)
    num_points = min(num_points, CLS_array.shape[0])
    indices = np.random.choice(CLS_array.shape[0], num_points, replace=False)

    CLS_sample = CLS_array[indices]
    energies_sample = energies[indices]
    formulas_sample = formulas[indices]

    CLS_2D = TSNE(n_components=2, random_state=random_state).fit_transform(CLS_sample)

    plt.figure(figsize=(14, 12))
    sc = plt.scatter(
        CLS_2D[:, 0],
        CLS_2D[:, 1],
        c=energies_sample,
        cmap='viridis',
        s=50,
        alpha=0.8,
        edgecolor='k'
    )

    texts = []
    for x, y, label in zip(CLS_2D[:, 0], CLS_2D[:, 1], formulas_sample):
        texts.append(plt.text(x, y, label, fontsize=8))

    adjust_text(
        texts,
        expand_text=(1.2, 1.4),
        expand_points=(1.2, 1.4),
        arrowprops=dict(
            arrowstyle="->",
            color='gray',
            lw=0.7,
            alpha=0.8
        )
    )

    cbar = plt.colorbar(sc)
    cbar.set_label("Predicted energy")

    plt.title(
        f"t-SNE of CLS Tokens ({num_points} random points)\n"
        "Color = predicted energy\n"
    )
    plt.xlabel("t-SNE dimension 1")
    plt.ylabel("t-SNE dimension 2")
    plt.tight_layout()
    plt.show()
