import torch
from tqdm import tqdm

import torch
from tqdm.auto import tqdm
import matplotlib.pyplot as plt


def Train_One_Epoch(Model, Train_Loader, Optimizer, Criterion, List_Train_Losses_Per_Batches, Device):
    """
    Entraîne le modèle pour une époque sur le jeu de données d'entraînement.

    Args:
        Model: Le modèle Transformer.
        Train_Loader: DataLoader pour les données d'entraînement.
        Optimizer: Optimiseur (par exemple, Adam).
        Criterion: Fonction de perte (par exemple, MSELoss).
        List_Train_Losses_Per_Batches: Liste pour stocker les pertes par batch.
        Device: Dispositif (par exemple, 'cuda' ou 'cpu').

    Returns:
        Train_Loss: Perte moyenne sur l'époque.
        List_Train_Losses_Per_Batches: Liste mise à jour des pertes par batch.
    """
    Model.to(Device)
    Model.train()

    Progress_Bar_Batch = tqdm(Train_Loader, desc="Batches")

    Train_Loss = 0
    Num_Batches = 0

    for Ids, symbols, positions, mask, energy in Progress_Bar_Batch:
        # Envoyer les tenseurs sur le dispositif
        symbols = symbols.to(Device)
        positions = positions.to(Device)
        mask = mask.to(Device)
        energy = energy.to(Device)

        # Vérifier que tous les tenseurs sont sur le même dispositif

        # Forward pass
        Optimizer.zero_grad()
        outputs = Model(symbols, positions, mask=mask)  # (batch_size, 1)
        loss = Criterion(outputs.squeeze(), energy)  # Squeeze pour (batch_size,)

        # Backward pass
        loss.backward()
        Optimizer.step()

        Running_Loss = loss.item()
        List_Train_Losses_Per_Batches.append(Running_Loss)

        Progress_Bar_Batch.set_description(f"Num Batches: {Num_Batches} Running Train Loss: {Running_Loss:.3f}")
        
        Train_Loss += Running_Loss
        Num_Batches += 1

    Train_Loss = Train_Loss / Num_Batches
    return Train_Loss, List_Train_Losses_Per_Batches

def Test_One_Epoch(Model, Test_Loader, Criterion, List_Test_Losses_Per_Batches, Device):
    """
    Évalue le modèle sur le jeu de données de test pour une époque.

    Args:
        Model: Le modèle Transformer.
        Test_Loader: DataLoader pour les données de test.
        Criterion: Fonction de perte.
        List_Test_Losses_Per_Batches: Liste pour stocker les pertes par batch.
        Device: Dispositif.

    Returns:
        Test_Loss: Perte moyenne sur l'époque.
        List_Test_Losses_Per_Batches: Liste mise à jour des pertes par batch.
    """
    Model.to(Device)
    Model.eval()

    Progress_Bar_Batch = tqdm(Test_Loader, desc="Batches", leave=False)

    Test_Loss = 0
    Num_Batches = 0

    with torch.no_grad():
        for Ids, symbols, positions, mask, energy in Progress_Bar_Batch:
            symbols = symbols.to(Device)
            positions = positions.to(Device)
            mask = mask.to(Device)
            energy = energy.to(Device)

            # Vérifier que tous les tenseurs sont sur le même dispositif
            outputs = Model(symbols, positions, mask=mask)  # (batch_size, 1)
            loss = Criterion(outputs.squeeze(), energy)  # Squeeze pour (batch_size,)

            Running_Loss = loss.item()
            List_Test_Losses_Per_Batches.append(Running_Loss)

            Test_Loss += Running_Loss
            Num_Batches += 1

            Progress_Bar_Batch.set_description(f"Num Batches: {Num_Batches} Running Test Loss: {Running_Loss:.3f}")

    Test_Loss = Test_Loss / Num_Batches
    return Test_Loss, List_Test_Losses_Per_Batches

def Train(Model, Train_Loader, Test_Loader, Optimizer, Criterion, Num_Epochs, Device, Save_Path="best_model.pth", Best_Test_Loss=float('inf')):
    """
    Entraîne et évalue le modèle sur plusieurs époques, avec sauvegarde du meilleur modèle basé sur la perte de test.

    Args:
        Model: Le modèle Transformer.
        Train_Loader: DataLoader pour l'entraînement.
        Test_Loader: DataLoader pour le test.
        Optimizer: Optimiseur.
        Criterion: Fonction de perte.
        Num_Epochs: Nombre d'époques.
        Device: Dispositif.
        Save_Path: Chemin pour sauvegarder le meilleur modèle (par défaut: "best_model.pth").
        Best_Test_Loss: Meilleure perte de test initiale (par défaut: float('inf')).

    Returns:
        List_Train_Losses_Per_Epochs: Pertes moyennes par époque (entraînement).
        List_Test_Losses_Per_Epochs: Pertes moyennes par époque (test).
        List_Train_Losses_Per_Batches: Pertes par batch (entraînement).
        List_Test_Losses_Per_Batches: Pertes par batch (test).
    """
    Model.to(Device)

    List_Train_Losses_Per_Epochs = []
    List_Test_Losses_Per_Epochs = []
    List_Train_Losses_Per_Batches = []
    List_Test_Losses_Per_Batches = []

    Progress_Bar_Epochs = tqdm(range(Num_Epochs), desc="Epochs")

    for epoch in Progress_Bar_Epochs:
        Train_Loss, List_Train_Losses_Per_Batches = Train_One_Epoch(
            Model, Train_Loader, Optimizer, Criterion, List_Train_Losses_Per_Batches, Device
        )

        Test_Loss, List_Test_Losses_Per_Batches = Test_One_Epoch(
            Model, Test_Loader, Criterion, List_Test_Losses_Per_Batches, Device
        )

        List_Train_Losses_Per_Epochs.append(Train_Loss)
        List_Test_Losses_Per_Epochs.append(Test_Loss)

        # Save model if test loss is the best so far
        if Test_Loss < Best_Test_Loss:
            Best_Test_Loss = Test_Loss
            torch.save(Model, Save_Path)
            print(f"New best test loss: {Test_Loss:.3f}, model saved to {Save_Path}")

        print(f"Epoch: {epoch+1}/{Num_Epochs} Train Loss: {Train_Loss:.3f} Test Loss: {Test_Loss:.3f}")
        Progress_Bar_Epochs.set_description(f"Epochs Train Loss: {Train_Loss:.3f} Test Loss: {Test_Loss:.3f}")

    return List_Train_Losses_Per_Epochs, List_Test_Losses_Per_Epochs, List_Train_Losses_Per_Batches, List_Test_Losses_Per_Batches

def Plot_Losses(List_Train_Losses_Per_Epochs,List_Test_Losses_Per_Epochs,List_Train_Losses_Per_Batches,List_Test_Losses_Per_Batches, Save = False, Save_Path = None):

    plt.figure(figsize=(15,15))

    fig, axes = plt.subplots(2, 2, figsize=(15, 15))

    axes[0, 0].plot(List_Train_Losses_Per_Epochs)
    axes[0, 0].set_title("Train Losses Per Epochs")
    axes[0, 0].set_xlabel("Epochs")
    axes[0, 0].set_ylabel("Losses")
    axes[0, 0].set_yscale('log')
    
    axes[0, 1].plot(List_Test_Losses_Per_Epochs)
    axes[0, 1].set_title("Test Losses Per Epochs")
    axes[0, 1].set_xlabel("Epochs")
    axes[0, 1].set_ylabel("Losses")
    axes[0, 1].set_yscale('log')

    axes[1, 0].plot(List_Train_Losses_Per_Batches)
    axes[1, 0].set_title("Train Losses Per Batches")
    axes[1, 0].set_xlabel("Batches")
    axes[1, 0].set_ylabel("Losses")
    axes[1, 0].set_yscale('log')


    axes[1, 1].plot(List_Test_Losses_Per_Batches)
    axes[1, 1].set_title("Test Losses Per Batches")
    axes[1, 1].set_xlabel("Batches")
    axes[1, 1].set_ylabel("Losses")
    axes[1, 1].set_yscale('log')

    plt.show()

    if Save:
        plt.savefig(Save_Path)

