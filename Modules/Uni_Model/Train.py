import torch
import torch.nn as nn
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

def Train_One_Epoch(Model, Train_Loader, Optimizer, Criterion, List_Train_Losses_Per_Batches, Device, Coord_Loss_Weight=1.0, Energy_Loss_Weight=1.0):
    """
    Entraîne le modèle pour une époque sur les données d'entraînement.
    """
    Model.train()
    Progress_Bar_Batch = tqdm(Train_Loader, desc="Batches")
    Train_Loss = 0.0
    Num_Batches = 0

    for batch in Progress_Bar_Batch:
        atom_types = batch['atom_types'].to(Device)
        coords = batch['coords'].to(Device)
        pair_types = batch['pair_types'].to(Device)
        mask = batch['mask'].to(Device)
        energies = batch['energies'].to(Device) if batch['energies'] is not None else None

        Optimizer.zero_grad()

        pred_coords, pred_energy = Model(atom_types, coords, pair_types, mask)

        loss = 0.0
        if mask is not None:
            masked_pred_coords = pred_coords * mask.unsqueeze(-1)
            masked_coords = coords * mask.unsqueeze(-1)
            coord_loss = Criterion(masked_pred_coords, masked_coords)
            loss += Coord_Loss_Weight * coord_loss
        else:
            coord_loss = Criterion(pred_coords, coords)
            loss += Coord_Loss_Weight * coord_loss

        if energies is not None:
            energy_loss = Criterion(pred_energy, energies)
            loss += Energy_Loss_Weight * energy_loss
        else:
            energy_loss = torch.tensor(0.0).to(Device)

        loss.backward()
        Optimizer.step()

        Running_Loss = loss.item()
        List_Train_Losses_Per_Batches.append(Running_Loss)
        Train_Loss += Running_Loss
        Num_Batches += 1

        Progress_Bar_Batch.set_description(f"Num Batches: {Num_Batches} Running Train Loss: {Running_Loss:.3f} (Coord: {coord_loss.item():.3f}, Energy: {energy_loss.item():.3f})")

    Train_Loss /= Num_Batches
    return Train_Loss, List_Train_Losses_Per_Batches


def Test_One_Epoch(Model, Val_Loader, Criterion, List_Val_Losses_Per_Batches, Device, Coord_Loss_Weight=1.0, Energy_Loss_Weight=1.0):
    """
    Évalue le modèle pour une époque sur les données de validation.
    """
    Model.eval()
    Progress_Bar_Batch = tqdm(Val_Loader, desc="Batches", leave=False)
    Val_Loss = 0.0
    Num_Batches = 0

    with torch.no_grad():
        for batch in Progress_Bar_Batch:
            atom_types = batch['atom_types'].to(Device)
            coords = batch['coords'].to(Device)
            pair_types = batch['pair_types'].to(Device)
            mask = batch['mask'].to(Device)
            energies = batch['energies'].to(Device) if batch['energies'] is not None else None

            pred_coords, pred_energy = Model(atom_types, coords, pair_types, mask)

            loss = 0.0
            if mask is not None:
                masked_pred_coords = pred_coords * mask.unsqueeze(-1)
                masked_coords = coords * mask.unsqueeze(-1)
                coord_loss = Criterion(masked_pred_coords, masked_coords)
                loss += Coord_Loss_Weight * coord_loss
            else:
                coord_loss = Criterion(pred_coords, coords)
                loss += Coord_Loss_Weight * coord_loss

            if energies is not None:
                energy_loss = Criterion(pred_energy, energies)
                loss += Energy_Loss_Weight * energy_loss
            else:
                energy_loss = torch.tensor(0.0).to(Device)

            Running_Loss = loss.item()
            List_Val_Losses_Per_Batches.append(Running_Loss)
            Val_Loss += Running_Loss
            Num_Batches += 1

            Progress_Bar_Batch.set_description(f"Num Batches: {Num_Batches} Running Val Loss: {Running_Loss:.3f} (Coord: {coord_loss.item():.3f}, Energy: {energy_loss.item():.3f})")

    Val_Loss /= Num_Batches
    return Val_Loss, List_Val_Losses_Per_Batches


def Train(Model, Train_Loader, Val_Loader, Optimizer, Criterion, Num_Epochs, Device, Scheduler=None, Save_Path="best_model.pth", Best_Val_Loss=float('inf'), Coord_Loss_Weight=1.0, Energy_Loss_Weight=1.0):
    """
    Entraîne et évalue le modèle sur plusieurs époques avec un Scheduler optionnel pour le taux d'apprentissage.
    Si le Scheduler est ReduceLROnPlateau, la perte de validation (Val_Loss) est utilisée comme métrique.
    Pour les autres Schedulers, step() est appelé sans métrique.
    """
    Model.train()
    Model.to(Device)

    List_Train_Losses_Per_Epochs = []
    List_Val_Losses_Per_Epochs = []
    List_Train_Losses_Per_Batches = []
    List_Val_Losses_Per_Batches = []

    Progress_Bar_Epochs = tqdm(range(Num_Epochs), desc="Epochs")

    for epoch in Progress_Bar_Epochs:
        Train_Loss, List_Train_Losses_Per_Batches = Train_One_Epoch(
            Model, Train_Loader, Optimizer, Criterion, List_Train_Losses_Per_Batches, Device, Coord_Loss_Weight, Energy_Loss_Weight
        )
        Val_Loss, List_Val_Losses_Per_Batches = Test_One_Epoch(
            Model, Val_Loader, Criterion, List_Val_Losses_Per_Batches, Device, Coord_Loss_Weight, Energy_Loss_Weight
        )

        # Mise à jour du Scheduler (s'il est fourni)
        if Scheduler is not None:
            if isinstance(Scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                Scheduler.step(Val_Loss)
            else:
                Scheduler.step()

        List_Train_Losses_Per_Epochs.append(Train_Loss)
        List_Val_Losses_Per_Epochs.append(Val_Loss)

        if Val_Loss < Best_Val_Loss:
            Best_Val_Loss = Val_Loss
            torch.save(Model, Save_Path)
            print(f"New best validation loss: {Val_Loss:.3f}, model saved to {Save_Path}")

        # Affichage du taux d'apprentissage actuel
        current_lr = Optimizer.param_groups[0]['lr']
        print(f"Epoch: {epoch+1}/{Num_Epochs} Train Loss: {Train_Loss:.3f} Val Loss: {Val_Loss:.3f} Learning Rate: {current_lr:.6f}")
        Progress_Bar_Epochs.set_description(f"Epochs Train Loss: {Train_Loss:.3f} Val Loss: {Val_Loss:.3f} LR: {current_lr:.6f}")

    return List_Train_Losses_Per_Epochs, List_Val_Losses_Per_Epochs, List_Train_Losses_Per_Batches, List_Val_Losses_Per_Batches


def Plot_Losses(List_Train_Losses_Per_Epochs, List_Val_Losses_Per_Epochs, List_Train_Losses_Per_Batches, List_Val_Losses_Per_Batches, Save=False, Save_Path=None):
    """
    Trace les courbes de pertes d'entraînement et de validation.
    """
    plt.figure(figsize=(15, 15))
    fig, axes = plt.subplots(2, 2, figsize=(15, 15))

    axes[0, 0].plot(List_Train_Losses_Per_Epochs)
    axes[0, 0].set_title("Train Losses Per Epochs")
    axes[0, 0].set_xlabel("Epochs")
    axes[0, 0].set_ylabel("Losses")

    axes[0, 1].plot(List_Val_Losses_Per_Epochs)
    axes[0, 1].set_title("Validation Losses Per Epochs")
    axes[0, 1].set_xlabel("Epochs")
    axes[0, 1].set_ylabel("Losses")

    axes[1, 0].plot(List_Train_Losses_Per_Batches)
    axes[1, 0].set_title("Train Losses Per Batch")
    axes[1, 0].set_xlabel("Batches")
    axes[1, 0].set_ylabel("Losses")

    axes[1, 1].plot(List_Val_Losses_Per_Batches)
    axes[1, 1].set_title("Validation Losses Per Batch")
    axes[1, 1].set_xlabel("Batches")
    axes[1, 1].set_ylabel("Losses")

    plt.tight_layout()

    if Save and Save_Path is not None:
        plt.savefig(Save_Path)
    else:
        plt.show()