import torch 
import torch.nn as nn
from tqdm.auto import tqdm
import sys
import matplotlib.pyplot as plt


def Train_One_Epoch(Model, Train_Loader, Optimizer, Criterion, List_Train_Losses_Per_Batches, Device):
    Progress_Bar_Batch = tqdm(Train_Loader, desc="Batches")

    Train_Loss = 0
    Num_Batches = 0
    for Ids, Atom_Indices, Atom_Positions, Sequence_Lengths, Energies in Progress_Bar_Batch:
        # Ids = Ids.to(Device)
        Atom_Indices = Atom_Indices.to(Device)
        Atom_Positions = Atom_Positions.to(Device)
        # Do NOT move Sequence_Lengths to Device (keep it on CPU for pack_padded_sequence)
        Energies = Energies.to(Device)

        # Forward pass
        Optimizer.zero_grad()

        outputs = Model(Atom_Indices, Atom_Positions, Sequence_Lengths)

        loss = Criterion(outputs, Energies)

        loss.backward()

        Optimizer.step()

        Running_Loss = loss.item()

        # Calculate the accuracy
        List_Train_Losses_Per_Batches.append(Running_Loss)

        Progress_Bar_Batch.set_description(f"Num Batches: {Num_Batches} Running Train Loss: {Running_Loss:.3f} ")

        Train_Loss += Running_Loss

        Num_Batches += 1

    Train_Loss = Train_Loss / Num_Batches
    return Train_Loss, List_Train_Losses_Per_Batches


def Test_One_Epoch(Model, Test_Loader, Criterion, List_Test_Losses_Per_Batches, Device):
    Progress_Bar_Batch = tqdm(Test_Loader, desc="Batches", leave=False)

    Test_Loss = 0
    Num_Batches = 0

    for Ids, Atom_Indices, Atom_Positions, Sequence_Lengths, Energies in Progress_Bar_Batch:
        # Ids = Ids.to(Device)
        Atom_Indices = Atom_Indices.to(Device)
        Atom_Positions = Atom_Positions.to(Device)
        # Do NOT move Sequence_Lengths to Device (keep it on CPU for pack_padded_sequence)
        Energies = Energies.to(Device)

        outputs = Model(Atom_Indices, Atom_Positions, Sequence_Lengths)

        loss = Criterion(outputs, Energies)

        Running_Loss = loss.item()
        List_Test_Losses_Per_Batches.append(Running_Loss)

        Test_Loss += Running_Loss

        Num_Batches += 1

        Progress_Bar_Batch.set_description(f"Num Batches: {Num_Batches} Running Test Loss: {Running_Loss:.3f} ")

    Test_Loss = Test_Loss / Num_Batches

    return Test_Loss, List_Test_Losses_Per_Batches


def Train(Model, Train_Loader, Test_Loader, Optimizer, Criterion, Num_Epochs, Device, Save_Path="best_model.pth", Best_Test_Loss=float('inf')):
    # Set the model to training mode
    Model.train()

    # Move the model to the device
    Model.to(Device)

    List_Train_Losses_Per_Epochs = []
    List_Test_Losses_Per_Epochs = []
    List_Train_Losses_Per_Batches = []
    List_Test_Losses_Per_Batches = []

    Progress_Bar_Epochs = tqdm(range(Num_Epochs), desc="Epochs")

    for epoch in Progress_Bar_Epochs:
        Train_Loss, List_Train_Losses_Per_Batches = Train_One_Epoch(Model, Train_Loader, Optimizer, Criterion, List_Train_Losses_Per_Batches, Device)

        Test_Loss, List_Test_Losses_Per_Batches = Test_One_Epoch(Model, Test_Loader, Criterion, List_Test_Losses_Per_Batches, Device)

        List_Train_Losses_Per_Epochs.append(Train_Loss)
        List_Test_Losses_Per_Epochs.append(Test_Loss)

        # Save model if test loss is the best so far
        if Test_Loss < Best_Test_Loss:
            Best_Test_Loss = Test_Loss
            torch.save(Model, Save_Path)
            print(f"New best test loss: {Test_Loss:.3f}, model saved to {Save_Path}")

        print(f"Epoch: {epoch+1}/{Num_Epochs} Train Loss: {Train_Loss:.3f} Test Loss: {Test_Loss:.3f} ")

        Progress_Bar_Epochs.set_description(f"Epochs Train Loss: {Train_Loss:.3f} Test Loss: {Test_Loss:.3f}")

    return List_Train_Losses_Per_Epochs, List_Test_Losses_Per_Epochs, List_Train_Losses_Per_Batches, List_Test_Losses_Per_Batches


def Plot_Losses(List_Train_Losses_Per_Epochs, List_Test_Losses_Per_Epochs, List_Train_Losses_Per_Batches, List_Test_Losses_Per_Batches, Save=False, Save_Path=None):
    plt.figure(figsize=(15,15))

    fig, axes = plt.subplots(2, 2, figsize=(15, 15))

    axes[0, 0].plot(List_Train_Losses_Per_Epochs)
    axes[0, 0].set_title("Train Losses Per Epochs")
    axes[0, 0].set_xlabel("Epochs")
    axes[0, 0].set_ylabel("Losses")

    axes[0, 1].plot(List_Test_Losses_Per_Epochs)
    axes[0, 1].set_title("Test Losses Per Epochs")
    axes[0, 1].set_xlabel("Epochs")
    axes[0, 1].set_ylabel("Losses")

    axes[1, 0].plot(List_Train_Losses_Per_Batches)
    axes[1, 0].set_title("Train Losses Per Batches")
    axes[1, 0].set_xlabel("Batches")
    axes[1, 0].set_ylabel("Losses")

    axes[1, 1].plot(List_Test_Losses_Per_Batches)
    axes[1, 1].set_title("Test Losses Per Batches")
    axes[1, 1].set_xlabel("Batches")
    axes[1, 1].set_ylabel("Losses")

    plt.show()

    if Save:
        plt.savefig(Save_Path)