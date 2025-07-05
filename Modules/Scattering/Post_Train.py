def plot_scattering_tsne(orders_1_and_2_train, energies_train, ids_train, num_points=180, random_state=42):
    """
    Visualise les coefficients de scattering avec t-SNE.
    
    Args:
        orders_1_and_2_train (np.ndarray): Coefficients de scattering (déjà flatten).
        energies_train (np.ndarray): Énergies associées.
        ids_train (np.ndarray): IDs des molécules.
        num_points (int): Nombre de points à afficher.
        random_state (int): Graine aléatoire pour la reproductibilité.
    """
    from sklearn.manifold import TSNE
    from adjustText import adjust_text
    import numpy as np
    import matplotlib.pyplot as plt

    np.random.seed(random_state)
    n = min(num_points, orders_1_and_2_train.shape[0])
    indices = np.random.choice(orders_1_and_2_train.shape[0], n, replace=False)

    scattering_sample = orders_1_and_2_train[indices]
    energies_sample = energies_train[indices]
    ids_sample = np.array(ids_train)[indices]

    scattering_2D = TSNE(n_components=2, random_state=random_state).fit_transform(scattering_sample)

    plt.figure(figsize=(14, 12))
    sc = plt.scatter(
        scattering_2D[:, 0],
        scattering_2D[:, 1],
        c=energies_sample,
        cmap='viridis',
        s=50,
        alpha=0.8,
        edgecolor='k'
    )

    texts = []
    for x, y, label in zip(scattering_2D[:, 0], scattering_2D[:, 1], ids_sample):
        texts.append(plt.text(x, y, str(label), fontsize=8))

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
    cbar.set_label("Énergie")

    plt.title(
        f"t-SNE des coefficients de scattering ({n} points au hasard)\n"
        "Couleur = énergie"
    )
    plt.xlabel("t-SNE dimension 1")
    plt.ylabel("t-SNE dimension 2")
    plt.tight_layout()
    plt.show()





def predict_and_save_test_energies(X_Test_With_Features, scaler, ridge_final, Ids_test, output_csv_path):
        # Scale les features pour le test set
    import pandas as pd
    X_test_with_features_scaled = scaler.transform(X_Test_With_Features)
    # Prédiction sur le test set
    energies_test_pred = ridge_final.predict(X_test_with_features_scaled)
    # Afficher les résultats
    for id, energy in zip(Ids_test, energies_test_pred):
        print(f"ID: {id.item()}, Predicted Energy: {energy:.4f}")
    # Save predictions as a CSV file
    predictions_df = pd.DataFrame({
        'id': Ids_test.numpy(),
        'energy': energies_test_pred
    })
    predictions_df.to_csv(output_csv_path, index=False)
    return predictions_df
