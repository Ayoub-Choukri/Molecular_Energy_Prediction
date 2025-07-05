from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
from tqdm import tqdm
from IPython.display import display

def ridge_cv_manual(X_train_with_features, y_train, nb_folds=5, alphas=np.logspace(-5, 5, 150), random_state=42):
    scaler = StandardScaler()
    X_train_with_features_scaled = scaler.fit_transform(X_train_with_features)

    kf = KFold(n_splits=nb_folds, shuffle=True, random_state=random_state)
    mse_per_alpha_per_fold = np.zeros((len(alphas), nb_folds))
    mean_mse_per_alpha = []

    for alpha_idx, alpha in enumerate(tqdm(alphas, desc="Alphas")):
        ridge = Ridge(alpha=alpha)
        fold_mse = []
        for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X_train_with_features_scaled)):
            X_train_fold, X_val_fold = X_train_with_features_scaled[train_idx], X_train_with_features_scaled[val_idx]
            y_train_fold, y_val_fold = y_train[train_idx], y_train[val_idx]
            ridge.fit(X_train_fold, y_train_fold)
            y_pred = ridge.predict(X_val_fold)
            mse = mean_squared_error(y_val_fold, y_pred)
            fold_mse.append(mse)
            mse_per_alpha_per_fold[alpha_idx, fold_idx] = mse
        mean_mse_per_alpha.append(np.mean(fold_mse))

    results_df = pd.DataFrame(mse_per_alpha_per_fold, columns=[f"MSE_Fold_{i+1}" for i in range(nb_folds)])
    results_df['Alpha'] = alphas
    results_df['Mean MSE'] = mean_mse_per_alpha
    cols = ['Alpha', 'Mean MSE'] + [f"MSE_Fold_{i+1}" for i in range(nb_folds)]
    results_df = results_df[cols]
    results_df = results_df.sort_values(by='Mean MSE')

    print("\n📊 Tableau des MSE pour chaque alpha et chaque fold (trié par Mean MSE croissant) :")
    display(results_df)

    best_alpha = results_df.iloc[0]['Alpha']
    print(f"\n✅ Meilleur alpha trouvé : {best_alpha:.6f}")

    ridge_final = Ridge(alpha=best_alpha)
    ridge_final.fit(X_train_with_features_scaled, y_train)
    energies_pred = ridge_final.predict(X_train_with_features_scaled)
    mse_val = mean_squared_error(y_train, energies_pred)
    print(f"✅ Mean Squared Error on Validation Set (avec meilleur alpha) : {mse_val:.4f}")

    return scaler, ridge_final, results_df, best_alpha, mse_val



def plot_ridge_cv_results(results_df, best_alpha):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6))
    plt.plot(results_df['Alpha'], results_df['Mean MSE'], marker='o')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Alpha (log scale)')
    plt.ylabel('Mean MSE')
    plt.title('Mean MSE vs Alpha')
    plt.grid(True)
    plt.axvline(x=best_alpha, color='r', linestyle='--', label=f'Best Alpha: {best_alpha:.6f}, Best Test MSE: {results_df["Mean MSE"].min():.6f}')
    plt.legend()
    plt.show()
    best_loss = results_df['Mean MSE'].min()
    print(f"Best Mean MSE: {best_loss:.6f}")
