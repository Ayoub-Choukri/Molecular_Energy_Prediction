import torch
import numpy as np
import xgboost as xgb
from torch import nn
from sklearn.metrics import mean_squared_error, accuracy_score
import logging

# Configuration du logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class XGBoostWrapper(nn.Module):
    def __init__(self, input_dim, n_estimators=100, max_depth=6, learning_rate=0.1,
                objective='reg:squarederror', model_type='regressor',
                reg_lambda=1.0, reg_alpha=0.0):
        super().__init__()
        self.input_dim = input_dim
        self.model_type = model_type.lower()
        self.objective = objective
        self.params = {
            'max_depth': max_depth,
            'eta': learning_rate,
            'objective': objective,
            'eval_metric': 'rmse' if self.model_type == 'regressor' else 'mlogloss',
            'tree_method': 'hist',
            'verbosity': 0,
            'seed': 42,
            'lambda': reg_lambda,   # régularisation L2
            'alpha': reg_alpha      # régularisation L1
        }
        self.n_estimators = n_estimators
        self.booster = None
        self.is_fitted = False
        logger.info(f"XGBoostWrapper initialisé avec input_dim={input_dim}, model_type={model_type}, reg_lambda={reg_lambda}, reg_alpha={reg_alpha}")
    def fit(self, X, y, eval_set=None, early_stopping_rounds=None, verbose=False):
        if isinstance(X, torch.Tensor):
            X = X.cpu().numpy()
        if isinstance(y, torch.Tensor):
            y = y.cpu().numpy().ravel()

        dtrain = xgb.DMatrix(X, label=y)

        evals = [(dtrain, 'train')]
        if eval_set is not None:
            eval_X, eval_y = eval_set[0]
            if isinstance(eval_X, torch.Tensor):
                eval_X = eval_X.cpu().numpy()
            if isinstance(eval_y, torch.Tensor):
                eval_y = eval_y.cpu().numpy().ravel()
            dval = xgb.DMatrix(eval_X, label=eval_y)
            evals.append((dval, 'eval'))

        try:
            logger.info(f"Early stopping activé avec {early_stopping_rounds} rounds")
            self.booster = xgb.train(
                params=self.params,
                dtrain=dtrain,
                num_boost_round=self.n_estimators,
                evals=evals,
                early_stopping_rounds=early_stopping_rounds,
                verbose_eval=verbose
            )
            self.is_fitted = True
            logger.info("Entraînement terminé avec succès")
        except Exception as e:
            logger.error(f"Erreur dans fit() : {e}")
            raise e

    def forward(self, x):
        if not self.is_fitted:
            raise RuntimeError("Le modèle doit être entraîné avant d'utiliser forward().")
        if isinstance(x, torch.Tensor):
            x_np = x.cpu().numpy()
        else:
            x_np = x
        dmatrix = xgb.DMatrix(x_np)
        preds = self.booster.predict(dmatrix)
        preds = torch.tensor(preds, dtype=torch.float32)
        if preds.dim() == 1:
            preds = preds.unsqueeze(-1)
        return preds

    def evaluate(self, X, y=None):
        if not self.is_fitted:
            raise RuntimeError("Le modèle doit être entraîné avant d'évaluer.")
        if isinstance(X, torch.Tensor):
            X = X.cpu().numpy()
        dmatrix = xgb.DMatrix(X)
        preds = self.booster.predict(dmatrix)
        preds_list = preds.tolist()

        metric = None
        if y is not None:
            if isinstance(y, torch.Tensor):
                y = y.cpu().numpy().ravel()
            if self.model_type == 'regressor':
                metric = mean_squared_error(y, preds)
                logger.info(f"MSE : {metric:.4f}")
            elif self.model_type == 'classifier':
                pred_labels = (preds > 0.5).astype(int)
                metric = accuracy_score(y, pred_labels)
                logger.info(f"Accuracy : {metric:.4f}")
        return preds_list, metric

    def save_model(self, path):
        if not self.is_fitted:
            raise RuntimeError("Le modèle n'est pas encore entraîné.")
        self.booster.save_model(path)
        logger.info(f"Modèle sauvegardé : {path}")

    def load_model(self, path):
        self.booster = xgb.Booster()
        self.booster.load_model(path)
        self.is_fitted = True
        logger.info(f"Modèle chargé : {path}")

    def get_params(self):
        return self.params


# ===========================
# TEST EN STANDALONE
# ===========================
if __name__ == "__main__":
    import torch
    print(torch.cuda.is_available())
    print(torch.cuda.device_count())
    print(torch.cuda.get_device_name(0))

    logger.info(f"Version de XGBoost : {xgb.__version__}")

    input_dim = 26
    model = XGBoostWrapper(input_dim=input_dim, model_type='regressor', n_estimators=100)

    X_train = torch.randn(1000, input_dim)
    y_train = torch.randn(1000, 1)
    X_val = torch.randn(200, input_dim)
    y_val = torch.randn(200, 1)
    X_test = torch.randn(32, input_dim)
    y_test = torch.randn(32, 1)

    model.fit(
        X=X_train,
        y=y_train,
        eval_set=[(X_val, y_val)],
        early_stopping_rounds=10,
        verbose=False
    )

    model.eval()
    with torch.no_grad():
        preds = model(X_test)
    print(f"Shape des prédictions : {preds.shape}")

    preds_list, mse = model.evaluate(X_test, y_test)
    print(f"MSE : {mse:.4f}")

    model.save_model("test_model.xgb")
    model.load_model("test_model.xgb")

    with torch.no_grad():
        preds_loaded = model(X_test)
    print(f"Écart moyen après chargement : {torch.mean(torch.abs(preds - preds_loaded))}")
