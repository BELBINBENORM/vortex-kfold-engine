import os
import cloudpickle
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, clone
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import roc_auc_score, r2_score

class VortexKFold(BaseEstimator):
    def __init__(self, base_estimator, task='classification', model_name="vortex_model", 
                 n_splits=5, path=".", load_saved=True, random_state=42, n_jobs=-1, verbose=1):
        self.base_estimator = base_estimator
        self.task = task.lower()
        self.model_name = model_name
        self.n_splits = n_splits
        self.path = path
        self.load_saved = load_saved
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.verbose = verbose 
        self.models_ = []
        self.oof_preds_ = None
        self.cv_score_ = None

    def _log(self, msg):
        if self.verbose == 1: 
            print(msg)

    def _get_paths(self, cv_score=None):
        score_str = f"{cv_score:.5f}" if cv_score is not None else "pending"
        model_file = f"{self.model_name}_{score_str}_{self.n_splits}_fold.cloudpickle"
        oof_file = f"{self.model_name}_{score_str}_{self.n_splits}_fold_oof.npy"
        return os.path.join(self.path, model_file), os.path.join(self.path, oof_file)

    def _find_existing_files(self):
        if not os.path.exists(self.path): return None, None
        files = os.listdir(self.path)
        model_files = [f for f in files if f.startswith(self.model_name) and f.endswith(".cloudpickle")]
        oof_files = [f for f in files if f.startswith(self.model_name) and f.endswith("_oof.npy")]
        if model_files and oof_files:
            # Sort to get the latest/relevant file if multiple exist
            return os.path.join(self.path, sorted(model_files)[-1]), os.path.join(self.path, sorted(oof_files)[-1])
        return None, None

    def _train_fold(self, fold, t_idx, v_idx, X, y):
        m = clone(self.base_estimator)
        m.fit(X[t_idx], y[t_idx])
        if self.task == 'classification':
            preds = m.predict_proba(X[v_idx])[:, 1]
        else:
            preds = m.predict(X[v_idx])
        return m, v_idx, preds

    def fit(self, X, y):
        if self.load_saved:
            m_path, o_path = self._find_existing_files()
            if m_path and o_path:
                self._log(f"🔍 [{self.model_name}] Found pre-trained files. Skipping training...")
                with open(m_path, "rb") as f: 
                    self.models_ = cloudpickle.load(f)
                self.oof_preds_ = np.load(o_path)
                self.cv_score_ = roc_auc_score(y, self.oof_preds_) if self.task == 'classification' else r2_score(y, self.oof_preds_)
                self._log(f"✅ [{self.model_name}] Load completed. Verified CV Score: {self.cv_score_:.5f}")
                return self

        X_val = X.values if hasattr(X, "values") else X
        y_val = y.values if hasattr(y, "values") else y
        
        self._log(f"🚀 [{self.model_name}] Initiating {self.task} training ({self.n_splits} folds)...")
        
        cv = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state) if self.task == 'classification' else KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
        
        results = Parallel(n_jobs=self.n_jobs)(
            delayed(self._train_fold)(i, t, v, X_val, y_val) 
            for i, (t, v) in enumerate(cv.split(X_val, y_val))
        )
        
        self.models_ = [r[0] for r in results]
        self.oof_preds_ = np.zeros(len(X_val))
        for i, (_, v_idx, preds) in enumerate(results):
            self.oof_preds_[v_idx] = preds
            self._log(f"✅ [{self.model_name}] Fold {i} training & OOF completed.")

        self.cv_score_ = roc_auc_score(y_val, self.oof_preds_) if self.task == 'classification' else r2_score(y_val, self.oof_preds_)
        m_path, o_path = self._get_paths(self.cv_score_)
        
        with open(m_path, "wb") as f: 
            cloudpickle.dump(self.models_, f)
        np.save(o_path, self.oof_preds_)
        
        self._log(f"💾 [{self.model_name}] Model & OOF Saved. Final CV: {self.cv_score_:.5f}")
        return self

    def predict_proba(self, X):
        if self.task != 'classification':
            raise AttributeError("predict_proba is only available for classification tasks.")
        X_val = X.values if hasattr(X, "values") else X
        all_probs = Parallel(n_jobs=self.n_jobs)(delayed(m.predict_proba)(X_val) for m in self.models_)
        return np.mean(all_probs, axis=0)

    def predict(self, X):
        X_val = X.values if hasattr(X, "values") else X
        if self.task == 'classification':
            return (self.predict_proba(X_val)[:, 1] >= 0.5).astype(int)
        else:
            all_preds = Parallel(n_jobs=self.n_jobs)(delayed(m.predict)(X_val) for m in self.models_)
            return np.mean(all_preds, axis=0)
