"""Untitled

"""

import sqlite3
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List

from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

import hdbscan
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import make_scorer, r2_score, mean_absolute_error

import os
import matplotlib.pyplot as plt

from itertools import product

# -----------------------
# Base class for training regression models
# -----------------------
class CCSML:
    def __init__(self, database_file: str):

        if os.path.exists(database_file):
            with sqlite3.connect(database_file) as conn:
                self.db_master = pd.read_sql_query("SELECT * FROM master", conn)
                self.db_mqns = pd.read_sql_query("SELECT * FROM mqns", conn)
                self.db = pd.merge(self.db_master, self.db_mqns, on="g_id", how="left")
        else:
            raise FileNotFoundError(f"Database file {database_file} not found")

        self.seed = 69 # random seed

        self.X_columns: Optional[List[str]] = None #columns for features
        self.y_column: Optional[List[str]] = None #column for target

        # split
        self.trainX = None
        self.trainY = None

        self.testX = None
        self.testY = None

    def build_features(self, *feature_columns: List[str], target_column: str, use_mqns: bool = True):
        #initialize the target column
        self.y_column = target_column

        #build the feature columns
        if use_mqns:
            self.mqn_pca_columns = self.mqn_pca_transform()

        if feature_columns:
            if "adduct" in feature_columns:
                feature_column.remove("adduct")
                pass

    def get_pca1_mqn_columns(self, n_components: int = 2, top_n: int = 10):
        """
        Get the MQN columns that contribute most to the first principal component
        
        Args:
            n_components: Number of PCA components to fit
            top_n: Number of top contributing MQN columns to return
            
        Returns:
            List of MQN column names that contribute most to PCA component 1
        """
        pca = PCA(n_components=n_components)
        pca.fit(self.db_mqns)
        
        # Get the loadings for the first principal component
        pca1_loadings = pca.components_[0]
        
        # Get the MQN column names (excluding g_id if it exists)
        mqn_columns = [col for col in self.db_mqns.columns if col != 'g_id']
        
        # Create pairs of (column_name, loading_value) and sort by absolute loading
        column_loadings = list(zip(mqn_columns, pca1_loadings))
        column_loadings.sort(key=lambda x: abs(x[1]), reverse=True)
        
        # Return the top N column names
        top_columns = [col for col, _ in column_loadings[:top_n]]
        return top_columns

    def train_test_split(self, *, test_size=0.2, random_state=26):
        """Optional global split (not used by the HDBSCAN per-cluster flow)."""
        if self.X is None or self.y is None:
            raise RuntimeError("Call build_features() first.")
        self.trainX, self.testX, self.trainy, self.testy = train_test_split(
            self.X, self.y, test_size=test_size, random_state=random_state
        )
        return self.trainX, self.testX, self.trainy, self.testy

    def train(self):
        raise NotImplementedError

    def predict(self, df_like: pd.DataFrame):
        raise NotImplementedError

    # ---- metrics utility ----
    @staticmethod
    def compute_metrics(y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
        r2 = r2_score(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        sqe = (y_true - y_pred) ** 2
        mdse = float(np.median(sqe))
        rel = np.abs((y_true - y_pred) / y_true) * 100
        mre = float(np.mean(rel))
        mdre = float(np.median(rel))
        return {
            "R^2": float(r2),
            "MAE": float(mae),
            "MdSE": mdse,
            "MRE": mre,
            "MdRE": mdre,
            "prop_within_1": float((rel < 1).mean() * 100),
            "prop_within_3": float((rel < 3).mean() * 100),
            "prop_within_5": float((rel < 5).mean() * 100),
            "prop_within_10": float((rel < 10).mean() * 100),
        }

class HDBSCANKMeans:
    # ---- knobs (you can widen these later) ----
    HDBSCAN_PARAMS = {
        'min_cluster_size': [300],
        'min_samples': [220]
    }
    RF_PARAM_GRID = {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 20, 30, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 5, 10]
    }
    RANDOM_STATE = 26
    NOISE_K = 6
    CLUSTER_MIN_SIZE = 500        # ignore tiny clusters
    CLUSTER_MAX_MRE = 2.0         # percent
    VERBOSE = True

    def __init__(self, database_file: str):
        self.db = database_file

        # raw data
        self.data: Optional[pd.DataFrame] = None

        # split
        self.df_train: Optional[pd.DataFrame] = None
        self.df_test: Optional[pd.DataFrame] = None

        # transformers fit on TRAIN ONLY
        self.encoder: Optional[OneHotEncoder] = None
        self.scaler: Optional[StandardScaler] = None

        # feature names
        self.descriptor_columns = [f"MQN_{i+1}" for i in range(42)] + ["LabuteASA", "Chi0v"]
        self.numeric_columns = ["monoisotopic_mass"] + self.descriptor_columns
        self.adduct_cols: List[str] = []

        # trained on TRAIN ONLY
        self.hdbscan_model: Optional[hdbscan.HDBSCAN] = None
        self.kmeans_noise: Optional[KMeans] = None
        self.cluster_models: Dict[str, RandomForestRegressor] = {}  # 'hdbscan_#' or 'kmeans_#'
        self.fallback_model: Optional[RandomForestRegressor] = None

    # ---------- utils ----------
    def _log(self, *a):
        if self.VERBOSE: print(*a)

    @staticmethod
    def _load_df(path: str) -> pd.DataFrame:
        if path.lower().endswith(".csv"):
            return pd.read_csv(path)
        if path.lower().endswith(".db"):
            with sqlite3.connect(path) as conn:
                return pd.read_sql_query("SELECT * FROM master", conn)
        raise ValueError("database_file must be a .csv or .db")

    @staticmethod
    def _compute_descriptors(smiles: str) -> List[float]:
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None: return [0.0] * 44
            mqns = rdMolDescriptors.MQNs_(mol)
            labute_asa = rdMolDescriptors.CalcLabuteASA(mol)
            chi0v = rdMolDescriptors.CalcChi0v(mol)
            return list(mqns) + [float(labute_asa), float(chi0v)]
        except Exception:
            return [0.0] * 44

    @staticmethod
    def _mre_scorer(y_true, y_pred):
        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)
        denom = np.where(y_true == 0, np.finfo(float).eps, y_true)
        return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)

    # ---------- public flow ----------
    def build_features(self):
        """Load raw table; just validates columns. No fitting here."""
        self.data = self._load_df(self.db).copy()
        need = {"smi", "adduct", "monoisotopic_mass", "ccs"}
        miss = need - set(self.data.columns)
        if miss:
            raise ValueError(f"Missing required columns: {miss}")
        self._log(f"[Build] Loaded {len(self.data)} rows.")

    def global_split(self, *, test_size=0.2, random_state=26):
        """Do ONE global split; fit encoder/scaler on TRAIN only; transform both."""
        if self.data is None:
            self.build_features()

        idx_train, idx_test = train_test_split(
            np.arange(len(self.data)), test_size=test_size, random_state=random_state, shuffle=True
        )
        self.df_train = self.data.iloc[idx_train].reset_index(drop=True)
        self.df_test  = self.data.iloc[idx_test].reset_index(drop=True)
        self._log(f"[Split] Train: {len(self.df_train)} | Test: {len(self.df_test)}")

        # fit transformers on TRAIN
        # 1) fit OHE on adduct
        try:
            self.encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        except TypeError:
            self.encoder = OneHotEncoder(sparse=False, handle_unknown="ignore")
        self.encoder.fit(self.df_train[["adduct"]])
        self.adduct_cols = list(self.encoder.get_feature_names_out(["adduct"]))

        # 2) fit scaler on numeric columns using TRAIN stats
        # (numeric columns exist after we compute descriptors; we’ll compute per-frame)
        # So nothing to fit yet; scaler.fit happens in _featurize(train_df, fit=True).
        self._log("[Split] Transformers initialized (encoder fitted on TRAIN).")

    def _featurize(self, df: pd.DataFrame, *, fit_scaler: bool) -> pd.DataFrame:
        """Make X from (smi, adduct, monoisotopic_mass). Fit scaler if requested."""
        desc = df["smi"].apply(self._compute_descriptors)
        desc_df = pd.DataFrame(desc.tolist(), columns=self.descriptor_columns, index=df.index)

        adduct_ohe = self.encoder.transform(df[["adduct"]])
        adduct_df = pd.DataFrame(adduct_ohe, columns=self.adduct_cols, index=df.index)

        X_raw = pd.concat([df[["monoisotopic_mass"]], desc_df, adduct_df], axis=1)

        if fit_scaler or (self.scaler is None):
            self.scaler = StandardScaler()
            self.scaler.fit(X_raw[self.numeric_columns])

        X_num_scaled = self.scaler.transform(X_raw[self.numeric_columns])
        X_num_scaled = pd.DataFrame(X_num_scaled, columns=self.numeric_columns, index=df.index)

        X = pd.concat([X_num_scaled, adduct_df], axis=1)
        return X

    def train(self):
      """Outer search over HDBSCAN params on TRAIN only. No test evaluation here."""
      if self.df_train is None:
          self.global_split()

      X_train = self._featurize(self.df_train, fit_scaler=True)
      y_train = self.df_train["ccs"].astype(float).reset_index(drop=True)

      mqn_cols = [f"MQN_{i+1}" for i in range(42)]
      scorer = make_scorer(self._mre_scorer, greater_is_better=False)

      # normalize param grid to lists
      mcs_list = self.HDBSCAN_PARAMS["min_cluster_size"]
      ms_list = self.HDBSCAN_PARAMS["min_samples"]
      if not isinstance(mcs_list, (list, tuple)): mcs_list = [mcs_list]
      if not isinstance(ms_list, (list, tuple)): ms_list = [ms_list]

      best = {"mre": np.inf}

      for mcs, ms in product(mcs_list, ms_list):
          mcs, ms = int(mcs), int(ms)
          self._log(f"\n[HDBSCAN] Trying min_cluster_size={mcs}, min_samples={ms}…")

          # 1) HDBSCAN on TRAIN
          hdb = hdbscan.HDBSCAN(
              min_cluster_size=mcs,
              min_samples=ms,
              cluster_selection_method="eom",
              prediction_data=True,
          )
          labels_train = hdb.fit_predict(X_train[mqn_cols])
          uniq, cnts = np.unique(labels_train, return_counts=True)
          self._log("[HDBSCAN] Cluster sizes on TRAIN:")
          for u, c in zip(uniq, cnts):
              self._log(f"  label={u:>2}  size={c}")

          # 2) Per-cluster RFs with CV
          per_cluster_models = {}
          cluster_cv_mres, cluster_sizes = [], []
          excluded_train_idx: List[int] = []

          for cl in sorted([c for c in np.unique(labels_train) if c != -1]):
              idx = np.where(labels_train == cl)[0]
              size = len(idx)
              self._log(f"\n[Cluster] HDBSCAN {cl}: size={size}")
              if size < self.CLUSTER_MIN_SIZE:
                  self._log(f"  -> Exclude (size < {self.CLUSTER_MIN_SIZE})")
                  excluded_train_idx.extend(idx.tolist()); continue

              Xc, yc = X_train.iloc[idx], y_train.iloc[idx]
              rf = RandomForestRegressor(random_state=self.RANDOM_STATE)
              grid = GridSearchCV(rf, self.RF_PARAM_GRID, scoring=scorer, cv=3, n_jobs=-1)
              grid.fit(Xc, yc)
              cv_mre = -grid.best_score_
              self._log(f"  CV MRE={cv_mre:.4f}% | Best RF params: {grid.best_params_}")

              if cv_mre >= self.CLUSTER_MAX_MRE:
                  self._log(f"  -> Exclude (CV MRE {cv_mre:.3f}% ≥ {self.CLUSTER_MAX_MRE}%)")
                  excluded_train_idx.extend(idx.tolist())
              else:
                  key = f"hdbscan_{int(cl)}"
                  per_cluster_models[key] = grid.best_estimator_
                  cluster_cv_mres.append(cv_mre); cluster_sizes.append(size)
                  self._log(f"  -> Accepted as {key}")

          # 3) KMeans on noise (TRAIN) → RF per subcluster with CV
          kmeans = None
          noise_idx = np.where(labels_train == -1)[0]
          if len(noise_idx) > 0:
              self._log(f"\n[KMeans] TRAIN noise size={len(noise_idx)}, k={self.NOISE_K}")
              X_noise = X_train.iloc[noise_idx]; y_noise = y_train.iloc[noise_idx]
              kmeans = KMeans(n_clusters=self.NOISE_K, random_state=self.RANDOM_STATE)
              noise_sub = kmeans.fit_predict(X_noise)

              for sub in sorted(np.unique(noise_sub)):
                  sub_local = np.where(noise_sub == sub)[0]
                  sub_idx = [noise_idx[i] for i in sub_local]
                  size = len(sub_idx)
                  self._log(f"[KMeans] Subcluster {sub}: size={size}")
                  if size < self.CLUSTER_MIN_SIZE:
                      self._log(f"  -> Exclude (size < {self.CLUSTER_MIN_SIZE})")
                      excluded_train_idx.extend(sub_idx); continue

                  Xc, yc = X_train.iloc[sub_idx], y_train.iloc[sub_idx]
                  rf = RandomForestRegressor(random_state=self.RANDOM_STATE)
                  grid = GridSearchCV(rf, self.RF_PARAM_GRID, scoring=scorer, cv=3, n_jobs=-1)
                  grid.fit(Xc, yc)
                  cv_mre = -grid.best_score_
                  self._log(f"  CV MRE={cv_mre:.4f}% | Best RF params: {grid.best_params_}")

                  if cv_mre < self.CLUSTER_MAX_MRE:
                      key = f"kmeans_{int(sub)}"
                      per_cluster_models[key] = grid.best_estimator_
                      cluster_cv_mres.append(cv_mre); cluster_sizes.append(size)
                      self._log(f"  -> Accepted as {key}")
                  else:
                      self._log(f"  -> Exclude (CV MRE {cv_mre:.3f}% ≥ {self.CLUSTER_MAX_MRE}%)")
                      excluded_train_idx.extend(sub_idx)
          else:
              self._log("[KMeans] No TRAIN noise; skipping.")

          # 4) Fallback RF (CV on TRAIN)
          excluded_train_idx = sorted(set(excluded_train_idx))
          if len(excluded_train_idx) < 100:
              self._log(f"\n[Fallback] Too few excluded rows ({len(excluded_train_idx)}). Train fallback on ALL TRAIN.")
              X_fallback, y_fallback = X_train, y_train
          else:
              self._log(f"\n[Fallback] Training on excluded TRAIN rows: {len(excluded_train_idx)}")
              X_fallback = X_train.iloc[excluded_train_idx]
              y_fallback = y_train.iloc[excluded_train_idx]

          rf = RandomForestRegressor(random_state=self.RANDOM_STATE)
          grid = GridSearchCV(rf, self.RF_PARAM_GRID, scoring=scorer, cv=3, n_jobs=-1)
          grid.fit(X_fallback, y_fallback)
          fallback = grid.best_estimator_
          fallback_cv_mre = -grid.best_score_
          self._log(f"[Fallback] CV MRE={fallback_cv_mre:.4f}% | Best RF params: {grid.best_params_}")

          # 5) Score this HDBSCAN config (lower is better)
          if cluster_cv_mres:
              weighted_mre = float(np.sum(np.array(cluster_cv_mres) * np.array(cluster_sizes)) / np.sum(cluster_sizes))
              self._log(f"[Score] Weighted per-cluster CV MRE: {weighted_mre:.4f}%")
          else:
              weighted_mre = np.inf
              self._log("[Score] No accepted per-cluster models.")

          chosen_mre = min(weighted_mre, fallback_cv_mre)
          chosen_bundle = "per-cluster" if weighted_mre <= fallback_cv_mre else "fallback"
          self._log(f"[Choose] {chosen_bundle} bundle (MRE={chosen_mre:.4f}%)")

          if chosen_mre < best["mre"]:
              best = {
                  "mre": chosen_mre,
                  "params": {"min_cluster_size": mcs, "min_samples": ms},
                  "hdb": hdb,
                  "kmeans": kmeans,
                  "per_cluster": per_cluster_models if chosen_bundle == "per-cluster" else {},
                  "fallback": fallback,
                  "bundle": chosen_bundle,
              }

      if best["mre"] == np.inf:
          raise RuntimeError("No valid HDBSCAN configuration produced evaluable clusters.")

      # Persist best artifacts
      self.hdbscan_model = best["hdb"]
      self.kmeans_noise = best["kmeans"]
      self.cluster_models = best["per_cluster"]
      self.fallback_model = best["fallback"]

      self._log(f"\n[Result] Best HDBSCAN params: {best['params']}")
      self._log(f"[Result] Best CV MRE (train only): {best['mre']:.4f}% | bundle={best['bundle']}")
      self._log(f"[Result] Models kept: {sorted(self.cluster_models.keys())} + Fallback")

      return {
          "best_hdbscan_params": best["params"],
          "best_cv_mre_train": best["mre"],
          "bundle": best["bundle"],
          "accepted_models": sorted(self.cluster_models.keys()),
      }


    def _featurize_for_predict(self, df_like: pd.DataFrame) -> pd.DataFrame:
        if self.encoder is None or self.scaler is None:
            raise RuntimeError("Call global_split() and train() first.")
        return self._featurize(df_like, fit_scaler=False)

    def predict(self, df_like: pd.DataFrame) -> np.ndarray:
        """Route each row to an accepted cluster RF, or KMeans RF, else fallback."""
        X_new = self._featurize_for_predict(df_like)
        mqn_cols = [f"MQN_{i+1}" for i in range(42)]

        # assign cluster from TRAIN clusters
        try:
            h_labels, _ = hdbscan.approximate_predict(self.hdbscan_model, X_new[mqn_cols].values)
        except Exception:
            h_labels = np.full(len(X_new), -1, dtype=int)

        preds = np.zeros(len(X_new), dtype=float)

        # calculate KMeans sublabels only for rows routed to noise
        noise_idx = np.where(h_labels == -1)[0]
        km_sub = None
        if len(noise_idx) and self.kmeans_noise is not None:
            km_sub = self.kmeans_noise.predict(X_new.iloc[noise_idx])

        for i in range(len(X_new)):
            # try HDBSCAN cluster RF
            if h_labels[i] != -1:
                key = f"hdbscan_{int(h_labels[i])}"
                rf = self.cluster_models.get(key)
                if rf is not None:
                    preds[i] = rf.predict(X_new.iloc[[i]])[0]
                    if self.VERBOSE: print(f"[Predict] row {i} → {key}")
                    continue

            # else try KMeans subcluster RF
            if km_sub is not None and i in noise_idx:
                sub = int(km_sub[list(noise_idx).index(i)])
                key = f"kmeans_{sub}"
                rf = self.cluster_models.get(key)
                if rf is not None:
                    preds[i] = rf.predict(X_new.iloc[[i]])[0]
                    if self.VERBOSE: print(f"[Predict] row {i} → {key}")
                    continue

            # fallback
            preds[i] = self.fallback_model.predict(X_new.iloc[[i]])[0]
            if self.VERBOSE: print(f"[Predict] row {i} → FALLBACK")

        return preds

    # ---------- metrics with optional plots ----------
    @staticmethod
    def compute_metrics(
        y_true: pd.Series,
        y_pred: np.ndarray,
        *,
        plots: bool = False,
        outdir: str = "plots",
        prefix: str = "overall",
        show: bool = False,
    ) -> Dict[str, float]:
        y_true = pd.Series(y_true).astype(float).reset_index(drop=True)
        y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
        if len(y_true) != len(y_pred):
            raise ValueError("y_true and y_pred must have same length")

        r2 = r2_score(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        sqe = (y_true - y_pred) ** 2
        mdse = float(np.median(sqe))
        denom = np.where(y_true == 0, np.finfo(float).eps, y_true)
        rel = np.abs((y_true - y_pred) / denom) * 100.0
        mre = float(np.mean(rel))
        mdre = float(np.median(rel))

        metrics = {
            "R^2": float(r2),
            "MAE": float(mae),
            "MdSE": mdse,
            "MRE": mre,
            "MdRE": mdre,
            "prop_within_1": float((rel < 1).mean() * 100),
            "prop_within_3": float((rel < 3).mean() * 100),
            "prop_within_5": float((rel < 5).mean() * 100),
            "prop_within_10": float((rel < 10).mean() * 100),
        }

        if not plots:
            return metrics

        os.makedirs(outdir, exist_ok=True)
        # 1) Parity
        fig = plt.figure(figsize=(5,5))
        ax = fig.gca()
        ax.scatter(y_true, y_pred, s=8, alpha=0.6)
        mn, mx = float(min(y_true.min(), y_pred.min())), float(max(y_true.max(), y_pred.max()))
        ax.plot([mn, mx], [mn, mx], linewidth=1)
        ax.set_xlabel("True"); ax.set_ylabel("Predicted"); ax.set_title(f"Parity: {prefix}")
        fig.savefig(os.path.join(outdir, f"{prefix}_parity.png"), dpi=200, bbox_inches="tight")
        if not show: plt.close(fig)

        # 2) |% error| hist
        fig = plt.figure(figsize=(6,4))
        ax = fig.gca()
        ax.hist(rel, bins=50)
        for thr in (1,3,5,10): ax.axvline(thr, linestyle="--", linewidth=1)
        ax.set_xlabel("|Percent error|"); ax.set_ylabel("Count"); ax.set_title(f"|% error| hist: {prefix}")
        fig.savefig(os.path.join(outdir, f"{prefix}_abs_pct_error_hist.png"), dpi=200, bbox_inches="tight")
        if not show: plt.close(fig)

        # 3) Residuals vs Pred
        resid = y_true - y_pred
        fig = plt.figure(figsize=(6,4))
        ax = fig.gca()
        ax.scatter(y_pred, resid, s=8, alpha=0.6); ax.axhline(0.0, linewidth=1)
        ax.set_xlabel("Predicted"); ax.set_ylabel("Residual (True - Pred)"); ax.set_title(f"Residuals: {prefix}")
        fig.savefig(os.path.join(outdir, f"{prefix}_residuals_vs_pred.png"), dpi=200, bbox_inches="tight")
        if not show: plt.close(fig)

        # 4) CDF of |% error|
        ape = np.sort(rel.values); cdf = np.arange(1, len(ape)+1) / len(ape)
        fig = plt.figure(figsize=(6,4))
        ax = fig.gca()
        ax.plot(ape, cdf)
        for thr in (1,3,5,10): ax.axvline(thr, linestyle="--", linewidth=1)
        ax.set_xlabel("|Percent error|"); ax.set_ylabel("CDF"); ax.set_title(f"|% error| CDF: {prefix}")
        fig.savefig(os.path.join(outdir, f"{prefix}_abs_pct_error_cdf.png"), dpi=200, bbox_inches="tight")
        if not show: plt.close(fig)

        return metrics

model = HDBSCANKMeans("ML_Dataset.csv")

# 1) Load and split once
model.build_features()
model.global_split(test_size=0.2, random_state=26)

# 2) Train EVERYTHING using TRAIN ONLY (CV happens strictly inside train)
model.train()

# 3) Predict on the GLOBAL TEST SET
yhat_test = model.predict(model.df_test)

# 4) Evaluate + (optionally) save plots
test_metrics = model.compute_metrics(
    model.df_test["ccs"], yhat_test,
    plots=True, outdir="plots", prefix="test"
)
print("\n[Test metrics]", test_metrics)