"""Untitled

"""

import sqlite3
import pickle
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List
import numpy.typing as npt

from sklearn.decomposition import PCA
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import StratifiedShuffleSplit

import os
import matplotlib.pyplot as plt

from itertools import product

# -----------------------
# Base class for training regression models
# -----------------------
class C3SDB:
    def __init__(self, database_file: str, subset_size: int = None):

        self.seed = 69 # random seed

        if os.path.exists(database_file):
            with sqlite3.connect(database_file) as conn:
                self.db_master = pd.read_sql_query("SELECT * FROM master", conn)
                self.db_mqns = pd.read_sql_query("SELECT * FROM mqns", conn)
                self.db = pd.merge(self.db_master, self.db_mqns, on="g_id", how="left")
                
                if subset_size > 0 and subset_size < len(self.db):
                    self.db = self.db.sample(n=subset_size, random_state=self.seed)
        else:
            raise FileNotFoundError(f"Database file {database_file} not found")

        

        self.X_columns: Optional[List[str]] = None #columns for features
        self.y_column: Optional[List[str]] = None #column for target

        self.X_df = None
        self.y_df = None

        # split
        self.trainX_df = None
        self.trainY_df = None

        self.testX_df = None
        self.testY_df = None

        self.validationX_df = None
        self.validationY_df = None

        self.g_id_train = None
        self.g_id_test = None
        self.g_id_validation = None

        self.X_train_ss_ = None
        self.X_test_ss_ = None
        self.X_validation_ss_ = None

        self._EXPLICIT_ADDUCTS = [
            "[M+H]+", "[M+Na]+", "[M-H]-", "[M+NH4]+", "[M+K]+",
            "[M+H-H2O]+", "[M+HCOO]-", "[M+CH3COO]-", "[M+Na-2H]-"
        ]

        self.SScaler_ = None

        # model and prediction-related state
        self.model_ = None
        self.last_predictions_: Optional[pd.Series] = None
        self.prediction_column_name_: Optional[str] = None
        self.last_true_: Optional[pd.Series] = None


    def build_features(self, feature_columns: List[str], target_column: str, use_mqns: bool = True):
        #initialize the target column
        self.y_column = [target_column]

        #build the feature columns
        if use_mqns:
            self.mqn_pca_columns = self.get_pca1_mqn_columns()

        if feature_columns:
            if "adduct" in feature_columns:
                feature_columns.remove("adduct")
                #one hot encode the adduct column
                self.adduct_column_names = self.one_hot_encode_adduct()
                        
            self.X_columns = feature_columns + self.adduct_column_names + self.mqn_pca_columns
                
    def get_pca1_mqn_columns(self, n_components: int = 2, top_n: int = 10):
        """
        Get the MQN columns that contribute most to the first principal component
        
        Args:
            n_components: Number of PCA components to fit
            top_n: Number of top contributing MQN columns to return
            
        Returns:
            List of MQN column names that contribute most to PCA component 1
        """
        # Get the MQN column names (excluding g_id if it exists)
        mqn_columns = [col for col in self.db_mqns.columns if col != 'g_id']

        pca = PCA(n_components=n_components)
        pca.fit(self.db_mqns[mqn_columns])
        
        # Get the loadings for the first principal component
        pca1_loadings = pca.components_[0]
                
        # Create pairs of (column_name, loading_value) and sort by absolute loading
        column_loadings = list(zip(mqn_columns, pca1_loadings))
        column_loadings.sort(key=lambda x: abs(x[1]), reverse=True)
        
        # Return the top N column names
        top_columns = [col for col, _ in column_loadings[:top_n]]
        return top_columns

    def _filter_common_adducts(self, adducts: np.ndarray
                            ) -> np.ndarray :
        """
        To reduce the number of adducts that have to get OneHot encoded, filter through an array of
        adducts and change any adduct that is not among common adducts (defined in _EXPLICIT_ADDUCTS
        constant) to a single label: 'other' 
        
        Parameters
        ----------
        adducts : ``np.ndarray(str)``
            numpy array containing adducts for the dataset
        
        Returns
        -------
        common_adducts : ``np.ndarrray(str)``
            numpy array containing adducts with uncommon adducts replaced with 'other'
        """
        common = adducts.copy()
        for i in range(common.shape[0]):
            if common[i] not in self._EXPLICIT_ADDUCTS:
                common[i] = 'other'
        return common

    def one_hot_encode_adduct(self):
        #Assemble Adduct
        ohe_adducts = None
        
        # convert adducts to OneHot vectors
        self.OHEncoder_ = OneHotEncoder(sparse_output=False, categories='auto')
        common_adducts = self._filter_common_adducts(self.db["adduct"].values.reshape(-1, 1)) #one column, and appropriate amount of rows
        ohe_adducts_matrix = self.OHEncoder_.fit_transform(common_adducts) #stores matrix of common adducts 
        
        # Create column names for the one-hot encoded adducts
        adduct_categories = self.OHEncoder_.categories_[0]
        adduct_column_names = [f"adduct_{cat}" for cat in adduct_categories]
        
        # Convert to DataFrame and concatenate with main DataFrame
        ohe_adducts_df = pd.DataFrame(ohe_adducts_matrix, columns=adduct_column_names, index=self.db.index)
        self.db = pd.concat([self.db, ohe_adducts_df], axis=1)
        
        return adduct_column_names #col names for the one hot encoded adducts

    def center_and_scale(self, fit_scaler: bool = True) -> None :
        """
        Centers and scales the training set features such that each has an average of 0 and variance of 1. Applies
        this transformation to the training and testing features, storing the results in the self.X_train_ss_ and 
        self.X_test_ss_ instance variables, respectively. Also stores a reference to the fitted StandardScaler
        instance for use with future data.

        sets the following instance variables:
            self.SScaler_       (StandardScaler instance)
            self.X_train_ss_    (centered/scaled training set features)
            self.X_test_ss_     (centered/scaled test set features)

        .. note:: 
            
            self.train_test_split(...) must be called first to generate the training features and 
            labels (self.X_train_, self.y_train_) that are used to initialize the StandardScaler
        """
        if self.trainX_df is None and self.testX_df is None:
            msg = 'C3SD: center_and_scale: self.trainX_df and self.testX_df are not initialized, self.train_test_split(...) must be ' + \
                  'called before calling self.center_and_scale(...)'
            raise RuntimeError(msg)

        # perform the scaling
        self.SScaler_ = StandardScaler()
        self.X_train_ss_ = self.SScaler_.fit_transform(self.trainX_df[self.X_columns])
        self.X_test_ss_ = self.SScaler_.transform(self.testX_df[self.X_columns])
        self.X_validation_ss_ = self.SScaler_.transform(self.validationX_df[self.X_columns])

    def train_test_split(self, 
                         stratify: str, 
                         test_frac: float = 0.3,
                         validation_frac: float = 0.1
                         ) -> None:
        """
        Shuffles the data then splits it into training, validation, and test sets, storing each in 
        self.trainX_df, self.trainY_df, self.validationX_df, self.validationY_df, self.testX_df, self.testY_df 
        instance variables. The splitting is done in a stratified manner based on either CCS or dataset source. 
        In the former case, the CCS distribution in the complete dataset is binned into a rough histogram (8 bins) 
        and the train/validation/test sets are split such that they each contain similar proportions of this 
        roughly binned CCS distribution. In the latter case, the train/validation/test sets are split such
        that they each preserve the rough proportions of all dataset sources present in the complete dataset.

        Sets the following instance variables:
        - self.trainX_df       (training set split of features)
        - self.trainY_df       (training set split of labels)
        - self.validationX_df  (validation set split of features)
        - self.validationY_df  (validation set split of labels)
        - self.testX_df        (test set split of features)
        - self.testY_df        (test set split of labels)
        - self.SSSplit_        (StratifiedShuffleSplit instance)

        .. note:: 
            self.build_features(...) must be called first to generate the features and labels (self.X_columns, self.y_column)

        Parameters
        ----------
        stratify : ``str``
            specifies the method of stratification to use when splitting the train/validation/test sets: 'source' 
            for stratification on data source, or 'ccs' for stratification on CCS 
        test_frac : ``float``, default=0.2
            fraction of the complete dataset to reserve as a test set
        validation_frac : ``float``, default=0.2
            fraction of the complete dataset to reserve as a validation set
        """
        # make sure self.build_features(...) has been called
        if self.X_columns is None or self.y_column is None:
            msg = 'CCSML: train_test_split: self.X_columns and self.y_column are not initialized, self.build_features(...) must be called before ' + \
                    'calling self.train_test_split(...)'
            raise RuntimeError(msg)
        
        # make sure stratify is a valid option (if provided)
        if stratify not in ['source', 'ccs']:
            msg = 'CCSML: train_test_split: stratify="{}" invalid, must be "source" or "ccs"'.format(stratify)
            raise RuntimeError(msg)
        
        if stratify == 'source':
            # stratify on dataset source
            y_cat = self.db['src_tag']  
        else:
            y_cat = self._get_categorical_y()

        # initialize StratifiedShuffleSplit for train/test split
        self.SSSplit_ = StratifiedShuffleSplit(n_splits=1, test_size=test_frac, random_state=self.seed)
        
        self.X_df = self.db[self.X_columns + ['g_id']]
        self.y_df = self.db[self.y_column + ['g_id']]
    
        # First split: separate test set from train+validation
        for train_val_index, test_index in self.SSSplit_.split(self.X_df, y_cat):
            self.train_val_X = self.X_df.iloc[train_val_index]
            self.train_val_y = self.y_df.iloc[train_val_index]
            
            self.testX_df = self.X_df.iloc[test_index]
            self.testY_df = self.y_df.iloc[test_index]
            
            # Get stratification categories for train+validation set (use positional indexing)
            train_val_cat = y_cat.iloc[train_val_index]
            
            # Second split: separate train and validation from train+validation set
            # Adjust validation_frac to be relative to the train+validation set
            adjusted_val_frac = validation_frac / (1 - test_frac)
            
            val_split = StratifiedShuffleSplit(n_splits=1, test_size=adjusted_val_frac, random_state=self.seed)
            
            for train_index, val_index in val_split.split(self.train_val_X, train_val_cat):
                self.trainX_df = self.train_val_X.iloc[train_index]
                self.trainY_df = self.train_val_y.iloc[train_index]
                self.validationX_df = self.train_val_X.iloc[val_index]
                self.validationY_df = self.train_val_y.iloc[val_index]
        
        return (self.trainX_df, self.trainY_df, 
                self.validationX_df, self.validationY_df,
                self.testX_df, self.testY_df)

    def _get_categorical_y(self) -> np.ndarray:
        """
        transforms the labels into 'categorical' data (required by StratifiedShuffleSplit) by performing a binning
        operation on the continuous label data. The binning is performed using the rough distribution of label values
        with the following bounds (based on quartiles):
                Q1            Q2             Q3
                |             |              |
          bin1  | bin2 | bin3 | bin4 | bin 5 | bin 6
                       |             |
              (Q2 - Q1 / 2) + Q1     |
                            (Q3 - Q2 / 2) + Q2
        Uses labels stored in the self.y instance variable
        
        Returns
        -------
        cat_y : ``np.ndarray(int)``
            categorical (binned) label data
        """
        from numpy import percentile, digitize
        # get the quartiles from the label distribution
        q1, q2, q3 = percentile(self.y_df, [25, 50, 75])
        # get midpoints
        mp12 = q1 + (q2 - q1) / 2.
        mp23 = q2 + (q3 - q2) / 2.
        # bin boundaries
        bounds = [q1, mp12, q2, mp23, q3]
        return digitize(self.y_df, bounds)

    def train(self, train_X_df: pd.DataFrame, train_Y_df: pd.Series):
        """
        Train the model on the provided training data
        
        Parameters
        ----------
        train_X_df : pd.DataFrame
            Training features
        train_Y_df : pd.Series
            Training labels/targets
        """
        raise NotImplementedError("Not implemented")

    def predict(self,
                X_df: Optional[pd.DataFrame] = None,
                Y_df: Optional[pd.Series] = None) -> pd.Series:
        """
        Make predictions on the provided test set and add predictions as a new column to the full dataset,
        aligning by g_id when provided.

        Parameters
        ----------
        test_X_df : pd.DataFrame
            Features to make predictions on. Must be provided.
        test_Y_series : pd.Series
            Ground truth targets for the provided features. Must be provided.
        g_id_series : pd.Series, optional
            Series of g_id values corresponding to rows in test_X_df, used to align predictions to self.db.
            If not provided, will look for a 'g_id' column in test_X_df.

        Returns
        -------
        pd.Series
            Predictions for the test set, indexed by g_id
        """
        if self.model_ is None:
            raise RuntimeError("No trained model found. Please implement/train in train(...) and set self.model_.")

        if X_df is None or Y_df is None:
            raise ValueError("predict requires both X_df and Y_df to be provided.")
        
        #now simply predict on the provided X_df and Y_df
        y_pred = self.model_.predict(X_df[self.X_columns].values)

        #store the prediction values to the self.testX_df based on g_id with column "prediction"
        X_df["prediction"] = y_pred
        return y_pred

    def save_predictions_to_csv(self, dataset: str):
        """
        export the test set and the predictions to a csv file
        """
        if dataset not in ["test", "validation"]:
            raise ValueError("dataset must be either 'test' or 'validation'.")
        
        if dataset == "test":
            df = pd.merge(self.testX_df, self.testY_df, on="g_id", how="left")
        elif dataset == "validation":
            df = pd.merge(self.validationX_df, self.validationY_df, on="g_id", how="left")
        
        df.to_csv(f"{dataset}_predictions.csv", index=False)

    def compute_metrics(self,
                        y_true: pd.Series,
                        y_pred: pd.Series,
                        dataset: str = "test"
                        ) -> Dict[str, float]:
        """
        Compute regression metrics. Use y_true/y_pred from the given dataset split.

        Parameters
        ----------
        y_true : pd.Series
            Ground truth.
        y_pred : pd.Series
            Predictions.
        dataset : str
            Name of dataset split (e.g., "test", "validation") used in the metric filename.
        """
        if y_true is None or y_pred is None:
            raise ValueError("y_true and y_pred must be provided.")
        
        if len(y_true) != len(y_pred):  
            raise ValueError("y_true and y_pred must have the same length.")    

        y_true = y_true[self.y_column[0]]

        r2 = r2_score(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        sqe = (y_true - y_pred) ** 2
        mdse = float(np.median(sqe))
        rel = np.abs((y_true - y_pred) / y_true) * 100
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
            "dataset": dataset
        }
        # Save metrics dictionary to a CSV file
        metrics_df = pd.DataFrame([metrics])
        metrics_df.to_csv(f"{dataset}_metrics.csv", index=False)
        return metrics

    def save_weights_pickle(self, weights_path: str = "weights.pkl"):
        """
        Save the model weights to a pickle file
        """
        if self.model_ is None:
            raise RuntimeError("No trained model to save. Train and set self.model_ before saving.")

        payload = {
            "model": self.model_,
            "trained_on_scaled": bool(self.X_train_ss_ is not None),
            "scaler": self.SScaler_,
            "x_columns": self.X_columns,
            "y_column": self.y_column,
            "adduct_categories": getattr(self, 'OHEncoder_', None).categories_[0].tolist() if hasattr(self, 'OHEncoder_') else None,
            "seed": self.seed,
        }

        with open(weights_path, "wb") as f:
            pickle.dump(payload, f)

if __name__ == "__main__":
    pass