from typing import Any, Callable, Dict, Iterable, List, Tuple, Optional, Union
import pandas as pd
import numpy as np
import os
import dgl
import yaml
import random

import torch
import torch.nn as nn

from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from mordred import Calculator, descriptors
import pubchempy as pcp

from huggingface_hub import snapshot_download
from transformers import AutoTokenizer, AutoModel, AutoModelForMaskedLM

from mol2vec.features import (
    MolSentence,
    DfVec,
    sentences2vec,
    mol2alt_sentence,
    mol2sentence,
)
from gensim.models import word2vec

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow import keras

from catboost import CatBoostRegressor
import xgboost as xgb
import lightgbm as lgb
import matplotlib.pyplot as plt


class PreprocessSMILES:

    def __init__(self, directory, base_on="smiles", task="train"):
        self.directory = directory
        self.base_on_smiles = base_on
        self.task = task

    def load_data(self, filename):
        return pd.read_csv(self.directory + filename)

    def preprocess_data(
        self,
        filename: str,
        filename_descriptors: str,
        property_: str = "ad7e6027-00b8-4c27-918c-d1561f949ad8",
    ) -> pd.DataFrame:
        """
        Load data from a csv file and process it.

        Parameters
        ----------
        filename : str
            Path to a csv file with the data.
        property_ : str, optional
            Property name to filter the data by, by default 'ad7e6027-00b8-4c27-918c-d1561f949ad8'

        Returns
        -------
        pd.DataFrame
            Processed data as a pandas DataFrame.
        """
        df = self.load_data(filename)
        descriptors = self.load_data(filename_descriptors)
        if "train" in filename and self.task == "train":
            df = df[(df["oil_property_param_title"] == property_)]
        df = df[df['smiles'].notna()]
        df = (
            df.groupby(["blend_id", "smiles"])
            .agg({"oil_property_param_value": "mean"})
            .reset_index()
        )
        df["mol"] = df["smiles"].apply(Chem.MolFromSmiles)
        df["canonical_smiles"] = df["smiles"].apply(
            lambda x: (
                Chem.MolToSmiles(
                    Chem.MolFromSmiles(x), isomericSmiles=True, canonical=True
                )
                if x is not None
                else None
            )
        )
        rows_to_drop = df[df['mol'].isnull()].index
        df = df.drop(df[df['blend_id'].isin(df.loc[rows_to_drop, 'blend_id'])].index)
        df["descriptors_array"] = (
            pd.merge(df, descriptors, on="smiles", how="inner")  # ! change
            .iloc[:, 6:]
            .values.tolist()
        )
        return df

    def calculate_similarity(self, smiles_list):
        """
        Calculates molecular similarity scores for a list of SMILES strings.

        Args:
        - smiles_list (list): List of SMILES strings

        Returns:
        - similarity_vectors (list): List of similarity vectors for each compound pair
        """
        fps = [
            AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(smiles), 2)
            for smiles in smiles_list
        ]
        similarity_vectors = []
        for i in range(len(smiles_list)):
            for j in range(i + 1, len(smiles_list)):
                similarity_score = DataStructs.TanimotoSimilarity(fps[i], fps[j])
                similarity_vectors.append(similarity_score)
        return np.array(similarity_vectors)

    def concatenate_sentences(self, vec_list) -> List:
        """
        Concatenate multiple lists of numpy arrays into a single list of numpy arrays.

        Parameters
        ----------
        column_list : List[List[np.ndarray]]
            List of lists of numpy arrays to concatenate.

        Returns
        -------
        List[np.ndarray]
            Concatenated numpy arrays.
        """
        vec_arrays = [x.sentence for x in vec_list]
        concatenated_vec = np.concatenate(vec_arrays)
        return concatenated_vec

    def preprocess_mol2vec(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess SMILES data to mol2vec embeddings.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with SMILES data, where each row is a sample and contains the SMILES string in a column named 'smiles'.

        Returns
        -------
        pd.DataFrame
            Preprocessed data with mol2vec embeddings, where each row is a sample and contains the mol2vec embeddings in a column named 'mol2vec'.
        """
        model = word2vec.Word2Vec.load(self.directory + "/embed_model/model_300dim.pkl")
        df["sentence"] = df.apply(
            lambda x: MolSentence(mol2alt_sentence(x["mol"], 1)), axis=1
        )
        grouped_df = (
            df.groupby("blend_id")["sentence"]
            .apply(self.concatenate_sentences)
            .reset_index()
        )
        grouped_df["mol2vec"] = [
            DfVec(x) for x in sentences2vec(grouped_df["sentence"], model, unseen="UNK")
        ]
        grouped_df["mol2vec"] = grouped_df["mol2vec"].apply(lambda x: x.vec)
        return grouped_df

    def mol_to_dgl_graph(self, mol: Chem.Mol) -> dgl.DGLGraph:
        """
        Convert RDKit molecule object to DGL graph.

        Parameters
        ----------
        mol : Chem.Mol
            RDKit molecule object.

        Returns
        -------
        dgl.DGLGraph
            DGL graph representing the molecule.
        """
        graph = dgl.DGLGraph()

        num_atoms = mol.GetNumAtoms()
        graph.add_nodes(num_atoms)

        for bond in mol.GetBonds():
            graph.add_edges(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
            graph.add_edges(bond.GetEndAtomIdx(), bond.GetBeginAtomIdx())

        node_feats = torch.tensor([[atom.GetAtomicNum()] for atom in mol.GetAtoms()], dtype=torch.float32)
        graph.ndata['feat'] = node_feats

        edge_feats = []
        for bond in mol.GetBonds():
            bond_type = bond.GetBondTypeAsDouble()
            edge_feats.append([bond_type])
            edge_feats.append([bond_type])
        graph.edata['type'] = torch.tensor(edge_feats, dtype=torch.float32)

        return graph

    def generate_graphs(self, df: pd.DataFrame) -> List[dgl.DGLGraph]:
        """
        Convert RDKit molecule objects in a dataframe to DGL graphs.

        Parameters
        ----------
        df : pd.DataFrame
            Dataframe with a column named 'mol' containing RDKit molecule objects.

        Returns
        -------
        List[dgl.DGLGraph]
            List of DGL graphs representing molecules in the input dataframe.
        """
        return df["mol"].apply(self.mol_to_dgl_graph).tolist()

    def embed_smiles(self, df: pd.DataFrame, model: str) -> pd.DataFrame:
        """
        Preprocess SMILES data to a specific language model.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with SMILES data, where each row is a sample and contains the SMILES string in a column named 'smiles'.
        model : str
            Hugging Face model identifier.

        Returns
        -------
        pd.DataFrame
            Preprocessed data with language model embeddings, where each row is a sample and contains the language model embeddings in a column named after the model identifier.
        """

        def process_row(row):
            smiles = row[self.base_on_smiles]
            encoded_inputs = tokenizer(
                smiles, padding=True, truncation=True, return_tensors="pt"
            )
            with torch.no_grad():
                outputs = model(**encoded_inputs)
            embeddings = outputs.pooler_output
            # if embeddings.size(1) < max_length:
            #     padding = torch.zeros(embeddings.size(0), max_length - embeddings.size(1), embeddings.size(2))
            #     embeddings = torch.cat((embeddings, padding), dim=1)
            return np.array(embeddings[0].tolist())

        model_path = "embed_model/" + model

        if not os.path.isdir(model_path):
            _ = snapshot_download(repo_id=model, cache_dir=self.directory + model_path)

        if model == "ibm/MoLFormer-XL-both-10pct":
            model_path = (
                self.directory
                + model_path
                + "/models--ibm--MoLFormer-XL-both-10pct/snapshots/7b12d946c181a37f6012b9dc3b002275de070314"
            )
            model = AutoModelForMaskedLM.from_pretrained(
                model_path
            )  # ? Why Error Tokenizer class MolformerTokenizer does not exist or is not currently imported.
        else:
            model_path = (
                self.directory
                + model_path
                + "/models--DeepChem--ChemBERTa-10M-MTR/snapshots/b65d0a6af3156071d9519e867d695aa265bb393f"
            )
            model = AutoModel.from_pretrained(model_path)
            model.eval()

        tokenizer = AutoTokenizer.from_pretrained(model_path)

        df["embeddings"] = df.apply(process_row, axis=1)
        return df

    def smiles2sentence(self, df: pd.DataFrame, task: str = "test") -> pd.DataFrame:
        """
        Aggregate SMILES data by blend_id and oil_property_param_title to form sentences.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with SMILES data, where each row is a sample and contains the SMILES string in a column named 'smiles'.

        Returns
        -------
        pd.DataFrame
            Aggregated data with sentences formed by concatenating SMILES strings by blend_id and oil_property_param_title.
        """
        def sum_arrays(arrays):
            arrays_np = np.array(arrays)
            summed_array = np.sum(arrays_np, axis=0)
            return np.array(summed_array)
        if task == "train":
            df = (
                df.groupby(by=["blend_id"])
                .agg(
                    func={
                        "canonical_smiles": lambda x: ", ".join(x),
                        "smiles": lambda x: ", ".join(x),
                        "descriptors_array": lambda x: [
                            sum_arrays(arr) for arr in zip(*x)
                        ],
                        "oil_property_param_value": "mean",
                    }
                )
                .reset_index()
            )
            df["similarity_vectors"] = df["smiles"].apply(
                lambda x: self.calculate_similarity(x.split(", "))
            )
        else:
            df = (
                df.groupby(by=["blend_id"])
                .agg(
                    func={
                        "canonical_smiles": lambda x: ", ".join(x),
                        "smiles": lambda x: ", ".join(x),
                        "descriptors_array": lambda x: [
                            sum_arrays(arr) for arr in zip(*x)
                        ],
                    }
                )
                .reset_index()
            )
            df["similarity_vectors"] = df["smiles"].apply(
                lambda x: self.calculate_similarity(x.split(", "))
            )
        return df

    def xy_split(
        self, df: pd.DataFrame, column: str = "mol2vec"
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Split data into X and y, where X is a numpy array of dtype=object and y is a numpy array of dtype=float64.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with SMILES data, where each row is a sample and contains the SMILES string in one or more columns named in columns.
        columns : List[str]
            List of column names containing SMILES strings.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            A tuple of X (an object numpy array of shape (n_samples, n_columns)) and y (a float64 numpy array of shape (n_samples,)).
        """
        target = df.groupby('blend_id')['oil_property_param_value'].mean().dropna()
        blend_id_without_nulls = df.groupby('blend_id')['oil_property_param_value'].mean().dropna().index.tolist()
        df = df[df["blend_id"].isin(blend_id_without_nulls)][column]
        X = np.vstack(np.array(df, dtype=object))
        y = target.values.astype(np.float64)
        return X, y


class GetDescriptors:
    def __init__(self, dataframe):
        self.dataframe = dataframe
        self.smiles_df = self.make_unique_smiles_df()

    def make_unique_smiles_df(self):
        unique_smiles = self.dataframe['smiles'].dropna().unique()
        smiles_df = pd.DataFrame({'id': range(len(unique_smiles)), 'smiles': unique_smiles})
        smiles_df = smiles_df.dropna(subset=['smiles'])
        smiles_df['mol'] = smiles_df['smiles'].apply(lambda x: Chem.MolFromSmiles(x))
        return smiles_df

    def number_of_atoms(self, atom_list):
        for i in atom_list:
            self.smiles_df['num_of_{}_atoms'.format(i)] = self.smiles_df['mol'].apply(lambda x: len(x.GetSubstructMatches(Chem.MolFromSmiles(i))))
        self.smiles_df['hs_mol'] = self.smiles_df['mol'].apply(lambda x: Chem.AddHs(x))
        self.smiles_df['num_of_atoms'] = self.smiles_df['hs_mol'].apply(lambda x: x.GetNumAtoms())
        self.smiles_df['num_of_heavy_atoms'] = self.smiles_df['hs_mol'].apply(lambda x: x.GetNumHeavyAtoms())

    def calculate_descriptors(self):
        calc = Calculator(descriptors, ignore_3D=True)
        df_desc = calc.pandas(self.smiles_df["mol"])
        df_desc = df_desc.dropna()
        #         df_desc = df_desc.select_dtypes(include=np.number).astype('float32')
        #         df_desc = df_desc.loc[:, df_desc.var() > 0.0]
        #         df_descN = pd.DataFrame(MinMaxScaler().fit_transform(df_desc), columns=df_desc.columns)
        self.smiles_df = pd.concat([self.smiles_df, df_desc], axis=1)

    def download_descriptors(self, property_list):
        from tqdm import tqdm
        data = []
        for i in tqdm(self.smiles_df['smiles'], desc="Download properties"):
            props = pcp.get_properties(property_list, i, "smiles")
            data.append(props)

        rows = []
        columns = data[0][0].keys()
        for i in tqdm(range(len(data)), desc="Processing data"):
            rows.append(data[i][0].values())

        props_df = pd.DataFrame(data=rows, columns=columns)
        self.smiles_df = pd.concat([self.smiles_df, props_df], axis=1)


class SimpleRegressions:

    def __init__(self, X, y, test_X=None):
        if X is None or y is None:
            raise ValueError("X and y cannot be None")

        if not isinstance(X, (list, np.ndarray)):
            raise ValueError("X must be a list or numpy array")

        if not isinstance(y, (list, np.ndarray)):
            raise ValueError("y must be a list or numpy array")

        if len(X) != len(y):
            raise ValueError("Length of X and y must be the same")

        self.test_X = test_X
        self.X = X
        self.y = y
        self.catboost = None
        self.lgbm = None
        self.gb_regressor = None
        self.ridge_regressor = None

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, shuffle=True, random_state=42
        )
        self.s_caler = StandardScaler()
        self.y_train_scaled = self.s_caler.fit_transform(
            self.y_train.reshape(-1, 1)
        ).flatten()
        self.y_test_scaled = self.s_caler.transform(
            self.y_test.reshape(-1, 1)
        ).flatten()

    def evaluation(self, model):
        if model is None:
            raise ValueError("Model cannot be None")

        prediction = model.predict(self.X_test)
        prediction = self.s_caler.inverse_transform(prediction.reshape(-1, 1)).flatten()
        mae = mean_absolute_error(self.y_test, prediction)
        mse = mean_squared_error(self.y_test, prediction)

        plt.figure(figsize=(15, 10))
        plt.plot(prediction, "red", label="prediction", linewidth=1.0)
        plt.plot(self.y_test, "green", label="actual", linewidth=1.0)
        plt.legend()
        plt.ylabel("oil_property_param_value")
        plt.title("MAE {}, MSE {}".format(round(mae, 4), round(mse, 4)))
        plt.show()

        print("MAE score:", round(mae, 4))
        print("MSE score:", round(mse, 4))

    def fit_and_evaluate(self):
        print("Catboost")
        self.catboost = CatBoostRegressor(
            iterations=2000, learning_rate=0.1, loss_function="RMSE", random_seed=1
        )
        self.catboost.fit(
            self.X_train,
            self.y_train_scaled,
            eval_set=(self.X_test, self.y_test_scaled),
            verbose=1,
        )
        self.evaluation(self.catboost)

        print("\nLGBMRegressor")
        self.lgbm = lgb.LGBMRegressor(
            n_estimators=2000, learning_rate=0.1, random_state=3
        )
        self.lgbm.fit(
            self.X_train,
            self.y_train_scaled,
            eval_set=[(self.X_test, self.y_test_scaled)],
        )
        self.evaluation(self.lgbm)

        print("\nGradientBoostingRegressor")
        self.gb_regressor = GradientBoostingRegressor(
            n_estimators=2000, learning_rate=0.1, random_state=45
        )
        self.gb_regressor.fit(self.X_train, self.y_train_scaled)
        self.evaluation(self.gb_regressor)

        print("\nStack model")
        self.ridge_train()

    def ridge_train(self):
        catboost_prediction_train = self.catboost.predict(self.X_train)
        lgbm_prediction_train = self.lgbm.predict(self.X_train)
        gb_prediction_train = self.gb_regressor.predict(self.X_train)
        predictions_df_train = pd.DataFrame(
            {
                "CatBoost_Prediction": catboost_prediction_train,
                "LGBM_Prediction": lgbm_prediction_train,
                "GradientBoosting_Prediction": gb_prediction_train,
            }
        )
        catboost_prediction_test = self.catboost.predict(self.X_test)
        lgbm_prediction_test = self.lgbm.predict(self.X_test)
        gb_prediction_test = self.gb_regressor.predict(self.X_test)
        predictions_df_test = pd.DataFrame(
            {
                "CatBoost_Prediction": catboost_prediction_test,
                "LGBM_Prediction": lgbm_prediction_test,
                "GradientBoosting_Prediction": gb_prediction_test,
            }
        )
        self.ridge_regressor = RandomForestRegressor(
            n_estimators=1000, criterion="absolute_error", verbose=1
        )
        self.ridge_regressor.fit(predictions_df_train, self.y_train_scaled)
        prediction = self.ridge_regressor.predict(predictions_df_test)
        prediction = self.s_caler.inverse_transform(prediction.reshape(-1, 1)).flatten()
        mae = mean_absolute_error(
            self.s_caler.inverse_transform(self.y_test_scaled.reshape(-1, 1)).flatten(),
            prediction,
        )
        mse = mean_squared_error(
            self.s_caler.inverse_transform(self.y_test_scaled.reshape(-1, 1)).flatten(),
            prediction,
        )

        plt.figure(figsize=(15, 10))
        plt.plot(prediction, "red", label="prediction", linewidth=1.0)
        plt.plot(
            self.s_caler.inverse_transform(self.y_test_scaled.reshape(-1, 1)).flatten(),
            "green",
            label="actual",
            linewidth=1.0,
        )
        plt.legend()
        plt.ylabel("oil_property_param_value")
        plt.title("MAE {}, MSE {}".format(round(mae, 4), round(mse, 4)))
        plt.show()

        print("MAE score:", round(mae, 4))
        print("MSE score:", round(mse, 4))
        return self.ridge_regressor

    def ridge_test(self):
        catboost_prediction_test = self.catboost.predict(self.test_X)
        lgbm_prediction_test = self.lgbm.predict(self.test_X)
        gb_prediction_test = self.gb_regressor.predict(self.test_X)
        predictions_df_test = pd.DataFrame(
            {
                "CatBoost_Prediction": catboost_prediction_test,
                "LGBM_Prediction": lgbm_prediction_test,
                "GradientBoosting_Prediction": gb_prediction_test,
            }
        )
        # ridge_regressor = self.ridge_train()
        preds = self.ridge_regressor.predict(predictions_df_test)
        return (
            self.ridge_regressor,
            self.catboost,
            self.lgbm,
            self.gb_regressor,
            self.s_caler.inverse_transform(preds.reshape(-1, 1)).flatten(),
        )


class SmallNN:

    def __init__(self, X: np.ndarray, y: np.ndarray, config):
        self.config = config
        self.model = None
        if X is None or y is None:
            raise ValueError("X and y cannot be None")

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.1, shuffle=True, random_state=42
        )
        self.s_caler = MinMaxScaler()
        self.y_train_scaled = self.s_caler.fit_transform(
            self.y_train.reshape(-1, 1)
        ).flatten()
        self.y_test_scaled = self.s_caler.transform(
            self.y_test.reshape(-1, 1)
        ).flatten()

    def load_config(self, file_path: str):
        with open(file_path, "r") as f:
            self.config = yaml.safe_load(f)
        return self.config["SmallNN"]

    def neural_model(self) -> keras.Model:
        np.random.seed(1)
        score = []
        kfold = KFold(n_splits=self.config["n_splits"], shuffle=True)

        model = Sequential()
        model.add(
            Dense(
                self.config["neurons"],
                input_dim=self.X_train.shape[1],
                activation=self.config["activation"],
            )
        )
        model.add(Dense(self.config["neurons"], activation=self.config["activation"]))
        model.add(Dense(1, activation=self.config["output_activation"]))

        opt = keras.optimizers.Adam(learning_rate=self.config["learning_rate"])
        model.compile(
            loss="mean_squared_error", optimizer=opt, metrics=["mean_absolute_error"]
        )

        rlrop = ReduceLROnPlateau(
            monitor="val_loss",
            factor=self.config["lr_reduction_factor"],
            patience=self.config["lr_patience"],
        )

        for train, validation in kfold.split(self.X_train, self.y_train_scaled):

            model.fit(
                self.X_train[train],
                self.y_train_scaled[train],
                epochs=self.config["epochs"],
                batch_size=self.config["batch_size"],
                callbacks=[rlrop],
                verbose=self.config["verbose"],
                validation_data=(
                    self.X_train[validation],
                    self.y_train_scaled[validation],
                ),
            )

            score.append(model.evaluate(self.X_test, self.y_test_scaled))

        return model, score

    def evaluation(self, model: keras.Model):
        if model is None:
            raise ValueError("Model cannot be None")

        prediction = model.predict(self.X_test)
        prediction = self.s_caler.inverse_transform(prediction.reshape(-1, 1)).flatten()
        mae = mean_absolute_error(self.y_test, prediction)
        mse = mean_squared_error(self.y_test, prediction)

        plt.figure(figsize=(15, 10))
        plt.plot(prediction, "red", label="prediction", linewidth=1.0)
        plt.plot(self.y_test, "green", label="actual", linewidth=1.0)
        plt.legend()
        plt.ylabel("oil_property_param_value")
        plt.title("MAE {}, MSE {}".format(round(mae, 4), round(mse, 4)))
        plt.show()

        print("MAE score:", round(mae, 4))
        print("MSE score:", round(mse, 4))

    def fit_and_evaluate(self):
        self.model, score = self.neural_model()
        self.evaluation(self.model)

    def predict(self, X):
        if self.model is None:
            raise ValueError("The model has not been trained yet.")

        y_pred_scaled = self.model.predict(X)
        y_pred_original = self.s_caler.inverse_transform(
            y_pred_scaled.reshape(-1, 1)
        ).flatten()
        return y_pred_original


class LstmRegressor:

    def __init__(
        self,
        units=50,
        dropout_rate=0.2,
        loss="loss",
        optimizer="rmsprop",
        epochs=20,
        batch_size=64,
        neurons_1=689,
        neurons_2=64,
        scaler=None,
    ):
        self.units = units
        self.neurons_1 = neurons_1
        self.neurons_2 = neurons_2
        self.dropout_rate = dropout_rate
        self.optimizer = optimizer
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = None
        self.reduce_lr = ReduceLROnPlateau(monitor="loss")
        self.scaler = scaler
        self.loss_ = loss

    def create_model(self, input_shape):
        model = Sequential()
        model.add(
            LSTM(units=self.neurons_1, return_sequences=True, input_shape=input_shape)
        )
        model.add(Dropout(self.dropout_rate))
        model.add(LSTM(units=self.neurons_2, return_sequences=True))
        model.add(Dropout(self.dropout_rate))
        model.add(LSTM(units=self.units, return_sequences=True))
        model.add(Dropout(self.dropout_rate))
        model.add(LSTM(units=self.units))
        model.add(Dropout(self.dropout_rate))
        model.add(Dense(units=1))
        model.compile(optimizer=self.optimizer, loss="mean_squared_error")
        return model

    def fit(self, X_train, y_train, X_test, y_test):
        input_shape = (X_train.shape[1], 1)
        if self.scaler:
            y_train_scaled = self.scaler.fit_transform(y_train.reshape(-1, 1)).flatten()
            y_test_scaled = self.scaler.transform(y_test.reshape(-1, 1)).flatten()
        else:
            y_train_scaled = y_train
            y_test_scaled = y_test

        self.model = KerasRegressor(
            build_fn=self.create_model,
            input_shape=input_shape,
            epochs=self.epochs,
            batch_size=self.batch_size,
            verbose=1,
        )
        checkpoint = self.create_model_checkpoint(
            filepath="best_model_weights.h5", monitor="val_loss", save_best_only=True
        )
        history = self.model.fit(
            X_train,
            y_train_scaled,
            validation_data=(X_test, y_test_scaled),
            callbacks=[self.reduce_lr, checkpoint],
        )

        self.plot_training_history(history)

        y_pred_scaled = self.model.predict(X_test)
        y_pred_original = self.scaler.inverse_transform(
            y_pred_scaled.reshape(-1, 1)
        ).flatten()
        return self.model, y_pred_original

    def predict(self, X):
        if self.model is None:
            raise ValueError("The model has not been trained yet.")

        y_pred_scaled = self.model.predict(X)
        y_pred_original = self.scaler.inverse_transform(
            y_pred_scaled.reshape(-1, 1)
        ).flatten()
        return y_pred_original

    def plot_training_history(self, history):
        plt.figure(figsize=(12, 6))

        plt.subplot(1, 2, 1)
        plt.plot(history.history[self.loss_], label="Training Loss", color="blue")
        plt.plot(
            history.history["val_" + self.loss_], label="Validation Loss", color="red"
        )
        plt.title("Loss")
        plt.xlabel("Epochs")
        plt.legend()
        plt.title("Metrics")
        plt.xlabel("Epochs")
        plt.legend(loc="upper right")

        plt.tight_layout()
        plt.show()

    def create_model_checkpoint(
        self,
        filepath,
        monitor="val_mean_absolute_error",
        save_best_only=True,
        save_weights_only=True,
        mode="auto",
        verbose=0,
    ):
        checkpoint = ModelCheckpoint(
            filepath=filepath,
            monitor=monitor,
            save_best_only=save_best_only,
            save_weights_only=save_weights_only,
            mode=mode,
            verbose=verbose,
        )
        return checkpoint


class Dataset:
    def __init__(self, data_x, data_y=None):
        super(Dataset, self).__init__()
        self.data_x = data_x
        self.data_y = data_y

    def __len__(self):
        return len(self.data_x)

    def __getitem__(self, idx):
        if self.data_y is not None:
            return self.data_x[idx], self.data_y[idx]
        else:
            return self.data_x[idx]

    def augment_data(self, x_, y_):
        copy_x = x_.copy()
        new_x = []
        new_y = y_.copy()
        dim = x_.shape[2]
        k = int(0.3 * dim)
        for i in range(x_.shape[0]):
            idx = random.sample(range(dim), k=k)
            copy_x[i, :, idx] = 0
            new_x.append(copy_x[i])
        return np.stack(new_x, axis=0), new_y


class DataLoader:

    def __init__(self, main_array, static_cols, dynamic_cols, task="train"):
        self.main_array = main_array
        self.static_cols = static_cols
        self.dynamic_cols = dynamic_cols
        self.data_x = self.combine_features()
        self.task = task
        if self.task == "train":
            self.data_y = np.array(main_array["oil_property_param_value"]).reshape(
                -1, 1
            )
        else:
            self.data_y = None

    def process_similarity_vectors(self, similarity_vectors):
        if (
            len(similarity_vectors) == 0
        ):  # если была только 1 молекула, то сходства нет - 0
            return np.array([0, 0, 0, 0, 0, 0])
        else:
            from scipy.stats import kurtosis

            min_value = np.min(similarity_vectors)
            mean_value = np.mean(similarity_vectors)
            median_value = np.median(similarity_vectors)
            max_value = np.max(similarity_vectors)
            kurtosis_value = kurtosis(similarity_vectors)
            std_value = np.std(similarity_vectors)
            processed_vector = np.array(
                [
                    min_value,
                    mean_value,
                    median_value,
                    max_value,
                    std_value,
                    kurtosis_value,
                ]
            )
            return processed_vector

    def combine_features(self):
        new_vecs = []
        add_dynamic_max_len = 0
        fixed_len = 0
        fixed_arrays = [self.main_array[col] for col in self.static_cols]
        fixed_len = sum(len(arr[0]) for arr in fixed_arrays)

        if self.dynamic_cols:
            add_dynamic_max_len = 6
            dynamic_col_data = self.main_array[self.dynamic_cols[0]]
            for i in range(len(self.main_array)):
                vec_ = np.array([])
                for arr in fixed_arrays:
                    vec_ = np.concatenate([vec_, arr[i]])

                if dynamic_col_data:
                    similarity_vectors = dynamic_col_data[i]
                    processed_vector = self.process_similarity_vectors(
                        similarity_vectors
                    )
                    pad_len = add_dynamic_max_len - processed_vector.shape[0]
                    processed_vector_padded = np.pad(
                        processed_vector,
                        (0, pad_len),
                        mode="constant",
                        constant_values=0,
                    )

                    vec_ = np.concatenate([vec_, processed_vector_padded])

                new_vecs.append(vec_)

        else:
            for i in range(len(self.main_array)):
                vec_ = np.array([])
                for arr in fixed_arrays:
                    vec_ = np.concatenate([vec_, arr[i]])
                new_vecs.append(vec_)

        max_len = fixed_len + add_dynamic_max_len
        return np.array(new_vecs).reshape(len(self.main_array), 1, max_len)

    def get_dataset(self):
        if self.data_y is not None:
            print(f"Shape of data_x: {self.data_x.shape}")
            print(f"Shape of data_y: {self.data_y.shape}")
            return Dataset(self.data_x, self.data_y)
        else:
            print(f"Shape of data_x: {self.data_x.shape}")
            return Dataset(self.data_x), None


class ConvRegressor(nn.Module):
    def __init__(self, config):
        super(ConvRegressor, self).__init__()
        self.name = "ConvRegressor"
        self.config = config
        self.conv_block = nn.Sequential(
            nn.Conv1d(1, 8, 5, stride=1, padding=0),
            nn.Dropout(0.3),
            nn.Conv1d(8, 8, 5, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv1d(8, 16, 5, stride=2, padding=0),
            nn.Dropout(0.3),
            nn.AvgPool1d(11),
            nn.Conv1d(16, 4, 3, stride=3, padding=0),
            nn.Flatten(),
        )

        self.linear = nn.Sequential(
            nn.Linear(self.config["Conv"]["input_layer"], 1024),
            nn.Dropout(0.3),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.Dropout(0.3),
            nn.ReLU(),
        )
        self.head1 = nn.Linear(512, 1)

        self.loss1 = nn.MSELoss()
        self.loss3 = nn.L1Loss()

    def forward(self, x, y=None):
        if y is None:
            out = self.conv_block(x)
            out = self.head1(self.linear(out))
            return out
        else:
            out = self.conv_block(x)
            out = self.head1(self.linear(out))
            loss1 = 0.4 * self.loss1(out, y) + 0.6 * self.loss3(out, y)
            return loss1


class LSTMRegressor(nn.Module):
    def __init__(self, config):
        super(LSTMRegressor, self).__init__()
        self.name = "LSTMRegressor"
        self.config = config
        self.lstm = nn.LSTM(
            self.config["alter_shapes"][1],
            self.config["batch_size_train"],
            num_layers=2,
            batch_first=True,
        )
        self.linear = nn.Sequential(
            nn.Linear(self.config["Lstm"]["input_layer"], 1024),
            nn.Dropout(0.3),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.Dropout(0.3),
            nn.ReLU(),
        )
        self.head1 = nn.Linear(512, 1)

        self.loss1 = nn.MSELoss()
        self.loss3 = nn.L1Loss()

    def forward(self, x, y=None):
        shape1, shape2 = self.config["alter_shapes"]
        x = x.reshape(x.shape[0], shape1, shape2)
        if y is None:
            out, (hn, cn) = self.lstm(x)
            out = out.reshape(out.shape[0], -1)
            out = torch.cat([out, hn.reshape(hn.shape[1], -1)], dim=1)
            out = self.head1(self.linear(out))
            return out
        else:
            out, (hn, cn) = self.lstm(x)

            out = out.reshape(out.shape[0], -1)
            out = torch.cat([out, hn.reshape(hn.shape[1], -1)], dim=1)
            out = self.head1(self.linear(out))
            loss1 = 0.4 * self.loss1(out, y) + 0.6 * self.loss3(out, y)
            return loss1


class GRURegressor(nn.Module):
    def __init__(self, config):
        super(GRURegressor, self).__init__()
        self.name = "GRURegressor"
        self.config = config
        self.gru = nn.GRU(
            self.config["alter_shapes"][1],
            self.config["batch_size_train"],
            num_layers=2,
            batch_first=True,
        )
        self.linear = nn.Sequential(
            nn.Linear(self.config["Gru"]["input_layer"], 1024),
            nn.Dropout(0.3),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.Dropout(0.3),
            nn.ReLU(),
        )
        self.head1 = nn.Linear(512, 1)

        self.loss1 = nn.MSELoss()
        self.loss3 = nn.L1Loss()

    def forward(self, x, y=None):
        shape1, shape2 = self.config["alter_shapes"]
        # print("Forward:: ДО::", x.shape)
        x = x.reshape(x.shape[0], shape1, shape2)
        # print("\nForward:: После::", x.shape)
        if y is None:
            out, (hn, cn) = self.gru(x)
            out = out.reshape(out.shape[0], -1)
            out = torch.cat([out, hn.reshape(hn.shape[1], -1)], dim=1)
            out = self.head1(self.linear(out))
            return out
        else:
            out, (hn, cn) = self.gru(x)
            out = out.reshape(out.shape[0], -1)
            out = torch.cat([out, hn.reshape(hn.shape[1], -1)], dim=1)
            out = self.head1(self.linear(out))
            loss1 = 0.4 * self.loss1(out, y) + 0.6 * self.loss3(out, y)
            return loss1


class Training:

    def __init__(self, config, X, y, test_X, cross_validation=True):
        self.scaler = StandardScaler()
        self.config = config
        self.cross_validation = cross_validation
        self.X = X
        self.test_X = test_X
        self.y = y
        self.y_scale = self.scaler.fit_transform(self.y)
        self.trained_models = self.cross_validate_models(
            self.X,
            self.y_scale,
            self.config["epochs"],
            self.config["clip_norm"],
            self.cross_validation,
        )

    def train_step(self, dataloader, model, opt, clip_norm):
        model.train()
        train_losses = []
        for x, target in dataloader:
            # print("Trainstep:: ДО::", x.shape, target.shape)
            if torch.cuda.is_available():
                model.cuda()
                x = x.cuda()
                target = target.cuda()
            loss = model(x, target)
            train_losses.append(loss.item())
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
            opt.step()
        return np.mean(train_losses)

    def validation_step(self, dataloader, model):
        model.eval()
        val_losses = []
        val_mse = []
        val_mae = []
        for x, target in dataloader:
            if torch.cuda.is_available():
                model.cuda()
                x = x.cuda()
                target = target.cuda()
            loss = model(x, target)
            pred = self.scaler.inverse_transform(model(x).detach().cpu().numpy())
            rescale_y = self.scaler.inverse_transform(target.cpu().numpy())
            val_mse.append(mean_squared_error(pred, rescale_y))
            val_mae.append(mean_absolute_error(pred, rescale_y))
            val_losses.append(loss.item())
        return np.mean(val_losses), np.mean(val_mse), np.mean(val_mae)

    def evaluate_steps(self, dataloader, model):
        model.eval()
        preds = []
        targets = []
        for x, target in dataloader:
            if torch.cuda.is_available():
                model.cuda()
                x = x.cuda()
                target = target.cuda()
            pred = self.scaler.inverse_transform(model(x).detach().cpu().numpy())
            rescale_y = self.scaler.inverse_transform(target.cpu().numpy())
            preds.append(pred)
            targets.append(rescale_y)
        return np.concatenate(preds, axis=0), np.concatenate(targets, axis=0)

    def train_function(
        self, model, x_train, y_train, x_val, y_val, epochs=20, clip_norm=1.0
    ):
        from torch.utils.data import DataLoader

        if model.name in ["GRURegressor"]:
            print("lr", self.config["learning_rate_gru"])
            opt = torch.optim.Adam(
                model.parameters(), lr=self.config["learning_rate_gru"]
            )
        else:
            print("lr", self.config["learning_rate"])
            opt = torch.optim.Adam(model.parameters(), lr=self.config["learning_rate"])
        if torch.cuda.is_available():
            model.cuda()

        data_x_train = torch.FloatTensor(x_train)
        data_y_train = torch.FloatTensor(y_train)
        data_x_val = torch.FloatTensor(x_val)
        data_y_val = torch.FloatTensor(y_val)
        train_dataloader = DataLoader(
            Dataset(data_x_train, data_y_train),
            num_workers=4,
            batch_size=self.config["batch_size_train"],
            shuffle=True,
        )
        val_dataloader = DataLoader(
            Dataset(data_x_val, data_y_val),
            num_workers=4,
            batch_size=self.config["batch_size_train"],
            shuffle=False,
        )
        best_loss = np.inf
        best_weights = None

        history = {"train_loss": [], "val_loss": [], "val_mse": [], "val_mae": []}

        for e in range(epochs):
            loss = self.train_step(train_dataloader, model, opt, clip_norm)
            val_loss, val_mse, val_mae = self.validation_step(val_dataloader, model)
            history["train_loss"].append(loss)
            history["val_loss"].append(val_loss)
            history["val_mse"].append(val_mse)
            history["val_mae"].append(val_mae)

            if val_mse < best_loss:
                best_loss = val_mse
                best_weights = model.state_dict()
                print("BEST ----> ")
            print(
                f"{model.name} Epoch {e}, train_loss {round(loss,3)}, val_loss {round(val_loss, 3)}, val_mae {val_mae}"
            )
        model.load_state_dict(best_weights)
        preds, targets = self.evaluate_steps(val_dataloader, model)
        self.evaluation(model.name, preds, targets)
        return model, history

    def plot_training_history(self, model_name, history):
        plt.figure(figsize=(24, 8))

        plt.subplot(1, 2, 1)
        plt.plot(history["train_loss"], label="Training Loss", color="blue")
        plt.plot(history["val_loss"], label="Validation Loss", color="red")
        plt.title(f"Loss for {model_name}")
        plt.xlabel("Epochs")
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(history["val_mae"], label="Validation MAE", color="green")
        plt.title(f"Metrics for {model_name}")
        plt.xlabel("Epochs")
        plt.legend(loc="upper right")

        plt.tight_layout()
        plt.show()

    def evaluation(self, model_name, prediction, y_test):
        mae = mean_absolute_error(y_test, prediction)
        mse = mean_squared_error(y_test, prediction)

        plt.figure(figsize=(15, 10))
        plt.plot(prediction, "red", label="prediction", linewidth=1.0)
        plt.plot(y_test, "green", label="actual", linewidth=1.0)
        plt.legend()
        plt.ylabel("oil_property_param_value")
        plt.title(
            "Model {}: MAE {}, MSE {}".format(
                (model_name), round(mae, 2), round(mse, 2)
            )
        )
        plt.show()

        print("MAE score:", round(mae, 4))
        print("MSE score:", round(mse, 4))

    def cross_validate_models(
        self, X, y, epochs=120, clip_norm=1.0, cross_validation=True
    ):

        trained_models = []

        if cross_validation:
            kf_cv = KFold(
                n_splits=self.config["n_splits"], shuffle=True, random_state=42
            )

            for Model in [GRURegressor, ConvRegressor, LSTMRegressor]:
                prev_train_len = None
                prev_val_len = None

                for i, (train_idx, val_idx) in enumerate(kf_cv.split(X)):
                    print(f"\nSplit {i+1}/{self.config['n_splits']}...")
                    train_len = len(train_idx)
                    val_len = len(val_idx)

                    if prev_train_len is not None and prev_train_len != train_len:
                        print("Train lengths are not consistent. Skipping this split.")
                        print(prev_train_len - train_len)
                        continue

                    if prev_val_len is not None and prev_val_len != val_len:
                        print(
                            "Validation lengths are not consistent. Skipping this split."
                        )
                        print(prev_val_len - val_len)
                        continue

                    prev_train_len = train_len
                    prev_val_len = val_len

                    x_train, x_val = X[train_idx], X[val_idx]
                    y_train, y_val = y[train_idx], y[val_idx]
                    model = Model(self.config)
                    model, history = self.train_function(
                        model,
                        x_train,
                        y_train,
                        x_val,
                        y_val,
                        epochs=epochs,
                        clip_norm=clip_norm,
                    )
                    self.plot_training_history(model.name, history)
                    model.to("cpu")
                    trained_models.append(model)
                    torch.cuda.empty_cache()

        else:
            x_train, x_val, y_train, y_val = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            for Model in [GRURegressor, ConvRegressor, LSTMRegressor]:
                model = Model(self.config)
                model, history = self.train_function(
                    model,
                    x_train,
                    y_train,
                    x_val,
                    y_val,
                    epochs=epochs,
                    clip_norm=clip_norm,
                )
                self.plot_training_history(model.name, history)
                model.to("cpu")
                trained_models.append(model)
                torch.cuda.empty_cache()

        return trained_models

    def inference_pytorch(self, model, dataloader):
        model.eval()
        preds = []
        for x in dataloader:
            if torch.cuda.is_available():
                model.cuda()
                x = x.cuda()
            pred = self.scaler.inverse_transform(model(x).detach().cpu().numpy())
            preds.append(pred)
        model.to("cpu")
        torch.cuda.empty_cache()
        return np.concatenate(preds, axis=0)

    def average_prediction(self):
        from torch.utils.data import DataLoader

        all_preds = []
        test_dataloader = DataLoader(
            Dataset(torch.FloatTensor(self.test_X)),
            num_workers=4,
            batch_size=self.config["batch_size_test"],
            shuffle=False,
        )
        for i, model in enumerate(self.trained_models):
            current_pred = self.inference_pytorch(model, test_dataloader)
            all_preds.append(current_pred)
        return np.stack(all_preds, axis=1).mean(axis=1)

    def weighted_average_prediction(self, model_wise=[0.5, 0.25, 0.25], fold_wise=None):
        all_preds = []
        test_dataloader = DataLoader(
            Dataset(torch.FloatTensor(self.test_X)),
            num_workers=4,
            batch_size=self.config["batch_size_test"],
            shuffle=False,
        )
        for i, model in enumerate(self.trained_models):
            current_pred = self.inference_pytorch(model, test_dataloader)
            current_pred = model_wise[i % 3] * current_pred
            if fold_wise:
                current_pred = fold_wise[i // 3] * current_pred
            all_preds.append(current_pred)
        return np.stack(all_preds, axis=1).sum(axis=1)


# -------------- ADD FUNCTIONS ------------------------------- #


def load_config(file_path: str, data: str, model_name: str):
    with open(file_path, "r") as f:
        config = yaml.safe_load(f)
    return config[data][model_name]


def seed_everything():
    seed = 42
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print("-----Seed Set!-----")
