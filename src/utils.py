from typing import Any, Callable, Dict, Iterable, List, Tuple, Optional, Union
import pandas as pd
import numpy as np
import os
import dgl
import yaml

from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from mordred import Calculator, descriptors

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
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.linear_model import RidgeCV, LassoCV

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.callbacks import ModelCheckpoint
import torch
from tensorflow import keras

from catboost import CatBoostRegressor
import xgboost as xgb
import lightgbm as lgb
import matplotlib.pyplot as plt


class PreprocessSMILES:

    def __init__(self, directory, base_on="smiles"):
        self.directory = directory
        self.base_on_smiles = base_on

    def load_data(self, filename):
        return pd.read_csv(self.directory + filename)

    def preprocess_data(
        self, filename: str, property_: str = "ad7e6027-00b8-4c27-918c-d1561f949ad8"
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
        df = df[(df['oil_property_param_title'] == property_)]
        df = df[df['smiles'].notna()]
        df = df.groupby(['blend_id', 'oil_property_param_title', 'smiles']).agg({'oil_property_param_value': 'mean'}).reset_index()
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
        return similarity_vectors

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
        model = word2vec.Word2Vec.load(self.directory + 'model_300dim.pkl')
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
            return embeddings[0].tolist()

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

    def smiles2sentence(self, df: pd.DataFrame) -> pd.DataFrame:
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
        df = (
            df.groupby(["blend_id", "oil_property_param_title"])
            .agg(
                {
                    "canonical_smiles": lambda x: ", ".join(x),
                    "smiles": lambda x: ", ".join(x),
                    "oil_property_param_value": "mean",
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
        if column == "mol2vec":
            X = np.array([x.vec for x in df[column]])
        else:
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
        df_desc = calc.pandas(self.smiles_df['mol'])
        df_desc = df_desc.dropna()
        #         df_desc = df_desc.select_dtypes(include=np.number).astype('float32')
        #         df_desc = df_desc.loc[:, df_desc.var() > 0.0]
        #         df_descN = pd.DataFrame(MinMaxScaler().fit_transform(df_desc), columns=df_desc.columns)
        self.smiles_df = pd.concat([self.smiles_df, df_desc], axis=1)

    def download_descriptors(self, property_list):
        from tqdm import tqdm
        data = []
        for i in tqdm(self.smiles_df['smiles'], desc="Download properties"):
            props = pcp.get_properties(property_list, i, 'smiles')
            data.append(props)

        rows = []
        columns = data[0][0].keys()
        for i in tqdm(range(len(data)), desc="Processing data"):
            rows.append(data[i][0].values())

        props_df = pd.DataFrame(data=rows, columns=columns)
        self.smiles_df = pd.concat([self.smiles_df, props_df], axis=1)


class SimpleRegressions:
    def __init__(self, X, y):
        if X is None or y is None:
            raise ValueError("X and y cannot be None")

        if not isinstance(X, (list, np.ndarray)):
            raise ValueError("X must be a list or numpy array")

        if not isinstance(y, (list, np.ndarray)):
            raise ValueError("y must be a list or numpy array")

        if len(X) != len(y):
            raise ValueError("Length of X and y must be the same")

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
        catboost = CatBoostRegressor(
            iterations=1000, learning_rate=0.1, loss_function="RMSE", random_seed=1
        )
        catboost.fit(
            self.X_train,
            self.y_train_scaled,
            eval_set=(self.X_test, self.y_test_scaled),
            verbose=1,
        )
        self.evaluation(catboost)

        print("RandomForestRegressor")
        rf = RandomForestRegressor(n_estimators=1000, random_state=42)
        rf.fit(self.X_train, self.y_train_scaled)
        self.evaluation(rf)

        print("XGBRegressor")
        xgboost = xgb.XGBRegressor(n_estimators=1000, learning_rate=0.1, random_state=8)
        xgboost.fit(
            self.X_train,
            self.y_train_scaled,
            eval_set=[(self.X_test, self.y_test_scaled)],
            early_stopping_rounds=10,
            verbose=1,
        )
        self.evaluation(xgboost)

        print("LGBMRegressor")
        lgbm = lgb.LGBMRegressor(n_estimators=1000, learning_rate=0.1, random_state=3)
        lgbm.fit(
            self.X_train,
            self.y_train_scaled,
            eval_set=[(self.X_test, self.y_test_scaled)],
        )
        self.evaluation(lgbm)

        print("GradientBoostingRegressor")
        gb_regressor = GradientBoostingRegressor(
            n_estimators=1000, learning_rate=0.1, random_state=45
        )
        gb_regressor.fit(self.X_train, self.y_train_scaled)
        self.evaluation(gb_regressor)

        print("SVR")
        svr = SVR(kernel="rbf")
        svr.fit(self.X_train, self.y_train_scaled)
        self.evaluation(svr)


class SmallNN:
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.config = self.load_config("config.yml")

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
        model, score = self.neural_model()
        self.evaluation(model)

class LstmRegressor:
    def __init__(
        self,
        units=50,
        dropout_rate=0.2,
        loss="loss",
        optimizer="rmsprop",
        epochs=20,
        batch_size=64,
        neurons_1=128,
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
            y_test_scaled = self.scaler.fit_transform(y_test.reshape(-1, 1)).flatten()
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
