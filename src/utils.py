# import pandas as pd
# from rdkit import Chem
# from mol2vec.features import MolSentence, DfVec, sentences2vec, mol2alt_sentence, mol2sentence
# from gensim.models import word2vec
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.linear_model import RidgeCV, LassoCV, RandomForestRegressor
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import mean_absolute_error, mean_squared_error

class SMILES2Wec:
    def __init__(self, directory):
        self.directory = directory
    
    def load_data(self, filename):
        return pd.read_csv(self.directory + filename)
    
    def preprocess_data(self, df):
        df = df[(df['oil_property_param_title'] == 'ad7e6027-00b8-4c27-918c-d1561f949ad8')]
        df = df[df['smiles'].notna()]
        df = df.groupby(['blend_id', 'oil_property_param_title', 'smiles']).agg({'oil_property_param_value': 'mean'}).reset_index()
        df['mol'] = df['smiles'].apply(lambda x: Chem.MolFromSmiles(x))
        rows_to_drop = df[df['mol'].isnull()].index
        df = df.drop(df[df['blend_id'].isin(df.loc[rows_to_drop, 'blend_id'])].index)
        return df
    
    def concatenate_sentences(self, vec_list):
        vec_arrays = [x.sentence for x in vec_list]
        concatenated_vec = np.concatenate(vec_arrays)
        return concatenated_vec
    
    def preprocess_mol2vec(self, df):
        model = word2vec.Word2Vec.load(self.directory + 'model_300dim.pkl')
        df['sentence'] = df.apply(lambda x: MolSentence(mol2alt_sentence(x['mol'], 1)), axis=1)
        grouped_df = df.groupby('blend_id')['sentence'].apply(self.concatenate_sentences).reset_index()
        grouped_df['mol2vec'] = [DfVec(x) for x in sentences2vec(grouped_df['sentence'], model, unseen='UNK')]
        return grouped_df
    
    def smiles2sentences(self, df):
        # grouped_df = df.groupby(['blend_id', 'oil_property_param_title', 'smiles']).agg({'oil_property_param_value': 'mean'}).reset_index()
        return df.groupby(['blend_id', 'oil_property_param_title']).agg({'smiles': list, 'oil_property_param_value': 'mean'}).reset_index()
    
    def xy_split(self, df):
        target = df.groupby('blend_id')['oil_property_param_value'].mean().dropna()
        blend_id_without_nulls = df.groupby('blend_id')['oil_property_param_value'].mean().dropna().index.tolist()
        df = df[df['blend_id'].isin(blend_id_without_nulls)]
        X = np.array([x.vec for x in df['mol2vec']])
        y = target.values
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