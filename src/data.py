import numpy as np
import pandas as pd
from typing import List
from sklearn.preprocessing import StandardScaler
from abc import ABC, abstractmethod


class Dataset(ABC):
    def __init__(self, seed: int = 0, folds: int = 5) -> None:
        self.seed = seed
        self.folds = folds
    
    def load_data(self, filepath: str, sep: str = ','):
        df = pd.read_csv(filepath, sep=sep).sample(frac=1, random_state=self.seed)
        return df
    
    def get_feat_types(self, df: pd.DataFrame):
        cat_feats, num_feats = [], []
        for feat in df.columns:
            if df[feat].dtype == object:
                cat_feats.append(feat)
            elif len(set(df[feat])) > 2:
                num_feats.append(feat) 
        return cat_feats, num_feats
    
    def scale_num_feats(self, df1: pd.DataFrame, df2: pd.DataFrame, num_feats: List[str]):
        self.scaler = StandardScaler()
        df1[num_feats] = self.scaler.fit_transform(df1[num_feats].values)
        df2[num_feats] = self.scaler.transform(df2[num_feats].values)
        return df1, df2
    
    def split_data(self, X: pd.DataFrame, y: pd.Series, fold: int):
        n = self.folds
        x_chunks, y_chunks = [], []
        for i in range(n):
            start = int(i/n * len(X))
            end = int((i+1)/n * len(X))
            x_chunks.append(X.iloc[start:end])
            y_chunks.append(y.iloc[start:end])

        X_test, y_test = x_chunks.pop(fold), y_chunks.pop(fold)
        X_train, y_train = pd.concat(x_chunks), pd.concat(y_chunks)
        
        return X_train, y_train, X_test, y_test
    
    def get_data(self, fold):
        fold = fold % self.folds
        data1 = self.split_data(self.X1, self.y1, fold)
        data2 = self.split_data(self.X2, self.y2, fold)
        return data1, data2
    
    
class SyntheticData(Dataset):
    def __init__(self, alpha: float = 1.5, beta: float = 3., n: int = 1000, dim: int = 2, seed: int = 0, folds: int = 5) -> None:
        self.c0_means = -2*np.ones(dim)
        self.c1_means = 2*np.ones(dim)
        self.c0_cov = 0.5*np.eye(dim)
        self.c1_cov = 0.5*np.eye(dim)
        super(SyntheticData, self).__init__(seed, folds)
        
        rng = np.random.default_rng(self.seed)
        
        X0 = rng.multivariate_normal(self.c0_means, self.c0_cov, int(n/2))
        X1 = rng.multivariate_normal(self.c1_means, self.c1_cov, int(n/2))
        X = np.vstack((X0,X1))
        y = np.array([0]*int(n/2) + [1]*int(n/2)).reshape(-1,1)
        data1 = np.hstack((X, y))
        
        X0 = rng.multivariate_normal(self.c0_means+alpha, (1+beta)*self.c0_cov, int(n/2))
        X1 = rng.multivariate_normal(self.c1_means+alpha, (1+beta)*self.c1_cov, int(n/2))
        X = np.vstack((X0,X1))
        y = np.array([0]*int(n/2) + [1]*int(n/2)).reshape(-1,1)
        data2 = np.hstack((X, y))
        
        rng.shuffle(data1)
        rng.shuffle(data2)
        
        df1 = pd.DataFrame(data1, columns=['X0', 'X1', 'y'])
        df2 = pd.DataFrame(data2, columns=['X0', 'X1', 'y'])
        
        self.X1, self.y1 = df1.drop(columns=['y']), df1['y']
        self.X2, self.y2 = df2.drop(columns=['y']), df2['y']
        
        
class CorrectionShift(Dataset):
    def __init__(self, fname1, fname2, seed):
        super(CorrectionShift, self).__init__(seed)

        df1 = self.load_data(fname1)
        df2 = self.load_data(fname2)

        #Using a reduced feature space due to causal baseline's SCM
        num_feat = ["duration", "amount", "age"]
        cat_feat = ["personal_status_sex"]
        target = "credit_risk"

        df1 = df1.drop(columns=[c for c in list(df1) if c not in num_feat+cat_feat+[target]])
        df2 = df2.drop(columns=[c for c in list(df2) if c not in num_feat+cat_feat+[target]])

        #Scale numerical features
        df1, df2 = self.scale_num_feats(df1, df2, num_feat)

        #One-hot encode categorical features
        df1 = pd.get_dummies(df1, columns=cat_feat, dtype=float)
        df2 = pd.get_dummies(df2, columns=cat_feat, dtype=float)

        self.X1, self.y1 = df1.drop(columns=[target]), df1[target]
        self.X2, self.y2 = df2.drop(columns=[target]), df2[target]


class TemporalShift(Dataset):
    def __init__(self, fname, seed):
        super(TemporalShift, self).__init__(seed)

        df = self.load_data(fname)
        df = df.fillna(-1)

        #Define target variable
        df["NoDefault"] = 1-df["Default"].values

        #Drop unique identifiers, constants, feature perfectly correlated
        #with outcome, and categorical variables that blow up the 
        #feature space
        df = df.drop(columns=["Selected", "State","Name", "BalanceGross", "LowDoc","BankState",
            "LoanNr_ChkDgt","MIS_Status","Default", "Bank", "City"])

        cat_feat, num_feat = self.get_feat_types(df)

        #One-hot encode categorical features
        df = pd.get_dummies(df, columns=cat_feat, dtype=float)

        #Get df1 and df2
        df1 = df[df["ApprovalFY"]<2006]
        df2 = df

        #Scale numerical features
        df1, df2 = self.scale_num_feats(df1, df2, num_feat)

        self.X1, self.y1 = df1.drop(columns=["NoDefault"]), df1["NoDefault"]
        self.X2, self.y2 = df2.drop(columns=["NoDefault"]), df2["NoDefault"]
        

class GeospatialShift(Dataset):
    def __init__(self, fname, seed):
        super(GeospatialShift, self).__init__(seed)

        df = self.load_data(fname, ';')

        # Define target variable
        df["Outcome"] = (df["G3"]<10).astype(int)

        # Drop variables highly correlated with target
        df = df.drop(columns=["G1","G2","G3"])

        cat_feat, num_feat = self.get_feat_types(df)

        # One-hot encode categorical features
        df = pd.get_dummies(df, columns=cat_feat, dtype=float)

        # Get df1 and df2
        df1 = df[df["school_GP"]==1]
        df2 = df
        
        # Scale numerical features
        df1, df2 = self.scale_num_feats(df1, df2, num_feat)

        self.X1, self.y1 = df1.drop(columns=["Outcome","school_GP","school_MS"]), df1["Outcome"]
        self.X2, self.y2 = df2.drop(columns=["Outcome","school_GP","school_MS"]), df2["Outcome"]
        