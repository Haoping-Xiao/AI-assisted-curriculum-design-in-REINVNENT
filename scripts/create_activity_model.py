from cgi import test
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from multiprocessing import Pool
import pandas as pd
import numpy as np
from rdkit.Chem import AllChem as Chem
from rdkit.Chem import DataStructs
from sklearn.metrics import accuracy_score, recall_score, precision_score
from rdkit.Chem.Scaffolds.MurckoScaffold import GetScaffoldForMol
import pickle
class Dataset2D:
    def __init__(self, file, y_field=None, id_field=None, ext='sdf'):
        # file = os.path.join(data_dir, data_file)
        self.smiles = []
        self.moles = []
        self.Y = [] if y_field is not None else None
        self.id = []
        temp_id = 1
        if ext == 'sdf':
            suppl = Chem.SDMolSupplier(file, strictParsing=False)
            for i in suppl:
                if i is None:
                    continue
                smi = Chem.MolToSmiles(i, isomericSmiles=False)
                if smi is not None and smi != '':
                    self.smiles.append(smi)
                    self.moles.append(i)
                    if y_field is not None:
                        self.Y.append(i.GetProp(y_field))
                    if id_field is not None:
                        self.id.append(i.GetProp(id_field))
                    else:
                        self.id.append('id{:0>5}'.format(temp_id))
                        temp_id += 1
        
        elif ext == 'csv':
            # df = pd.read_csv(file)
            df=file
            try:
                df['moles'] = df['SMILES'].apply(lambda x: Chem.MolFromSmiles(x))
            except KeyError:
                df['SMILES'] = df['canonical']
                df['moles'] = df['SMILES'].apply(lambda x: Chem.MolFromSmiles(x))
            df = df.dropna()
            self.smiles = df['SMILES'].tolist()
            self.moles = df['moles'].tolist()
            self.Y = df[y_field].tolist() if y_field is not None else None
            self.id = df[id_field].tolist() if id_field is not None else np.arange(len(self.smiles))
            
        else:
            raise ValueError('file extension not supported!')
            
                    
        assert(len(self.smiles) == len(self.moles) == len(self.id))
        if self.Y is not None:
            assert(len(self.smiles) == len(self.Y))
            self.Y = np.array(self.Y)

    
    def __getitem__(self, index):
        if self.Y is not None:
            ret = self.id[index], self.smiles[index], self.moles[index], self.Y[index]
        else:
            ret = self.id[index], self.smiles[index], self.moles[index]
        return ret
    
    def __len__(self):
        return len(self.smiles)
    
    def __add__(self, other):
        pass


class DataStructure:
    def __init__(self, dataset, feat_fn, y_transforms=None, num_proc=1):
        self.dataset = dataset
        self.feat_fn = feat_fn
        self.Y = dataset.Y
        self.id = dataset.id
        self.num_proc = num_proc
        self.feat_names = []
        self.name_to_idx = {}

        x_s = []
        for fname in self.feat_fn.keys():
            f = self.feat_fn[fname]
            with Pool(self.num_proc) as p:
                arr = np.array(p.map(f, self.dataset.moles))
            x_s.append(arr)
            length = arr.shape[1]
            names = list('{}_{}'.format(fname, x+1) for x in range(length))
            self.feat_names += names
        x_s = tuple(x_s)
        self.X_ = np.concatenate(x_s, axis=1)
        
        # remove any nan rows
        nans = np.isnan(self.X_)
        mask = np.any(nans, axis=1)
        self.X_ = self.X_[~mask, :]
        self.name_to_idx = dict(zip(self.feat_names, range(len(self.feat_names))))
        self.id = list(self.id[j] for j in range(len(mask)) if not mask[j])
        if self.Y is not None:
            self.Y = self.Y[~mask]
        if y_transforms is not None:
            for t in y_transforms:
                self.Y = np.array(list(map(t, self.Y)))
    
    def __len__(self):
        return self.X_.shape[0]
    
    @property
    def shape(self):
        return self.X_.shape
    
    def X(self, feats=None):
        '''
        Use a list of to select feature columns
        '''
        if feats is None:
            return self.X_
        else:
            mask = list(map(lambda x: self.name_to_idx[x], feats))
            return self.X_[:, mask]


#morgan fingerprints
def morganX(mol, bits=2048, radius=3):
    morgan = np.zeros((1, bits))
    fp = Chem.GetMorganFingerprintAsBitVect(mol, radius, nBits=bits)
    DataStructs.ConvertToNumpyArray(fp, morgan)
    return morgan


def get_scaffold(smiles):
    scaffolds=[]
    for smile in smiles:
        mol=Chem.MolFromSmiles(smile)
        res=Chem.MolToSmarts(GetScaffoldForMol(mol))
        # res=Chem.MolToSmiles(GetScaffoldForMol(mol))
        print(smile, res)
        print(mol.HasSubstructMatch(Chem.MolFromSmarts(res)))
        scaffolds.append(res)
    df=pd.DataFrame(scaffolds)
    df.to_csv("substructures.csv")


def read_data():
  train_file="/scratch/work/xiaoh2/human-component/data/drd2.train.csv"
  test_file="/scratch/work/xiaoh2/human-component/data/drd2.test.csv"
  train_data=pd.read_csv(train_file)
  test_data=pd.read_csv(test_file)
  train_active=train_data[train_data['activity']==1].sample(n=10,random_state=0)
  train_inactive=train_data[train_data['activity']==0].sample(n=90,random_state=0)
  train_data=pd.concat([train_active,train_inactive])
  print(train_data.shape)
#   get_scaffold(train_active["canonical"])

  train_ds=Dataset2D(train_data, y_field='activity', ext="csv")
  test_ds=Dataset2D(test_data, y_field='activity', ext="csv")
  train_str = DataStructure(train_ds, dict(physchem=morganX), num_proc=8)
  test_str = DataStructure(test_ds, dict(physchem=morganX), num_proc=8)
  X_train = train_str.X()
  y_train = train_str.Y
  X_test= test_str.X()
  y_test=test_str.Y
  return X_train, y_train, X_test, y_test



def read_sample_data(path):
    df=pd.read_csv(path,header=None).squeeze()

    mols = [Chem.MolFromSmiles(smile) for smile in df]
    valid_mask = [mol is not None for mol in mols]
    valid_idxs = [idx for idx, is_valid in enumerate(valid_mask) if is_valid]
    valid_mols = [mols[idx] for idx in valid_idxs]

    mols = valid_mols 
    print(len(mols))
    with Pool(8) as p:
        X = np.array(p.map(morganX, mols))
    return X

if __name__=="__main__":
  
  clf = RandomForestClassifier(n_estimators=100,max_depth=2, random_state=0)

  

  X_train, y_train, X_test, y_test=read_data()  
  clf.fit(X_train, y_train)
  pred=clf.predict(X_test)

  acc=accuracy_score(y_test,pred)
  recall=recall_score(y_test,pred)
  print(acc,recall)
  pred=clf.predict(X_train)
  acc=accuracy_score(y_train,pred)
  recall=recall_score(y_train,pred)
  print(acc,recall)


