import pandas as pd
import numpy as np
from rdkit.Chem.Descriptors import MolWt,TPSA,MolLogP
from rdkit.Chem.Lipinski import NumHAcceptors, NumHDonors, NumRotatableBonds
from rdkit.Chem import GetDistanceMatrix
from rdkit import Chem
from rdkit.Chem.rdMolDescriptors import CalcNumRings, CalcNumAtomStereoCenters
from matplotlib import pyplot as plt
import os

def analysis_property(property):
  maximum=np.max(property)
  minimum=np.min(property)
  average=np.mean(property)
  median=np.mean(property)
  return (maximum,minimum,average,median)

def visualize_property(property_score,label,path):
  plt.figure()
  plt.hist(property_score,color='red',label=label)
  plt.legend()
  plt.savefig(os.path.join(path,"{}.png".format(task_name)))
  # plt.figure()
  # plt.hist(property_score[labels==0],color='blue',label='inactive')
  # plt.legend()
  # plt.savefig(os.path.join(path,"{}_{}.png".format('inactive',property)))

def maximum_graph_length(mol):
  return int(np.max(GetDistanceMatrix(mol)))


def read_data(file):
  df=pd.read_csv(file,dtype={'canonical':str,'activity':int})
  df['mols'] = df['canonical'].apply(lambda x: Chem.MolFromSmiles(x))
  return  df['mols'], df['activity']

if __name__=='__main__':

  data_path='/scratch/work/xiaoh2/human-component/data/drd2.train.csv'
  output_path=os.path.join(os.path.dirname(os.path.realpath(__file__)),'../data')

  data=pd.read_csv(data_path)
  mols,activity=read_data(data_path)
  tasks={
    'weight':MolWt,
    'hba':NumHAcceptors,
    'hbd': NumHDonors,
    'ring':CalcNumRings,
    'nrb':NumRotatableBonds,
    'slog':MolLogP,
    'tpsa':TPSA,
    'graph':maximum_graph_length,
    'stereo':CalcNumAtomStereoCenters,
  }

  stat=[]

  for key, task in tasks.items():
    scores=np.array([task(mol) for mol in mols])
    category=['inactive','active']
    for c in category:
      filtered_scores=scores[activity==(c=='active')]
      task_name="{}_{}".format(key,c)
      res=analysis_property(filtered_scores)
      stat.append((task_name,)+res)
      visualize_property(filtered_scores,task_name,output_path)


  df=pd.DataFrame(stat, columns=['task', 'max','min','mean','median'])
  df.to_csv(os.path.join(output_path,'statistic.csv'))

