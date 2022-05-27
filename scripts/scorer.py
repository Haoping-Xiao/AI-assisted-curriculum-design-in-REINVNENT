from reinvent_scoring.scoring.scoring_function_factory import ScoringFunctionFactory
from reinvent_scoring.scoring.scoring_function_parameters import ScoringFuncionParameters

from utils import read_scaffold_smiles, read_component_config, read_sample_smiles
import os
import numpy as np
import pandas as pd
import warnings


class Scorer:
    

    def __init__(self,scoring_function_parameters):
        # set up scoring function
        scoring_function_parameters = ScoringFuncionParameters(**scoring_function_parameters)
        self.scoring_function_instance = ScoringFunctionFactory(scoring_function_parameters)
        
        
    def get_score(self,smiles):
        summary=self.scoring_function_instance.get_final_score(smiles)
        return summary.total_score
    
def get_scorer(mode):
    scorer_config={
        'qed':'qed_scorer.json',
        'activity':'drd2_activity_scorer.json',
        'sa':'sa_scorer1.json'
    }
    dir_path = os.path.dirname(os.path.realpath(__file__))
    component_config_path=os.path.join(dir_path,'../component_config',scorer_config[mode])
    component_config=read_component_config(component_config_path)
    scorer=Scorer(component_config)
    return scorer


def analyze_curriculum_score(curriculum_name,curriculum_path):
    
    file_path=os.path.dirname(os.path.realpath(__file__))
    smiles_path=os.path.join(curriculum_path,'results_0/sampled.csv')
    smiles=  read_sample_smiles(smiles_path) 
    modes=['activity','qed','sa']
    record=[]
    for mode in modes:
        scorer=get_scorer(mode)
        score=scorer.get_score(smiles)
        record.append((score.shape[0], np.max(score), np.min(score), np.mean(score), np.median(score)))

    df=pd.DataFrame(record, columns=['number_of_smiles', 'max','min','mean','median'],index=modes)
    output_path=os.path.join(file_path,'../data/{}_record.csv'.format(curriculum_name))
    df.to_csv(output_path)



if __name__=='__main__':

    # curriculum_name="run_hba"
    # analyze_curriculum_score(curriculum_name,"/scratch/work/xiaoh2/Thesis/results/run_qed_26-04-2022")
    # scorer_registry={
    #     'qed':get_qed_scorer(),
    #     'activity':get_activity_scorer(),
    #     'sa':get_sa_scorer()
    # }
    warnings.filterwarnings(action='ignore', category=UserWarning)
    MAX_CONFIGURATION_NUM=1
    file_path=os.path.dirname(os.path.realpath(__file__))
    # jobname='run_curriculum_activity_tspa'
    jobname='activity_qed_sa'
    jobid='23-04-2022'
    # smiles_filename='scaffold_memory.csv'
    smiles_filename='sampled.csv' 
    # mode='qed' #'qed' or 'activity' or 'sa'
    mode='sa' #'qed' or 'activity'  or 'sa'
    # mode='activity' #'qed' or 'activity'  or 'sa'
    output_filename=smiles_filename[:-4]+'_'+mode
    # result_path=os.path.join(file_path,'../results/prior')
    result_path=os.path.join(file_path,'../results/{}_{}'.format(jobname,jobid))
    output_path=os.path.join(file_path,'../data/{}_{}_record.csv'.format(jobname,output_filename))

    scorer=get_scorer(mode)

    record=[]
    for i in range(MAX_CONFIGURATION_NUM):
        # data_path=os.path.join(result_path,'results_{}'.format(i),smiles_filename)
        data_path="/scratch/work/xiaoh2/Thesis/results/prior/sampled.csv"
        if not os.path.exists(data_path):
            print("{} not exist, exiting...".format(data_path))
            break
        smiles=  read_sample_smiles(data_path)  if smiles_filename=='sampled.csv' else  read_scaffold_smiles(data_path) 
        score=scorer.get_score(smiles)
        record.append((score.shape[0], np.max(score), np.min(score), np.mean(score), np.median(score)))

    df=pd.DataFrame(record, columns=['number_of_smiles', 'max','min','mean','median'])
    df.to_csv(output_path)
    