from typing import Dict, List
from run_curriculum import execute_curriculum
import os
from dataclasses import dataclass,field
from scorer import get_scorer
import numpy as np
import pandas as pd
from utils import get_component_statistic, Performance, read_sample_smiles

from AI_assistant import AI_assistant
from enums import ComponentEnum
from pathlib import Path
import logging
def first_curriculum(curriculum_name:List)->str:
  # agent="/scratch/work/xiaoh2/Thesis/scripts/../../reinventcli/data/augmented.prior"
  agent="/scratch/work/xiaoh2/Thesis/results/run_tpsa_11-04-2022/results_2/Agent.ckpt"
  output_dir=execute_curriculum(curriculum_name,agent,using_gpu=True)
  return output_dir


def next_curriculum(last_output_dir:str,curriculum_name:List)->str:
  agent=os.path.join(last_output_dir,"results_0/Agent.ckpt")
  output_dir=execute_curriculum(curriculum_name,agent,last_output_dir,using_gpu=True)
  return output_dir





@dataclass
class Human():
  components_data:Dict[ComponentEnum,Performance]
  prior_data:Performance
  current_performance: Performance=field(init=False)
  hypothesis_classes: List[ComponentEnum]=field(default_factory=lambda:[ComponentEnum.ACT,ComponentEnum.QED,ComponentEnum.SA]) #default=['activity','qed','sa']
  # max_component_evaluation: int=3
  # standard: Performance = Performance(**{'activity':0.67,'qed':0.77,'sa':0.97})
  weights: Performance=Performance(**{ComponentEnum.ACT.value:1,ComponentEnum.QED.value:1,ComponentEnum.SA.value:1})
  
  def __post__init__(self):
    self.current_performance= self.prior_data.copy()

  def get_logger(self):
    self.logger=logging.getLogger(__name__)

  def get_AI_assistant(self):
    self.ai=AI_assistant()

  def get_component_list(self):
    return self.compoents_data.keys()


  def get_performance(self,smiles:pd.DataFrame)->Performance:
    scores=Performance()
    for hypothesis_class in self.hypothesis_classes:
      scorer=get_scorer(hypothesis_class.value)
      setattr(scores,hypothesis_class.value,np.mean(scorer.get_score(smiles)))
    return scores*self.weights

  def evaluate_components(self)->Dict[ComponentEnum,float]:
  # def evaluate_components(self,components_data:Dict[str,Performance],prior_data:Performance)->Dict[str,float]:
    evaluation={}
    for component_name, component_performance in self.components_data.items():
      weighted_performance=self.weights*(component_performance-self.prior_data)  # weighted normalized performance
      evaluation[component_name]=weighted_performance.sum/self.weights.sum
    return evaluation
  

  def evaluate_advice(self, human_choice:ComponentEnum, advice:ComponentEnum):
    
    for component_name in [human_choice,advice]:
      try:
        jobname=self.get_jobname(component_name)
        curriculum_path=Path(self.output_dir,jobname)
        if curriculum_path.exists():
          curriculum_path=self.setup_curriculum(component_name,jobname)


        production_path=Path(curriculum_path,"production")
        if not production_path.exists():
          production_path=self.setup_production(component_name,curriculum_path)

        smiles_path=Path(production_path,self.curriculum_sample_path)
        smiles=  read_sample_smiles(smiles_path) 
        weighted_performance:Performance=self.infer_performance(smiles)
        self.save_performance(component_name,weighted_performance)
        return weighted_performance.sum
      except Exception as e:
        self.logger.debug(e)
        return np.nan

  

  def make_decision(self,evaluation:Dict[str,float])->str:
    
    def softmax(logit:List[float],beta:int=10):
      logit=np.array(logit)
      exp_logit=np.exp(beta*logit)
      prob=exp_logit/np.sum(exp_logit)
      return prob
    
    prob=softmax(list(evaluation.values()))
    human_choice=np.random.choice(list(evaluation.keys()),p=prob)
    #TODO: AI advice
    advice=self.ai.recommend_component(current_performance=self.current_performance)


    return human_choice


  def get_component_candidates(self,current_performance:Performance)->str:
    """
      a biased policy to get candidates:
      big improvement in single property
    """
    evaluation=self.evaluate_components()
    component_name=self.make_decision(evaluation)
    return component_name
    
  def get_reward(self,current_performance:Performance):
    pass


  def policy(self):
    """
      step1: biased components
      step2: estimated Q values
      step3: best component out of biased components
    """
    pass



if __name__=="__main__":
  # output_dir=first_curriculum(["drd2_activity_1"])
  # output_dir=first_curriculum(["drd2_activity_1","QED","sa"])

  data={"activity":0.12,"qed":0.55,"sa":0.93}
  # prob=
  # print(np.random.choice(list(data.keys()),p=prob))
  # test=Performance(**data)
  # print(test.activity)
  # print(output_dir)
  # next_curriculum("/scratch/work/xiaoh2/Thesis/scripts/../results/run_curriculum_drd2_activity_1",["drd2_activity_1","QED"])
  # next_curriculum("/scratch/work/xiaoh2/Thesis/results/run_curriculum_drd2_activity_1/run_curriculum_QED",["drd2_activity_1","QED","sa"])
  # next_curriculum("/scratch/work/xiaoh2/Thesis/results/curriculum_drd2_activity_1/curriculum_QED","sa")


# data["sa"]=  {"tpsa_1":0.93,
    #               "tpsa_3":0.90,
    #               "Custom_alerts":0.95,
    #               "number_of_stereo_centers_1":0.93,
    #               "graph_length_1":0.93,
    #               "graph_length_2":0.93,
    #               "num_hba_lipinski_1":0.93,
    #               "num_hba_lipinski_2":0.93,
    #               "num_hba_lipinski_3":0.94,
    #               "num_hbd_lipinski_1":0.93,
    #               "num_hbd_lipinski_3":0.94,
    #               "num_rotatable_bonds_1":0.92,
    #               "num_rings_1":0.94,
    #               "num_rings_3":0.92,
    #               "slogp_1":0.93,
    #               "slogp_2":0.94,
    #               "Molecular_mass_4":0.94,
    #               "Molecular_mass_1":0.95,
    #               }
    # data["qed"]= {"tpsa_1":0.55,
    #               "tpsa_3":0.71,
    #               "Custom_alerts":0.59,
    #               "number_of_stereo_centers_1":0.59,
    #               "graph_length_1":0.57,
    #               "graph_length_2":0.54,
    #               "num_hba_lipinski_1":0.61,
    #               "num_hba_lipinski_2":0.60,
    #               "num_hba_lipinski_3":0.57,
    #               "num_hbd_lipinski_1":0.59,
    #               "num_hbd_lipinski_3":0.57,
    #               "num_rotatable_bonds_1":0.64,
    #               "num_rings_1":0.64,
    #               "num_rings_3":0.49,
    #               "slogp_1":0.68,
    #               "slogp_2":0.62,
    #               "Molecular_mass_4":0.65,
    #               "Molecular_mass_1":0.59,
    #               }
    # data["activity"]={"tpsa_1":0.12,
    #                   "tpsa_3":0.36,
    #                   "Custom_alerts":0.13,
    #                   "number_of_stereo_centers_1":0.15,
    #                   "graph_length_1":0.13,
    #                   "graph_length_2":0.10,
    #                   "num_hba_lipinski_1":0.13,
    #                   "num_hba_lipinski_2":0.14,
    #                   "num_hba_lipinski_3":0.10,
    #                   "num_hbd_lipinski_1":0.15,
    #                   "num_hbd_lipinski_3":0.12,
    #                   "num_rotatable_bonds_1":0.14,
    #                   "num_rings_1":0.12,
    #                   "num_rings_3":0.13,
    #                   "slogp_1":0.13,
    #                   "slogp_2":0.14,
    #                   "Molecular_mass_4":0.12,
    #                   "Molecular_mass_1":0.10,
    #                   }