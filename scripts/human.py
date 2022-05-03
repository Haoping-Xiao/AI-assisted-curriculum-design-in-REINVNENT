from typing import Dict, List, Optional, Tuple,Union
from dataclasses import dataclass,field
from scorer import get_scorer
import numpy as np
import pandas as pd
from utils import get_component_statistic, Performance, get_prior_statistic, softmax

from AI_assistant import AI_assistant
from enums import ComponentEnum,HypothesisEnum
from pathlib import Path
import logging
# def first_curriculum(curriculum_name:List)->str:
#   # agent="/scratch/work/xiaoh2/Thesis/scripts/../../reinventcli/data/augmented.prior"
#   agent="/scratch/work/xiaoh2/Thesis/results/run_tpsa_11-04-2022/results_2/Agent.ckpt"
#   output_dir=execute_curriculum(curriculum_name,agent,using_gpu=True)
#   return output_dir


# def next_curriculum(last_output_dir:str,curriculum_name:List)->str:
#   agent=os.path.join(last_output_dir,"results_0/Agent.ckpt")
#   output_dir=execute_curriculum(curriculum_name,agent,last_output_dir,using_gpu=True)
#   return output_dir





@dataclass
class Human():
  weights: Performance=Performance(**{HypothesisEnum.ACT.value:1,HypothesisEnum.QED.value:1,HypothesisEnum.SA.value:1})
  bias: Tuple[float,float]=(10,10)
  components_data:Dict[ComponentEnum,Performance]=field(default_factory=get_component_statistic)
  prior_data:Performance=field(default_factory=get_prior_statistic)
  curriculum: List[ComponentEnum]=field(default_factory=lambda:[])
  max_curriculum_num:int=3
  current_performance: Performance=field(init=False)
  hypothesis_classes: List[ComponentEnum]=field(default_factory=lambda:[HypothesisEnum.ACT,HypothesisEnum.QED,HypothesisEnum.SA]) #default=['activity','qed','sa']
  ai_output_dir:Path="/scratch/work/xiaoh2/Thesis/results/curriculum"
  curriculum_sample_path:Path="results_0/sampled.csv"
  def __post_init__(self):
    self.logger=self.get_logger()
    self.current_performance= self.prior_data.copy()
    self.ai=self.get_AI_assistant()

  def get_logger(self)->logging.Logger:
    logger=logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    return logger

  def get_AI_assistant(self)->AI_assistant:
    ai=AI_assistant()
    return ai

  def get_performance(self,smiles:pd.DataFrame)->Performance:
    scores=Performance()
    for hypothesis_class in self.hypothesis_classes:
      scorer=get_scorer(hypothesis_class.value)
      setattr(scores,hypothesis_class.value,np.mean(scorer.get_score(smiles)))
    return scores*self.weights
    
  def setup_curriculum_pool(self)->List[ComponentEnum]:
    pool=[]
    for component in ComponentEnum:
      if component not in (self.hypothesis_classes + self.curriculum):
        pool.append(component)
    return pool

  def evaluate_components(self)->Dict[ComponentEnum,float]:
    pool=self.setup_curriculum_pool()
    evaluation={}
    for component_name in pool:
      weighted_performance=self.weights*(self.components_data[component_name]-self.prior_data)  # weighted normalized performance
      evaluation[component_name]=weighted_performance.sum/self.weights.sum
    return evaluation
  

  def get_jobname(self, component_name:ComponentEnum)->str:
    evaluated_curriclum=self.curriculum+[component_name]
    return '_'.join(list(map(lambda enums: enums.value,evaluated_curriclum)))

  def read_component_performance(self, component_name:ComponentEnum)->Performance:
    jobname=self.get_jobname(component_name)
    performance_path=Path(self.ai_output_dir,"_performance","{}_performance.csv".format(jobname))
    df=pd.read_csv(performance_path,index_col=0)
    weighted_performance:Performance=Performance(df[ComponentEnum.ACT.value][0],df[ComponentEnum.QED.value][0],df[ComponentEnum.SA.value][0])
    return weighted_performance

  def evaluate_advice(self, human_choice:ComponentEnum, advice:Union[ComponentEnum,None])->ComponentEnum:
    # TODO: if update prior statistic
    p1=self.read_component_performance(human_choice)
    p2=self.read_component_performance(advice) if advice else self.current_performance
    prob=np.exp(self.bias[1]*(p2.sum-p1.sum))/(1+np.exp(self.bias[1]*(p2.sum-p1.sum)))
    
    self.logger.info("prob swtich from {} to {} :{}".format(human_choice,advice,prob))
    choice=np.random.choice([human_choice,advice],p=[1-prob,prob])
    self.logger.info("In round {}, human choose {}".format(len(self.curriculum),choice))
    # update current performance
    self.current_performance=p1 if choice==human_choice else p2
    return choice



  def make_decision(self,evaluation:Dict[ComponentEnum,float],advice:Optional[ComponentEnum])->ComponentEnum:    
    prob=softmax(list(evaluation.values()),beta=self.bias[0])
    human_choice=np.random.choice(list(evaluation.keys()),p=prob)
    if not advice:
      advice=self.ai.recommend_component(current_performance=self.current_performance)
    if human_choice!=advice:
      decision=self.evaluate_advice(human_choice,advice)
    else:
      decision=human_choice
    return decision


  def create_curriculum(self)->List[ComponentEnum]:
    """
      a biased policy to get candidates:
      big improvement in single property
    """
    count=0
    evaluation=self.evaluate_components()
    while evaluation or count<self.max_curriculum_num: # there is no component in pool or human decide to terminate
      component_name=self.make_decision(evaluation)
      if component_name:
        self.curriculum.append(component_name)
        self.ai.curriculum.append(component_name)
      else:
        break
      evaluation=self.evaluate_components()
      count+=1
      
    return self.curriculum
    
  



if __name__=="__main__":
  
  components_data=get_component_statistic()
  prior_data=get_prior_statistic()
  human=Human(components_data=components_data,prior_data=prior_data)
  print(human.create_curriculum())
  # output_dir=first_curriculum(["drd2_activity_1"])
  # output_dir=first_curriculum(["drd2_activity_1","QED","sa"])
  # path:Path="/scratch/work/xiaoh2/Thesis/results/curriculum/_performance/hba1_performance.csv"
  # df=pd.read_csv(path)
  # print(df)
  # data={"activity":0.12,"qed":0.55,"sa":0.93}
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