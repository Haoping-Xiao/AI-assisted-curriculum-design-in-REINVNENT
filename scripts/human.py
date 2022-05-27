from __future__ import annotations
from secrets import choice
import pystan
from typing import Dict, List, Optional, Tuple,Union
from dataclasses import dataclass,field
from scorer import get_scorer
from copy import deepcopy
import numpy as np
import pandas as pd
from utils import get_component_statistic, Performance, get_prior_statistic, softmax, read_sample_smiles
from run_curriculum import CurriculumBroker
# from AI_assistant import AI_assistant
from mdp import Node, State, Action
from enums import ComponentEnum,HypothesisEnum,ProjectConfig
from pathlib import Path
from stan_model import SampleParameter
import logging
from multiprocessing import Pool
from datetime import datetime
import warnings



@dataclass
class Human():
  weights: Performance=Performance(**{HypothesisEnum.ACT.value:0.5,HypothesisEnum.QED.value:0.3,HypothesisEnum.SA.value:0.2})
  bias: List[float]=field(default_factory=lambda:[2])
  components_data:Dict[ComponentEnum,Performance]=field(default_factory=get_component_statistic)
  prior_data:Performance=field(default_factory=get_prior_statistic)
  curriculum: List[ComponentEnum]=field(default_factory=lambda:[])
  max_curriculum_num:int=10
  current_performance: Performance=field(default_factory=get_prior_statistic)
  hypothesis_classes: List[ComponentEnum]=field(default_factory=lambda:[HypothesisEnum.ACT,HypothesisEnum.QED,HypothesisEnum.SA])
  __human_performance: List[float]=field(default_factory=lambda:[])
  __human_ai_performance: List[float]=field(default_factory=lambda:[])
  def __post_init__(self):
    self.__config=ProjectConfig()
    self.ai_output_dir=self.__config.OUT_DIR
    self.curriculum_sample_path=self.__config.SAMPLE_PATH
    self.logger=self.get_logger()
    # self.current_performance= self.prior_data.copy()
    self.ai=self.get_AI_assistant()
    self.broker=CurriculumBroker(weights=self.weights, curriculum=self.curriculum, hypothesis_classes=self.hypothesis_classes)
    
  def get_logger(self)->logging.Logger:
    logger=logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    return logger

  def get_AI_assistant(self)->AI:
    ai=AI(current_performance=self.current_performance,curriculum=self.curriculum)
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
      # human have to think about components until they satisfy with current performance. When setup the pool, it means human still want improvement.
      # Therefore, End is not the option.
      # For curriculum settings, hypothesis components are not considered.
      if component != ComponentEnum.END:
      # if component not in (self.hypothesis_classes + self.curriculum + [ComponentEnum.END]):
        pool.append(component)
    return pool

  def evaluate_components(self)->Dict[ComponentEnum,float]:
    pool=self.setup_curriculum_pool()
    evaluation={}
    
    for component_name in pool:
      try:
        weighted_performance=self.weights*(self.components_data[component_name]-self.prior_data)  # weighted normalized performance
        evaluation[component_name]=weighted_performance.sum
      except Exception as e:
        self.logger.info(e)
        self.logger.info("weights {}".format(self.weights))
        self.logger.info("data {}".format(self.components_data[component_name]))
        self.logger.info("prior data {}".format(self.prior_data))
    return evaluation
  

  # def get_jobname(self, component_name:ComponentEnum)->str:
  #   evaluated_curriclum=self.curriculum+[component_name]
  #   return '_'.join(list(map(lambda enums: enums.value,evaluated_curriclum)))

  def read_component_performance(self, component_name:ComponentEnum)->Performance:
    jobname=self.broker.get_jobname(component_name)
    performance_path=Path(self.ai_output_dir,"_evaluation","{}_performance.csv".format(jobname))
    if not performance_path.exists():
      raise FileExistsError("{} not exist".format(performance_path))
    df=pd.read_csv(performance_path,index_col=0)
    performance:Performance=Performance(df[HypothesisEnum.ACT.value][0],df[HypothesisEnum.QED.value][0],df[HypothesisEnum.SA.value][0])
    return performance

  def evaluate_advice(self, human_choice:ComponentEnum, advice:ComponentEnum, save:Optional[bool]=False)->ComponentEnum:
    # TODO: if update prior statistic
    p1=self.read_component_performance(human_choice)
    p2=self.read_component_performance(advice)

    weighted_p1=self.weights*p1
    weighted_p2=self.weights*p2
    weighted_cur_p=self.weights*self.current_performance
    # prob=np.exp(self.bias[1]*(p2.sum-p1.sum))/(1+np.exp(self.bias[1]*(p2.sum-p1.sum)))
    
    # choice=np.random.choice([human_choice,advice],p=[1-prob,prob])
    if weighted_cur_p.sum>=weighted_p1.sum and weighted_cur_p.sum>=weighted_p2.sum:
      self.logger.info("There is no improvement! Ending the selection!")
      choice=ComponentEnum.END
      
    else:
      choice=human_choice if weighted_p1.sum>weighted_p2.sum else advice
      # update current performance, note, this is not shallow copy. has to inform ai
      self.current_performance=p1 if choice==human_choice else p2
      # inform AI
      self.ai.current_performance=self.current_performance
      weighted_cur_p=self.weights*self.current_performance

    self.logger.info("p1 {}, p2 {}, cur {}".format(weighted_p1.sum,weighted_p2.sum,weighted_cur_p.sum))
    if save:
      self.logger.info("human choice {} adive {} ".format(human_choice,advice))
      self.logger.info("For curriculum {}, human choose {}".format(len(self.curriculum),choice))
      self.logger.info("current_performance{}".format(self.current_performance))
      self.logger.info("save comparison results after human make decision")
      self.__human_performance.append(weighted_p1.sum)
      self.__human_ai_performance.append(weighted_cur_p.sum)

    return choice



  def make_decision(self,evaluation:Dict[ComponentEnum,float],advice:Optional[ComponentEnum]=None)->ComponentEnum:    
    prob=softmax(list(evaluation.values()),beta=self.bias[0])
    human_choice=np.random.choice(list(evaluation.keys()),p=prob)
    
    if not advice:
      advice=self.ai.plan()
      #inform AI
      self.ai.prior_choice.append(self.ai.sampler.component_to_int(human_choice))

      decision=self.evaluate_advice(human_choice,advice,save=True)
    else: #served as user model in AI, the AI estimate how human will choose
      decision=self.evaluate_advice(human_choice,advice,save=False)
    
    return decision



  def create_curriculum(self)->List[ComponentEnum]:
    """
      a biased policy to get candidates:
      big improvement in single property
    """
    count=0
    evaluation=self.evaluate_components()
    while evaluation and count<self.max_curriculum_num: # there is no component in pool or human decide to terminate
      decision=self.make_decision(evaluation)

      self.curriculum.append(decision)
      if decision==ComponentEnum.END:
        break
      evaluation=self.evaluate_components()
      count+=1
    self.save_performance()
    return self.curriculum
    
  def save_performance(self):
    data={"human":self.__human_performance,"human_ai":self.__human_ai_performance,"human_choice":self.ai.prior_choice,"decision":self.curriculum}
    df=pd.DataFrame(data=data)
    performance_path=Path(self.__config.OUT_DIR,"_performance/result.csv")
    if performance_path.exists():
      # in case re-start after some interruptions
      df.to_csv(performance_path,mode="a")
    else:
      df.to_csv(performance_path)
    




@dataclass
class AI():
    current_performance: Performance
    prior_choice: List[int]=field(default_factory=lambda:[])
    curriculum: List[ComponentEnum]=field(default_factory=lambda:[])
    n_inter:int=500
    hypothesis_classes: List[HypothesisEnum]=field(default_factory=lambda:[HypothesisEnum.ACT,HypothesisEnum.QED,HypothesisEnum.SA])
    exploration_constant:float=1/np.sqrt(2)
    max_depth:int=1
    gamma:float=1

    def __post_init__(self):
        self.logger=logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        self.sampler=SampleParameter(curriculum=self.curriculum,prior_choice=self.prior_choice)

    
    def plan(self)->ComponentEnum:
        """
          give a state, search for best actions
        """

        if len(self.prior_choice)==0:
          #prior knowledge
          distribution=([0.5,0.5,0.5],3,[0.2,0.2,0.2],2)
        else:
          distribution= self.sampler.get_parameter_distribution()
          self.logger.info("w1 {} w2 {} w3 {}".format(distribution[0][0],distribution[0][1],distribution[0][2]))
          self.logger.info("std: w1 {} w2 {} w3 {}".format(distribution[2][0],distribution[2][1],distribution[2][2]))
          self.logger.info("bias {} std {}".format(distribution[1],distribution[3]))

        # self.inferred_weights,self.inferred_bias=self.sampler.get_parameters(distribution)
        # create root state with mean values
        self.inferred_weights,self.inferred_bias=Performance(**{HypothesisEnum.ACT.value:distribution[0][0],HypothesisEnum.QED.value:distribution[0][1],HypothesisEnum.SA.value:distribution[0][2]}),[distribution[1]]

        state=State(weights=self.inferred_weights,curriculum=self.curriculum)

        root = Node(state=state,parent=None,actions=state.get_possible_actions())
            
        # create state space
        self.state_space=self.__get_state_space(root_state=state)

        if root.is_terminal:
          self.logger.info("root node is a terminal node")
          return None
        
        for i in range(self.n_inter):
            if i>0:
              weights,bias=self.sampler.get_parameters(distribution)
              # trick: update weights in state space due to shallow copy
              self.inferred_weights.activity=weights.activity
              self.inferred_weights.qed=weights.qed
              self.inferred_weights.sa=weights.sa
              self.inferred_bias[0]=bias[0]

            
            self.simulate(root,depth=0)
        self.check_node(root)
        best_advice=self.get_best_action(root,exploration_value=0).name
        return best_advice
    


    def check_space(self):
        for state_pair in self.state_space.values():
          states=state_pair.values()
          for state in states:
            self.logger.info(state)

    def check_node(self, node:Node):
        self.logger.info("checking node!")
        self.logger.info(node.num_visits)
        for action in node.actions:
          self.logger.info(action)

    def simulate(self,root:Node, depth:int):
        """
            execute a selection-expansion-simulation-backpropagation round
        """
        action=self.get_best_action(root,self.exploration_constant)
        node=self.transition(root,action,depth)
        root.children[action.name]=node
        reward = node.state.get_reward()
        depth+=1
        if depth==self.max_depth or node.num_visits==0 or node.is_terminal:
            q=reward
        else:
            q=reward+self.gamma*self.simulate(node,depth)
        root.num_visits+=1
        #action is a pointer, can update info by this pointer
        action.num_visits+=1
        action.total_reward+=(q-action.total_reward)/action.num_visits


    def get_best_action(self, node:Node, exploration_value: float)->Action:
        best_value = float("-inf")
        best_actions=[]
        for action in node.actions:
          if action.num_visits==0 and exploration_value!=0:
            value=np.inf
          elif action.num_visits==0:
            value=action.total_reward
          else:
            value=action.total_reward+exploration_value*np.sqrt(np.log(node.num_visits)/action.num_visits)
          if value>best_value:
            best_value=value
            best_actions=[action]
          elif value==best_value:
            best_actions.append(action)
        return np.random.choice(best_actions)

 
    def transition(self,node:Node, action:ComponentEnum, depth:int)->Node:
        #AI's action for human is advice
        self.logger.info("simulate transition advice {}".format(action.name))
        # user model, know the hypothesis class but not the parameters.
        # user model will not use AI. 
        # user model is used to infer the next state.
        user_model=Human(weights=self.inferred_weights,bias=self.inferred_bias, curriculum=self.curriculum, current_performance=self.current_performance)
        evaluation=user_model.evaluate_components()
        inferred_human_action=user_model.make_decision(evaluation,action.name)
        node.state.take_action(inferred_human_action)
        name=self.__get_state_name(node.state.curriculum+[inferred_human_action])
        state=self.state_space[depth+1][name]
        
        return Node(state=state,parent=node,actions=state.get_possible_actions())


    def __get_state_name(self, curriculum: List[ComponentEnum])->str:
      curriculum_list=list(map(lambda enums: enums.value,curriculum))
      return '_'.join(filter(None,curriculum_list))

    def __get_state_space(self, root_state:State)->Dict[int,Dict[str,State]]:
      
      root_name=self.__get_state_name(root_state.curriculum)
      state_space:Dict[int,Dict[str,State]]={0:{root_name:root_state}}
      
      for depth in range(self.max_depth):
        current_states=state_space[depth].values()
        state_space[depth+1]={}
        for state in current_states:
          actions=state.get_possible_actions()
          for action in actions:
            next_state=State(weights=state.weights,curriculum=state.curriculum+[action.name])
            next_state_name=self.__get_state_name(next_state.curriculum)
            state_space[depth+1][next_state_name]=next_state
          # take action in advance here for parallel efficicy
          # MCTS will only take one action at each time and thus not able to run jobs in parallel
          actions_name=list(map(lambda action:action.name, actions))
          with Pool(len(actions)) as p:
            p.map(state.take_action,actions_name)

        for next_state in state_space[depth+1].values():
          next_state.get_reward(for_evaluation=True)
      return state_space



if __name__=="__main__":
  # suppress scikit model userwarning due to inconsistent scikit version. The drd2 model require 0.21.2 while sa component require 0.21.3
  warnings.simplefilter('ignore', UserWarning)


  human=Human()
  res=human.create_curriculum()
  print("res is {}".format(res))
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