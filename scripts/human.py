from __future__ import annotations

from typing import Dict, List, Optional
from dataclasses import dataclass,field
from scorer import get_scorer
import numpy as np
import pandas as pd
from utils import get_component_statistic, Performance, get_prior_statistic, softmax, read_sample_smiles
from run_curriculum import CurriculumBroker
from mdp import Node, State, Action
from enums import ComponentEnum,HypothesisEnum,ProjectConfig
from pathlib import Path
from stan_model import SampleParameter
import logging
from multiprocessing import Pool
import warnings
import argparse


@dataclass
class Human():
  weights: Performance
  id: int
  bias: List[float]=field(default_factory=lambda:[])
  components_data:Dict[ComponentEnum,Performance]=field(default_factory=get_component_statistic,repr=False)
  prior_data:Performance=field(default_factory=get_prior_statistic,repr=False)
  curriculum: List[ComponentEnum]=field(default_factory=lambda:[])
  current_performance: Performance=field(default_factory=get_prior_statistic)
  hypothesis_classes: List[ComponentEnum]=field(default_factory=lambda:[HypothesisEnum.ACT,HypothesisEnum.QED,HypothesisEnum.SA])
  objective_values: List[float]=field(default_factory=lambda:[])
  activity_values: List[float]=field(default_factory=lambda:[])
  def __post_init__(self):
    self.__config=ProjectConfig()
    self.ai_output_dir=self.__config.OUT_DIR
    self.curriculum_sample_path=self.__config.SAMPLE_PATH
    self.logger=self.get_logger()
    # self.current_performance= self.prior_data.copy()
    self.ai=self.get_AI_assistant()
    self.broker=CurriculumBroker(curriculum=self.curriculum, hypothesis_classes=self.hypothesis_classes)
    
  def get_logger(self)->logging.Logger:
    logger=logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    return logger

  def get_AI_assistant(self)->AI:
    ai=AI(current_performance=self.current_performance,id=self.id,curriculum=self.curriculum)
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
  


  def read_component_performance(self, component_name:ComponentEnum)->Performance:
    jobname=self.broker.get_jobname(component_name)
    performance_path=Path(self.ai_output_dir,"_performance","{}_performance.csv".format(jobname))
    if not performance_path.exists():
      production_path=Path(self.__config.OUT_DIR,jobname,"production")
      smiles_path=Path(production_path,self.__config.SAMPLE_PATH)
      if not smiles_path.exists():
        curriculum_path=Path(self.__config.OUT_DIR,jobname)
        try:
          if not Path(curriculum_path,self.__config.MODEL_PATH).exists():
            try:
              curriculum_path=self.broker.setup_curriculum(component_name,jobname)
            except Exception as e:
              raise Exception("bugs in setup curriculum: {}".format(e))

          try:
            production_path=self.broker.setup_production(component_name,curriculum_path,epoch=self.__config.PRODUCTION_EPOCH)
          except Exception as e:
            raise Exception("bugs in setup production: {}".format(e))
        except Exception as e:
          self.broker.logger.debug(e)
      smiles= read_sample_smiles(smiles_path)     
      performance:Performance=self.broker.infer_performance(smiles)
      self.broker.save_performance(jobname,performance)

    df=pd.read_csv(performance_path,index_col=0)
    performance:Performance=Performance(df[HypothesisEnum.ACT.value][0],df[HypothesisEnum.QED.value][0],df[HypothesisEnum.SA.value][0])
    return performance

  def evaluate_advice(self, human_choice:ComponentEnum, advice:ComponentEnum, save:Optional[bool]=False)->ComponentEnum:
    p1=self.read_component_performance(human_choice)
    p2=self.read_component_performance(advice)

    weighted_p1=self.weights*p1
    weighted_p2=self.weights*p2
    weighted_cur_p=self.weights*self.current_performance

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
      if choice!=ComponentEnum.END:
        self.objective_values.append(weighted_cur_p.sum)
        self.activity_values.append(self.current_performance.activity)
        self.curriculum.append(choice)
      self.save_performance()
    return choice

  def evaluate_prior_choice(self, human_choice:ComponentEnum)->bool:
    p1=self.read_component_performance(human_choice)
    weighted_p1=self.weights*p1
    weighted_cur_p=self.weights*self.current_performance
    if weighted_cur_p.sum>=weighted_p1.sum:
      return False
    else:
      self.current_performance=p1
      print(self.current_performance)
      print("cur {}, p1 {}, select {}".format(weighted_cur_p.sum,weighted_p1.sum,weighted_p1.sum))
      self.objective_values.append(weighted_p1.sum)
      self.activity_values.append(self.current_performance.activity)
      return True


  def make_assisted_decision(self,evaluation:Dict[ComponentEnum,float],advice:Optional[ComponentEnum]=None,save:Optional[bool]=True)->ComponentEnum:    
    prob=softmax(list(evaluation.values()),beta=self.bias[0])
    human_choice=np.random.choice(list(evaluation.keys()),p=prob)
    
    if not advice:
      advice=self.ai.plan()
      #inform AI
      self.ai.prior_choice.append(self.ai.sampler.component_to_int(human_choice))

      decision=self.evaluate_advice(human_choice,advice,save=save)
    else: #served as user model in AI, the AI estimate how human will choose
      decision=self.evaluate_advice(human_choice,advice,save=save)
    
    return decision

  def make_unassisted_decision(self,evaluation:Dict[ComponentEnum,float])->ComponentEnum:
    prob=softmax(list(evaluation.values()),beta=self.bias[0])
    human_choice=np.random.choice(list(evaluation.keys()),p=prob)
    if self.evaluate_prior_choice(human_choice):
      return human_choice
    else:
      return ComponentEnum.END


  def create_unassisted_curriculum(self)->List[ComponentEnum]:
    evaluation=self.evaluate_components()
    patience=3
    count=0
    while True:
      decision=self.make_unassisted_decision(evaluation)
      if decision!=ComponentEnum.END:
        self.curriculum.append(decision)
        self.save_performance()
      elif count<=patience:
        count+=1
      else:
        break
    return self.curriculum

  def create_assisted_curriculum(self,random_advice:Optional[bool]=False)->List[ComponentEnum]:
    evaluation=self.evaluate_components()
    while True:
      if not random_advice:
        decision=self.make_assisted_decision(evaluation)
      else:
        advice=np.random.choice(ComponentEnum)
        decision=self.make_assisted_decision(evaluation,advice=advice)
      if decision==ComponentEnum.END:
        break
    return self.curriculum
    
  def save_performance(self):
    #prior model performance in production phase
    prior=get_prior_statistic()
    prior_objective=(self.weights*prior).sum
    obj=[prior_objective]
    act=[prior.activity]
    decision=["prior"]
    obj.extend(self.objective_values)
    act.extend(self.activity_values)
    decision.extend(self.curriculum)
    data={"objective_value":obj, "activity":act, "decision":decision}
    df=pd.DataFrame(data=data)
    performance_path=Path(self.__config.OUT_DIR,"_performance/assisted_result_{}.csv".format(self.id))
    df.to_csv(performance_path)
    




@dataclass
class AI():
    current_performance: Performance
    id:int
    prior_choice: List[int]=field(default_factory=lambda:[])
    curriculum: List[ComponentEnum]=field(default_factory=lambda:[])
    n_inter:int=1000
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
        weights,beta= self.sampler.get_parameter(iter=self.n_inter)
        self.inferred_weights,self.inferred_bias=Performance(**{HypothesisEnum.ACT.value:weights[0][0],HypothesisEnum.QED.value:weights[0][1],HypothesisEnum.SA.value:weights[0][2]}),[beta[0]]


        state=State(curriculum=self.curriculum)

        root = Node(state=state,parent=None,actions=state.get_possible_actions())
            
        # create state space
        self.state_space=self.__get_state_space(root_state=state)

        if root.is_terminal:
          self.logger.info("root node is a terminal node")
          return None
        
        for i in range(self.n_inter):
            if i>0:
              # trick: update weights in state space due to shallow copy
              self.inferred_weights.activity=weights[i][0]
              self.inferred_weights.qed=weights[i][1]
              self.inferred_weights.sa=weights[i][2]
              self.inferred_bias[0]=beta[i]

            self.logger.info("iter {}: get parameters w {} bias {}".format(i, self.inferred_weights, self.inferred_bias))
            
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
        reward = (reward * self.inferred_weights).sum
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
        user_model=Human(weights=self.inferred_weights,id=self.id,bias=self.inferred_bias, curriculum=self.curriculum, current_performance=self.current_performance)
        evaluation=user_model.evaluate_components()
        inferred_human_action=user_model.make_assisted_decision(evaluation,action.name,save=False)
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
            next_state=State(curriculum=state.curriculum+[action.name])
            next_state_name=self.__get_state_name(next_state.curriculum)
            state_space[depth+1][next_state_name]=next_state
          # take action in advance here for parallel efficicy
          # MCTS will only take one action at each time and thus not able to run jobs in parallel
          actions_name=list(map(lambda action:action.name, actions))
          with Pool(len(actions)) as p:
            p.map(state.take_action,actions_name)
      return state_space



if __name__=="__main__":
  # suppress scikit model userwarning due to inconsistent scikit version. The drd2 model require 0.21.2 while sa component require 0.21.3
  warnings.simplefilter('ignore', UserWarning)

  parser = argparse.ArgumentParser(prog="AI-assisted curriculum learning",description="require reward parameters and bias parameter")
  parser.add_argument('w1',type=float,  help="reward parameter 1")
  parser.add_argument('w2',type=float,  help="reward parameter 2")
  parser.add_argument('w3',type=float,  help="reward parameter 3")
  parser.add_argument('beta',type=float,  help="bias parameter")
  parser.add_argument('id',type=int,  help="human id")
  args=parser.parse_args()
  human=Human(weights=Performance(**{HypothesisEnum.ACT.value:args.w1,HypothesisEnum.QED.value:args.w2,HypothesisEnum.SA.value:args.w3}),id=args.id, bias=[args.beta])
  # res=human.create_assisted_curriculum(random_advice=True)
  # res=human.create_unassisted_curriculum()
  res=human.create_assisted_curriculum()
  print("res is {}".format(res))