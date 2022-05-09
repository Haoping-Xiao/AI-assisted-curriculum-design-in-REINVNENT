from __future__ import annotations
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
  weights: Performance=Performance(**{HypothesisEnum.ACT.value:0.5,HypothesisEnum.QED.value:0.3,HypothesisEnum.SA.value:0.2})
  bias: List[float]=[2,20]
  components_data:Dict[ComponentEnum,Performance]=field(default_factory=get_component_statistic)
  prior_data:Performance=field(default_factory=get_prior_statistic)
  curriculum: List[ComponentEnum]=field(default_factory=lambda:[])
  max_curriculum_num:int=2
  current_performance: Performance=field(default_factory=get_prior_statistic)
  hypothesis_classes: List[ComponentEnum]=field(default_factory=lambda:[HypothesisEnum.ACT,HypothesisEnum.QED,HypothesisEnum.SA])
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
    ai=AI(current_performance=self.current_performance)
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
      evaluation[component_name]=weighted_performance.sum
    return evaluation
  

  def get_jobname(self, component_name:ComponentEnum)->str:
    evaluated_curriclum=self.curriculum+[component_name]
    return '_'.join(list(map(lambda enums: enums.value,evaluated_curriclum)))

  def read_component_performance(self, component_name:ComponentEnum)->Performance:
    jobname=self.get_jobname(component_name)
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
            production_path=self.broker.setup_production(component_name,curriculum_path)
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

  def evaluate_advice(self, human_choice:ComponentEnum, advice:Union[ComponentEnum,None])->ComponentEnum:
    # TODO: if update prior statistic
    p1=self.read_component_performance(human_choice)
    p2=self.read_component_performance(advice) if advice else self.current_performance

    p1=self.weights*p1
    p2=self.weights*p2

    prob=np.exp(self.bias[1]*(p2.sum-p1.sum))/(1+np.exp(self.bias[1]*(p2.sum-p1.sum)))
    
    self.logger.info("prob swtich from {} to {} :{}".format(human_choice,advice,prob))

    choice=np.random.choice([human_choice,advice],p=[1-prob,prob])
    self.logger.info("For curriculum {}, human choose {}".format(len(self.curriculum),choice))
    # update current performance
    self.current_performance=p1 if choice==human_choice else p2
    # inform AI
    self.ai.current_performance=self.current_performance
    return choice



  def make_decision(self,evaluation:Dict[ComponentEnum,float],advice:Optional[ComponentEnum]=None)->ComponentEnum:    
    prob=softmax(list(evaluation.values()),beta=self.bias[0])
    human_choice=np.random.choice(list(evaluation.keys()),p=prob)
    
    if not advice:
      advice=self.ai.plan()
      #inform AI
      self.ai.prior_choice.append(self.ai.component_to_int(human_choice))

    decision=self.evaluate_advice(human_choice,advice)
    
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
      #inform AI
      self.ai.curriculum.append(decision)
      if decision:
        self.curriculum.append(decision)
      else:
        break
      evaluation=self.evaluate_components()
      count+=1
      
    return self.curriculum
    
  




@dataclass
class AI():
    current_performance: Performance
    prior_choice: List[int]=field(default_factory=lambda:[])
    advice: List[int]=field(default_factory=lambda:[])
    curriculum: List[ComponentEnum]=field(default_factory=lambda:[])
    n_inter:int=100
    hypothesis_classes: List[HypothesisEnum]=field(default_factory=lambda:[HypothesisEnum.ACT,HypothesisEnum.QED,HypothesisEnum.SA])
    exploration_constant:float=1/np.sqrt(2)
    max_depth:int=1
    gamma:float=1

    def __post_init__(self):
        self.logger=logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        self.__config=ProjectConfig()

        prior=get_component_statistic()
        self.pri_activity=[]
        self.pri_qed=[]
        self.pri_sa=[]

        for component_name in ComponentEnum:
          p=prior[component_name]
          self.pri_activity.append(p.activity)
          self.pri_qed.append(p.qed)
          self.pri_sa.append(p.sa)


    def get_prior_parameters(self)->Tuple[Performance,List[float]]:
        w=np.random.normal(loc=0.5,scale=0.15,size=3)
        w=w/np.sum(w)
        bias1=np.random.uniform(low=1,high=4)
        bias2=np.random.uniform(low=10,high=40)
        w=Performance(**{HypothesisEnum.ACT.value:w[0],HypothesisEnum.QED.value:w[1],HypothesisEnum.SA.value:w[2]})
        bias:List[float]=[bias1,bias2]
        self.logger.info("get prior parameters w {} bias {}".format(w, bias))
        return w, bias
        
    
    def get_observed_data(self)->Tuple(List[List[float]],List[List[float]],List[List[float]]):
        observed_activity=[]
        observed_qed=[]
        observed_sa=[]

        for k in range(len(self.curriculum)):
          ob_a=[]
          ob_q=[]
          ob_s=[]
          for component_name in ComponentEnum:
            curriculum_in_k_round=self.curriculum[:k]
            # the component name has been included. AI doesn't have to observe it. 
            if component_name not in curriculum_in_k_round:
              jobname='_'.join(list(map(lambda enums: enums.value,curriculum_in_k_round+[component_name])))
              performance=self.read_component_performance(jobname)
              ob_a.append(performance.activity)
              ob_q.append(performance.qed)
              ob_s.append(performance.sa)
            else:
              #if the component has been introduce. the create a random big negative number, so softmax will reduce the prob to be chosen.
              ob_a.append(-10000)
              ob_q.append(-10000)
              ob_s.append(-10000)
          observed_activity.append(ob_a)
          observed_qed.append(ob_q)
          observed_sa.append(ob_s)
        return (observed_activity,observed_qed,observed_sa)

    def component_to_int(self,component_name:ComponentEnum)->int:
        components=list(ComponentEnum)
        return components.index(component_name)+1

    def prepare_data(self)->Dict:
        K=len(self.curriculum)
        J=len(ComponentEnum)
        
        decision=[]
        prior_activity=[]
        prior_qed=[]
        prior_sa=[]

        for k in range(K):
          if self.component_to_int(self.curriculum[k])==self.prior_choice[k]:
            decision.append(0)
          else:
            decision.append(1)
          
          if len(self.prior_choice)>1:
            self.pri_activity[self.prior_choice[k-1]]=-10000
            self.pri_qed[self.prior_choice[k-1]]=-10000
            self.pri_sa[self.prior_choice[k-1]]=-10000

          prior_activity.append(deepcopy(self.pri_activity))
          prior_qed.append(deepcopy(self.pri_qed))
          prior_sa.append(deepcopy(self.pri_sa))



        observed_activity,observed_qed,observed_sa=self.get_observed_data()

        self.logger.info("curriculum {}".format(self.curriculum))
        self.logger.info("advice {}".format(self.advice))
        self.logger.info("prior_choice {}".format(self.prior_choice))
        self.logger.info("decision {}".format(decision))
        self.logger.info("prior_activity {}".format(prior_activity))
        self.logger.info("prior_qed {}".format(prior_qed))
        self.logger.info("prior_sa {}".format(prior_sa))
        self.logger.info("observed_activity {}".format(observed_activity))
        self.logger.info("observed_qed {}".format(observed_qed))
        self.logger.info("observed_sa {}".format(observed_sa))
        data={"J":J,
              "K":K,
              "prior_activity":prior_activity,
              "prior_qed":prior_qed,
              "prior_sa":prior_sa,
              "observed_activity":observed_activity,
              "observed_qed":observed_qed,
              "observed_sa":observed_sa,
              "prior_choice":self.prior_choice,
              "advice":self.advice,
              "decision":decision}
        return data
    
    def get_parameter(self)->Tuple[Performance,List[float]]:

        if len(self.advice)==0:
          return self.get_prior_parameters()
        
        data=self.prepare_data()
        model=pystan.StanModel(file=self.__config.STAN_PATH)
        res=model.sampling(data=data).extract()
        weights=res["w"] 
        beta1=res["beta1"]
        beta2=res["beta2"]

        mean_weights=np.mean(weights,axis=0)
        mean_beta1=np.mean(beta1)
        mean_beta2=np.mean(beta2)
        w:Performance=Performance(**{HypothesisEnum.ACT.value:mean_weights[0],HypothesisEnum.QED.value:mean_weights[1],HypothesisEnum.SA.value:mean_weights[2]})
        bias:List[float]=[mean_beta1,mean_beta2]
        self.logger.info("sample posterior parameters w {} bias {}".format(w, bias))
        return w, bias

    
    def plan(self)->ComponentEnum:
        """
          give a state, search for best actions
        """
        
        self.inferred_weights,self.inferred_bias=self.get_parameter()

        state=State(weights=self.inferred_weights,curriculum=self.curriculum)


        # create state space
        self.state_space=self.__get_state_space(root_state=state)
        
        
        for _ in range(self.n_inter):
            weights,bias=self.get_parameter()
            # trick: update weights in state space due to shallow copy
            self.inferred_weights.activity=weights.activity
            self.inferred_weights.qed=weights.qed
            self.inferred_weights.sa=weights.sa
            self.inferred_bias[0]=bias[0]
            self.inferred_bias[1]=bias[1]

            
            root = Node(state=state,parent=None,actions=state.get_possible_actions())
            if root.is_terminal:
              self.logger.info("root node is a terminal node")
              return None

            self.simulate(root,depth=0)
        self.check_node(root)
        best_advice=self.get_best_action(root,exploration_value=0).name
        self.advice.append(self.component_to_int(best_advice))
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
        user_model=Human(weights=self.inferred_weights,bias=self.inferred_bias, curriculum=self.curriculum, current_performance=self.current_performance)
        evaluation=user_model.evaluate_components()
        human_action=user_model.make_decision(evaluation,action.name)
        node.state.take_action(human_action)
        name=self.__get_state_name(node.state.curriculum+[human_action])
        state=self.state_space[depth+1][name]
        
        return Node(state=state,parent=node,actions=state.get_possible_actions())


    def __get_state_name(self, curriculum: List[ComponentEnum])->str:
      return '_'.join(list(map(lambda enums: enums.value,curriculum)))

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
      return state_space

    # def __update_weights(self,state_space:Dict[int,Dict[str,State]])->Node:
    #   root=None
    #   for depth,state_pair in state_space.items():
    #     for state in state_pair.values():
    #       state.weights=self.inferred_weights

    #   pass

    def read_component_performance(self, jobname:str)->Performance:
      performance_path=Path(self.__config.OUT_DIR,"_performance","{}_performance.csv".format(jobname))
      df=pd.read_csv(performance_path,index_col=0)
      #TODO: save non-weighted performance
      performance:Performance=Performance(df[HypothesisEnum.ACT.value][0],df[HypothesisEnum.QED.value][0],df[HypothesisEnum.SA.value][0])
      return performance


if __name__=="__main__":
  
  # components_data=get_component_statistic()
  # prior_data=get_prior_statistic()
  # human=Human(components_data=components_data,prior_data=prior_data)
  # print(human.create_curriculum())

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