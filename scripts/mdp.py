from __future__ import annotations
from math import gamma
from secrets import choice
from enums import ComponentEnum, HypothesisEnum, ProjectConfig
from typing import Dict, List, Optional, Union,Tuple
from dataclasses import dataclass,field
from pathlib import Path
import logging
import numpy as np
from utils import Performance, read_component_config, read_sample_smiles,softmax
from run_curriculum import execute_curriculum, run_job
import pandas as pd
from scorer import get_scorer
import os
from human import Human

@dataclass
class State():
    """
        In this project, a state is Agent parameter, represeted by its curriculums.
        Taking Action is choosing and running a component.
        Possible actions is possible component. 
    """
    weights: Performance
    curriculum: List[ComponentEnum]=field(default_factory=lambda:[])
    hypothesis_classes: List[HypothesisEnum]=field(default_factory=lambda:[HypothesisEnum.ACT,HypothesisEnum.QED,HypothesisEnum.SA]) #default=['activity','qed','sa']
    max_curriculum: int=5
    


    def __post_init__(self):
        self.__logger=self.__get_logger()
        self.__config=ProjectConfig()

    def get_possible_actions(self)->List[ComponentEnum]:
        actions=[]
        for component in ComponentEnum:
            if component not in self.curriculum:
                actions.append(component)
        return actions
        

    def take_action(self, action:ComponentEnum)->State:
        """
        input action
        return state
        """
        try:
            jobname=self.__get_jobname(action)
            curriculum_path=Path(self.__config.OUT_DIR,jobname)

            if not curriculum_path.exists():
                try:
                    curriculum_path=self.__setup_curriculum(action,jobname)
                except Exception as e:
                    raise Exception("bugs in setup curriculum: {}".format(e))

            production_path=Path(curriculum_path,"production")
            self.__logger.info("production_path {}".format(production_path))
            if not production_path.exists():
                try:
                    production_path=self.__setup_production(action,curriculum_path)
                except Exception as e:
                    raise Exception("bugs in setup production: {}".format(e))
        except Exception as e:
            self.__logger.debug(e)
        #TODO: check if affect __get_jobname 
        # self.curriculum.append(action)
        return State(weights=self.weights,curriculum=self.curriculum+[action])

    def is_terminal(self)->bool:
        actions=self.get_possible_actions()
        return len(self.curriculum)==self.max_curriculum or len(actions)==0
        
    def get_reward(self):
        # assume a new component is added into self.currciculum
        # means get_reward should only happen after take_action
        self.__logger.critical("get_reward should only happen after take_action")
        jobname=self.__get_jobname()
        production_path=Path(self.__config.OUT_DIR,jobname,"production")
        smiles_path=Path(production_path,self.__config.SAMPLE_PATH)
        smiles=  read_sample_smiles(smiles_path) 
        weighted_performance:Performance=self.__infer_performance(smiles,self.weights)
        self.__save_performance(jobname,weighted_performance)
        return weighted_performance.sum

    def __get_logger(self):
        __logger=logging.getLogger(__name__)
        return __logger
    
    def __get_jobname(self, component_name:Optional[ComponentEnum]=None,evaluated_curriclum:Optional[Union[ComponentEnum,HypothesisEnum]]=None):
        if not evaluated_curriclum:
            evaluated_curriclum=self.curriculum+[component_name] if component_name else self.curriculum
        return '_'.join(list(map(lambda enums: enums.value,evaluated_curriclum)))

    def __setup_component(self, component_name:ComponentEnum,weight:int)->Dict:
        configs=read_component_config(self.__config.CONFIG_PATH)["components"]
        for component_config in configs:
            if component_config["name"]==component_name.value:
                component_config["weight"]=weight
                break
        return component_config

    def __setup_curriculum(self,component_name:ComponentEnum,jobname:str)->Path:
        # train a component on top of a given prior agent, then train inferred scoring function
        if len(self.curriculum):
            prior_agent=Path(self.__config.OUT_DIR,self.__get_jobname(),self.__config.MODEL_PATH)
        else:
            prior_agent=Path(self.__config.PRIOR_DIR)
        component_config=self.__setup_component(component_name,weight=1) # 1 is a default value
        curriculum_path=execute_curriculum(jobname,component_config,prior_agent,self.__config.OUT_DIR)
        success=run_job(jobname+" training",curriculum_path,self.__config.TRAIN_ENDING_MSG, self.__config.TRAIN_SCRIPT)

        if not success:
            raise Exception("some error occurs in set up curriculum") 
        return curriculum_path

    def __infer_scoring_function(self)->List[Dict]:
        # TODO: learn weights
        # return a list of weighted components
        components=[]
        for hypothesis_class in self.hypothesis_classes:
            weight=getattr(self.weights,hypothesis_class.value)
            components.append(self.__setup_component(hypothesis_class,weight))
        return components

    def __setup_production(self,component_name:ComponentEnum,curriculum_path:Path)->Path:
         # self.curriculum remains the same in the evaluation stage, append it only when human make a decision.
        jobname=self.__get_jobname(component_name)+" production"

        prior_agent=Path(curriculum_path,self.__config.MODEL_PATH)
        component_config=self.__infer_scoring_function()

        production_path=execute_curriculum(self.__get_jobname(evaluated_curriclum=self.hypothesis_classes),component_config,prior_agent,curriculum_path,production_mode=True)

        success=run_job(jobname+" training",production_path,self.__config.TRAIN_ENDING_MSG,self.__config.TRAIN_SCRIPT)
        if not success:
            raise Exception("some error occurs in set up production training") 
        
        success=run_job(jobname+" sampling",production_path,self.__config.SAMPLE_ENDING_MSG,self.__config.SAMPLE_SCRIPT)
        if not success:
            raise Exception("some error occurs in set up production sampling") 
        return production_path


    def __infer_performance(self,smiles:pd.DataFrame,weights:Performance)->Performance:
        """
            evaluated smiles using each hypothesis class
        """
        scores=Performance()
        for hypothesis_class in self.hypothesis_classes:
            scorer=get_scorer(hypothesis_class.value)
            setattr(scores,hypothesis_class.value,np.mean(scorer.get_score(smiles)))
        return scores*weights

    def __save_performance(self,jobname:ComponentEnum,performance:Performance):
        #create a new folder to save
        path=Path(self.__config.OUT_DIR,"_performance")
        try:
            os.makedirs(path)
        except FileExistsError:
            pass
        performance.to_csv(jobname,path)






@dataclass
class Node():
    state:State
    parent:Union[Node,None]
    num_visits: int=0
    total_reward:float=0
    children:Dict[ComponentEnum,Node]=field(default_factory=dict)
    is_terminal:bool=field(init=False)
    is_fully_expanded:bool=field(init=False)

    def __post_init__(self):
        self.is_terminal=self.state.is_terminal()
        self.is_fully_expanded=self.is_terminal

@dataclass
class MDP():
    inferred_weights:Performance
    inferred_bias:Tuple[float,float]=(10,10)
    n_inter:int=100
    curriculum: List[ComponentEnum]=field(default_factory=lambda:[])
    exploration_constant:float=1/np.sqrt(2)
    max_depth:int=1
    gamma:float=1

    def plan(self, state:State):
        # give a state, search for best actions
        root = Node(state,None)
        if root.is_terminal:
            return None
        
        for _ in range(self.n_inter):
        # TODO: sample parameter from distribution
            self.simulate(root,depth=0)
        return self.get_best_child(root,exploration_value=0)
        


    def simulate(self,root:Node, depth:int):
        """
            execute a selection-expansion-simulation-backpropagation round
        """
        node=self.select_node(root)
        reward = node.state.get_reward()
        depth+=1
        if depth==self.max_depth or node.num_visits==0 or node.is_terminal:
            q=reward
        else:
            q=reward+self.gamma*self.simulate(node,depth)
        root.num_visits+=1
        node.num_visits+=1
        node.total_reward+=(q-node.total_reward)/node.num_visits




    def select_node(self, node:Node)->Node:
        """
            given a node, expand children if not fully expanded or choose its children by exploitation and exploration
        """
        if not node.is_fully_expanded:
            self.expand(node)
        return self.get_best_child(node,self.exploration_constant)

            
                

    def expand(self, node:Node):
        """
            expand all children node
        """
        actions=node.state.get_possible_actions()
        for action in actions:
            new_node=self.transition(node,action)
            node.children[action]=new_node
        node.is_fully_expanded=True
    

    def transition(self,node:Node, action:ComponentEnum)->State:
        #AI's action for human is advice
        user_model=Human(weights=self.inferred_weights,bias=self.inferred_bias)
        evaluation=user_model.evaluate_components()
        human_action=user_model.make_decision(evaluation,action)
        return Node(node.state.take_action(human_action),node)


    def backpropogate(self, node:Node, reward:float):
        while node:
            node.num_visits+=1
            node.total_reward+=(reward-node.total_reward)/node.num_visits
            node=node.parent

    def get_best_child(self, node:Node, exploration_value:float)->Node:
        best_value = float("-inf")
        best_nodes=[]
        for child in node.children.values():
            node_value=child.total_reward+exploration_value*np.sqrt(np.log(node.num_visits)/child.num_visits)
            if node_value>best_value:
                best_value=node_value
                best_nodes=[child]
            elif node_value==best_value:
                best_nodes.append(child)
        return np.random.choice(best_nodes)


    # def simulate(self,)
    


