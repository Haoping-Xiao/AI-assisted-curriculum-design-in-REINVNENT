from __future__ import annotations
from enums import ComponentEnum, HypothesisEnum, ProjectConfig
from typing import Dict, List,  Union
from dataclasses import dataclass,field
from pathlib import Path
from utils import Performance, read_sample_smiles
from run_curriculum import CurriculumBroker
import time
import pandas as pd
@dataclass
class Action():
    name:ComponentEnum
    total_reward:float=0
    num_visits:int=0


@dataclass
class State():
    """
        In this project, a state is Agent parameter, represeted by its curriculums.
        Taking Action is choosing and running a component.
        Possible actions is possible component. 
    """
    curriculum: List[ComponentEnum]=field(default_factory=lambda:[])
    hypothesis_classes: List[HypothesisEnum]=field(default_factory=lambda:[HypothesisEnum.ACT,HypothesisEnum.QED,HypothesisEnum.SA])
    
   

    def __post_init__(self):
        # self.__logger=self.__get_logger()
        self.config=ProjectConfig()
        self.broker=CurriculumBroker(curriculum=self.curriculum, hypothesis_classes=self.hypothesis_classes)

    def get_possible_actions(self)->List[Action]:
        if ComponentEnum.END in self.curriculum:
            return []
        actions=[]
        for component in ComponentEnum:
            # if component not in self.curriculum:
            actions.append(Action(name=component))
        return actions
        

    def take_action(self, action:ComponentEnum):
        """
        input action
        return state
        """
        try:
            jobname=self.broker.get_jobname(action)
            curriculum_path=Path(self.config.OUT_DIR,jobname)
            if jobname!="prior":
                # only none action(curriculum) doesn't need to run curriculum
                #check if the agent exist
                if not Path(curriculum_path,self.config.MODEL_PATH).exists():
                    try:
                        t=time.time()
                        curriculum_path=self.broker.setup_curriculum(action,jobname)
                        self.broker.logger.info("curriculum spent {} secs".format(time.time()-t))
                    except Exception as e:
                        raise Exception("bugs in setup curriculum: {}".format(e))
            
            production_path=Path(curriculum_path,"production")
            #check if sample file exists
            if not Path(production_path,self.config.SAMPLE_PATH).exists():
                try:
                    t=time.time()
                    production_path=self.broker.setup_production(action,curriculum_path)
                    self.broker.logger.info("production spent {} secs".format(time.time()-t))
                except Exception as e:
                    raise Exception("bugs in setup production: {}".format(e))
        except Exception as e:
            self.broker.logger.info(e)


    def is_terminal(self)->bool:
        actions=self.get_possible_actions()
        return len(actions)==0
        
    def get_reward(self)->Performance:
        # assume a new component is added into self.currciculum
        # means get_reward should only happen after take_action
        jobname=self.broker.get_jobname()
        performance_path=Path(self.config.OUT_DIR,"_performance","{}_performance.csv".format(jobname))
        if not performance_path.exists():
            production_path=Path(self.config.OUT_DIR,jobname,"production")
            smiles_path=Path(production_path,self.config.SAMPLE_PATH)
            smiles=  read_sample_smiles(smiles_path) 
            performance:Performance=self.broker.infer_performance(smiles)
            self.broker.save_performance(jobname,performance)
        else:
            df=pd.read_csv(performance_path,index_col=0)
            performance:Performance=Performance(df[HypothesisEnum.ACT.value][0],df[HypothesisEnum.QED.value][0],df[HypothesisEnum.SA.value][0])
        return performance





@dataclass
class Node():
    state:State
    parent:Union[Node,None]
    actions:List[Action]
    num_visits: int=0
    total_reward:float=0
    children:Dict[ComponentEnum,Node]=field(default_factory=dict)
    is_terminal:bool=field(init=False)
    is_fully_expanded:bool=field(init=False)

    def __post_init__(self):
        self.is_terminal=self.state.is_terminal()
        self.is_fully_expanded=self.is_terminal
