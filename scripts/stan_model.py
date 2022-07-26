import pystan
import numpy as np
from utils import get_component_statistic,get_prior_statistic,softmax
from typing import Dict, List,  Tuple

from enums import ComponentEnum,ProjectConfig
from dataclasses import dataclass,field
import logging


@dataclass
class SampleParameter():
    curriculum:List[ComponentEnum]=field(default_factory=lambda:[])
    prior_choice: List[int]=field(default_factory=lambda:[])

    def __post_init__(self):
        self.logger=logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        self.__config=ProjectConfig()
        components_data=get_component_statistic()
        prior=get_prior_statistic()
        self.pri_activity=[]
        self.pri_qed=[]
        self.pri_sa=[]

        for component_name in ComponentEnum:
            p=components_data[component_name]-prior #normalized component_data
            self.pri_activity.append(p.activity)
            self.pri_qed.append(p.qed)
            self.pri_sa.append(p.sa)

    def get_jobname(self):
        return '_'.join(list(map(lambda enums: enums.value,self.curriculum)))
    


    def get_parameter(self,iter:int)->Tuple[List,List]:
        data=self.prepare_data()
        model=pystan.StanModel(file=self.__config.STAN_PATH)
        fit=model.sampling(data=data,iter=iter)
        res=fit.extract()
        weights=res["w"] 
        beta=res["beta1"]

        return weights,beta

        
    
    def component_to_int(self,component_name:ComponentEnum)->int:
        components=list(ComponentEnum)
        return components.index(component_name)+1

    def prepare_data(self)->Dict:
        K=len(self.curriculum)
        J=len(ComponentEnum)

        data={"J":J,
              "K":K,
              "prior_activity":self.pri_activity,
              "prior_qed":self.pri_qed,
              "prior_sa":self.pri_sa,
              "prior_choice":np.array(self.prior_choice,dtype=int)}
        return data