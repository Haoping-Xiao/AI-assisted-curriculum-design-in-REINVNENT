import pystan
import numpy as np
import pandas as pd
from pathlib import Path
from copy import deepcopy
from utils import get_component_statistic,get_prior_statistic,softmax
from typing import Dict, List, Optional, Tuple
from utils import  Performance
from enums import ComponentEnum,HypothesisEnum,ProjectConfig
from dataclasses import dataclass,field
import logging
import pickle

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
          if component_name!=ComponentEnum.END:
            p=components_data[component_name]-prior #normalized component_data
            self.pri_activity.append(p.activity)
            self.pri_qed.append(p.qed)
            self.pri_sa.append(p.sa)

    def get_jobname(self):
        return '_'.join(list(map(lambda enums: enums.value,self.curriculum)))
    

    # def get_piror_parameter(self,distribution:Tuple[List[float],float,List[float],float])->Tuple[List[float],float]:
    #     w1,w2,w3=-1,-1,-1
    #     while w1<0:
    #         w1=np.random.normal(loc=distribution[0][0],scale=distribution[2][0])
    #     while w2<0:
    #         w2=np.random.normal(loc=distribution[0][1],scale=distribution[2][1])
    #     while w3<0:
    #         w3=np.random.normal(loc=distribution[0][2],scale=distribution[2][2])
    #     w=np.array([w1,w2,w3])
    #     w=w/np.sum(w)
    #     bias=np.random.uniform(low=distribution[1]-distribution[3],high=distribution[1]+distribution[3])
    #     return w,bias

    def get_parameter(self,iter:int)->Tuple[List,List]:
        data=self.prepare_data()
        model=pystan.StanModel(file=self.__config.STAN_PATH)
        fit=model.sampling(data=data,iter=iter)
        with open(Path(self.__config.OUT_DIR,"{}.pkl".format(self.get_jobname())),"wb") as f:
          pickle.dump({"model":model,"fit":fit},f,protocol=-1)
          
        res=fit.extract()
        weights=res["w"] 
        beta=res["beta1"]

        return weights,beta

    # def get_parameters(self,distribution:Tuple[List[float],float,List[float],float])->Tuple[Performance,List[float]]:
        
    #     # if not distribution:
    #     #   w=np.random.normal(loc=0.5,scale=0.2,size=3)
    #     #   w=softmax(w,beta=1)
    #     #   bias=np.random.uniform(low=1,high=4)
    #     #   self.logger.info("get prior parameters w {} bias {}".format(w, bias))
    #     # else:
    #     w1=np.random.normal(loc=distribution[0][0],scale=distribution[2][0])
    #     w2=np.random.normal(loc=distribution[0][1],scale=distribution[2][1])
    #     w3=np.random.normal(loc=distribution[0][2],scale=distribution[2][2])
    #     w=np.array([w1,w2,w3])
    #     # w=w/np.sum(w)
    #     self.logger.info("w {}".format(w))
    #     w=softmax(w,beta=1)
    #     bias=np.random.uniform(low=distribution[1]-distribution[3],high=distribution[1]+distribution[3])
    #     w=Performance(**{HypothesisEnum.ACT.value:w[0],HypothesisEnum.QED.value:w[1],HypothesisEnum.SA.value:w[2]})
    #     bias:List[float]=[bias]
        
    #     return w, bias

        
    
    def component_to_int(self,component_name:ComponentEnum)->int:
        components=list(ComponentEnum)
        return components.index(component_name)+1

    def prepare_data(self)->Dict:
        K=len(self.curriculum)
        J=len(ComponentEnum)-1
        
        # # decision=[]
        # prior_activity=[]
        # prior_qed=[]
        # prior_sa=[]

        # for k in range(K):
        #   if len(self.prior_choice)>1:
        #     self.pri_activity[self.prior_choice[k-1]]=0
        #     self.pri_qed[self.prior_choice[k-1]]=0
        #     self.pri_sa[self.prior_choice[k-1]]=0

        #   prior_activity.append(deepcopy(self.pri_activity))
        #   prior_qed.append(deepcopy(self.pri_qed))
        #   prior_sa.append(deepcopy(self.pri_sa))



        # self.logger.info("curriculum {}".format(self.curriculum))
        # self.logger.info("prior_choice {}".format(self.prior_choice))
        # self.logger.info("prior_activity {}".format(prior_activity))
        # self.logger.info("prior_qed {}".format(prior_qed))
        # self.logger.info("prior_sa {}".format(prior_sa))

        data={"J":J,
              "K":K,
              # "prior_activity":prior_activity,
              # "prior_qed":prior_qed,
              # "prior_sa":prior_sa,
              "prior_activity":self.pri_activity,
              "prior_qed":self.pri_qed,
              "prior_sa":self.pri_sa,
              "prior_choice":np.array(self.prior_choice,dtype=int)}
        return data