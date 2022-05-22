import pystan
import numpy as np
import pandas as pd
from pathlib import Path
from copy import deepcopy
from utils import get_component_statistic,get_prior_statistic
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
    advice: List[int]=field(default_factory=lambda:[])

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

    def get_parameter_distribution(self)->Tuple[List[float],float,float,List[float],float,float]:
        data=self.prepare_data()
        model=pystan.StanModel(file=self.__config.STAN_PATH)
        fit=model.sampling(data=data)
        with open(Path(self.__config.OUT_DIR,"{}.pkl".format(self.get_jobname())),"wb") as f:
          pickle.dump({"model":model,"fit":fit},f,protocol=-1)
          
        res=fit.extract()
        weights=res["w"] 
        beta1=res["beta1"]
        beta2=res["beta2"]

        mean_weights=np.mean(weights,axis=0)
        mean_beta1=np.mean(beta1)
        mean_beta2=np.mean(beta2)
        std_weights=np.std(weights,axis=0)
        std_beta1=np.std(beta1)
        std_beta2=np.std(beta2)
        
        return mean_weights,mean_beta1,mean_beta2,std_weights,std_beta1,std_beta2

    def get_parameters(self,distribution:Optional[Tuple[List[float],float,float,List[float],float,float]]=None)->Tuple[Performance,List[float]]:
        
        if not distribution:
          w=np.random.normal(loc=0.5,scale=0.15,size=3)
          w=w/np.sum(w)
          bias1=np.random.uniform(low=1,high=4)
          bias2=np.random.uniform(low=10,high=40)
          self.logger.info("get prior parameters w {} bias {}".format(w, [bias1,bias2]))
        else:
          w1=np.random.normal(loc=distribution[0][0],scale=distribution[3][0],size=1)
          w2=np.random.normal(loc=distribution[0][1],scale=distribution[3][1],size=1)
          w3=np.random.normal(loc=distribution[0][2],scale=distribution[3][2],size=1)
          w=np.array([w1[0],w2[0],w3[0]])
          w=w/np.sum(w)
          bias1=np.random.uniform(low=distribution[1]-distribution[4],high=distribution[1]+distribution[4])
          bias2=np.random.uniform(low=distribution[2]-distribution[5],high=distribution[2]-distribution[5])
          self.logger.info("get posterior parameters w {} bias {}".format(w, [bias1,bias2]))
        w=Performance(**{HypothesisEnum.ACT.value:w[0],HypothesisEnum.QED.value:w[1],HypothesisEnum.SA.value:w[2]})
        bias:List[float]=[bias1,bias2]
        
        return w, bias

        
    
    def get_observed_data(self)->Tuple[List[List[float]],List[List[float]],List[List[float]]]:
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
              ob_a.append(0)
              ob_q.append(0)
              ob_s.append(0)
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
            self.pri_activity[self.prior_choice[k-1]]=0
            self.pri_qed[self.prior_choice[k-1]]=0
            self.pri_sa[self.prior_choice[k-1]]=0

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
    
    

    def read_component_performance(self, jobname:str)->Performance:
        performance_path=Path(self.__config.OUT_DIR,"_performance","{}_performance.csv".format(jobname))
        df=pd.read_csv(performance_path,index_col=0)
        performance:Performance=Performance(df[HypothesisEnum.ACT.value][0],df[HypothesisEnum.QED.value][0],df[HypothesisEnum.SA.value][0])
        return performance
    