from pathlib import Path
from typing import Dict, List
from dataclasses import dataclass,field
import pandas as pd
from scorer import get_scorer
import numpy as np
from enums import ComponentEnum
from run_curriculum import execute_curriculum, run_workflow
from utils import Performance, read_component_config, read_sample_smiles,get_prior_statistic
import os
from multiprocessing import Pool
import logging
@dataclass
class AI_assistant():
    hypothesis_classes: List[ComponentEnum]=field(default_factory=lambda:[ComponentEnum.ACT,ComponentEnum.QED,ComponentEnum.SA]) #default=['activity','qed','sa']
    curriculum: List[ComponentEnum]=field(default_factory=lambda:[])
    weights: Performance=Performance(**{ComponentEnum.ACT.value:1,ComponentEnum.QED.value:1,ComponentEnum.SA.value:1})
    output_dir:Path="/scratch/work/xiaoh2/Thesis/results/curriculum"
    prior_path:Path="/scratch/work/xiaoh2/Thesis/models/augmented.prior"
    config_path:Path="/scratch/work/xiaoh2/Thesis/component_config/component_lib.json"
    curriculum_model_path:Path="results_0/Agent.ckpt"
    curriculum_sample_path:Path="results_0/sampled.csv"


    
    def get_logger(self):
        self.logger=logging.getLogger(__name__)

    def setup_curriculum_pool(self)->List[ComponentEnum]:
        pool=[]
        for component in ComponentEnum:
            if component not in self.hypothesis_classes:
                pool.append(component)
        return pool

    def setup_component(self, component_name:ComponentEnum,weight:int)->Dict:
        configs=read_component_config(self.config_path)["components"]
        for component_config in configs:
            if component_config["name"]==component_name.value:
                component_config["weight"]=weight
                break
        return component_config

    def setup_curriculum(self,component_name:ComponentEnum,jobname:str)->Path:
        # train a component on top of a given prior agent, then train inferred scoring function
        if len(self.curriculum):
            prior_agent=Path(self.output_dir ,"_".join(self.curriculum),self.curriculum_model_path)
        else:
            prior_agent=Path(self.prior_path)
        component_config=self.setup_component(component_name,weight=1) # 1 is a default value
        curriculum_path=execute_curriculum(jobname,component_config,prior_agent,self.output_dir)
        success=run_workflow(id=jobname,output_dir=curriculum_path)
        if not success:
            raise Exception("some error occurs in set up curriculum") 
        return curriculum_path

    def infer_scoring_function(self)->List[Dict]:
        # return a list of weighted components
        components=[]
        for hypothesis_class in self.hypothesis_classes:
            weight=getattr(self.weights,hypothesis_class.value)
            components.append(self.setup_component(hypothesis_class,weight))
        return components
    
    def setup_production(self,component_name:ComponentEnum,curriculum_path:Path)->Path:
        evaluated_curriclum=self.curriculum+[component_name] # self.curriculum remains the same in the evaluation stage, append it only when human make a decision.
        id='_'.join(list(map(lambda enums: enums.value,evaluated_curriclum)))+" production"

        prior_agent=Path(curriculum_path,self.curriculum_model_path)
        component_config=self.infer_scoring_function()
        production_path=execute_curriculum(self.hypothesis_classes,component_config,prior_agent,curriculum_path,production_mode=True)
        success=run_workflow(id=id,output_dir=production_path)
        if not success:
            raise Exception("some error occurs in set up production") 
        return production_path

    

    def infer_performance(self,smiles:pd.DataFrame)->Performance:
        scores=Performance()
        for hypothesis_class in self.hypothesis_classes:
            scorer=get_scorer(hypothesis_class.value)
            setattr(scores,hypothesis_class.value,np.mean(scorer.get_score(smiles)))
        return scores*self.weights

    def save_performance(self,component_name:ComponentEnum,performance:Performance):
        #create a new folder to save
        path=Path(self.output_dir,"_performance")
        try:
            os.makedirs(path)
        except FileExistsError:
            pass
        evaluated_curriclum=self.curriculum+[component_name]
        performance.to_csv(evaluated_curriclum,path)
        
    def get_jobname(self, component_name:List[ComponentEnum]):
        evaluated_curriclum=self.curriculum+[component_name]
        return '_'.join(list(map(lambda enums: enums.value,evaluated_curriclum)))

    def evaluate_component(self,component_name:ComponentEnum):
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

    
    def recommend_component(self,current_performance:Performance)->str:
        component_pool=self.setup_curriculum_pool()
        performance={}
        best_component=None
        with Pool(len(component_pool)) as p:
            res=p.map(self.evaluate_component,component_pool)
            
        for i,component_name in enumerate(component_pool):
            #check if it is better
            if res[i]>current_performance.sum:
                performance[component_name.value]=res[i]
        if performance:
            sorted_component=sorted(performance, key=performance.get)
            evaluated_curriclum=self.curriculum if len(self.curriculum) else ["prior"]
            filename='_'.join(list(map(lambda enums: enums.value,evaluated_curriclum)))
            with open("{}.txt".format(Path(self.output_dir,"_performance",filename)),"w") as f:
                for component in sorted_component:
                    f.write("{} {}\n".format(component,performance[component]))
                f.close()
            #TODO: softmax decision
            best_component=sorted_component[-1]
        return best_component        
        

# hypothesis_classes=['activity','qed','sa']

path="/scratch/work/xiaoh2/Thesis/results/sampled.csv"
smiles= read_sample_smiles(path)


current_performance=get_prior_statistic()
ai=AI_assistant()
# print(ai.infer_performance(smiles))
best=ai.recommend_component(current_performance)
print(best)
