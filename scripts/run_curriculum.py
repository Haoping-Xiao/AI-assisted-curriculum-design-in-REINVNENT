from pathlib import Path
import os
from typing import Dict, List,Optional, Union
from utils import  write_curriculum_file, write_sample_file, write_run_train, write_run_sample,Performance,read_component_config
from datetime import datetime
import subprocess
from time import sleep
from enums import ComponentEnum,HypothesisEnum
from dataclasses import dataclass,field
import logging
from enums import ProjectConfig
import pandas as pd
import numpy as np
from scorer import get_scorer


logging.basicConfig(format=' %(levelname)s %(asctime)s %(name)s %(message)s',level = logging.INFO)

def execute_curriculum(jobname:str, component_config:List[Dict], agent:Path,output_dir:Path, using_gpu:Optional[bool]=True ,production_mode:Optional[bool]=False)->Path:
  # jobname = '_'.join(list(map(lambda enums: enums.value,curriculum_name)))
  jobid=datetime.now().strftime("%d-%m-%Y")
  if production_mode:
    output_dir=os.path.join(output_dir,"production")
  else:
    output_dir=os.path.join(output_dir,"{}".format(jobname))
  if not os.path.exists(output_dir):
    os.makedirs(output_dir)

  
  dir_path = os.path.dirname(os.path.realpath(__file__)) 
  reinvent_dir= os.path.join(dir_path,"../../reinventcli")
  reinvent_env="/home/xiaoh2/.conda/envs/shared"

  write_curriculum_file(jobid, jobname, str(agent), reinvent_dir, output_dir, component_config, gpu=using_gpu)
  write_sample_file(jobid, jobname,  output_dir)
  write_run_train(output_dir, reinvent_env, reinvent_dir,gpu=using_gpu)
  write_run_sample(output_dir, reinvent_env, reinvent_dir) 
  return output_dir



def successful_end(jobname:str, logfile_path: Path, ending_message:str, slurm_output_path: str, check_interval_time: int = 30)->bool:
  is_end=False
  has_error=False
  
  while (not is_end):
    try:
      with open (Path(logfile_path,slurm_output_path)) as f: # may throw FileNotFoundError if wait for resources to compute, then no log file exist
        lines=f.readlines()
        if lines: #non-empty file
          last_line=lines[-1]
          if ending_message in last_line: #Custom message in REINVENT : "Finish training"  or "Finish sampling"
            logging.info("Finish {}".format(jobname))
            is_end=True
          elif "Terminating" in last_line: #Aalto slurm default output when error occurs. "Terminating" 
            logging.error("Some error occurs during {}!".format(jobname))
            is_end=True
            has_error=True
          else: # in training/sampling process
            logging.info("Executing {}!".format(jobname))
            sleep(check_interval_time)
        f.close()
    except Exception as error:
      if type(error).__name__=="FileNotFoundError":
        logging.info('{}: File does not exist yet'.format(jobname))
        sleep(check_interval_time)
      else:
        logging.debug("{}: {}".format(jobname,error))
        break
  return not has_error

def run_job(jobname:str, output_dir:Path, ending_message:str, script:str, slurm_output_path: str = "slurm/out_0.out")->bool:
  '''
    ending_message:str="Finish training" or "Finish sampling"
    script:str="runs.sh" or "run_sample.sh"
  '''
  if Path(output_dir,slurm_output_path).is_file():
    #clean old slurm output
    os.remove(Path(output_dir,slurm_output_path))
  command=['sbatch',Path(output_dir,script)] 
  subprocess.run(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
  logging.info("Start {}".format(jobname))
  if successful_end(jobname, output_dir,ending_message,slurm_output_path):
      return True
  else:
    logging.debug("Failed to finish {}!".format(jobname))
  return False



@dataclass
class CurriculumBroker():
  weights: Performance
  curriculum: List[ComponentEnum]=field(default_factory=lambda:[])
  hypothesis_classes: List[HypothesisEnum]=field(default_factory=lambda:[HypothesisEnum.ACT,HypothesisEnum.QED,HypothesisEnum.SA])


  def __post_init__(self):
    self.logger=self.get_logger()
    self.__config=ProjectConfig()

  def get_logger(self):
    __logger=logging.getLogger(__name__)
    return __logger

  def setup_curriculum(self,component_name:ComponentEnum,jobname:str)->Path:
    # train a component on top of a given prior agent, then train inferred scoring function
    if len(self.curriculum):
        prior_agent=Path(self.__config.OUT_DIR,self.get_jobname(),self.__config.MODEL_PATH)
    else:
        prior_agent=Path(self.__config.PRIOR_DIR)
    component_config=self.setup_component(component_name,weight=1) # 1 is a default value
    curriculum_path=execute_curriculum(jobname,component_config,prior_agent,self.__config.OUT_DIR)
    success=run_job(jobname+" training",curriculum_path,self.__config.TRAIN_ENDING_MSG, self.__config.TRAIN_SCRIPT)

    if not success:
        raise Exception("some error occurs in set up curriculum") 
    return curriculum_path

  
  def get_jobname(self, component_name:Optional[ComponentEnum]=None,evaluated_curriclum:Optional[Union[ComponentEnum,HypothesisEnum]]=None):
    if not evaluated_curriclum:
        evaluated_curriclum=self.curriculum+[component_name] if component_name else self.curriculum
    return '_'.join(list(map(lambda enums: enums.value,evaluated_curriclum)))

  def setup_component(self, component_name:ComponentEnum,weight:int)->Dict:
    configs=read_component_config(self.__config.CONFIG_PATH)["components"]
    for component_config in configs:
        if component_config["name"]==component_name.value:
            component_config["weight"]=weight
            break
    return component_config

  def setup_production(self,component_name:ComponentEnum,curriculum_path:Path)->Path:
    # self.curriculum remains the same in the evaluation stage, append it only when human make a decision.
    jobname=self.get_jobname(component_name)+" production"

    prior_agent=Path(curriculum_path,self.__config.MODEL_PATH)
    component_config=self.infer_scoring_function()

    production_path=execute_curriculum(self.get_jobname(evaluated_curriclum=self.hypothesis_classes),component_config,prior_agent,curriculum_path,production_mode=True)

    success=run_job(jobname+" training",production_path,self.__config.TRAIN_ENDING_MSG,self.__config.TRAIN_SCRIPT)
    if not success:
        raise Exception("some error occurs in set up production training") 
    
    success=run_job(jobname+" sampling",production_path,self.__config.SAMPLE_ENDING_MSG,self.__config.SAMPLE_SCRIPT)
    if not success:
        raise Exception("some error occurs in set up production sampling") 
    return production_path

  def infer_performance(self,smiles:pd.DataFrame)->Performance:
    """
        evaluated smiles using each hypothesis class
    """
    scores=Performance()
    for hypothesis_class in self.hypothesis_classes:
        scorer=get_scorer(hypothesis_class.value)
        setattr(scores,hypothesis_class.value,np.mean(scorer.get_score(smiles)))
    return scores


  def infer_scoring_function(self)->List[Dict]:
    # TODO: learn weights
    # return a list of weighted components
    components=[]
    for hypothesis_class in self.hypothesis_classes:
        weight=getattr(self.weights,hypothesis_class.value)
        components.append(self.setup_component(hypothesis_class,weight))
    return components
  
  def save_performance(self,jobname:ComponentEnum,performance:Performance):
    #create a new folder to save
    path=Path(self.__config.OUT_DIR,"_performance")
    try:
        os.makedirs(path)
    except FileExistsError:
        pass
    performance.to_csv(jobname,path)











# def run_workflow(id:str, output_dir:Path, train_ending_message:str="Finish training",train_script:str="runs.sh", sample_ending_message:str="Finish sampling", sample_script:str="run_sample.sh", slurm_output_path: str = "slurm/out_0.out")->bool:
#   if Path(output_dir,slurm_output_path).is_file():
#     #clean old slurm output
#     os.remove(Path(output_dir,slurm_output_path))
#   command=['sbatch',Path(output_dir,train_script)] 
#   subprocess.run(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
#   logging.info("{}: Start training".format(id))
#   if successful_end("{} training".format(id), output_dir,train_ending_message,slurm_output_path):
#     command=['sbatch',Path(output_dir,sample_script)]
#     subprocess.run(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
#     logging.info("{}: Start sampling".format(id))
#     if successful_end("{} sampling".format(id), output_dir,sample_ending_message,slurm_output_path):
#       logging.info("{}: Finish the workflow".format(id))
#       return True
#   else:
#     logging.debug("{}: Failed to finish the wrokflow!".format(id))
#   return False


