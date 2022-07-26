from __future__ import annotations
from pathlib import Path
from typing import Dict, List
import pandas as pd
import numpy as np
import json
import os
from dataclasses import dataclass,field,asdict
from enums import ComponentEnum

@dataclass()
class Performance():
  activity: float=0
  qed:float=0
  sa: float=0
  sum: float=field(init=False,repr=False)
  def __post_init__(self)->None:
    self.sum=self.activity+self.qed+self.sa
  def __eq__(self, __o: Performance) -> bool:
      return self.activity==__o.activity and self.qed==__o.qed and self.sa==__o.sa
  def __gt__(self, __o: Performance) -> bool:
    return self.activity>__o.activity or self.qed>__o.qed or self.sa>__o.sa
  def __lt__(self, __o: Performance) -> bool:
    return not (self.__eq__(__o) or self.__gt__(__o))
  def __sub__(self, __o: Performance) -> Performance:
    #round due to float point substraction
    return Performance(round(self.activity-__o.activity,4),round(self.qed-__o.qed,4), round(self.sa-__o.sa,4))
  def __mul__(self, __o: Performance) -> Performance:
    return Performance(round(self.activity*__o.activity,4),round(self.qed*__o.qed,4), round(self.sa*__o.sa,4))
  def copy(self)-> Performance:
    return Performance(self.activity,self.qed,self.sa)

  def to_csv(self,jobname:str,filepath:Path):
    data=asdict(self)
    df=pd.DataFrame(data=data,index=[jobname])
    df.to_csv(Path(filepath,"{}_performance.csv".format(jobname)))

def read_sample_smiles(path:Path)->pd.DataFrame:
    assert Path(path).is_file(), '{} is not exist'.format(path)
    smiles=pd.read_csv(path,header=None).squeeze()
    return smiles

def read_scaffold_smiles(path:Path)->pd.DataFrame:
    assert Path(path).is_file(), '{} is not exist'.format(path)
    df=pd.read_csv(path)
    return df['SMILES']

def read_component_config(path:Path):
    assert Path(path).is_file(), '{} is not exist'.format(path)
    with open(path,'r') as config_file:
        component_config=json.load(config_file)
    return component_config

def write_curriculum_file(jobid, jobname, agent, reinvent_dir, output_dir, components, epoch=300, id=0, gpu=False):
  # if required, generate a folder to store the results
  try:
      os.makedirs(output_dir)
  except FileExistsError:
      pass
  
  diversity_filter = {
    "name": "IdenticalMurckoScaffold",
    "bucket_size": 25,
    "minscore": 0.2,
    "minsimilarity": 0.4
  }

  inception = {
    "memory_size": 20,
    "sample_size": 5,
    "smiles": []
  }

  scoring_function= {
   "name": "custom_sum",
    "parallel": not gpu,
    "parameters": [
      components
    ] 
  } if not isinstance(components,list) else {
    "name": "custom_sum",
    "parallel": not gpu,
    "parameters": components
  }

  configuration = {
    "version": 2,
    "run_type": "reinforcement_learning",
    "parameters": {
      "scoring_function": scoring_function
    }
  }

  configuration["parameters"]["diversity_filter"] = diversity_filter
  configuration["parameters"]["inception"] = inception

  configuration["parameters"]["reinforcement_learning"] = {
      "prior": os.path.join(reinvent_dir, "data/augmented.prior"),
      "agent": agent,
      "n_steps": epoch,
      "sigma": 128,
      "learning_rate": 0.0001,
      "batch_size": 128,
      "reset": 0,
      "reset_score_cutoff": 0.5,
      "margin_threshold": 50
  }

  configuration["logging"] = {
      "sender": "http://127.0.0.1",
      "recipient": "local",
      "logging_frequency": 0,
      "logging_path": os.path.join(output_dir, "progress_train_{}.log".format(id)),
      "resultdir": os.path.join(output_dir, "results_{}".format(id)),
      "job_name": "{}".format(jobname),
      "job_id": jobid
  }
  
  # write the configuration file to the disc
  conf_filename = os.path.join(output_dir, "config_{}.json".format(id))
  with open(conf_filename, 'w') as f:
      json.dump(configuration, f, indent=4, sort_keys=True)
  return conf_filename


def write_train_file(jobid, jobname, reinvent_dir, output_dir, components, epoch=300, id=0, gpu=False):
  # if required, generate a folder to store the results
  try:
      os.makedirs(output_dir)
  except FileExistsError:
      pass
  diversity_filter = {
    "name": "IdenticalMurckoScaffold",
    "bucket_size": 25,
    "minscore": 0.2,
    "minsimilarity": 0.4
  }

  inception = {
    "memory_size": 20,
    "sample_size": 5,
    "smiles": []
  }

  scoring_function= {
   "name": "custom_sum",
    "parallel": not gpu,
    "parameters": [
      components
    ] 
  } if not isinstance(components,list) else {
    "name": "custom_sum",
    "parallel": not gpu,
    "parameters": components
  }

  configuration = {
    "version": 2,
    "run_type": "reinforcement_learning",
    "parameters": {
      "scoring_function": scoring_function
    }
  }

  configuration["parameters"]["diversity_filter"] = diversity_filter
  configuration["parameters"]["inception"] = inception

  configuration["parameters"]["reinforcement_learning"] = {
      "prior": os.path.join(reinvent_dir, "data/augmented.prior"),
      "agent": os.path.join(reinvent_dir, "data/augmented.prior"),
      "n_steps": epoch,
      "sigma": 128,
      "learning_rate": 0.0001,
      "batch_size": 128,
      "reset": 0,
      "reset_score_cutoff": 0.5,
      "margin_threshold": 50
  }

  configuration["logging"] = {
      "sender": "http://127.0.0.1",
      "recipient": "local",
      "logging_frequency": 0,
      "logging_path": os.path.join(output_dir, "progress_train_{}.log".format(id)),
      "resultdir": os.path.join(output_dir, "results_{}".format(id)),
      "job_name": "{}".format(jobname),
      "job_id": jobid
  }
  
  # write the configuration file to the disc
  conf_filename = os.path.join(output_dir, "config_{}.json".format(id))
  with open(conf_filename, 'w') as f:
      json.dump(configuration, f, indent=4, sort_keys=True)
  return conf_filename

def write_sample_file(jobid, jobname, output_dir, id=0):
  try:
      os.makedirs(output_dir)
  except FileExistsError:
      pass
  configuration={
    "logging": {
        "job_id": jobid,
        "job_name":  "{}".format(jobname),
        "logging_path": os.path.join(output_dir, "progress_sample_{}.log".format(id)),
        "recipient": "local",
        "sender": "http://127.0.0.1"
    },
    "parameters": {
        "model_path": os.path.join(output_dir, "results_{}/Agent.ckpt".format(id)),
        "output_smiles_path": os.path.join(output_dir, "results_{}/sampled.csv".format(id)),
        "num_smiles": 1000,
        "batch_size": 128,                          
        "with_likelihood": False
    },
    "run_type": "sampling",
    "version": 2
  }
  conf_filename = os.path.join(output_dir, "sample_config_{}.json".format(id))
  with open(conf_filename, 'w') as f:
      json.dump(configuration, f, indent=4, sort_keys=True)
  return conf_filename

def write_run_train(output_dir, reinvent_env, reinvent_dir, n_component_configs=0, gpu=False):
  runfile = output_dir + '/runs.sh'
  try:
    os.mkdir(output_dir + '/slurm')
  except FileExistsError:
      pass
  with open(runfile, 'w') as f:
      f.write("#!/bin/bash -l \n")
      f.write("#SBATCH --gres=gpu:1 --exclude=gpu[11-17]\n") if gpu else f.write("#SBATCH --cpus-per-task=4 \n")
      f.write('#SBATCH --time=01:00:00 \n')
      f.write('#SBATCH -o {}/slurm/train_%a.out\n'.format(output_dir))
      f.write('#SBATCH --array=0-{}\n'.format(n_component_configs))
      # f.write('#SBATCH -p short\n')
      f.write('\n')
      f.write('module purge\n')
      f.write('module load anaconda\n')
      f.write('source activate {}\n'.format(reinvent_env))
      f.write('\n')
      f.write('conf_filename="{}/config_$SLURM_ARRAY_TASK_ID.json"\n'.format(output_dir))
      f.write('srun python {}/input.py $conf_filename\n'.format(reinvent_dir))
  
def write_run_sample(output_dir, reinvent_env, reinvent_dir, n_component_configs=0):
  runfile = output_dir + '/run_sample.sh'
  try:
    os.mkdir(output_dir + '/slurm')
  except FileExistsError:
      pass
  with open(runfile, 'w') as f:
      f.write("#!/bin/bash -l \n")
      f.write("#SBATCH --cpus-per-task=4 \n")
      f.write('#SBATCH --time=00:05:00 --mem-per-cpu=4000\n')
      f.write('#SBATCH -o {}/slurm/sample_%a.out\n'.format(output_dir))
      f.write('#SBATCH --array=0-{}\n'.format(n_component_configs))
      f.write('#SBATCH -p short\n')
      f.write('\n')
      f.write('module purge\n')
      f.write('module load anaconda\n')
      f.write('source activate {}\n'.format(reinvent_env))
      f.write('\n')
      f.write('conf_filename="{}/sample_config_$SLURM_ARRAY_TASK_ID.json"\n'.format(output_dir))
      f.write('srun python {}/input.py $conf_filename\n'.format(reinvent_dir))

def get_component_statistic()->Dict[ComponentEnum,Performance]:
    components_data={}
    components_data[ComponentEnum.BOND]=Performance(**{"activity":0.0814,"qed":0.5528,"sa":0.7063})
    components_data[ComponentEnum.HBD1]=Performance(**{"activity":0.0845,"qed":0.5871,"sa":0.7536})
    components_data[ComponentEnum.HBD4]=Performance(**{"activity":0.1102,"qed":0.0855,"sa":0.5718})
    components_data[ComponentEnum.ALERT]=Performance(**{"activity":0.0818,"qed":0.5866,"sa":0.7998})
    components_data[ComponentEnum.MASS3]=Performance(**{"activity":0.0805,"qed":0.4723,"sa":0.7372})
    components_data[ComponentEnum.SLOGP2]=Performance(**{"activity":0.0803,"qed":0.6220,"sa":0.7344})
    components_data[ComponentEnum.RING3]=Performance(**{"activity":0.0920,"qed":0.4863,"sa":0.7087})
    components_data[ComponentEnum.HBA2]=Performance(**{"activity":0.0800,"qed":0.6002,"sa":0.7459})
    components_data[ComponentEnum.TPSA3]=Performance(**{"activity":0.0963,"qed":0.71051,"sa":0.7505})
    components_data[ComponentEnum.GRAPH1]=Performance(**{"activity":0.0809,"qed":0.5695,"sa":0.7117})
    components_data[ComponentEnum.CENTER]=Performance(**{"activity":0.0844,"qed":0.5904,"sa":0.7276})
    components_data[ComponentEnum.SIM0]=Performance(**{"activity":0.2835,"qed":0.4718,"sa":0.4718})
    components_data[ComponentEnum.SIM1]=Performance(**{"activity":0.2597,"qed":0.4306,"sa":0.7599})
    components_data[ComponentEnum.SIM2]=Performance(**{"activity":0.1341,"qed":0.3128,"sa":0.7408})
    components_data[ComponentEnum.SIM3]=Performance(**{"activity":0.3355,"qed":0.4005,"sa":0.8521})
    components_data[ComponentEnum.SIM4]=Performance(**{"activity":0.3156,"qed":0.3307,"sa":0.7158})
    components_data[ComponentEnum.SIM5]=Performance(**{"activity":0.1010,"qed":0.5367,"sa":0.8610})
    components_data[ComponentEnum.SIM6]=Performance(**{"activity":0.2204,"qed":0.3924,"sa":0.8357})
    components_data[ComponentEnum.SIM7]=Performance(**{"activity":0.1420,"qed":0.7338,"sa":0.7924})
    components_data[ComponentEnum.SIM8]=Performance(**{"activity":0.3153,"qed":0.4646,"sa":0.8360})
    components_data[ComponentEnum.SIM9]=Performance(**{"activity":0.1484,"qed":0.4927,"sa":0.8158})
    components_data[ComponentEnum.SUB]=Performance(**{"activity":0.0813,"qed":0.5713,"sa":0.7237})
    components_data[ComponentEnum.ACT]=Performance(**{"activity":0.5,"qed":0.6830,"sa":0.8845})
    components_data[ComponentEnum.SA]=Performance(**{"activity":0.0607,"qed":0.6482,"sa":0.9031})
    components_data[ComponentEnum.QED]=Performance(**{"activity":0.0707,"qed":0.7708,"sa":0.8145})
    components_data[ComponentEnum.END]=get_prior_statistic()
    return components_data

def get_prior_statistic()->Performance:
    return Performance(**{"activity":0.0750,"qed":0.5599,"sa":0.7358})

def softmax(logit:List[float],beta:float=10):
    logit=np.array(logit)
    exp_logit=np.exp(beta*logit)
    prob=exp_logit/np.sum(exp_logit)
    return prob

if __name__=='__main__':
    projectPath=os.getcwd()
    dataPath=os.path.join(projectPath,'data')
    filename='sampled.csv'
    smiles=read_sample_smiles(dataPath,filename)
    print(smiles)

         
