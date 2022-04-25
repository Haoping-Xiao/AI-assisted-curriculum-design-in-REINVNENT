import os
from typing import List,Optional
from utils import read_component_config, write_curriculum_file, write_sample_file, write_run_train, write_run_sample
from datetime import datetime
import subprocess




def execute_curriculum(curriculum_name:List,agent:str,output_dir:Optional[str]=None,training_mode:Optional[bool]=True)->str:
  jobname = 'run_curriculum_{}'.format(curriculum_name[-1])
  jobid=datetime.now().strftime("%d-%m-%Y")
  dir_path = os.path.dirname(os.path.realpath(__file__))
  output_dir=os.path.join(output_dir,"{}".format(jobname)) if output_dir else os.path.join(dir_path,"../results/{}".format(jobname))

  if not os.path.exists(output_dir):
    os.makedirs(output_dir)

  if training_mode:
    config_filename="component_lib.json"
    config_path=os.path.join(dir_path,"../component_config")
    reinvent_dir= os.path.join(dir_path,"../../reinventcli")
    reinvent_env="/home/xiaoh2/.conda/envs/shared"
    config=read_component_config(config_path,config_filename)
    components_config=config["components"]
    components=[]
    for component in components_config:
      if component["name"] in curriculum_name:
        components.append(component)
        # write_curriculum_file(jobid, jobname, agent, reinvent_dir, output_dir, component) if training_mode else write_sample_file(jobid, jobname,  output_dir)
    write_curriculum_file(jobid, jobname, agent, reinvent_dir, output_dir, components)
    write_sample_file(jobid, jobname,  output_dir)
    # write_run_train(output_dir, reinvent_env, reinvent_dir,gpu=False) if training_mode else write_run_sample(output_dir, reinvent_env, reinvent_dir)
    write_run_train(output_dir, reinvent_env, reinvent_dir,gpu=False)
    write_run_sample(output_dir, reinvent_env, reinvent_dir) 

  
  command=['sbatch',output_dir+'/runs.sh'] if training_mode else ['sbatch',output_dir+'/run_sample.sh']
  subprocess.run(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)

  return output_dir

# if __name__=="__main__":

  # jobid=datetime.now().strftime("%d-%m-%Y")
  # curriculum_name="drd2 activity"
  # jobname = 'curriculum_{}'.format(curriculum_name)

  # training_mode=False # 'train' or 'sample'
  # config_filename="component_lib.json"
  

  # dir_path = os.path.dirname(os.path.realpath(__file__))
  # # agent in the first curriculum is the prior.
  # agent=os.path.join(dir_path,"../results/{}_{}".format("run_tpsa", "11-04-2022"),"results_2/Agent.ckpt")


  # config_path=os.path.join(dir_path,"../component_config")
  # reinvent_dir= os.path.join(dir_path,"../../reinventcli")
  # reinvent_env="/home/xiaoh2/.conda/envs/shared"
  # output_dir=os.path.join(dir_path,"../results/{}".format(jobname))
  
  
  

  # if not os.path.exists(output_dir):
  #   os.makedirs(output_dir)

  # config=read_component_config(config_path,config_filename)
  # components_config=config["components"]


  # for id,component in enumerate(components_config):
  #   if component["name"]==curriculum_name:
  #     write_curriculum_file(jobid, jobname, agent, reinvent_dir, output_dir, component, id) if training_mode else write_sample_file(jobid, jobname,  output_dir,  id)
  #     break
  # write_run_train(output_dir, reinvent_env, reinvent_dir,gpu=True) if training_mode else write_run_sample(output_dir, reinvent_env, reinvent_dir)

  
  # command=['sbatch',output_dir+'/runs.sh'] if training_mode else ['sbatch',output_dir+'/run_sample.sh']
  # res=subprocess.run(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)