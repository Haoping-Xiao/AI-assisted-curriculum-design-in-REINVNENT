import os
from utils import read_component_config, write_train_file, write_sample_file, write_run_train, write_run_sample
from datetime import datetime
import subprocess


if __name__=="__main__":

  jobid=datetime.now().strftime("%d-%m-%Y")
  # jobid="v1"
  jobname = 'activity_qed_sa'
  training_mode=True # 'train' or 'sample'
  single_mode=False # 'single component' or 'multi components'
  config_filename="activity_qed_sa.json"


  dir_path = os.path.dirname(os.path.realpath(__file__))
  config_path=os.path.join(dir_path,"../component_config")
  reinvent_dir= os.path.join(dir_path,"../../reinventcli")
  reinvent_env="/home/xiaoh2/.conda/envs/shared"
  output_dir=os.path.join(dir_path,"../results/{}_{}".format(jobname, jobid))

  
  if not os.path.exists(output_dir):
    os.makedirs(output_dir)

  config=read_component_config(os.path.join(config_path,config_filename))
  components_config=config["components"]

  if training_mode:
    if single_mode:
      for id,component in enumerate(components_config):
        write_train_file(jobid, jobname, reinvent_dir, output_dir, component, id,gpu=True)
      write_sample_file(jobid, jobname,  output_dir,  id)
      write_run_train(output_dir, reinvent_env, reinvent_dir, len(components_config)-1,gpu=True)
      write_run_sample(output_dir, reinvent_env, reinvent_dir, len(components_config)-1)
    else:
      write_train_file(jobid, jobname, reinvent_dir, output_dir, components_config,gpu=True)
      write_sample_file(jobid, jobname,  output_dir)
      write_run_train(output_dir, reinvent_env, reinvent_dir,gpu=True)
      write_run_sample(output_dir, reinvent_env, reinvent_dir)
  
  command=['sbatch',output_dir+'/runs.sh'] if training_mode else ['sbatch',output_dir+'/run_sample.sh']
  res=subprocess.run(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)