import pandas as pd
import json
import os


def check_file_exist(path,filename):
    assert os.path.exists(path), '{} is not exist'.format(path)
    assert os.path.exists(os.path.join(path,filename)), '{} is not exist'.format(filename)


def read_sample_smiles(path,filename):
    check_file_exist(path,filename)
    smiles=pd.read_csv(os.path.join(path,filename),header=None).squeeze()
    return smiles

def read_scaffold_smiles(path,filename):
    check_file_exist(path,filename)
    df=pd.read_csv(os.path.join(path,filename))
    return df['SMILES']

def read_component_config(path,filename):
    check_file_exist(path,filename)
    with open(os.path.join(path,filename),'r') as config_file:
        component_config=json.load(config_file)
    return component_config

def write_curriculum_file(jobid, jobname, agent, reinvent_dir, output_dir, components, id=0):
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
    "parallel": True,
    "parameters": [
      components
    ] 
  } if not isinstance(components,list) else {
    "name": "custom_sum",
    "parallel": True,
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
      "n_steps": 300,
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


def write_train_file(jobid, jobname, reinvent_dir, output_dir, components, id=0):
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
    "parallel": True,
    "parameters": [
      components
    ] 
  } if not isinstance(components,list) else {
    "name": "custom_sum",
    "parallel": True,
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
      "n_steps": 300,
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
        "num_smiles": 1024,
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
      f.write("#SBATCH --cpus-per-task=4 --gres=gpu:1\n") if gpu else f.write("#SBATCH --cpus-per-task=4 \n")
      f.write('#SBATCH --time=02:00:00 --mem-per-cpu=4000\n')
      f.write('#SBATCH -o {}/slurm/out_%a.out\n'.format(output_dir))
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
      f.write('#SBATCH --time=02:00:00 --mem-per-cpu=4000\n')
      f.write('#SBATCH -o {}/slurm/out_%a.out\n'.format(output_dir))
      f.write('#SBATCH --array=0-{}\n'.format(n_component_configs))
      f.write('#SBATCH -p short\n')
      f.write('\n')
      f.write('module purge\n')
      f.write('module load anaconda\n')
      f.write('source activate {}\n'.format(reinvent_env))
      f.write('\n')
      f.write('conf_filename="{}/sample_config_$SLURM_ARRAY_TASK_ID.json"\n'.format(output_dir))
      f.write('srun python {}/input.py $conf_filename\n'.format(reinvent_dir))


if __name__=='__main__':
    projectPath=os.getcwd()
    dataPath=os.path.join(projectPath,'data')
    filename='sampled.csv'
    smiles=read_sample_smiles(dataPath,filename)
    print(smiles)

         
