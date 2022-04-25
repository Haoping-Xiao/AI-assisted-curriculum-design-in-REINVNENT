from typing import List
from run_curriculum import execute_curriculum
import os
from dataclasses import dataclass
from scorer import get_scorer
import numpy as np


def first_curriculum(curriculum_name:List)->str:
  agent="/scratch/work/xiaoh2/Thesis/scripts/../../reinventcli/data/augmented.prior"
  output_dir=execute_curriculum(curriculum_name,agent)
  return output_dir


def next_curriculum(last_output_dir:str,curriculum_name:List)->str:
  agent=os.path.join(last_output_dir,"results_0/Agent.ckpt")
  output_dir=execute_curriculum(curriculum_name,agent,last_output_dir)
  return output_dir


@dataclass
class Human():
  

  def get_component_statistic(self):
    self.compoents_data={}
    self.compoents_data["tpsa_1"]={"activity":0.12,"qed":0.55,"sa":0.93}
    self.compoents_data["tpsa_3"]={"activity":0.36,"qed":0.71,"sa":0.90}
    self.compoents_data["Custom_alerts"]={"activity":0.13,"qed":0.59,"sa":0.95}
    self.compoents_data["number_of_stereo_centers_1"]={"activity":0.15,"qed":0.59,"sa":0.93}
    self.compoents_data["graph_length_1"]={"activity":0.13,"qed":0.57,"sa":0.93}
    self.compoents_data["graph_length_2"]={"activity":0.10,"qed":0.54,"sa":0.93}
    self.compoents_data["num_hba_lipinski_1"]={"activity":0.13,"qed":0.61,"sa":0.93}
    self.compoents_data["num_hba_lipinski_2"]={"activity":0.14,"qed":0.60,"sa":0.93}
    self.compoents_data["num_hba_lipinski_3"]={"activity":0.10,"qed":0.57,"sa":0.94}
    self.compoents_data["num_hbd_lipinski_1"]={"activity":0.15,"qed":0.59,"sa":0.93}
    self.compoents_data["num_hbd_lipinski_3"]={"activity":0.12,"qed":0.57,"sa":0.94}
    self.compoents_data["num_rotatable_bonds_1"]={"activity":0.14,"qed":0.64,"sa":0.92}
    self.compoents_data["num_rings_1"]={"activity":0.12,"qed":0.64,"sa":0.94}
    self.compoents_data["num_rings_3"]={"activity":0.13,"qed":0.49,"sa":0.92}
    self.compoents_data["slogp_1"]={"activity":0.13,"qed":0.68,"sa":0.93}
    self.compoents_data["slogp_2"]={"activity":0.14,"qed":0.62,"sa":0.94}
    self.compoents_data["Molecular_mass_4"]={"activity":0.12,"qed":0.65,"sa":0.94}
    self.compoents_data["Molecular_mass_1"]={"activity":0.10,"qed":0.59,"sa":0.95}

  def get_component_list(self):
    return self.compoents_data.keys()


  def get_values(self,smiles):
    modes=['qed','sa','activity']
    weights=[0.3,0.1,0.6]
    scorers=[get_scorer(mode) for mode in modes]
    mean_scores=[np.mean(scorer.get_score(smiles)) for scorer in scorers]

  def policy(self):
    """
      step1: biased components
      step2: estimated Q values
      step3: best component out of biased components
    """
    



if __name__=="__main__":
  # output_dir=first_curriculum(["drd2_activity_1"])
  # print(output_dir)
  # next_curriculum("/scratch/work/xiaoh2/Thesis/scripts/../results/run_curriculum_drd2_activity_1",["drd2_activity_1","QED"])
  next_curriculum("/scratch/work/xiaoh2/Thesis/results/run_curriculum_drd2_activity_1/run_curriculum_QED",["drd2_activity_1","QED","sa"])
  # next_curriculum("/scratch/work/xiaoh2/Thesis/results/curriculum_drd2_activity_1/curriculum_QED","sa")


# data["sa"]=  {"tpsa_1":0.93,
    #               "tpsa_3":0.90,
    #               "Custom_alerts":0.95,
    #               "number_of_stereo_centers_1":0.93,
    #               "graph_length_1":0.93,
    #               "graph_length_2":0.93,
    #               "num_hba_lipinski_1":0.93,
    #               "num_hba_lipinski_2":0.93,
    #               "num_hba_lipinski_3":0.94,
    #               "num_hbd_lipinski_1":0.93,
    #               "num_hbd_lipinski_3":0.94,
    #               "num_rotatable_bonds_1":0.92,
    #               "num_rings_1":0.94,
    #               "num_rings_3":0.92,
    #               "slogp_1":0.93,
    #               "slogp_2":0.94,
    #               "Molecular_mass_4":0.94,
    #               "Molecular_mass_1":0.95,
    #               }
    # data["qed"]= {"tpsa_1":0.55,
    #               "tpsa_3":0.71,
    #               "Custom_alerts":0.59,
    #               "number_of_stereo_centers_1":0.59,
    #               "graph_length_1":0.57,
    #               "graph_length_2":0.54,
    #               "num_hba_lipinski_1":0.61,
    #               "num_hba_lipinski_2":0.60,
    #               "num_hba_lipinski_3":0.57,
    #               "num_hbd_lipinski_1":0.59,
    #               "num_hbd_lipinski_3":0.57,
    #               "num_rotatable_bonds_1":0.64,
    #               "num_rings_1":0.64,
    #               "num_rings_3":0.49,
    #               "slogp_1":0.68,
    #               "slogp_2":0.62,
    #               "Molecular_mass_4":0.65,
    #               "Molecular_mass_1":0.59,
    #               }
    # data["activity"]={"tpsa_1":0.12,
    #                   "tpsa_3":0.36,
    #                   "Custom_alerts":0.13,
    #                   "number_of_stereo_centers_1":0.15,
    #                   "graph_length_1":0.13,
    #                   "graph_length_2":0.10,
    #                   "num_hba_lipinski_1":0.13,
    #                   "num_hba_lipinski_2":0.14,
    #                   "num_hba_lipinski_3":0.10,
    #                   "num_hbd_lipinski_1":0.15,
    #                   "num_hbd_lipinski_3":0.12,
    #                   "num_rotatable_bonds_1":0.14,
    #                   "num_rings_1":0.12,
    #                   "num_rings_3":0.13,
    #                   "slogp_1":0.13,
    #                   "slogp_2":0.14,
    #                   "Molecular_mass_4":0.12,
    #                   "Molecular_mass_1":0.10,
    #                   }