from dataclasses import dataclass
from enum import Enum
from pathlib import Path

# from typing_extensions import Literal


class ComponentEnum(Enum):
    TPSA3:str="tpsa3"
    ALERT:str="alerts"
    CENTER:str="center"
    GRAPH1:str="graph1"
    HBA2:str="hba2"
    HBD1:str="hbd1"
    HBD4:str="hbd4"
    BOND:str="bond"
    RING3:str="ring3"
    SLOGP2:str="slogp2"
    MASS3:str="mass3"
    SIM0:str="sim0"
    SIM1:str="sim1"
    SIM2:str="sim2"
    SIM3:str="sim3"
    SIM4:str="sim4"
    SIM5:str="sim5"
    SIM6:str="sim6"
    SIM7:str="sim7"
    SIM8:str="sim8"
    SIM9:str="sim9"
    SUB:str="sub"
    ACT:str="activity"
    SA:str="sa"
    QED:str="qed"
    END:None=None

class HypothesisEnum(Enum):
    ACT:str="activity"
    SA:str="sa"
    QED:str="qed"

@dataclass()
class ProjectConfig():
    OUT_DIR:Path="/scratch/work/xiaoh2/Thesis/results/curriculum"
    PRIOR_DIR:Path="/scratch/work/xiaoh2/Thesis/models/augmented.prior"
    CONFIG_PATH:Path="/scratch/work/xiaoh2/Thesis/component_config/component_lib.json"
    STAN_PATH:Path="/scratch/work/xiaoh2/Thesis/scripts/user_model.stan"
    RESULT_FOLDER:str="results_0"
    TRAIN_ENDING_MSG:str="Finish training"
    SAMPLE_ENDING_MSG:str="Finish sampling"
    TRAIN_LOG:str="slurm/train_0.out"
    SAMPLE_LOG:str="slurm/sample_0.out"
    TRAIN_SCRIPT:str="runs.sh"
    SAMPLE_SCRIPT:str="run_sample.sh"
    ESTIMATE_PRODUCTION_EPOCH:int=200
    PRODUCTION_EPOCH:int=300
    def __post_init__(self):
        self.MODEL_PATH=Path(self.RESULT_FOLDER,"Agent.ckpt")
        self.SAMPLE_PATH=Path(self.RESULT_FOLDER,"sampled.csv")
