from dataclasses import dataclass
from enum import Enum
from pathlib import Path

# from typing_extensions import Literal


class ComponentEnum(Enum):
    TPSA1:str="tpsa1"
    TPSA3:str="tpsa3"
    ALERT:str="alerts"
    CENTER:str="center"
    GRAPH1:str="graph1"
    GRAPH2:str="graph2"
    HBA1:str="hba1"
    HBA2:str="hba2"
    HBA3:str="hba3"
    HBD1:str="hbd1"
    HBD3:str="hbd3"
    BOND:str="bond"
    RING1:str="ring1"
    RING3:str="ring3"
    SLOGP1:str="slogp1"
    SLOGP2:str="slogp2"
    SLOGP3:str="slogp3"
    MASS1:str="mass1"
    MASS4:str="mass4"
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
    TRAIN_SCRIPT:str="runs.sh"
    SAMPLE_SCRIPT:str="run_sample.sh"

    def __post_init__(self):
        self.MODEL_PATH=Path(self.RESULT_FOLDER,"Agent.ckpt")
        self.SAMPLE_PATH=Path(self.RESULT_FOLDER,"sampled.csv")
