# from dataclasses import dataclass,asdict
from enum import Enum

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
    ACT:str="activity"
    SA:str="sa"
    QED:str="qed"


