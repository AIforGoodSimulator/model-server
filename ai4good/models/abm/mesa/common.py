from enum import Enum


class DiseaseStage(Enum):
    SUSCEPTIBLE = 0
    EXPOSED = 1
    PRESYMPTOMATIC = 2
    SYMPTOMATIC = 3
    ASYMPTOMATIC1 = 4
    ASYMPTOMATIC2 = 5
    MILD = 6
    SEVERE = 7
    RECOVERED = 8
    DECEASED = 9


class Route(Enum):
    HOUSEHOLD = 0
    TOILET = 1
    FOOD_LINE = 2
    WANDERING = 3