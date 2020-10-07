from enum import Enum

# Side length of square shaped camp
# In tucker model, CAMP_SIZE=1.0. However, any value can be set here since all distance calculations are done
# relative to `CAMP_SIZE`
CAMP_SIZE = 100.0

# Disease states
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


# Agent's status
HOUSEHOLD = 0
TOILET = 1
FOOD_LINE = 2
WANDERING = 3
QUARANTINED = 4  # same as isolation
