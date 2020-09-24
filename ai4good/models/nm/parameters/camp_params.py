from scipy.stats import poisson
from ai4good.models.nm.utils import network_utils, stats_utils

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Camp parameters for Moria
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# Total number of people
n_pop = 18700

# Sample the population age, and parameter rates
sample_pop = stats_utils.sample_population(
    n_pop, "ai4good/models/nm/data/augmented_population.csv")

# Grid info for isoboxes
dims_isoboxes = (29, 28)  # 812

# Grid info for tents
dims_block1 = (20, 67)  # 1340
dims_block2 = (53, 15)  # 795
dims_block3 = (19, 28)  # 532

# Isoboxes
pop_isoboxes = 8100
pop_per_isobox = 10
n_isoboxes = dims_isoboxes[0]*dims_isoboxes[1]

# Tents
n_tents = 2650
pop_tents = 10600
pop_per_tent = 4

# Others
n_bathrooms = 144
n_ethnic_groups = 8

# We define neighboring structures within a range of 2 in the structure grid
proximity = 2

# Define the maximum population per structures (tents and isoboxes) drawn from a poisson distribution
max_pop_per_struct_isoboxes = list(
    poisson.rvs(mu=pop_per_isobox, size=n_isoboxes))
max_pop_per_struct_block1 = list(poisson.rvs(
    mu=pop_per_tent, size=dims_block1[0]*dims_block1[1]))
max_pop_per_struct_block2 = list(poisson.rvs(
    mu=pop_per_tent, size=dims_block2[0]*dims_block2[1]))
max_pop_per_struct_block3 = list(poisson.rvs(
    mu=pop_per_tent, size=dims_block3[0]*dims_block3[1]))

max_pop_per_struct = max_pop_per_struct_isoboxes\
    + max_pop_per_struct_block1\
    + max_pop_per_struct_block2\
    + max_pop_per_struct_block3

n_structs = len(max_pop_per_struct)

grid_isoboxes = network_utils.create_grid(
    dims_isoboxes[0], dims_isoboxes[1], 0)
grid_block1 = network_utils.create_grid(
    dims_block1[0], dims_block1[1], grid_isoboxes[-1][-1] + 1)
grid_block2 = network_utils.create_grid(
    dims_block2[0], dims_block2[1], grid_block1[-1][-1] + 1)
grid_block3 = network_utils.create_grid(
    dims_block3[0], dims_block3[1], grid_block2[-1][-1] + 1)

household_weight = 0.98  # Edge weight for connections within each structure
neighbor_weight = 0.017  # Edge weight for friendship connections
# The radius of nearby structures whose inhabitants a person has connections with
neighbor_proximity = 2
food_weight = 0.407  # Edge weight for connections in the food queue
