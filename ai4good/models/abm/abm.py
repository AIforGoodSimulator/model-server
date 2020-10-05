import math
import random
import numpy as np
import pandas as pd
from numba import njit
from scipy import optimize
from scipy.special import factorial

from ai4good.utils.path_utils import get_am_aug_pop
from ai4good.models.abm.ops.spatial_ops import distance_matrix, assign_block


def read_age_gender(num_ppl):
    """
    Read file containing people's age and gender.

    Parameters
    ----------
        num_ppl : Number of records to return
    
    Returns
    -------
        out : Numpy array of size (`num_ppl`, 2) containing rows: (age, gender)

    """
    # Data frame. V1 = age, V2 is sex (1 = male?, 0  = female?).
    age_and_gender = pd.read_csv(get_am_aug_pop())
    age_and_gender = age_and_gender.loc[:, ~age_and_gender.columns.str.contains('^Unnamed')]
    age_and_gender = age_and_gender.values
    age_and_gender = age_and_gender[np.random.randint(age_and_gender.shape[0], size=num_ppl)]
    return age_and_gender


def create_household_column(num_hh_type1, num_ppl_type1, num_hh_type2, num_ppl_type2):
    """
    Creates an initial household distribution for each individual.

    Parameters
    ----------
        num_hh_type1  : Number of iso-boxes (hb)
        num_ppl_type1 : Number of people in iso-boxes (Nb)
        num_hh_type2  : Number of tents (ht)
        num_ppl_type2 : Number of people in tents (Nt)

    Returns
    -------
        out : An array of size (Nb+Nt,) containing household distribution.
            Each element of `out` will be in range [0, hb+ht-1]

    """

    # Random allotments of people to households
    # `ppl_hh_index_draw` will be of size [Nb + Nt] with each element ∈ [1, hb + ht]
    ppl_hh_index_draw = np.concatenate([
        # values ∈ [1, hb] for iso-boxes
        np.ceil(num_hh_type1 * np.random.uniform(0, 1, num_ppl_type1)),
        # values ∈ [hb + 1, hb + ht] for tents
        num_hh_type1 + np.ceil(num_hh_type2 * np.random.uniform(0, 1, num_ppl_type2))
    ])

    # get unique allotments ids
    hh_unique, ppl_to_hh_index = np.unique(ppl_hh_index_draw, return_inverse=True)
    
    # ui - indices from the unique sorted array that would reconstruct rN
    assert hh_unique[ppl_to_hh_index].all() == ppl_hh_index_draw.all()
    
    # return indices to indicate household
    return np.sort(ppl_to_hh_index)

    # NOTE: This function can be replaced with just > (np.random.sample(Nb+Nt) * (hb+ht)).astype(np.int32)


def create_diseasestate_column(num_ppl, seed=1):
    """
    Creates a binary (0/1) disease state column for the population with given initial infections.

    Parameters
    ----------
        num_ppl : Number of people in the population (N)
        seed    : Total number of initial infections (I)
    
    Returns
    -------
        out : An binary array of size (N,) with exactly I elements = 1 (i.e. exactly I people with initial infection)

    """
    # initialize all 0s array
    initial_diseasestate = np.zeros(num_ppl)
    # infect exactly `seed` number of elements
    initial_diseasestate[np.random.choice(num_ppl, seed)] = 1
    # return infection state
    return initial_diseasestate


def create_daystosymptoms_column(num_ppl):
    # The time from exposure until symptoms appear (i.e., the incubation period) is
    # drawn from a Weibull distribution with a mean of 6.4 days and a standard deviation of 2.3
    # days (Backer et al. 2020)
    k = (2.3/6.4)**(-1.086)
    L = 6.4 / (math.gamma(1 + 1/k))
    return np.array([random.weibullvariate(L, k) for ppl in np.arange(num_ppl)])


def create_daycount_column(num_ppl):
    return np.zeros(num_ppl)


def create_asymp_column(num_ppl, asymp_rate, age_column=None, num_ppl_chro=300):
    """
    Create asymptomatic flag for population. People with chronic diseases won't be asymptomatically infected.

    Parameters
    ----------
        num_ppl      : Total number of people
        asymp_rate   : Permanently asymptomatic cases
        age_column   : Array containing age of people [optional]
        num_ppl_chro : Total number of people with chronic disease (pre-exisitng medical conditions)
    
    Returns
    -------
        out : A boolean array indicating if person is asymptomatically infected or not

    Notes
    -----
        Asymptomatic people are those who are infected BUT don't show any symptoms. Such people can infect others
        without showing any symptoms on their own
    """
    
    if age_column is not None:
        raise NotImplementedError("Age has not been considered for asymptomatic infection calculations yet!")
    else:
        # TODO: age < 16 should be asymptomatic as per tucker model
        return np.random.uniform(0, 1, num_ppl) < (asymp_rate * (num_ppl/(num_ppl-num_ppl_chro)))


def create_age_column(age_data):
    return age_data


def create_gender_column(gender_data):
    return gender_data


def create_chronic_column(num_ppl, age_column, num_ppl_chro=300):
    myfunction = lambda x: np.absolute(num_ppl_chro-np.sum((1+np.exp(-(x-11.69+.2191*age_column-0.001461*age_column**2))**(-1))))-num_ppl
    xopt = optimize.fsolve(myfunction, x0=[2])
    rchron = (1+np.exp(-(xopt-11.69+.2191*age_column-0.001461*age_column**2)))**(-1)
    chroncases = (np.random.uniform(np.min(rchron), 1, num_ppl) < rchron)
    return chroncases


def adjust_asymp_with_chronic(asymp_column, chronic_column):
    """
    Remove asymptomatic flag for people with chronic disease

    Parameters
    ----------
        asymp_column   : A boolean array storing people with asymptomatic infection
        chronic_column : A boolean array storing people with chronic disease
    
    Returns
    -------
        out : Updated asymptomatic array
    """
    new_asymp_column = asymp_column.copy()
    new_asymp_column[chronic_column == 1] = 0
    return new_asymp_column


def create_wanderer_column(gender, age):
    """
    Male of age greater than 10 are the wanderers in the camp

    Parameters
    ----------
        gender : Array of 0/1 values indicating if person is female/male
        age    : Array of float age values
    
    Returns
    -------
        out : Boolean array indicating if a person is wanderer (True) or not (False)

    """
    return np.logical_and([gender == 1], [age >= 10]).transpose()


def form_population_matrix(N, hb, Nb, ht, Nt, pac, age_and_gender):
    """
    Create population matrix based on parameters.

    Parameters
    ----------
        N              : Total population
        hb             : Number of iso-boxes
        Nb             : Number of people in iso-boxes
        ht             : Number of tents
        Nt             : Number of people in tents
        pac            : Proportion of permanently asymptomatic cases (Mizumoto et al 2020 Eurosurveillance)
        age_and_gender : (N, 2) sized array containing age and gender of population
    
    Returns
    -------
        out : 2D array containing population info. Each row will contain following columns:
           0: Initial household distribution for each individual [0, hb+ht)
           1: Disease state for each individual
           2: Days to symptoms
           3: Day count
           4: Binary state indicating if individual is asymptomatic {0, 1}
           5: Age of the individual
           6: Gender of the individual {0: female, 1: male}
           7: Binary state indicating if individual suffers from chronic disease {0, 1}
           8: Binary state indicating if individual is wanderer or not {0, 1}

    """
    
    # 1 allocated to iso-boxes and tents
    household_column = create_household_column(hb, Nb, ht, Nt)
    # 2 allocate an infector
    disease_column = create_diseasestate_column(N)
    # 3 weibull distribution
    dsymptom_column = create_daystosymptoms_column(N)
    # 4 np.zeros(num_ppl)
    daycount_column = create_daycount_column(N)
    # 5 hard coded number of chronics
    asymp_column = create_asymp_column(N, pac)
    # 6 age
    age_column = create_age_column(age_and_gender[:,0])
    # 7 gender
    gender_column = create_gender_column(age_and_gender[:,1])
    # 8 formula based on age
    chronic_column = create_chronic_column(N, age_column)
    # 5 new_asymp_column[chronic_column==1]=0
    new_asymp_column = adjust_asymp_with_chronic(asymp_column, chronic_column)
    # 9
    wanderer_column = create_wanderer_column(gender_column, age_column)

    pop_matrix = np.column_stack((household_column, disease_column, dsymptom_column,
                                daycount_column, new_asymp_column, age_column,
                                gender_column, chronic_column, wanderer_column))
    
    assert pop_matrix.shape == (N, 9)
    return pop_matrix


def place_households(ppl_to_hh_index, prop_type1, num_hh_type1):
    """
    Create population matrix based on parameters.

    Parameters
    ----------
        ppl_to_hh_index : Household distribution within the population
        prop_type1      : Proportion of area covered by iso-boxes (iba=0.5)
        num_hh_type1    : Number of iso-boxes (hb)
    
    Returns
    -------
        out : 2D array containing location of households (iso-boxes + tents)

    """
    
    # count of people per household index
    pph = np.bincount(ppl_to_hh_index)
    
    # total number of households
    maxhh = pph.size

    # Assign x and y coordinates to isoboxes (there are hb total isoboxes). 
    hhloc1 = 0.5 * (1 - np.sqrt(prop_type1)) + np.sqrt(prop_type1) * np.random.uniform(0, 1, (int(num_hh_type1), 2) )
    
    # Repeat for tents.
    hhloc2 = np.random.uniform(0, 1, (int(maxhh-num_hh_type1), 2) ) # Note: Nb-1 and N-1 to account for zero-indexing.

    ######## assert (hhloc1.shape[0]+hhloc2.shape[0] == maxhh)
    
    # Randomly move tents to the edges of the camp. Assign randomly a side to each of the household.
    hhloc2w = np.random.randint(1, 5, size=int(maxhh-num_hh_type1))
    assert len(hhloc2w) == hhloc2.shape[0]

    # This block moves some tents to the right edge.
    shift = 0.5*(1-np.sqrt(prop_type1)) #this is the width of the gap assuming isobox occupies a square in the middle with half the area 
    #(interesting parameter to tune)
    hhloc2[np.where(hhloc2w==1),0] = shift*hhloc2[np.where(hhloc2w == 1),0]+(1-shift)
    hhloc2[np.where(hhloc2w==1),1] = (1-shift)*hhloc2[np.where(hhloc2w == 1),1] #shrink towards bottom right
    # This block moves some tents to the upper edge.
    hhloc2[np.where(hhloc2w==2),0] = hhloc2[np.where(hhloc2w == 2),0]*(1-shift)+shift #push towards top right
    hhloc2[np.where(hhloc2w==2),1] = shift*hhloc2[np.where(hhloc2w == 2),1]+(1-shift)
    # This block moves some tents to the left edge.
    hhloc2[np.where(hhloc2w==3),0] = shift*hhloc2[np.where(hhloc2w == 3),0]
    hhloc2[np.where(hhloc2w==3),1] = hhloc2[np.where(hhloc2w == 3),1]*(1-shift)+shift #push it towards top left
    # This block moves some tents to the bottom edge.
    hhloc2[np.where(hhloc2w==4),0] = (1-shift)*hhloc2[np.where(hhloc2w == 4),0] #push it towards bottom left
    hhloc2[np.where(hhloc2w==4),1] = shift*hhloc2[np.where(hhloc2w == 4),1] 
    
    hhloc = np.vstack((hhloc1, hhloc2))
    assert hhloc.shape[0] == maxhh
    return hhloc


def position_toilet(hhloc, nx=12, ny=12):
    """
    Position toilets in the camp.
    From the doc: "Toilets are placed at the centres of the squares that form a 12 x 12 grid covering the camp"
    Parameters
    ----------
        hhloc : A (hb+ht, 2) array containing position of isoboxes and tents
        nx    : Width of hypothetical grid used for toilet placement
        ny    : Height of hypothetical grid used for toilet placement
    """

    # Grid dimensions for toilet blocks
    tblocks = np.array([nx, ny])

    # assign toilets to households
    return assign_block(hhloc, tblocks)


def position_foodline(hhloc, nx=1, ny=1):
    """
    Position foodline in the camp.
    From the doc: "The camp has one food line. The position of the food line is not explicitly modelled."
    Parameters
    ----------
        hhloc : A (hb+ht, 2) array containing position of isoboxes and tents
        nx    : Width of hypothetical grid used for foodine placement
        ny    : Height of hypothetical grid used for foodline placement
    """

    # Grid dimensions for foodline blocks
    fblocks = np.array([nx, ny])

    # assign foodline to households
    return assign_block(hhloc, fblocks)

    
def create_ethnic_groups(hh_loc, int_eth):
    """

    Parameters
    ----------
        hh_loc : A (hb+ht, 2) array containing position of households (iso-boxes and tents)
        int_eth: A scalar representing relative strength interactions between ethnicities (external parameter)

    Returns
    -------
        eth_cor: ethnicities correlation

    Notes
    -----
    "In Moria, the homes of people with the same ethnic or national background are spatially clustered,
    and people interact more frequently with others from the same background as themselves.   ...
    To simulate ethnicities or nationalities in our camp, we assigned each household to one of eight (?) “backgrounds”
    in proportion to the self-reported countries of origin of people.. For each of the eight simulated backgrounds,
    we randomly selected one tent or isobox to be he seed for the cluster. We assigned the x nearest unassigned
    households to that background, where x is the number of households with that background."

    """
    
    # TODO: should be parameterized
    Afghan = 7919 ; Cameroon = 149 ; Congo = 706 ;Iran = 107 ;Iraq = 83 ; Somalia = 442 ; Syria = 729
    g = np.array([Afghan, Cameroon, Congo, Iran, Iraq, Somalia, Syria])  
    total_pop = sum(g)
    hh_size = hh_loc.shape[0]

    # Number of households per group
    g_hh = np.round(hh_size*g/total_pop)
    np.random.shuffle(g_hh)

    # Unassigned households. First column is the index of hh
    hh_index = np.arange(0, hh_size)
    hh_unassigned = np.column_stack((hh_index, hh_loc))

    # to store which ethnic group is allocated to which household
    hh_eth = np.zeros((hh_size, 1))

    for i, g in enumerate(g_hh):  # iterate for each ethnic group

        # Chose an unassigned household as the group (cluster) center.
        g_center = hh_unassigned[np.random.randint(hh_unassigned.shape[0]), 1:]

        # Squared distance to cluster centre.
        dfromc = np.sum((hh_unassigned[:, 1:]-np.tile(g_center, (hh_unassigned.shape[0], 1)))**2, 1)

        # Get the indices of the `g` closest households
        cloind = np.argsort(dfromc)[0: int(g)]

        # Assign i-th ethnic group to those households
        hh_eth[hh_unassigned[cloind, 0].astype(int)] = i

        # Remove those households (remove the i-th cluster/ethnic group)
        hh_unassigned = np.delete(hh_unassigned, cloind, axis=0)

    # find out which households have same ethnic groups
    eth_match = (np.tile(hh_eth, (1, hh_size)) == np.tile(hh_eth, (1, hh_size)).T)

    # scale down the connection for people of different background
    # TODO: is scale-down happening here? Since `eth_match` will be either 0 or 1, eth_cor will be either `int_eth` or 1
    eth_cor = eth_match + int_eth * (1 - eth_match)

    return eth_cor


def interaction_neighbours(household_coordinates, r, R, lrtol, ethnic_groups):
    """
    Parameters
    ----------
        household_coordinates : A 2D array containing co-ordinates of the households
        r                     : Smaller movement radius [0-1]
        R                     : Larger movement radius [0-1]
        lrtol                 : scale value for interactions within household
        ethnic_groups         : Scale values for interaction between ethnic groups (output of `create_ethnic_groups`())
    
    Returns
    -------
        out : Local interaction space
    """

    # create distance matrix for distance in between households
    household_distance_matrix = distance_matrix(household_coordinates)

    # the case where person with movement_radius_small is interacting with person with same radius
    relative_encounter_small = relative_encounter_rate(household_distance_matrix, r, r)

    # the case where person with movement_radius_large is interacting with person with same radius
    relative_encounter_large = relative_encounter_rate(household_distance_matrix, R, R)

    # the case where person with movement_radius_large is interacting with person with movement_radius_small
    relative_encounter_small_large = relative_encounter_rate(household_distance_matrix, r, R)

    lis = np.multiply(
        math.pi * lrtol ** 2 * np.dstack((
            relative_encounter_small,
            relative_encounter_small_large,
            relative_encounter_large
        )),
        np.dstack((ethnic_groups, ethnic_groups, ethnic_groups))
    )
    return lis


def interaction_neighbours_fast(hhloc, lr1, lr2, lrtol, ethcor):
    # use the formula from https://mathworld.wolfram.com/Circle-CircleIntersection.html
    # create distance matrix for distance in between households
    hhdm = distance_matrix(hhloc)
    # the case where lr1 is interacting with lr1
    area_overlap11=2*(lr1**2*np.arccos(np.clip(0.5*hhdm/lr1,a_min=None,a_max=1))-np.nan_to_num(hhdm/2*np.sqrt(lr1**2-hhdm**2/4)))
    relative_encounter11=area_overlap11/(math.pi**2*lr1**4)
    # the case where lr2 is interacting with lr2
    area_overlap22=2*(lr2**2*np.arccos(np.clip(hhdm/(2*lr2),a_min=None,a_max=1))-np.nan_to_num(hhdm/2*np.sqrt(lr2**2-hhdm**2/4)))
    relative_encounter22=area_overlap22/(math.pi**2*lr2**4)
    # the case where lr1 is interacting with lr2
    area_overlap12=np.nan_to_num((lr1**2*np.arccos(np.clip((hhdm**2+lr1**2-lr2**2)/(2*hhdm*lr1),a_min=None,a_max=1)))
    +(lr2**2*np.arccos(np.clip((hhdm**2+lr2**2-lr1**2)/(2*hhdm*lr2),a_min=None,a_max=1)))
    -0.5*np.sqrt((-hhdm+lr1+lr2)*(hhdm+lr1-lr2)*(hhdm-lr1+lr2)*(hhdm+lr1+lr2)))
    relative_encounter12=area_overlap12/(math.pi**2*lr2**2*lr1**2)
    lis = np.multiply(math.pi*lrtol**2*np.dstack((relative_encounter11,relative_encounter12,relative_encounter22)),np.dstack((ethcor,ethcor,ethcor)))
    return lis


def epidemic_finish(states, iteration):
    """
    Finish the simulation when no person is in any state other than recovered or susceptible
    """
    return np.sum(states) == 0 and iteration > 10


def disease_state_update(pop_matrix, mild_rec, sev_rec, pick_sick, thosp, quarantined=False):
    """
    Disease progress from one state to another among susceptible, exposed, presymptomatic, symptomatic, mild, severe
    and recovered for Quarantine and Normal situation.

    Parameters
    ----------
        pop_matrix: Population matrix (created in `form_population_matrix`)
        mild_rec: Probability that person's disease state will change from mild->recovered
        sev_rec: Probability that person's disease state will change from severe->recovered
        pick_sick: Health state
        thosp: Total number of hospitalized individuals
        quarantined: Boolean indicating if quarantined population is considered or not-quarantined population

    Returns
    -------
        pop_matrix: Updated population matrix
        thosp: Updated number of hospitalized people

    Notes
    -----
        abc_to_xyz_ind = abc state to xyz state's indices in pop_matrix

        Columns of pop_matrix (relevant to this method):
        {1: disease state, 2: days to symptomatic, 3: day count, 5: age, 7: chronic flag}

        Possible values of disease state (pop_matrix[:, 1]):
        {
            0: susceptible, 1: exposed, 2: presymptomatic, 3: symptomatic, 4: mild, 5: severe, 6: recovered,
            7: qua_susceptible, 8: qua_exposed, 9: qua_presymptomatic, 10: qua_symptomatic, 11: qua_mild,
            12: qua_severe, 13: qua_recovered
        }

    """

    # get susceptible index value. Refer values of pop_matrix[:, 1] for more details
    qua_add = 0  # for normal situation
    if quarantined:
        qua_add = 7  # for quarantined situation

    # Move exposed to presymptomatic
    exposed_to_presym_ind = np.logical_and(
        pop_matrix[:, 1] == (1 + qua_add),  # returns exposed/qua_exposed people
        pop_matrix[:, 3] >= np.floor(0.5 * pop_matrix[:, 2])  # returns people who passed half of incubation period
    )
    # update exposed->presymptomatic state
    pop_matrix[exposed_to_presym_ind, 1] = 2 + qua_add

    # Move presymptomatic to symptomatic but not yet severe.
    presymp_to_symp_ind = np.logical_and(
        pop_matrix[:, 1] == (2 + qua_add),  # returns presymptomatic/qua_presymptomatic people
        pop_matrix[:, 3] >= pop_matrix[:, 2],  # returns people for which incubation period is over
        # TODO: verify (Added by Ankit)
        pop_matrix[:, 4] == 0  # returns non-asymptomatic people. Asymptomatic people don't become symptomatic
    )
    # update presymptomatic->symptomatic state
    pop_matrix[presymp_to_symp_ind, 1] = 3 + qua_add
    # NOTE: Reset the day count when incubation period is over
    pop_matrix[presymp_to_symp_ind, 3] = 0

    # Move individuals with 6 days of symptoms to mild.
    symp_to_mild_ind = np.logical_and(
        pop_matrix[:, 1] == (3 + qua_add),  # returns symptomatic/qua_symptomatic people
        pop_matrix[:, 3] == 6  # people on their 6th day after incubation period was over
    )
    pop_matrix[symp_to_mild_ind, 1] = 4 + qua_add

    # Move people with mild symptoms to recovered state
    mild_to_recovered_ind = np.logical_and(
        pop_matrix[:, 1] == (4 + qua_add),  # returns mild/qua_mild people
        mild_rec  # returns True if mild->recovered is valid based on probabilities defined in Lui et al. 2020
    )
    # update mild->recovered state
    pop_matrix[mild_to_recovered_ind, 1] = 6 + qua_add

    # Move people with severe symptoms to recovered state
    severe_to_recovered_ind = np.logical_and(
        pop_matrix[:, 1] == (5+qua_add),  # returns severe/qua_severe people
        sev_rec  # returns True if severe->recovered is valid based on probabilities defined in Cai et al.
    )
    # update severe->recovered state
    pop_matrix[severe_to_recovered_ind, 1] = 6 + qua_add

    # symptomatic to the “mild” or “severe”
    # Verity et al. hospitalisation.
    asp = np.array([0, .000408, .0104, .0343, .0425, .0816, .118, .166, .184])
    # Verity et al. corrected for Tuite
    aspc = np.array([.0101, .0209, .0410, .0642, .0721, .2173, .2483, .6921, .6987])

    age_bucket = 9  # age ranges from (0-10) to (90+)
    for buc in range(age_bucket):

        # Assign individuals with mild symptoms for six days, sick, between 10*sci and 10*sci+1 years old to severe
        # and count as hospitalized.

        severe_ind = np.logical_and.reduce((
            pop_matrix[:, 1] == 4 + qua_add,  # returns mild/qua_mild people
            pop_matrix[:, 3] == 6,  # 6th day after incubation period ??
            pick_sick < asp[buc],  # Verity and colleagues data (low-risk)
            pop_matrix[:, 5] >= 10 * buc,  # people age lower bound
            pop_matrix[:, 5] < (10 * buc + 10),  # people age upper bound
            pop_matrix[:, 7] == 0  # people not having chronic disease
        ))
        # consider new severe people as hospitalized
        thosp += np.sum(severe_ind)
        # update mild->severe state
        pop_matrix[severe_ind, 1] = 5 + qua_add

        # Move individuals with Chronic diseases with 6 days of mild to severe.
        severe_chronic_ind = np.logical_and.reduce((
            pop_matrix[:, 1] == 4 + qua_add,  # returns mild/qua_mild people
            pop_matrix[:, 3] == 6,  # 6th day after incubation period ??
            pick_sick < aspc[buc],  # Tuite and colleagues data (high-risk)
            pop_matrix[:, 5] >= (10 * buc),  # people age lower bound
            pop_matrix[:, 5] < (10 * buc + 10),  # people age upper bound
            pop_matrix[:, 7] == 1  # person having chronic disease
        ))
        # consider new severe people as hospitalized
        thosp += np.sum(severe_chronic_ind)
        # update mild->severe state
        pop_matrix[severe_chronic_ind, 1] = 5 + qua_add
                           
    return pop_matrix, thosp


@njit
def accumarray(subs, val):
    """Construct Array with accumulation. https://www.mathworks.com/help/matlab/ref/accumarray.html"""
    unq = np.unique(subs)
    n = subs.shape[0]
    out = []
    for h in unq:
        h_sum = 0
        for i in range(n):
            if subs[i] == h:
                h_sum = h_sum + val[i]
        out.append(h_sum)

    return np.array(out)

    # return np.array([np.sum(val[np.where(subs == i)]) for i in np.unique(subs)])


def identify_contagious_active(pop_matrix):
    """

    Parameters
    ----------
        pop_matrix: Population matrix (created in `form_population_matrix`)

    Returns
    -------
        contagious_hhl:         people with 2-5 states
        contagious_hhl_qua:     people with 9-12 states
        contagious_camp:        symp+asymp or presymp or symp+teen
        contagious_sitters:     not wanderers
        contagious_wanderers:   wanderers
        active_camp:            symp+asymp or presymp or symp+teen or exposed+susep or recovered

    Notes
    -----
    Possible values of disease state (pop_matrix[:, 1]):
        0: susceptible, 1: exposed, 2: presymptomatic, 3: symptomatic, 4: mild, 5: severe, 6: recovered,
        7: qua_susceptible, 8: qua_exposed, 9: qua_presymptomatic, 10: qua_symptomatic, 11: qua_mild,
        12: qua_severe, 13: qua_recovered

    """

    contagious_hhl = np.logical_and(pop_matrix[:, 1] > 1, pop_matrix[:, 1] < 6)
    contagious_hhl_qua = np.logical_and(pop_matrix[:, 1] > 8, pop_matrix[:, 1] < 13)

    # get asymptomatic population with contagious infection
    contagious_asymp = np.logical_and.reduce((
        pop_matrix[:, 1] > 2, pop_matrix[:, 1] < 5,  # returns people who are symptomatic/mild
        # TODO: should 10,11 stages also included here?
        pop_matrix[:, 4] == 1  # returns asymptomatic people
    ))
    # get presymptomatic population
    # TODO: should qua_presymptomatic also included here?
    contagious_presymp = pop_matrix[:, 1] == 2
    contagious_teen = np.logical_and.reduce((pop_matrix[:, 1] > 2, pop_matrix[:, 1] < 5, pop_matrix[:, 5] < 16))
    contagious_camp = np.logical_or.reduce((
        contagious_asymp,
        contagious_presymp,
        contagious_teen
    ))

    # NOTE: pop_matrix[:, 8] returns 0/1 if person is sitter/wanderer
    contagious_sitters = np.logical_and(contagious_camp, pop_matrix[:, 8] == 0)
    contagious_wanderers = np.logical_and(contagious_camp, pop_matrix[:, 8] == 1)

    active_camp = np.logical_or.reduce((
        contagious_camp,
        pop_matrix[:, 1] < 2,  # susceptible/exposed people (i.e. uninfected) TODO: should we include 7/8 as well?
        pop_matrix[:, 1] == 6  # recovered people (cannot get infection again) TODO: should we include 13 as well?
    ))

    assert sum(contagious_camp) == (sum(contagious_sitters)+sum(contagious_wanderers))
    return contagious_hhl, contagious_hhl_qua, contagious_camp, contagious_sitters, contagious_wanderers, active_camp


def infected_and_sum_by_households(pop_matrix, contagious):

    contagious_hhl = contagious[0]
    contagious_hhl_qua = contagious[1]
    contagious_camp = contagious[2]
    contagious_sitters = contagious[3]
    contagious_wanderers = contagious[4]
    active_camp = contagious[5]

    inf_h = accumarray(pop_matrix[:, 0], contagious_hhl)         # All infected in house and at toilets, population
    inf_hq = accumarray(pop_matrix[:, 0], contagious_hhl_qua)    # All infected in house, quarantine
    inf_l = accumarray(pop_matrix[:, 0], contagious_camp)        # presymptomatic and asymptomatic for food lines
    inf_ls = accumarray(pop_matrix[:, 0], contagious_sitters)    # All sitters for local transmission
    inf_lw = accumarray(pop_matrix[:, 0], contagious_wanderers)  # All wanderers for local transmission
    all_fl = accumarray(pop_matrix[:, 0], active_camp)           # All people in food lines
    return inf_h, inf_hq, inf_l, inf_ls, inf_lw, all_fl


def infected_prob_inhhl(inf_prob_hhl, trans_prob_hhl):
    return 1-(1-trans_prob_hhl)**np.array(inf_prob_hhl) 


def infected_prob_activity(
        # Toilets or foodline activity shared by households
        household_sharing_matrix,
        # households_with_asymp_presymp_symp_sympteen
        people_infected_probablity_activity,
        # households_with_asymp_presymp_symp_sympteen_exposed_suseptible
        people_per_household,
        # visits per person per day for activity (toilet or foodline)
        visits_per_day,
        # contacts made per person for every visit at toilet or foodline
        num_contacts_per_visit,
        # probability of transmission at toilet or foodline
        probability_of_transmission_for_activity,
        # initial_transmission_reduction
        transmission_reduction,
        # probability of individual performing activity
        factor=1):
    # this could be infections at the toilet or the foodline
    # infected individuals without symptoms
    infected_individuals=household_sharing_matrix.dot(people_infected_probablity_activity)
    # total individuals without symptoms
    total_individuals=household_sharing_matrix.dot(people_per_household)
    proportion_infected=infected_individuals/total_individuals
    activity_factor=visits_per_day*num_contacts_per_visit
    activity_coeff = np.arange(activity_factor+1)
    transmission_during_activity = 1-factorial(activity_factor)*np.sum(((factorial(activity_factor-activity_coeff)*factorial(activity_coeff))**-1)*
                                (np.transpose(np.array([(1-proportion_infected)**(activity_factor-i) for i in activity_coeff])))*
                                (np.transpose(np.array([proportion_infected**i for i in activity_coeff])))*
                                (np.power(1-probability_of_transmission_for_activity*transmission_reduction,activity_coeff)),1)
    return transmission_during_activity*factor


def infected_prob_movement(pop_matrix, neighbour_inter, aip, tr, contagious_households):
    """

    Parameters
    ----------
        pop_matrix: 2D Population matrix
        neighbour_inter: Local interaction space
        aip: the probability of infecting other people moving about
        tr: Transmission reduction
        contagious_households: Contagious households

    Returns
    -------
        out: Local transmission rates i.e. probability of transmission during movement
    """

    households_with_no_wanderers = contagious_households[3]
    households_with_wanderers = contagious_households[4]

    lr1_exp_contacts = neighbour_inter[:, :, 0].dot(households_with_no_wanderers) + \
                       neighbour_inter[:, :, 1].dot(households_with_wanderers)
    lr2_exp_contacts = neighbour_inter[:, :, 1].dot(households_with_no_wanderers) + \
                       neighbour_inter[:, :, 2].dot(households_with_wanderers)

    # But contacts are roughly Poisson distributed (assuming a large population), so transmission rates are:
    trans_for_lr1 = 1-np.exp(-lr1_exp_contacts*aip*tr)
    trans_for_lr2 = 1-np.exp(-lr2_exp_contacts*aip*tr)    

    # Now, assign the appropriate local transmission rates to each person.
    trans_local_inter = trans_for_lr1[pop_matrix[:, 0].astype(int)]*(1-pop_matrix[:, 8]) + \
                        trans_for_lr2[pop_matrix[:, 0].astype(int)]*(pop_matrix[:, 8])

    return trans_local_inter


def prob_of_transmission_within_household(Ph, Pt, Pf):
    """
    Get the probability of transmission of infection within household

    Parameters
    ----------
        Ph: Probability for members of each household to contract from their housemates
        Pt: Probability for members of each household to contract during a toilet visit
        Pf: Probability for members of each household to contract during a food line visit

    Returns
    -------
        out: Probability of transmission of infection within household

    Notes
    -----
    From paper: "𝑝𝑖𝑑 = 1 − Π(1 − 𝑝𝑖𝑑𝑤) 𝑤∈{ℎ,𝑡,𝑓,𝑚}."
    """
    return 1 - ((1 - Ph) * (1 - Pt) * (1 - Pf))


def prob_of_transmission(Ph, Pm):
    """
    Total probability of transmission
    Parameters
    ----------
        Ph: Probability for members of each household to contract from their housemates
        Pm: Probability for members of each household to contract during the movement (wandering around)

    Returns
    -------
        out: Total probability of transmission

    Notes
    -----
    From paper: "𝑝𝑖𝑑 = 1 − Π(1 − 𝑝𝑖𝑑𝑤) 𝑤∈{ℎ,𝑡,𝑓,𝑚}."
    """
    return 1 - ((1 - Ph) * (1 - Pm))


def impose_infection(infection_status, newly_infected):
    """
    Get Infection status for newly infected people.

    Parameters
    ----------
        infection_status: Disease state for each individual.
        newly_infected: Boolean array where ith element is True if ith person is newly infected

    Returns
    -------
        out: Updated infection status for newly infected people

    Notes
    -----
    Only possible update is from "susceptible (0) -> exposed (1)" in this function
    """
    infection_status += (1 - np.sign(infection_status)) * newly_infected
    return infection_status


def impose_infection_in_quarantine(infection_probability_per_person_in_quarantine, population_total):
    random_uniform = np.random.uniform(0, 1, population_total)
    newly_infected_in_quarantine = infection_probability_per_person_in_quarantine > random_uniform
    return newly_infected_in_quarantine


def update_infection_status(
        # Population matrix of the camp
        population,
        # Probability of the transmission
        probability_of_transmission,
        # Probability of the transmission in quarantin
        infection_probability_per_person_in_quarantin
):
    population_total = population.shape[0]
    infection_status = population[:, 1]
    households = population[:, 0].astype(int)

    # Find new infections by person, population
    newly_infected = probability_of_transmission > np.random.uniform(0, 1, population_total)

    # Impose infections, population. Only infect susceptible individuals
    infection_status = impose_infection(infection_status, newly_infected)

    # Find new infections by person, quarantine
    newly_infected_in_quarantin = impose_infection_in_quarantine(
        infection_probability_per_person_in_quarantin[households],
        population_total)

    # Impose infections, quarantine
    infection_status += (infection_status == 7) * newly_infected_in_quarantin

    return population


def transmission_within_household(
        # probability_of_infection_household
        probability_of_infection_household,
        # toilets_shared_by_households
        toilets_shared_by_households,
        # toilet_visits_per_person_per_day
        toilet_visits_per_person_per_day,
        # contacts_per_toilet_visit
        contacts_per_toilet_visit,
        # probability_of_infection_toilet
        probability_of_infection_toilet,
        # initial_transmission_reduction
        initial_transmission_reduction,
        # foodpoints_shared_by_households
        foodpoints_shared_by_households,
        # foodline_visits_per_person_per_day
        foodline_visits_per_person_per_day,
        # contacts_per_foodline_visit
        contacts_per_foodline_visit,
        # probability_of_infection_food_line
        probability_of_infection_food_line,
        # pct_days_with_foodline_visits
        pct_days_with_foodline_visits,
        # contagious_households
        contagious_households
):
    households_with_contagious_person = contagious_households[0]
    households_with_asymp_presymp_symp_sympteen = contagious_households[2]
    households_with_asymp_presymp_symp_sympteen_exposed_suseptible = contagious_households[5]

    # In population
    probability_of_transmission_in_household = infected_prob_inhhl(
        households_with_contagious_person,
        probability_of_infection_household
    )

    # Compute proportions infecteds at toilets and in food lines.
    probability_of_transmission_at_toilet = infected_prob_activity(
        toilets_shared_by_households,
        households_with_asymp_presymp_symp_sympteen,
        households_with_asymp_presymp_symp_sympteen_exposed_suseptible,
        toilet_visits_per_person_per_day,
        contacts_per_toilet_visit,
        probability_of_infection_toilet,
        initial_transmission_reduction
    )
    # Compute transmission in food lines by household.
    # Assume each person goes to the food line once per day on 75% oft_factor days.
    # Other days someone brings food to them (with no additional contact).
    probability_of_transmission_in_foodline = infected_prob_activity(
        foodpoints_shared_by_households,
        households_with_asymp_presymp_symp_sympteen,
        households_with_asymp_presymp_symp_sympteen_exposed_suseptible,
        foodline_visits_per_person_per_day,
        contacts_per_foodline_visit,
        probability_of_infection_food_line,
        initial_transmission_reduction,
        factor=pct_days_with_foodline_visits
    )

    # Households in quarantine don't get these exposures, but that is taken care of below
    # because this is applied only to susceptible in the population with these, we can calculate
    # the probability of all transmissions that calculated at the household level.
    probability_of_transmission_within_household = prob_of_transmission_within_household(
        probability_of_transmission_in_household,
        probability_of_transmission_at_toilet,
        probability_of_transmission_in_foodline
    )

    return probability_of_transmission_within_household


def assign_new_infections(
        population, toilets_shared_by_households, foodpoints_shared_by_households,
        toilet_visits_per_person_per_day, contacts_per_toilet_visit, foodline_visits_per_person_per_day,
        contacts_per_foodline_visit, pct_days_with_foodline_visits, initial_transmission_reduction,
        local_interaction_space, probability_of_infection_household, probability_of_infection_food_line,
        probability_of_infection_toilet, probability_of_infection_wandering):

    """

    Parameters
    ----------
        population:                             Population matrix (created in `form_population_matrix`)
        toilets_shared_by_households:           a matrix (2D array) of shared toilets at the household level
        foodpoints_shared_by_households:        a matrix (2D array) of shared food points at the household level
        toilet_visits_per_person_per_day:       the number of toilet visits per person per day
        contacts_per_toilet_visit:              the number of contacts per a toilet visit
        foodline_visits_per_person_per_day:     the number of food point visits per person per day
        contacts_per_foodline_visit:            the number of contacts per a food point visit
        pct_days_with_foodline_visits:          percentage of food point visits ppd (once per day on 3 out of 4 days)
        initial_transmission_reduction:         the initial transmission reduction
        local_interaction_space:                Local interaction space
        probability_of_infection_household:     the probability of infecting each person in your household per day
        probability_of_infection_food_line:     the probability of infecting other people in the food line
        probability_of_infection_toilet:        the probability of infecting other people in the toilet
        probability_of_infection_wandering:     the probability of infecting other people moving about

    Returns
    -------

    """

    ##########################################################
    # 1. IDENTIFY CONTAGIOUS AND ACTIVE PEOPLE IN DIFFERENT CONTEXTS
    # Contagious in the house and at toilets, food points, outside, in population.
    # At least presymptomatic AND at most severe.

    households = population[:, 0].astype(int)
    contagious_people = identify_contagious_active(population)
    contagious_households = infected_and_sum_by_households(population, contagious_people)

    ##########################################################
    # 2. COMPUTE INFECTION PROBABILITIES FOR EACH PERSON BY HOUSEHOLD
    # Probability for members of each household to contract from their housemates

    probability_of_transmission_within_household = transmission_within_household(
        # probability_of_infection_household
        probability_of_infection_household,
        # toilets_shared_by_households
        toilets_shared_by_households,
        # toilet_visits_per_person_per_day
        toilet_visits_per_person_per_day,
        # contacts_per_toilet_visit
        contacts_per_toilet_visit,
        # probability_of_infection_toilet
        probability_of_infection_toilet,
        # initial_transmission_reduction
        initial_transmission_reduction,
        # foodpoints_shared_by_households
        foodpoints_shared_by_households,
        # foodline_visits_per_person_per_day
        foodline_visits_per_person_per_day,
        # contacts_per_foodline_visit
        contacts_per_foodline_visit,
        # probability_of_infection_food_line
        probability_of_infection_food_line,
        # pct_days_with_foodline_visits
        pct_days_with_foodline_visits,
        contagious_households
    )

    # Transmissions during movement around the residence must be calculated at the individual level,
    # because they do not depend on what movement radius the individual uses. So...
    # Compute expected contacts with infected individuals for individuals that use small and
    # large movement radii.
    probability_of_transmission_during_movement = infected_prob_movement(
        population,
        local_interaction_space,
        probability_of_infection_wandering,
        initial_transmission_reduction,
        contagious_households
    )
    # Finally, compute the full per-person infection probability within households,
    # at toilets and food lines.
    probability_of_transmission = prob_of_transmission(
        probability_of_transmission_within_household[households],
        probability_of_transmission_during_movement
    )

    # In quarantine
    probability_of_transmission_in_quarantin = infected_prob_inhhl(
        # Households with contagious person in quarantine
        contagious_households[1],
        probability_of_infection_household
    )

    ##########################################################
    # 3. ASSIGN NEW INFECTIONS

    population = update_infection_status(
        population,
        probability_of_transmission,
        probability_of_transmission_in_quarantin
    )

    return population


def move_hhl_quarantine(pop_matrix, prob_spot_symp):
    """

    Parameters
    ----------
    pop_matrix: Population matrix (from `form_population_matrix`)
    prob_spot_symp: Probability of spotting symptoms

    Returns
    -------

    Notes
    -----
    Possible values of disease state (pop_matrix[:, 1]):
        0: susceptible, 1: exposed, 2: presymptomatic, 3: symptomatic, 4: mild, 5: severe, 6: recovered,
        7: qua_susceptible, 8: qua_exposed, 9: qua_presymptomatic, 10: qua_symptomatic, 11: qua_mild,
        12: qua_severe, 13: qua_recovered

    """

    # Individuals in camp with symptoms spotted given some probability
    spot_symp = np.random.uniform(0, 1, pop_matrix.shape[0]) < prob_spot_symp

    # Current state of all individuals in camp
    states = pop_matrix[:, 1]

    # Filter conditions for next operation
    symptomatic = states > 2                # Symptomatic individuals
    not_quarantined = states < 6            # Individuals not in quarantine
    not_new_asymp = pop_matrix[:, 4] == 0   # Individuals who are not newly asymptomatic
    aged_above_15 = pop_matrix[:, 5] >= 16  # Individuals aged above 15 yrs

    # Individuals not quarantined who should be...
    symp = np.logical_and.reduce((
        symptomatic,
        not_quarantined,
        not_new_asymp,
        aged_above_15
    ))

    # ... of which those who discover they have symptoms and should be quarantined
    spotted_per_day = spot_symp * symp

    # IDS of households containing quarantined individuals
    symp_house = pop_matrix[spotted_per_day == 1, 0]

    # Individuals in quarantined households
    qua_hh = np.in1d(pop_matrix[:, 0], symp_house)

    # Update individuals in quarantined households to 'qua_*'
    pop_matrix[qua_hh, 1] += 7

    return pop_matrix


def relative_encounter_rate(d, R, r):
    """
    Relative encounter rate between individuals in overlapping area

    Parameters
    ----------
        d : 2D distance matrix where (i, j) element holds the Euclidean distance between household (i) and household (j)
        R : Movement radius - 1
        r : Movement radius - 2
    
    Returns
    -------
        out : 
    """

    # angle between center of circle centered at (1) and point of intersections of circles centered at (1) and (2)
    angle1 = 2*np.arccos(
        np.nan_to_num( # `nan_to_num` will replace `nan` values with 0
            np.clip(
                ( d**2 + R**2 - r**2 ) / ( 2*d*R ),
                a_min=None,
                a_max=1
            )
        )
    )  # nan means no overlap in this case

    # angle between center of circle centered at (2) and point of intersections of circles centered at (1) and (2)
    angle2 = 2*np.arccos(
        np.nan_to_num(
            np.clip(
                (d**2 + r**2 - R**2) / (2*d*r),
                a_min=None,
                a_max=1
            )
        )
    )  # nan means no overlap in this case

    area_sector1 = 0.5 * (r**2) * angle2
    area_sector2 = 0.5 * (R**2) * angle1

    area_overlap12 = np.nan_to_num(area_sector1 + area_sector2 - r * np.sin(angle2/2) * d)
    
    relative_encounter12 = area_overlap12 / (math.pi**2 * R**2 * r**2)
    
    return relative_encounter12
