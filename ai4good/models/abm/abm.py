import numpy as np
import pandas as pd
import random
import math
from scipy import optimize
from scipy.special import factorial

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
    path_to_file = 'age_and_sex.csv'
    
    # Data frame. V1 = age, V2 is sex (1 = male?, 0  = female?).
    age_and_gender = pd.read_csv(path_to_file)
    age_and_gender = age_and_gender.loc[:, ~age_and_gender.columns.str.contains('^Unnamed')]
    age_and_gender = age_and_gender.values
    age_and_gender = age_and_gender[np.random.randint(age_and_gender.shape[0], size=num_ppl)]
    return age_and_gender


def create_household_column(num_hh_type1, num_ppl_type1, num_hh_type2, num_ppl_type2):
    """
    Creates an initial household distribution for each individual.

    Parameters
    ----------
        num_hh_type1  : Number of isoboxes (hb)
        num_ppl_type1 : Number of people in isoboxes (Nb)
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
        # values ∈ [1, hb] for isoboxes
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
    #weibull distribution parameters following (Backer et al. 2020 Eurosurveillance)
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
    """
    
    if age_column is not None:
        raise NotImplementedError("Age has not been considered for asymptomatic infection calculations yet!")
    else:
        return np.random.uniform(0, 1, num_ppl) < ( asymp_rate * (num_ppl/(num_ppl-num_ppl_chro)) )

def create_age_column(age_data):
    return age_data

def create_gender_column(gender_data):
    return gender_data

def create_chronic_column(num_ppl, age_column, num_ppl_chro=300):
    myfunction = lambda x: np.absolute(num_ppl_chro-np.sum((1+np.exp(-(x-11.69+.2191*age_column-0.001461*age_column**2))**(-1))))-num_ppl
    xopt = optimize.fsolve(myfunction, x0=[2])
    rchron = (1+np.exp(-(xopt-11.69+.2191*age_column-0.001461*age_column**2)))**(-1)
    chroncases = (np.random.uniform(np.min(rchron),1,num_ppl) < rchron)
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
    new_asymp_column[chronic_column==1] = 0
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
        out : Boolean array of same size as of `gender` and `age`, indicating if a person is wanderer (True) or not (False)

    """

    return np.logical_and([gender == 1], [age >= 10]).transpose()

def form_population_matrix(N, hb, Nb, ht, Nt, pac, age_and_gender):
    """
    Create population matrix based on parameters.

    Parameters
    ----------
        N              : Total population
        hb             : Number of isoboxes
        Nb             : Number of people in isoboxes
        ht             : Number of tents
        Nt             : Number of people in tents
        pac            : Proportion of permanently asymptomatic cases (Mizumoto et al 2020 Eurosurveillance)
        age_and_gender : (N, 2) sized array containing age and gender of population
    
    Returns
    -------
        out : 2D array containing population info. Each row will contain following columns:
           0: Initial household distribution for each individual [0, hb+ht)
           1: Binary disease state for each individual {0, 1}
           2: Days to symptoms
           3: Daycount
           4: Binary state indicating if individual is asymptomatic {0, 1}
           5: Age of the individual
           6: Gender of the individual {0: female, 1: male}
           7: Binary state indicating if individual suffers from chronic disease {0, 1}
           8: Binary state indicating if individual is wanderer or not {0, 1}

    """
    
    #1 allocated to isoboxes and tents
    household_column = create_household_column(hb, Nb, ht, Nt)
    #2 allocate an infector
    disease_column = create_diseasestate_column(N)
    #3 weibull distribution
    dsymptom_column = create_daystosymptoms_column(N)
    #4 np.zeros(num_ppl)
    daycount_column = create_daycount_column(N)
    #5 hard coded number of chronics
    asymp_column = create_asymp_column(N, pac)
    #6 age
    age_column = create_age_column(age_and_gender[:,0])
    #7 gender
    gender_column = create_gender_column(age_and_gender[:,1])
    #8 formula based on age
    chronic_column = create_chronic_column(N, age_column)
    #5 new_asymp_column[chronic_column==1]=0
    new_asymp_column = adjust_asymp_with_chronic(asymp_column, chronic_column)
    #9
    wanderer_column = create_wanderer_column(gender_column, age_column)

    pop_matrix=np.column_stack((household_column, disease_column, dsymptom_column,
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
        prop_type1      : Proportion of area covered by isoboxes (iba=0.5)
        num_hh_type1    : Number of isoboxes (hb)
    
    Returns
    -------
        out : 2D array containing location of households (isoboxes + tents)

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

def assign_block(hhloc, blocks):
    """
    Assign households to blocks based on a hypothetical grid size

    Parameters
    ----------
        hhloc  : A (hb+ht, 2) array containing position of isoboxes and tents
        blocks : [nx, ny] pair containing grid size
    
    Returns
    -------
        label  : A 1D array containing block numbers for each household
        shared : A 2D boolean array of shared blocks at the household level.
                 shared[i, j] will be `True` if household i and j share the toilet or food line else `False`
    """

    # total number of households (isoboxes + tents)
    hh_size = hhloc.shape[0]

    # assign x coordinate of household to one of the x-axis blocks
    limit_x = np.arange(1, blocks[0] + 1) / blocks[0] # x-axis grid lines considering 1x1 camp
    label_x = np.digitize(hhloc[:, 0], limit_x)       # returns indices of x-axis grid line where household belongs

    # assign y coordinate of household to one of the y-axis blocks
    limit_y = np.arange(1, blocks[1] + 1) / blocks[1] # y-axis grid lines considering 1x1 camp
    label_y = np.digitize(hhloc[:, 1], limit_y)       # returns indices of y-axis grid line where household belongs
    
    # based on integral grid position (label_x, label_y), calculate the grid number [1, nx*ny]
    label = label_y * blocks[0] + label_x + 1

    # find out which households share the same toilet or foodline
    TEMP = np.tile(label, (len(label), 1))
    shared = (TEMP.T == TEMP) - np.eye(hh_size)

    # TODO: this assert statement will not be always True! Confirm with others.
    # assert np.max(label) == np.prod(blocks)
    assert shared.shape == (hh_size, hh_size)
    location = np.vstack((limit_x, limit_y))
    
    return location, label, shared

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
    
def create_ethnic_groups(hhloc,int_eth):
    """
    From the paper:
    -----------------
    "In Moria, the homes of people with the same ethnic or national background are spatially clustered, and people interact more frequently      with others from the same background as themselves.   ... To simulate ethnicities or nationalities in our camp, we assigned each            household to one of eight (?) “backgrounds” in proportion to the    self-reported countries of origin of people.. For each of the          eight simulated backgrounds, we randomly selected one tent or isobox to be    the seed for the cluster. We assigned the x nearest          unassigned households to that background, where x is the number of households with that background."

    Parameters:
    -----------
    hhloc: A (hb+ht, 2) array containing position of isoboxes and tents
    int_eth: scalar representing relative strength interactions between ethnicities (external parameter)

    Returns:
    ----------
    ethcor: ethnicities correlation

    """
    Afghan = 7919 ; Cameroon = 149 ; Congo = 706 ;Iran = 107 ;Iraq = 83 ; Somalia = 442 ; Syria = 729
    g = np.array([Afghan,Cameroon,Congo,Iran,Iraq,Somalia,Syria])  
    totEthnic = sum(g) 
    hh_size=hhloc.shape[0]
    g_hh = np.round(hh_size*g/totEthnic)              # Number of households per group.
    np.random.shuffle(g_hh) #shuffle the array
    hhunass= np.column_stack((np.arange(0,hh_size), hhloc))   # Unassigned households. Firsto column is the index of the hh.
    hheth = np.zeros((hh_size,1))
    i=0
    for g in g_hh:
        gcen = hhunass[np.random.randint(hhunass.shape[0]),1:] # Chose an unassigned household as the group (cluster) centre.
        dfromc = np.sum((hhunass[:,1:]-np.tile(gcen,(hhunass.shape[0],1)))**2,1) # Squared distance to cluster centre.
        cloind = np.argsort(dfromc)                            # Get the indices of the closest households (cloind).
        hheth[hhunass[cloind[0:int(g)],0].astype(int)] = i  # Assign i-th ethnic group to those households.
        hhunass = np.delete(hhunass,cloind[0:int(g)],0)     # Remove those hoseholds (remove the i-th cluster/ethnic group)
        i+=1
    ethmatch = (np.tile(hheth,(1,len(hheth)))==np.tile(hheth,(1,len(hheth))).T )
    #scale down the connection for poeple of different background
    ethcor = ethmatch+int_eth*(1-ethmatch)
    return ethcor

def interaction_neighbours(household_coordinates, movement_radius_small, movement_radius_large, lrtol, ethnic_groups):
    """
    Parameters
    ----------
        household_coordinates : A 2D array containing co-ordinates of the households
        movement_radius_small : Smaller movement radius [0-1]
        movement_radius_large : Larger movement radius [0-1]
        lrtol                 : scale value for interactions within household
        ethnic_groups         : Scale values for interaction between ethnic groups (output of create_ethnic_groups())
    
    Returns
    -------
        out : Local interaction space
    """

    # create distance matrix for distance in between households
    household_distance_matrix = distance_between_households(household_coordinates);

    # the case where person with movement_radius_small is inteacting with person with same radius
    relative_encounter_small = relative_encounter_rate(household_distance_matrix, movement_radius_small,
                                                       movement_radius_small)

    # the case where person with movement_radius_large is inteacting with person with same radius
    relative_encounter_large = relative_encounter_rate(household_distance_matrix, movement_radius_large,
                                                       movement_radius_large)
    # the case where person with movement_radius_large is inteacting with person with movement_radius_small
    relative_encounter_small_large = relative_encounter_rate(household_distance_matrix, movement_radius_small,
                                                             movement_radius_large)

    lis = np.multiply(math.pi * lrtol ** 2 * np.dstack(
        (relative_encounter_small, relative_encounter_small_large, relative_encounter_large)),
                      np.dstack((ethnic_groups, ethnic_groups, ethnic_groups)))
    return lis

def interaction_neighbours_fast(hhloc, lr1, lr2, lrtol, ethcor):
    #use the formula from https://mathworld.wolfram.com/Circle-CircleIntersection.html 
    #create distance matrix for distance in between households
    hhdm_x=(np.tile(hhloc[:,0],(hhloc.shape[0],1)).T - np.tile(hhloc[:,0],(hhloc.shape[0],1)))**2
    hhdm_y=(np.tile(hhloc[:,1],(hhloc.shape[0],1)).T - np.tile(hhloc[:,1],(hhloc.shape[0],1)))**2
    hhdm=np.sqrt(hhdm_x+hhdm_y)
    #the case where lr1 is inteacting with lr1
    area_overlap11=2*(lr1**2*np.arccos(np.clip(0.5*hhdm/lr1,a_min=None,a_max=1))-np.nan_to_num(hhdm/2*np.sqrt(lr1**2-hhdm**2/4)))
    relative_encounter11=area_overlap11/(math.pi**2*lr1**4)
    #the case where lr2 is inteacting with lr2
    area_overlap22=2*(lr2**2*np.arccos(np.clip(hhdm/(2*lr2),a_min=None,a_max=1))-np.nan_to_num(hhdm/2*np.sqrt(lr2**2-hhdm**2/4)))
    relative_encounter22=area_overlap22/(math.pi**2*lr2**4)
    #the case where lr1 is interacting with lr2
    area_overlap12=np.nan_to_num((lr1**2*np.arccos(np.clip((hhdm**2+lr1**2-lr2**2)/(2*hhdm*lr1),a_min=None,a_max=1)))
    +(lr2**2*np.arccos(np.clip((hhdm**2+lr2**2-lr1**2)/(2*hhdm*lr2),a_min=None,a_max=1)))
    -0.5*np.sqrt((-hhdm+lr1+lr2)*(hhdm+lr1-lr2)*(hhdm-lr1+lr2)*(hhdm+lr1+lr2)))
    relative_encounter12=area_overlap12/(math.pi**2*lr2**2*lr1**2)
    lis = np.multiply(math.pi*lrtol**2*np.dstack((relative_encounter11,relative_encounter12,relative_encounter22)),np.dstack((ethcor,ethcor,ethcor)))
    return lis

def epidemic_finish(states, iteration):
    '''
    Finish the simulation when no person is in any state other than recovered or susceptible
    '''
    return (np.sum(states) == 0 and iteration > 10)

def disease_state_update(pop_matrix, mild_rec, sev_rec, pick_sick, thosp, quarantined=False):
    '''
    Disease progress from one state to another among susceptible, exposed, presymptomatic,
    symptomatic, mild, severe and recovered for Quarantine and Normal situation
    '''
    # thosp = Total number of hospitalized individuals.
    # abc_to_xyz_ind = abc state to xyz state 's indices in pop_matrix

    # columns for pop_matrix
    # 0 "household",
    # 1 "disease",
    # 2 "dsymptom",
    # 3 "daycount",
    # 4 "new_asymp",
    # 5 "age",
    # 6 "gender",
    # 7 "chronic",
    # 8 "wanderer"

    # values for pop_matrix[:,1]:
    # 0 'susceptible',
    # 1 'exposed',
    # 2 'presymptomatic',
    # 3 'symptomatic',
    # 4 'mild',
    # 5 'severe',
    # 6 'recovered',
    # 7 'qua_susceptible',
    # 8 'qua_exposed',
    # 9 'qua_presymptomatic',
    # 10'qua_symptomatic',
    # 11'qua_mild',
    # 12'qua_severe',
    # 13'qua_recovered'

    qua_add=0
    if quarantined:
        qua_add=7
    # Move exposed to presymptomatic
    exposed_to_presym_ind=np.logical_and(pop_matrix[:,1]==1+qua_add,pop_matrix[:,3]>=np.floor(0.5*pop_matrix[:,2]))
    pop_matrix[exposed_to_presym_ind,1] =2+qua_add
    # Move presymptomatic to symptomatic but not yet severe.
    presymp_to_symp_ind = np.logical_and(pop_matrix[:,1]==2+qua_add,pop_matrix[:,3]>=pop_matrix[:,2])
    pop_matrix[presymp_to_symp_ind,1] = 3+qua_add
    pop_matrix[presymp_to_symp_ind,3] = 0
    # Move individuals with 6 days of symptoms to mild.
    symp_to_mild_ind=np.logical_and(pop_matrix[:,1]==(3+qua_add),pop_matrix[:,3]==6) 
    pop_matrix[symp_to_mild_ind,1] = 4+qua_add   
    # Move Mild Symptoms to recovered
    mild_to_recovered_ind = np.logical_and(pop_matrix[:,1]==(4+qua_add),mild_rec)
    pop_matrix[mild_to_recovered_ind,1] = 6+qua_add                             
    # Move Severe symptoms to recovered.
    severe_to_recovered_ind = np.logical_and(pop_matrix[:,1]==(5+qua_add),sev_rec)
    pop_matrix[severe_to_recovered_ind,1] = 6+qua_add     
    #  symptomatic to the “mild” or “severe”
    asp = np.array([0,.000408,.0104,.0343,.0425,.0816,.118,.166,.184])        # Verity et al. hospitalisation.
    aspc = np.array([.0101,.0209,.0410,.0642,.0721,.2173,.2483,.6921,.6987])  # Verity et al. corrected for Tuite.
    AGE_BUCKET=9
    for buc in range(AGE_BUCKET):
        # Assign individuals with mild symptoms for six days, sick, between 10*sci and 10*sci+1 years old to severe and count as hospitalized.
        if buc==8:# For all individual with Age 80 and above.
            # Move individuals with Chronic diseases with 6 days of mild to severe.
            mild_to_severe_ind=np.logical_and.reduce((pop_matrix[:,1]==4+qua_add,pop_matrix[:,3]==6,pick_sick<asp[buc],pop_matrix[:,5]>=10*buc,pop_matrix[:,7]==0))
            thosp += np.sum(mild_to_severe_ind)
            pop_matrix[mild_to_severe_ind,1] = 5+qua_add
            # Move individuals with Chronic diseases with 6 days of mild to severe.
            mild_to_severe_chronic_ind=np.logical_and.reduce((pop_matrix[:,1]==4+qua_add,pop_matrix[:,3]==6,pick_sick<aspc[buc],pop_matrix[:,5]>=10*buc,pop_matrix[:,7]==1))
            thosp += np.sum(mild_to_severe_chronic_ind)
            pop_matrix[mild_to_severe_chronic_ind,1] = 5+qua_add 
        else:
            severe_ind = np.logical_and.reduce((
                pop_matrix[:, 1] == 4 + qua_add, #mild
                pop_matrix[:, 3] == 6, #6 days
                pick_sick < asp[buc], #probability of passage to next disease state for low risk people
                pop_matrix[:, 5] >= 10 * buc,
                pop_matrix[:, 5] < (10 * buc + 10)
            ))
            thosp += np.sum(severe_ind)
            pop_matrix[severe_ind, 1] = 5 + qua_add
            # Wouldnt this step double count previous individuals? Is this step the one that adjusts for pre-existing conditions?
            severe_chronic_ind = np.logical_and.reduce((
                pop_matrix[:, 1] == 4 + qua_add,
                pop_matrix[:, 3] == 6,
                pick_sick < aspc[buc], #probability of passage to next disease state for high risk people
                pop_matrix[:, 5] >= (10 * buc),
                pop_matrix[:, 5] < (10 * buc + 10),
                pop_matrix[:, 7] == 1
            ))
            thosp += np.sum(severe_chronic_ind)
            pop_matrix[severe_chronic_ind, 1] = 5 + qua_add

            # Move individuals with Non Chronic diseases with 6 days of mild to severe for current age group.
            # cond1 = pop_matrix[:,1]==4+qua_add
            # cond2 = pop_matrix[:,3]==6
            # cond3 = pick_sick<asp[buc]
            # cond4 = pop_matrix[:,5]>=10*buc
            # cond5 = pop_matrix[:,5]<(10*buc+10)
            # cond6 = pop_matrix[:,7]==0
            # mild_to_severe_ind=np.logical_and.reduce((pop_matrix[:,1]==4+qua_add,pop_matrix[:,3]==6,pick_sick<asp[buc],pop_matrix[:,5]>=10*buc, pop_matrix[:,5]<(10*buc+10)))
            #np.logical_and.reduce((pop_matrix[:,1]==4+qua_add,pop_matrix[:,3]==6,pick_sick<asp[buc],pop_matrix[:,5]>=10*buc, pop_matrix[:,5]<(10*buc+10), pop_matrix[:,7]==0))
            # thosp += np.sum(mild_to_severe_ind)
            # pop_matrix[mild_to_severe_ind,1] = 5+qua_add
            # # Move individuals with Chronic diseases with 6 days of mild to severe for current age group.
            # mild_to_severe_chronic_ind=np.logical_and.reduce((pop_matrix[:,1]==4+qua_add,pop_matrix[:,3]==6,pick_sick<aspc[buc],pop_matrix[:,5]>=10*buc,pop_matrix[:,5]<(10*buc+10),pop_matrix[:,7]==1))
            # thosp += np.sum(mild_to_severe_chronic_ind)
            # pop_matrix[mild_to_severe_chronic_ind,1] = 5+qua_add
                           
    return pop_matrix,thosp

def disease_state_update_for_agent(agent,mild_rec,sev_rec,pick_sick,thosp,quarantined=False):
    '''
    Disease progress from one state to another among susceptible, exposed, presymptomatic,
    symptomatic, mild, severe and recovered for Quarantine and Normal situation
    '''
    # thosp = Total number of hospitalized individuals.
    # abc_to_xyz_ind = abc state to xyz state 's indices in pop_matrix

    # columns for pop_matrix
    # 0 "household",
    # 1 "disease",
    # 2 "dsymptom",
    # 3 "daycount",
    # 4 "new_asymp",
    # 5 "age",
    # 6 "gender",
    # 7 "chronic",
    # 8 "wanderer"

    # values for pop_matrix[:,1]:
    # 0 'susceptible',
    # 1 'exposed',
    # 2 'presymptomatic',
    # 3 'symptomatic',
    # 4 'mild',
    # 5 'severe',
    # 6 'recovered',
    # 7 'qua_susceptible',
    # 8 'qua_exposed',
    # 9 'qua_presymptomatic',
    # 10'qua_symptomatic',
    # 11'qua_mild',
    # 12'qua_severe',
    # 13'qua_recovered'

    pop_matrix = np.array([[
        agent.household,
        agent.disease,
        agent.dsymptom,
        agent.daycount,
        agent.new_asymp,
        agent.age,
        agent.gender,
        agent.chronic,
        agent.wanderer
    ]])

    qua_add=0
    if quarantined:
        qua_add=7
    # Move exposed to presymptomatic
    exposed_to_presym_ind=np.logical_and(pop_matrix[:,1]==1+qua_add,pop_matrix[:,3]>=np.floor(0.5*pop_matrix[:,2]))
    pop_matrix[exposed_to_presym_ind,1] =2+qua_add
    # Move presymptomatic to symptomatic but not yet severe.
    presymp_to_symp_ind = np.logical_and(pop_matrix[:,1]==2+qua_add,pop_matrix[:,3]>=pop_matrix[:,2])
    pop_matrix[presymp_to_symp_ind,1] = 3+qua_add
    pop_matrix[presymp_to_symp_ind,3] = 0
    # Move individuals with 6 days of symptoms to mild.
    symp_to_mild_ind=np.logical_and(pop_matrix[:,1]==(3+qua_add),pop_matrix[:,3]==6)
    pop_matrix[symp_to_mild_ind,1] = 4+qua_add
    #  symptomatic to the “mild” or “severe”
    asp = np.array([0,.000408,.0104,.0343,.0425,.0816,.118,.166,.184])        # Verity et al. hospitalisation.
    aspc = np.array([.0101,.0209,.0410,.0642,.0721,.2173,.2483,.6921,.6987])  # Verity et al. corrected for Tuite.
    AGE_BUCKET=9
    for buc in range(AGE_BUCKET):
        # Assign individuals with mild symptoms for six days, sick, between 10*sci and 10*sci+1 years old to severe and count as hospitalized.
        if buc==8:# For all individual with Age 80 and above.
            # Move individuals with Chronic diseases with 6 days of mild to severe.
            mild_to_severe_ind=np.logical_and.reduce((pop_matrix[:,1]==4+qua_add,pop_matrix[:,3]==6,pick_sick<asp[buc],pop_matrix[:,5]>=10*buc,pop_matrix[:,7]==0))
            thosp += np.sum(mild_to_severe_ind)
            pop_matrix[mild_to_severe_ind,1] = 5+qua_add
            # Move individuals with Chronic diseases with 6 days of mild to severe.
            mild_to_severe_chronic_ind=np.logical_and.reduce((pop_matrix[:,1]==4+qua_add,pop_matrix[:,3]==6,pick_sick<aspc[buc],pop_matrix[:,5]>=10*buc,pop_matrix[:,7]==1))
            thosp += np.sum(mild_to_severe_chronic_ind)
            pop_matrix[mild_to_severe_chronic_ind,1] = 5+qua_add
        else:
            # Move individuals with Non Chronic diseases with 6 days of mild to severe for current age group.
            mild_to_severe_ind=np.logical_and.reduce((pop_matrix[:,1]==4+qua_add,pop_matrix[:,3]==6,pick_sick<asp[buc],pop_matrix[:,5]>=10*buc, pop_matrix[:,5]<(10*buc+10, pop_matrix[:,7]==0)))
            thosp += np.sum(mild_to_severe_ind)
            pop_matrix[mild_to_severe_ind,1] = 5+qua_add
            # Move individuals with Chronic diseases with 6 days of mild to severe for current age group.
            mild_to_severe_chronic_ind=np.logical_and.reduce((pop_matrix[:,1]==4+qua_add,pop_matrix[:,3]==6,pick_sick<aspc[buc],pop_matrix[:,5]>=10*buc,pop_matrix[:,5]<(10*buc+10),pop_matrix[:,7]==1))
            thosp += np.sum(mild_to_severe_chronic_ind)
            pop_matrix[mild_to_severe_chronic_ind,1] = 5+qua_add
    # Move Mild Symptoms to recovered
    mild_to_recovered_ind = np.logical_and(pop_matrix[:,1]==(4+qua_add),mild_rec)
    pop_matrix[mild_to_recovered_ind,1] = 6+qua_add
    # Move Severe symptoms to recovered.
    severe_to_recovered_ind = np.logical_and(pop_matrix[:,1]==(5+qua_add),sev_rec)
    pop_matrix[severe_to_recovered_ind,1] = 6+qua_add
    return pop_matrix,thosp

def accumarray(subs, val):
    '''Construct Array with accumulation.
    https://www.mathworks.com/help/matlab/ref/accumarray.html'''
    return np.array([np.sum(val[np.where(subs==i)]) for i in np.unique(subs)])

def identify_contagious_active(pop_matrix):
    # values for pop_matrix[:,1]:
    # 0 'susceptible',
    # 1 'exposed',
    # 2 'presymptomatic',
    # 3 'symptomatic',
    # 4 'mild',
    # 5 'severe',
    # 6 'recovered',
    # 7 'qua_susceptible',
    # 8 'qua_exposed',
    # 9 'qua_presymptomatic',
    # 10'qua_symptomatic',
    # 11'qua_mild',
    # 12'qua_severe',
    # 13'qua_recovered'

    # contagious_hhl = people with 2-5 states
    # contagious_hhl_qua = people with 9-12 states
    # contagious_camp = symp+asymp or presymp or symp+teen
    # contagious_sitters = not wanderers
    # contagious_wanderers = wanderers
    # active_camp = symp+asymp or presymp or symp+teen or exposed+susep or recovered

    contagious_hhl = np.logical_and(pop_matrix[:,1]>1,pop_matrix[:,1]<6)
    contagious_hhl_qua = np.logical_and(pop_matrix[:,1]>8,pop_matrix[:,1]<13) 
    contagious_asymp=np.logical_and.reduce((pop_matrix[:,1]>2,pop_matrix[:,1]<5,pop_matrix[:,4]==1))
    contagious_presymp=(pop_matrix[:,1]==2)
    contagious_teen=np.logical_and.reduce((pop_matrix[:,1]>2,pop_matrix[:,1]<5,pop_matrix[:,5]<16))
    contagious_camp=np.logical_or.reduce((contagious_asymp,contagious_presymp,contagious_teen))
    contagious_sitters = np.logical_and(contagious_camp,pop_matrix[:,8]==False)
    contagious_wanderers = np.logical_and(contagious_camp,pop_matrix[:,8]==True)
    active_camp=np.logical_or.reduce((contagious_camp,pop_matrix[:,1]<2,pop_matrix[:,1]==6))
    assert(sum(contagious_camp)==(sum(contagious_sitters)+sum(contagious_wanderers)))
    return contagious_hhl,contagious_hhl_qua,contagious_camp,contagious_sitters,contagious_wanderers,active_camp

def infected_and_sum_by_households(pop_matrix, contagious):

    contagious_hhl = contagious[0]
    contagious_hhl_qua = contagious[1]
    contagious_camp = contagious[2]
    contagious_sitters = contagious[3]
    contagious_wanderers = contagious[4]
    active_camp = contagious[5]

    infh = accumarray(pop_matrix[:,0],contagious_hhl)   # All infected in house and at toilets, population 
    infhq = accumarray(pop_matrix[:,0],contagious_hhl_qua) # All infected in house, quarantine 
    infl = accumarray(pop_matrix[:,0],contagious_camp)   # presymptomatic and asymptomatic for food lines
    infls = accumarray(pop_matrix[:,0],contagious_sitters) # All sedentaries for local transmission
    inflw = accumarray(pop_matrix[:,0],contagious_wanderers) # All wanderers for local transmission
    allfl = accumarray(pop_matrix[:,0],active_camp)  # All people in food lines
    return infh,infhq,infl,infls,inflw,allfl

def infected_prob_inhhl(inf_prob_hhl,trans_prob_hhl):
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
    households_with_no_wanderers = contagious_households[3]
    households_with_wanderers = contagious_households[4]
    lr1_exp_contacts = neighbour_inter[:,:,0].dot(households_with_no_wanderers)+\
                       neighbour_inter[:,:,1].dot(households_with_wanderers)
    lr2_exp_contacts = neighbour_inter[:,:,1].dot(households_with_no_wanderers)+\
                       neighbour_inter[:,:,2].dot(households_with_wanderers)
    # But contacts are roughly Poisson distributed (assuming a large population), so transmission rates are:
    trans_for_lr1 = 1-np.exp(-lr1_exp_contacts*aip*tr)
    trans_for_lr2 = 1-np.exp(-lr2_exp_contacts*aip*tr)    
    # Now, assign the appropriate local transmission rates to each person.
    trans_local_inter = trans_for_lr1[pop_matrix[:,0].astype(int)]*(1-pop_matrix[:,8])+trans_for_lr2[pop_matrix[:,0].astype(int)]*(pop_matrix[:,8])
    return trans_local_inter

def prob_of_transmission_within_household(
        # Probability for members of each household to contract from their housemates
        infection_probability_per_person,
        # Probability for members of each household to contract during a toilet visit
        transmission_at_toilet,
        # Probability for members of each household to contract during a food line visit
        transmission_in_foodline
):
    probability_of_transmission_within_household = 1 - \
           (1 - infection_probability_per_person) * \
           (1 - transmission_at_toilet) * \
           (1 - transmission_in_foodline)
    return probability_of_transmission_within_household

def prob_of_transmission(
        # Probability for members of each household to contract from their housemates
        probability_of_transmission_within_household,
        # Probability for members of each household to contract during the movement (wandering around)
        probability_of_transmission_during_movement
):
    full_inf_prob = 1 - (
            (1 - probability_of_transmission_within_household) *
            (1 - probability_of_transmission_during_movement)
    )
    return full_inf_prob

def impose_infection(
        # The existing infection status
        infection_status,
        # New infections
        newly_infected
):
    infection_status += (1 - np.sign(infection_status)) * newly_infected
    return infection_status

def infection_transmission(
        # probability_of_transmission
        probability_of_transmission,
        # population_total
        population_total
):
    random_uniform = np.random.uniform(0, 1, population_total)
    newly_infected = probability_of_transmission > random_uniform
    return newly_infected


def impose_infection_in_quarantin(
        # Probability of infection per person in quaranteen
        infection_probability_per_person_in_quarantin,
        # The total number of people in the camp
        population_total
):
    random_uniform = np.random.uniform(0, 1, population_total)
    newly_infected_in_quarantine = infection_probability_per_person_in_quarantin > random_uniform
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
    # newly_infected = probability_of_transmission > np.random.uniform(0, 1, population_total)
    newly_infected = infection_transmission(probability_of_transmission, population_total)

    # Impose infections, population. Only infect susceptible individuals
    infection_status = impose_infection(infection_status, newly_infected)

    # Find new infections by person, quarantine
    newly_infected_in_quarantin = impose_infection_in_quarantin(
        infection_probability_per_person_in_quarantin[households],
        population_total)

    # Impose nfections, quarantine
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
        factor = pct_days_with_foodline_visits
    )

    # Households in quarantine don't get these exposures, but that is taken care of below
    # because this is applied only to susceptibles in the population with these, we can calculate
    # the probability of all transmissions that calculated at the household level.
    probability_of_transmission_within_household = prob_of_transmission_within_household(
        probability_of_transmission_in_household,
        probability_of_transmission_at_toilet,
        probability_of_transmission_in_foodline
    )

    return probability_of_transmission_within_household

def assign_new_infections(
        # people in the camp: a matrix (2D array) with the following columns:
        # 0.household, 1.disease, 2.dsymptom, 3.daycount, 4.new_asymp,
        # 5.age, 6.gender, 7.chronic, 8.wanderer
        population,
        # a matrix (2D array) of shared toilets at the household level.
        toilets_shared_by_households,
        # a matrix (2D array) of shared food points at the household level.
        foodpoints_shared_by_households,
        # the number of toilet visits per person per day
        toilet_visits_per_person_per_day,
        # the number of contacts per a toilet visit
        contacts_per_toilet_visit,
        # the number of food point visits per person per day
        foodline_visits_per_person_per_day,
        # the number of contacts per a food point visit
        contacts_per_foodline_visit,
        # percentage of food point visits ppd (once per day on 3 out of 4 days)
        pct_days_with_foodline_visits,
        # the initial transmission reduction
        initial_transmission_reduction,
        # a matrix (2D array) for distance in between households
        distance_between_households,
        # the probability of infecting each person in your household per day
        probability_of_infection_household,
        # the probability of infecting other people in the food line
        probability_of_infection_food_line,
        # the probability of infecting other people in the toilet
        probability_of_infection_toilet,
        # the probability of infecting other people moving about
        probability_of_infection_wandering
):

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
        distance_between_households,
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
        # Households with contagious person in quarantin
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

def assign_new_infections_for_agent(
        # people in the camp: a matrix (2D array) with the following columns:
        # 0.household, 1.disease, 2.dsymptom, 3.daycount, 4.new_asymp,
        # 5.age, 6.gender, 7.chronic, 8.wanderer
        agent,
        # a matrix (2D array) of shared toilets at the household level.
        toilets_shared_by_households,
        # a matrix (2D array) of shared food points at the household level.
        foodpoints_shared_by_households,
        # the number of toilet visits per person per day
        toilet_visits_per_person_per_day,
        # the number of contacts per a toilet visit
        contacts_per_toilet_visit,
        # the number of food point visits per person per day
        foodline_visits_per_person_per_day,
        # the number of contacts per a food point visit
        contacts_per_foodline_visit,
        # percentage of food point visits ppd (once per day on 3 out of 4 days)
        pct_days_with_foodline_visits,
        # the initial transmission reduction
        initial_transmission_reduction,
        # a matrix (2D array) for distance in between households
        distance_between_households,
        # the probability of infecting each person in your household per day
        probability_of_infection_household,
        # the probability of infecting other people in the food line
        probability_of_infection_food_line,
        # the probability of infecting other people in the toilet
        probability_of_infection_toilet,
        # the probability of infecting other people moving about
        probability_of_infection_wandering
):
    population = np.array([[
        agent.household,
        agent.disease,
        agent.dsymptom,
        agent.daycount,
        agent.new_asymp,
        agent.age,
        agent.gender,
        agent.chronic,
        agent.wanderer
    ]])

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
        toilets_shared_by_households, # need to call model level function
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
        distance_between_households,
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
        # Households with contagious person in quarantin
        contagious_households[1],
        probability_of_infection_household
    )

    ##########################################################
    # 3. ASSIGN NEW INFECTIONS

    population = update_infection_status(
        population, #should have hh + status
        probability_of_transmission,
        probability_of_transmission_in_quarantin
    )

    return population

def move_hhl_quarantine(pop_matrix, prob_spot_symp):
    # Individuals in camp with symptons spotted given some probability
    spot_symp = np.random.uniform(0, 1, pop_matrix.shape[0]) < prob_spot_symp

    # Current state of all individuals in camp
    states = pop_matrix[:,1]

    # Filter conditions for next operation
    symptomatic = states > 2              # Symptomatic individuals
    not_quarantined = states < 6          # Individuals not in quarantine
    not_new_asymp = pop_matrix[:,4] == 0  # Individuals who are not newly asymptomatic
    aged_above_15 = pop_matrix[:,5] >= 16 # Individuals aged 15 and above

    # Individuals not quarantined who should be...
    symp = np.logical_and.reduce((
        symptomatic,
        not_quarantined,
        not_new_asymp,
        aged_above_15))

    # ... of which those who discover they have symptons and should be quarantined
    spotted_per_day = spot_symp * symp

    # IDS of households containing quarantined individuals
    symp_house = pop_matrix[spotted_per_day==1,0]

    # Individuals in quarantined households
    qua_hh = np.in1d(pop_matrix[:,0], symp_house)

    # Update individuals in quarantined households to 'qua_*'
    pop_matrix[qua_hh,1] += 7

    return pop_matrix

def move_hhl_quarantine_for_agent(agent, prob_spot_symp):

    pop_matrix = np.array([[
        agent.household,
        agent.disease,
        agent.dsymptom,
        agent.daycount,
        agent.new_asymp,
        agent.age,
        agent.gender,
        agent.chronic,
        agent.wanderer
    ]])

    # Individuals in camp with symptons spotted given some probability
    spot_symp = np.random.uniform(0, 1, pop_matrix.shape[0]) < prob_spot_symp

    # Current state of all individuals in camp
    states = pop_matrix[:,1]

    # Filter conditions for next operation
    symptomatic = states > 2              # Symptomatic individuals
    not_quarantined = states < 6          # Individuals not in quarantine
    not_new_asymp = pop_matrix[:,4] == 0  # Individuals who are not newly asymptomatic
    aged_above_15 = pop_matrix[:,5] >= 16 # Individuals aged 15 and above

    # Individuals not quarantined who should be...
    symp = np.logical_and.reduce((
        symptomatic,
        not_quarantined,
        not_new_asymp,
        aged_above_15))

    # ... of which those who discover they have symptons and should be quarantined
    spotted_per_day = spot_symp * symp

    # IDS of households containing quarantined individuals
    symp_house = pop_matrix[spotted_per_day==1,0]
    
    # Individuals in quarantined households
    qua_hh = np.in1d(pop_matrix[:,0], symp_house)

    # Update individuals in quarantined households to 'qua_*'
    pop_matrix[qua_hh,1] += 7

    return pop_matrix

def distance_between_households(household_coordinates):
    """
    Calculate distance between households

    Parameters
    ----------
        household_coordinates : A 2D array containing (X, Y) coordinates of the households
    
    Returns
    -------
        out : 2D distance matrix where (i, j) element holds the Euclidean distance between household (i) and household (j)
    """

    # number of households
    num = household_coordinates.shape[0]

    # (X, Y) coordintes of the households
    X = household_coordinates[:, 0]
    Y = household_coordinates[:, 1]
    
    # calculate delta X and delta Y for Euclidean distance
    delta_x = np.tile(X, (num, 1)).T - np.tile(X, (num, 1)) # delta_x is a 2D array containing diff between X coordinates for each pair of households
    delta_y = np.tile(Y, (num, 1)).T - np.tile(Y, (num, 1)) # delta_y is a 2D array containing diff between Y coordinates for each pair of households

    # return Euclidean distance
    return np.sqrt(delta_x ** 2 + delta_y ** 2)

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
    ) #nan means no overlap in this case

    # angle between center of circle centered at (2) and point of intersections of circles centered at (1) and (2)
    angle2 = 2*np.arccos(
        np.nan_to_num(
            np.clip(
                ( d**2 + r**2 - R**2 ) / ( 2*d*r ),
                a_min=None,
                a_max=1
            )
        )
    ) #nan means no overlap in this case
    
    area_sector1 = 0.5 * (r**2) * angle2
    area_sector2 = 0.5 * (R**2) * angle1

    area_overlap12 = np.nan_to_num(area_sector1 + area_sector2 - r * np.sin(angle2/2) * d)
    
    relative_encounter12 = area_overlap12 / (math.pi**2 * R**2 * r**2)
    
    return relative_encounter12

