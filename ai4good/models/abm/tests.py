import ai4good.models.abm.abm
import numpy as np
import pickle

def test_assign_new_infections():

    # people in the camp: a matrix (2D array) with the following columns:
    # 0.household, 1.disease, 2.dsymptom, 3.daycount, 4.new_asymp,
    # 5.age, 6.gender, 7.chronic, 8.wanderer
    population = np.array([
        [0.,0.,8.17139954,0.,0.,29.5,1.,0.,1.],
        [1.,0.,6.71160589,0.,0., 8.8,1.,0.,0.],
        [2.,1.,7.36001027,0.,0.,15.3,1.,0.,1.],
        [2.,0.,8.34934896,0.,0.,19.7,1.,0.,1.],
        [3.,0.,2.89684634,0.,0., 9.1,0.,0.,0.],
        [3.,0.,5.79198912,0.,0.,24.7,0.,1.,0.],
        [3.,0.,7.18514551,0.,0., 2.1,1.,0.,0.],
        [4.,0.,8.53961958,0.,0.,50.7,0.,0.,0.],
        [5.,0.,9.36731656,0.,0.,22.6,1.,0.,1.],
        [5.,0.,9.04148481,0.,0.,60.4,0.,0.,0.]
        ])
    # a matrix (2D array) of shared toilets at the household level.
    toilets_shared_by_households = np.array([
        [0., 0., 0., 0., 0., 1.],
        [0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 1., 1., 0.],
        [0., 0., 1., 0., 1., 0.],
        [0., 0., 1., 1., 0., 0.],
        [1., 0., 0., 0., 0., 0.]
    ])
    # a matrix (2D array) of shared food points at the household level.
    foodpoints_shared_by_households = np.array([
        [0., 1., 1., 1., 1., 1.],
        [1., 0., 1., 1., 1., 1.],
        [1., 1., 0., 1., 1., 1.],
        [1., 1., 1., 0., 1., 1.],
        [1., 1., 1., 1., 0., 1.],
        [1., 1., 1., 1., 1., 0.]
    ])
    # the number of toilet visits per person per day
    toilet_visits_per_person_per_day = 3
    # the number of contacts per a toilet visit
    contacts_per_toilet_visit = 2
    # the number of food point visits per person per day
    foodline_visits_per_person_per_day = 1
    # the number of contacts per a food point visit
    contacts_per_foodline_visit = 2
    # percentage of food point visits ppd (once per day on 3 out of 4 days)
    pct_days_with_foodline_visits = 3/4
    # the initial transmission reduction
    initial_transmission_reduction = 1
    # a matrix (2D array) for distance in between households
    distance_between_households = np.array([[
        [1., 0., 0.04],
        [0., 0., 0.],
        [0., 0., 0.],
        [0., 0., 0.],
        [0., 0., 0.],
        [0., 0., 0.]],

        [[0., 0., 0.],
         [1., 0., 0.04],
         [0., 0., 0.],
         [0., 0., 0.],
         [0., 0., 0.],
         [0., 0., 0.]],

        [[0., 0., 0.],
         [0., 0., 0.],
         [1., 0., 0.04],
         [0., 0., 0.00721094],
         [0., 0., 0.],
         [0., 0., 0.]],

        [[0., 0., 0.],
         [0., 0., 0.],
         [0., 0., 0.00721094],
         [1., 0., 0.04],
         [0., 0., 0.],
         [0., 0., 0.]],

        [[0., 0., 0.],
         [0., 0., 0.],
         [0., 0., 0.],
         [0., 0., 0.],
         [1., 0., 0.04],
         [0., 0., 0.]],

        [[0., 0., 0.],
         [0., 0., 0.],
         [0., 0., 0.],
         [0., 0., 0.],
         [0., 0., 0.],
         [1., 0., 0.04]
    ]])

    # the probability of infecting each person in your household per day
    probability_of_infection_household = 0.33
    # the probability of infecting other people in the food line
    probability_of_infection_food_line = 0.407
    # the probability of infecting other people in the toilet
    probability_of_infection_toilet = 0.099
    # the probability of infecting other people moving about
    probability_of_infection_wandering = 0.017

    expected = np.array([
        [0., 0., 8.17139954, 0., 0., 29.5, 1., 0., 1.],
        [1., 0., 6.71160589, 0., 0.,  8.8, 1., 0., 0.],
        [2., 1., 7.36001027, 0., 0., 15.3, 1., 0., 1.],
        [2., 0., 8.34934896, 0., 0., 19.7, 1., 0., 1.],
        [3., 0., 2.89684634, 0., 0.,  9.1, 0., 0., 0.],
        [3., 0., 5.79198912, 0., 0., 24.7, 0., 1., 0.],
        [3., 0., 7.18514551, 0., 0.,  2.1, 1., 0., 0.],
        [4., 0., 8.53961958, 0., 0., 50.7, 0., 0., 0.],
        [5., 0., 9.36731656, 0., 0., 22.6, 1., 0., 1.],
        [5., 0., 9.04148481, 0., 0., 60.4, 0., 0., 0.]
    ])

    result = abm.assign_new_infections(
        population,
        toilets_shared_by_households,
        foodpoints_shared_by_households,
        toilet_visits_per_person_per_day,
        contacts_per_toilet_visit,
        foodline_visits_per_person_per_day,
        contacts_per_foodline_visit,
        pct_days_with_foodline_visits,
        initial_transmission_reduction,
        distance_between_households,
        probability_of_infection_household,
        probability_of_infection_food_line,
        probability_of_infection_toilet,
        probability_of_infection_wandering
    )

    assert np.array_equal(expected, result)


def test_move_hhl_quarantine():
    population = np.array([
        [0.,0.,8.17139954,0.,0.,29.5,1.,0.,1.],
        [1.,0.,6.71160589,0.,0., 8.8,1.,0.,0.],
        [2.,1.,7.36001027,0.,0.,15.3,1.,0.,1.],
        [2.,0.,8.34934896,0.,0.,19.7,1.,0.,1.],
        [3.,0.,2.89684634,0.,0., 9.1,0.,0.,0.],
        [3.,0.,5.79198912,0.,0.,24.7,0.,1.,0.],
        [3.,0.,7.18514551,0.,0., 2.1,1.,0.,0.],
        [4.,0.,8.53961958,0.,0.,50.7,0.,0.,0.],
        [5.,4.,9.36731656,0.,0.,22.6,1.,0.,1.],
        [5.,0.,9.04148481,0.,0.,60.4,0.,0.,0.]
        ])

    expected = np.array([
        [0.,0.,8.17139954,0.,0.,29.5,1.,0.,1.],
        [1.,0.,6.71160589,0.,0., 8.8,1.,0.,0.],
        [2.,1.,7.36001027,0.,0.,15.3,1.,0.,1.],
        [2.,0.,8.34934896,0.,0.,19.7,1.,0.,1.],
        [3.,0.,2.89684634,0.,0., 9.1,0.,0.,0.],
        [3.,0.,5.79198912,0.,0.,24.7,0.,1.,0.],
        [3.,0.,7.18514551,0.,0., 2.1,1.,0.,0.],
        [4.,0.,8.53961958,0.,0.,50.7,0.,0.,0.],
        [5.,11.,9.36731656,0.,0.,22.6,1.,0.,1.],
        [5.,7.,9.04148481,0.,0.,60.4,0.,0.,0.]
        ])

    siprob = 1.0 # deterministic
    result = abm.move_hhl_quarantine(population, siprob)

    assert np.array_equal(expected, result)


def test_interaction_neighbours():
    #Coordinates of households
    household_coordinates =  np.array([
        [ 0.55438108,  0.36686012],
        [ 0.70129778,  0.60187273],
        [ 0.68618674,  0.70042585],
        [ 0.96297229,  0.81769586],
        [ 0.13318782,  0.43613684],
        [ 0.09827639,  0.70896226]
    ])
    #Smaller movement radius
    movement_radius_small = 0.02
    #Larger movement radius
    movement_radius_large = 0.1
    #Scale interactions
    lrtol = 0.02
    #households ethnic groups matrix
    ethcor = np.array([
        [ 1.,  1.,  1.,  1.,  1.,  1.],
        [ 1.,  1.,  1.,  1.,  1.,  1.],
        [ 1.,  1.,  1.,  1.,  1.,  1.],
        [ 1.,  1.,  1.,  1.,  1.,  1.],
        [ 1.,  1.,  1.,  1.,  1.,  1.],
        [ 1.,  1.,  1.,  1.,  1.,  1.]
    ])

    expected = np.array([
        [[ 1.        ,  0.        ,  0.04      ],
         [ 0.        ,  0.        ,  0.        ],
         [ 0.        ,  0.        ,  0.        ],
         [ 0.        ,  0.        ,  0.        ],
         [ 0.        ,  0.        ,  0.        ],
         [ 0.        ,  0.        ,  0.        ]],

        [[ 0.        ,  0.        ,  0.        ],
         [ 1.        ,  0.        ,  0.04      ],
         [ 0.        ,  0.01952447,  0.01570521],
         [ 0.        ,  0.        ,  0.        ],
         [ 0.        ,  0.        ,  0.        ],
         [ 0.        ,  0.        ,  0.        ]],

        [[ 0.        ,  0.        ,  0.        ],
         [ 0.        ,  0.01952447,  0.01570521],
         [ 1.        ,  0.        ,  0.04      ],
         [ 0.        ,  0.        ,  0.        ],
         [ 0.        ,  0.        ,  0.        ],
         [ 0.        ,  0.        ,  0.        ]],

        [[ 0.        ,  0.        ,  0.        ],
         [ 0.        ,  0.        ,  0.        ],
         [ 0.        ,  0.        ,  0.        ],
         [ 1.        ,  0.        ,  0.04      ],
         [ 0.        ,  0.        ,  0.        ],
         [ 0.        ,  0.        ,  0.        ]],

        [[ 0.        ,  0.        ,  0.        ],
         [ 0.        ,  0.        ,  0.        ],
         [ 0.        ,  0.        ,  0.        ],
         [ 0.        ,  0.        ,  0.        ],
         [ 1.        ,  0.        ,  0.04      ],
         [ 0.        ,  0.        ,  0.        ]],

        [[ 0.        ,  0.        ,  0.        ],
         [ 0.        ,  0.        ,  0.        ],
         [ 0.        ,  0.        ,  0.        ],
         [ 0.        ,  0.        ,  0.        ],
         [ 0.        ,  0.        ,  0.        ],
         [ 1.        ,  0.        ,  0.04      ]]
    ])

    result = abm.interaction_neighbours(
            household_coordinates,
            movement_radius_small,
            movement_radius_large,
            lrtol,
            ethcor)

#    assert np.array_equal(expected, result)
    np.testing.assert_allclose(result, expected)

test_assign_new_infections()
test_move_hhl_quarantine()
test_interaction_neighbours()
