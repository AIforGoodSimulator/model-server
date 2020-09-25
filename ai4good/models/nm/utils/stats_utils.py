import numpy as np
import pandas as pd


# Death rate per age calculation - parameters found by fitting a sigmoid curve
A_MALES = -9.58814632
B_MALES = 0.61453804
A_FEMALES = -9.91023535
B_FEMALES = 0.47451181
A_HOSP = -10.94003941
B_HOSP = 6.36278121


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def deathrate_male(x):
    return sigmoid(np.sqrt(x) + A_MALES) * B_MALES


def deathrate_female(x):
    return sigmoid(np.sqrt(x) + A_FEMALES) * B_FEMALES


def get_deathrate(row):
    if row["sex"] == 0:
        return deathrate_male(row["age"])

    else:
        return deathrate_female(row["age"])


def sympt_prob(x):
    return 0.41731169 + -1.38217963e-02*x**1 + 5.19931131e-04*x**2 + -3.66388844e-06*x**3


def get_prob_symptomatic(row):
    if row["age"] > 79:
        return sympt_prob(79)
    return sympt_prob(row["age"])


def hosp_prob(x):
    return sigmoid(np.sqrt(x) + A_HOSP) * B_HOSP


def get_prob_hospitalisation(row):
    return min(1, hosp_prob(row["age"]))


def increase_population(old_pop, new_pop, pop_df_file, **kwargs):

    # This is the number of people we now need
    diff_pop = new_pop - old_pop

    # Dataframe including age (V1) and sex (V2)
    pop_df = pd.read_csv(pop_df_file)

    # Camp parameters
    camp_params = pd.read_csv(kwargs["camp_params_file"])

    # Get the age ranges and their respective prevalence in the camp
    age_ranges = list(camp_params["Age"].iloc[1:9])
    age_prevalence = list(camp_params["Population_structure"].iloc[1:9] / 100)

    # Get the sex prevalence given the data in the age_sex csv --> Reversed because 1 has a higher probability than 0
    sex_prevalence = list(pop_df["sex"].value_counts() / len(pop_df["sex"]))[::-1]

    # Sample age ranges and sex according to the previous prevalences
    new_pop_ranges = np.random.choice(range(len(age_ranges)), p=age_prevalence, size=diff_pop, replace=True)
    new_pop_sex = np.random.choice([0, 1], p=sex_prevalence, size=diff_pop, replace=True)

    # We will store the actual new ages here
    new_pop_ages = []

    # For every new member, we will get their age
    for i in range(diff_pop):

        # Get the age range of the selected new person
        age_range = age_ranges[new_pop_ranges[i]].split("-")

        # Check if the range is 70+ --> Upper bound is currently 90
        if "70+" in age_range:
            person_age = np.random.choice(range(70, 90))
        else:
            person_age = np.random.choice(range(int(age_range[0]), int(age_range[1]) + 1))

        new_pop_ages.append(person_age)

    # Update the dataframe
    new_rows = [{"age": a, "sex": s} for a, s in zip(new_pop_ages, new_pop_sex)]
    pop_df = pop_df.append(new_rows, ignore_index=True)

    return pop_df


def sample_population(n_sample, pop_df_file):
    # Dataframe including age (V1) and sex (V2)
    pop_df = pd.read_csv(pop_df_file)

    # Sample only the number of people in isoboxes (n_samples)
    pop_df['death_rate'] = pop_df.apply(lambda row: get_deathrate(row), axis=1)
    pop_df['prob_symptomatic'] = pop_df.apply(lambda row: get_prob_symptomatic(row), axis=1)
    pop_df['prob_hospitalisation'] = pop_df.apply(lambda row: get_prob_hospitalisation(row), axis=1)
    sample = pop_df.sample(n=n_sample, random_state=69420)
    # print(sample['death_rate'].values.shape)

    return sample
