#from ai4good.models.cm.initialise_parameters import params, control_data, categories, calculated_categories, change_in_categories

from ai4good.models.cm.initialise_parameters import Parameters
from math import ceil, floor
import numpy as np
from scipy.integrate import ode
import pandas as pd
import statistics
import logging
from tqdm import tqdm
import dask
from dask.diagnostics import ProgressBar


def timing_function(t,time_vector):
    for ii in range(ceil(len(time_vector)/2)):
        if t>=time_vector[2*ii] and t<time_vector[2*ii+1]:
            return True
    # if wasn't in any of these time interval
    return False

##
# -----------------------------------------------------------------------------------
##


class Simulator:

    def __init__(self, params: Parameters):
        self.params = params

    def ode_system2d(self, t, y,  # state of system
                   infection_matrix, age_categories, symptomatic_prob, hospital_prob, critical_prob, beta,  # params
                   latentRate, removalRate, hospRate, deathRateICU, deathRateNoIcu,  # more params
                   better_hygiene, remove_symptomatic, remove_high_risk, ICU_capacity  # control
                   ):
        ##
        params = self.params

        y2d = y.reshape(age_categories, self.params.number_compartments).T
        dydt2d = np.zeros(y2d.shape)

        index_I = params.categories['I']['index']
        index_S = params.categories['S']['index']
        index_E = params.categories['E']['index']
        index_C = params.categories['C']['index']
        index_Q = params.categories['Q']['index']
        index_U = params.categories['U']['index']
        index_H = params.categories['H']['index']
        index_A = params.categories['A']['index']

        I_vec = y2d[index_I, :]
        H_vec = y2d[index_H, :]
        A_vec = y2d[index_A, :]
        C_vec = y2d[index_C, :]
        S_vec = y2d[index_S, :]

        E_latent = latentRate * y2d[index_E, :]
        I_removed = removalRate * I_vec
        Q_quarantined = params.quarant_rate * y2d[index_Q, :]

        total_I = sum(I_vec)
        total_H = sum(H_vec)

        # better hygiene
        if timing_function(t, better_hygiene['timing']):  # control in place
            control_factor = better_hygiene['value']
        else:
            control_factor = 1

        # removing symptomatic individuals
        if timing_function(t, remove_symptomatic['timing']):  # control in place
            remove_symptomatic_rate = min(total_I, remove_symptomatic['rate'])  # if total_I too small then can't take this many off site at once
        else:
            remove_symptomatic_rate = 0

        first_high_risk_category_n = age_categories - remove_high_risk['n_categories_removed']
        S_removal = sum(y2d[index_S, first_high_risk_category_n:])  # add all old people to remove
        # removing symptomatic individuals
        # these are put into Q ('quarantine');
        quarantine_sicks = (remove_symptomatic_rate / total_I) * I_vec  # no age bias in who is moved

        # removing susceptible high risk individuals
        # these are moved into O ('offsite')
        high_risk_people_removal_rates = np.zeros(age_categories);
        if timing_function(t, remove_high_risk['timing']):
            high_risk_people_removal_rates[first_high_risk_category_n:] = min(remove_high_risk['rate'],
                                      S_removal)  # only removing high risk (within time control window). Can't remove more than we have

        # ICU capacity
        if total_H > 0: # can't divide by 0
            hospitalized_on_icu = ICU_capacity['value'] / total_H * H_vec
            # ICU beds allocated on a first come, first served basis based on the numbers in hospital
        else:
            hospitalized_on_icu = np.full(age_categories, ICU_capacity['value'])

        # ODE system:
        # S
        infection_I = np.dot(infection_matrix, I_vec)
        infection_A = np.dot(infection_matrix, A_vec)
        infection_total = (infection_I + params.AsymptInfectiousFactor * infection_A)
        offsite = high_risk_people_removal_rates / S_removal * S_vec
        dydt2d[index_S, :] = (- control_factor * beta * S_vec * infection_total - offsite)

        # E
        dydt2d[index_E, :] = (control_factor * beta * S_vec * infection_total - E_latent)

        # I
        dydt2d[index_I, :] = ((1 - symptomatic_prob) * E_latent - I_removed- quarantine_sicks)

        # A
        A_removed = removalRate * A_vec
        dydt2d[index_A, :] = (symptomatic_prob * E_latent- A_removed)

        # H
        dydt2d[index_H, :] = (hospital_prob * I_removed - hospRate * H_vec
                + deathRateICU * (1 - params.death_prob_with_ICU) *
                np.minimum(C_vec, hospitalized_on_icu)  # recovered from ICU
                + hospital_prob * Q_quarantined          # proportion of removed people who were hospitalised once returned
        )

        # Critical care (ICU)
        deaths_on_icu = deathRateICU * C_vec
        without_deaths_on_icu = C_vec - deaths_on_icu
        needing_care = hospRate * critical_prob * H_vec # number needing care

        # number who get icu care (these entered category C)
        icu_cared = np.minimum(needing_care, hospitalized_on_icu - without_deaths_on_icu)

        # amount entering is minimum of: amount of beds available**/number needing it
        # **including those that will be made available by new deaths
        # without ICU treatment
        dydt2d[index_C, :] = (icu_cared - deaths_on_icu)

        # Uncared - no ICU
        deaths_without_icu = deathRateNoIcu * y2d[index_U, :] # died without ICU treatment (all cases that don't get treatment die)
        dydt2d[index_U, :] = (needing_care - icu_cared - deaths_without_icu)  # without ICU treatment

        # R
        # proportion of removed people who recovered once returned
        dydt2d[params.categories['R']['index'], :] = (
                (1 - hospital_prob) * I_removed + A_removed + hospRate * (1 - critical_prob) * H_vec + (1 - hospital_prob) * Q_quarantined
        )

        # D
        dydt2d[params.categories['D']['index'], :] = (
                    deaths_without_icu + params.death_prob_with_ICU * deaths_on_icu # died despite attempted ICU treatment
        )
        # O
        dydt2d[params.categories['O']['index'], :] = offsite

        # Q
        dydt2d[index_Q, :] = quarantine_sicks - Q_quarantined

        return dydt2d.T.reshape(y.shape)

    # deprecated: old slow function. you should use ode_system2d
    def ode_system(self, t, y,  # state of system
                   infection_matrix, age_categories, symptomatic_prob, hospital_prob, critical_prob, beta,  # params
                   latentRate, removalRate, hospRate, deathRateICU, deathRateNoIcu,  # more params
                   better_hygiene, remove_symptomatic, remove_high_risk, ICU_capacity  # control
                   ):
        ##
        params = self.params
        dydt = np.zeros(y.shape)

        I_vec = [y[params.categories['I']['index'] + i * params.number_compartments] for i in range(age_categories)]
        H_vec = [y[params.categories['H']['index'] + i * params.number_compartments] for i in range(age_categories)]
        A_vec = [y[params.categories['A']['index'] + i * params.number_compartments] for i in range(age_categories)]

        total_I = sum(I_vec)

        # better hygiene
        if timing_function(t, better_hygiene['timing']):  # control in place
            control_factor = better_hygiene['value']
        else:
            control_factor = 1

        # removing symptomatic individuals
        if timing_function(t, remove_symptomatic['timing']):  # control in place
            remove_symptomatic_rate = min(total_I, remove_symptomatic[
                'rate'])  # if total_I too small then can't take this many off site at once
        else:
            remove_symptomatic_rate = 0

        S_removal = 0
        for i in range(age_categories - remove_high_risk['n_categories_removed'], age_categories):
            S_removal += y[params.categories['S']['index'] + i * params.number_compartments]  # add all old people to remove

        for i in range(age_categories):
            # removing symptomatic individuals
            # these are put into Q ('quarantine');
            quarantine_sick = remove_symptomatic_rate * y[
                params.categories['I']['index'] + i * params.number_compartments] / total_I  # no age bias in who is moved

            # removing susceptible high risk individuals
            # these are moved into O ('offsite')
            if i in range(age_categories - remove_high_risk['n_categories_removed'],
                          age_categories) and timing_function(t, remove_high_risk['timing']):
                remove_high_risk_people = min(remove_high_risk['rate'],
                                              S_removal)  # only removing high risk (within time control window). Can't remove more than we have
            else:
                remove_high_risk_people = 0

            # ICU capacity
            if sum(H_vec)>0: # can't divide by 0
                ICU_for_this_age = ICU_capacity['value'] * y[params.categories['H']['index'] + i*params.number_compartments]/sum(H_vec)
                # ICU beds allocated on a first come, first served basis based on the numbers in hospital
            else:
                ICU_for_this_age = ICU_capacity['value']

            # ODE system:
            # S
            dydt[params.categories['S']['index'] + i * params.number_compartments] = (
                        - y[params.categories['S']['index'] + i * params.number_compartments] * control_factor * beta * (
                            np.dot(infection_matrix[i, :], I_vec) + params.AsymptInfectiousFactor * np.dot(
                        infection_matrix[i, :], A_vec))
                        - remove_high_risk_people * y[params.categories['S']['index'] + i * params.number_compartments] / S_removal)
            # E
            dydt[params.categories['E']['index'] + i * params.number_compartments] = (
                        y[params.categories['S']['index'] + i * params.number_compartments] * control_factor * beta * (
                            np.dot(infection_matrix[i, :], I_vec) + params.AsymptInfectiousFactor * np.dot(
                        infection_matrix[i, :], A_vec))
                        - latentRate * y[params.categories['E']['index'] + i * params.number_compartments])
            # I
            dydt[params.categories['I']['index'] + i * params.number_compartments] = (
                        latentRate * (1 - symptomatic_prob[i]) * y[params.categories['E']['index'] + i * params.number_compartments]
                        - removalRate * y[params.categories['I']['index'] + i * params.number_compartments]
                        - quarantine_sick
                        )
            # A
            dydt[params.categories['A']['index'] + i * params.number_compartments] = (
                        latentRate * symptomatic_prob[i] * y[params.categories['E']['index'] + i * params.number_compartments]
                        - removalRate * y[params.categories['A']['index'] + i * params.number_compartments])
            # H
            dydt[params.categories['H']['index'] + i * params.number_compartments] = (
                        removalRate * (hospital_prob[i]) * y[params.categories['I']['index'] + i * params.number_compartments]
                        - hospRate * y[params.categories['H']['index'] + i * params.number_compartments]
                        #  + deathRateNoIcu * (1 - params.death_prob) * max(0,y[params.categories['C']['index'] + i*params.number_compartments] - ICU_for_this_age) # recovered despite no ICU (0, since now assume death_prob is 1)
                        + deathRateICU * (1 - params.death_prob_with_ICU) * min(
                    y[params.categories['C']['index'] + i * params.number_compartments], ICU_for_this_age)  # recovered from ICU
                        + (hospital_prob[i]) * params.quarant_rate * y[params.categories['Q']['index'] + i * params.number_compartments]
                        # proportion of removed people who were hospitalised once returned
                        )
            # Critical care (ICU)
            dydt[params.categories['C']['index'] + i*params.number_compartments] = ( min(hospRate  * (critical_prob[i]) * y[params.categories['H']['index'] + i*params.number_compartments], ICU_for_this_age - y[params.categories['C']['index'] + i*params.number_compartments] + deathRateICU * y[params.categories['C']['index'] + i*params.number_compartments]
                # with ICU treatment 
                # max(0, 
                # )
                )
                # amount entering is minimum of: amount of beds available**/number needing it
                # **including those that will be made available by new deaths
                - deathRateICU * y[params.categories['C']['index'] + i*params.number_compartments]  # with ICU treatment
                )

            # Uncared - no ICU
            dydt[params.categories['U']['index'] + i * params.number_compartments] = (hospRate  * (critical_prob[i]) * y[params.categories['H']['index'] + i * params.number_compartments] # number needing care
                - min(hospRate  * (critical_prob[i]) * y[params.categories['H']['index'] + i*params.number_compartments],
                ICU_for_this_age - y[params.categories['C']['index'] + i*params.number_compartments]
                + deathRateICU * y[params.categories['C']['index'] + i*params.number_compartments]
                # max(0,
                # )
                ) # minus number who get it (these entered category C)
                - deathRateNoIcu * y[params.categories['U']['index'] + i*params.number_compartments] # without ICU treatment
                )

            # R
            dydt[params.categories['R']['index'] + i * params.number_compartments] = (
                        removalRate * (1 - hospital_prob[i]) * y[params.categories['I']['index'] + i * params.number_compartments]
                        + removalRate * y[params.categories['A']['index'] + i * params.number_compartments]
                        + hospRate * (1 - critical_prob[i]) * y[params.categories['H']['index'] + i * params.number_compartments]
                        + (1 - hospital_prob[i]) * params.quarant_rate * y[
                            params.categories['Q']['index'] + i * params.number_compartments]
                        # proportion of removed people who recovered once returned
                        )

            # D
            dydt[params.categories['D']['index'] + i * params.number_compartments] = (deathRateNoIcu * y[
                params.categories['U']['index'] + i * params.number_compartments]  # died without ICU treatment (all cases that don't get treatment die)
                                                                   + deathRateICU * (params.death_prob_with_ICU) * y[
                                                                       params.categories['C']['index'] + i * params.number_compartments]
                                                                   # died despite attempted ICU treatment
                                                                   )
            # O
            dydt[params.categories['O']['index'] + i * params.number_compartments] = remove_high_risk_people * y[
                params.categories['S']['index'] + i * params.number_compartments] / S_removal

            # Q
            dydt[params.categories['Q']['index'] + i * params.number_compartments] = quarantine_sick - params.quarant_rate * y[
                params.categories['Q']['index'] + i * params.number_compartments]
        return dydt
    ##
    #--------------------------------------------------------------------
    ##

    def run_model(self, T_stop, beta, latent_rate=None, removal_rate=None, hosp_rate=None, death_rate_ICU=None, death_rate_no_ICU=None):
        population = self.params.population
        population_frame = self.params.population_frame
        control_dict = self.params.control_dict
        if latent_rate is None:
            latent_rate = self.params.latent_rate
        if removal_rate is None:
            removal_rate = self.params.removal_rate
        if hosp_rate is None:
            hosp_rate = self.params.hosp_rate
        if death_rate_ICU is None:
            death_rate_ICU = self.params.death_rate_with_ICU
        if death_rate_no_ICU is None:
            death_rate_no_ICU = self.params.death_rate  # more params

        seirvars = np.zeros((self.params.number_compartments, 1));

        seirvars[self.params.categories['I']['index'], 0] = 1 / population  # sympt
        seirvars[self.params.categories['A']['index'], 0] = 1 / population  # asympt

        seirvars[self.params.categories['S']['index'], 0] = 1-seirvars.sum()

        age_categories = int(population_frame.shape[0])

        population_vector = population_frame.Population_structure.to_numpy()

        y1 = np.dot(seirvars, population_vector.reshape(1, age_categories)/100)
        # initial conditions
        y0 = y1.T.reshape(self.params.number_compartments*age_categories)

        symptomatic_prob = np.asarray(population_frame.p_symptomatic)
        hospital_prob = np.asarray(population_frame.p_hospitalised)
        critical_prob = np.asarray(population_frame.p_critical)

        sol = ode(self.ode_system2d).set_f_params(
            self.params.infection_matrix,
            age_categories,
            symptomatic_prob,
            hospital_prob,
            critical_prob,
            beta, # params
            latent_rate,removal_rate,hosp_rate,death_rate_ICU,death_rate_no_ICU, # more params
            control_dict['better_hygiene'],control_dict['remove_symptomatic'],control_dict['remove_high_risk'],control_dict['ICU_capacity']
        )


        tim = np.linspace(0,T_stop, T_stop+1) # 1 time value per day

        sol.set_initial_value(y0,tim[0])

        y_out = np.zeros((len(y0),len(tim)))

        i2 = 0
        y_out[:,0] = sol.y
        for t in tim[1:]:
                if sol.successful():
                    sol.integrate(t)
                    i2=i2+1
                    y_out[:,i2] = sol.y
                else:
                    raise RuntimeError('ode solver unsuccessful')

        y_plot = np.zeros((len(self.params.categories.keys()), len(tim) ))
        for name in self.params.calculated_categories:

            y_plot[self.params.categories[name]['index'],:] = y_out[self.params.categories[name]['index'],:]
            for i in range(1, population_frame.shape[0]): # age_categories
                y_plot[self.params.categories[name]['index'],:] = y_plot[self.params.categories[name]['index'],:] + y_out[self.params.categories[name]['index'] + i*self.params.number_compartments,:]

        for name in self.params.change_in_categories: # daily change in
            name_changed_var = name[-1] # name of the variable we want daily change of
            y_plot[self.params.categories[name]['index'],:] = np.concatenate([[0],np.diff(y_plot[self.params.categories[name_changed_var]['index'],:])])

        # finally, 
        E = y_plot[self.params.categories['CE']['index'],:]
        I = y_plot[self.params.categories['CI']['index'],:]
        A = y_plot[self.params.categories['CA']['index'],:]

        y_plot[self.params.categories['Ninf']['index'],:] = [E[i] + I[i] + A[i] for i in range(len(E))] # change in total number of people with active infection

        return {'y': y_out,'t': tim, 'y_plot': y_plot}

#--------------------------------------------------------------------

    def generate_percentiles(self, sols):
        n_time_points = len(sols[0]['t'])

        y_plot = np.zeros((len(self.params.categories.keys()), len(sols) , n_time_points ))

        for k, sol in enumerate(sols):
            sol['y'] = np.asarray(sol['y'])
            for name in self.params.categories.keys():
                y_plot[self.params.categories[name]['index'],k,:] = sol['y_plot'][self.params.categories[name]['index']]

        y_L95, y_U95, y_LQ, y_UQ, y_median = [np.zeros((len(self.params.categories.keys()),n_time_points)) for i in range(5)]

        for name in self.params.categories.keys():
            y_L95[self.params.categories[name]['index'],:] = np.asarray([ np.percentile(y_plot[self.params.categories[name]['index'],:,i],2.5) for i in range(n_time_points) ])
            y_LQ[self.params.categories[name]['index'],:] = np.asarray([ np.percentile(y_plot[self.params.categories[name]['index'],:,i],25) for i in range(n_time_points) ])
            y_UQ[self.params.categories[name]['index'],:] = np.asarray([ np.percentile(y_plot[self.params.categories[name]['index'],:,i],75) for i in range(n_time_points) ])
            y_U95[self.params.categories[name]['index'],:] = np.asarray([ np.percentile(y_plot[self.params.categories[name]['index'],:,i],97.5) for i in range(n_time_points) ])

            y_median[self.params.categories[name]['index'],:] = np.asarray([statistics.median(y_plot[self.params.categories[name]['index'],:,i]) for i in range(n_time_points) ])
        return [y_U95, y_UQ, y_LQ, y_L95, y_median]

    def simulate_range_of_R0s(self, t_stop=200): # gives solution for middle R0, as well as solutions for a range of R0s between an upper and lower bound
        beta_list = self.params.im_beta_list
        largest_eigenvalue = self.params.largest_eigenvalue

        sols = []
        sols_raw = {}
        for beta in beta_list:
            result = self.run_model(T_stop=t_stop, beta=beta)
            sols.append(result)
            sols_raw[beta*largest_eigenvalue/self.params.removal_rate]=result

        [y_U95, y_UQ, y_LQ, y_L95, y_median] = self.generate_percentiles(sols)

        standard_sol = [self.run_model(T_stop=t_stop, beta=self.params.beta_list[1])]

        return sols_raw, standard_sol, [y_U95, y_UQ, y_LQ, y_L95, y_median]


    def simulate_over_parameter_range(self, numberOfIterations, t_stop=200):

        sols = []
        config_dict = []
        sols_raw = {}
        for ii in tqdm(range(min(numberOfIterations,len(self.params.generated_disease_vectors)))):
            latentRate  = 1/self.params.generated_disease_vectors.LatentPeriod[ii]
            removalRate = 1/self.params.generated_disease_vectors.RemovalPeriod[ii]

            beta        = removalRate*self.params.generated_disease_vectors.R0[ii]/self.params.largest_eigenvalue

            hospRate       = 1/self.params.generated_disease_vectors.HospPeriod[ii]
            deathRateICU   = 1/self.params.generated_disease_vectors.DeathICUPeriod[ii]
            deathRateNoIcu = 1/self.params.generated_disease_vectors.DeathNoICUPeriod[ii]

            result = self.run_model(T_stop=t_stop, beta=beta,
                                    latent_rate=latentRate,
                                    removal_rate=removalRate,
                                    hosp_rate=hospRate,
                                    death_rate_ICU=deathRateICU,
                                    death_rate_no_ICU=deathRateNoIcu
                                    )
            sols.append(result)
            Dict = dict(beta       = beta,
                    latentRate     = latentRate,
                    removalRate    = removalRate,
                    hospRate       = hospRate,
                    deathRateICU   = deathRateICU,
                    deathRateNoIcu = deathRateNoIcu
                    )
            config_dict.append(Dict)
            sols_raw[(self.params.generated_disease_vectors.R0[ii],latentRate,removalRate,hospRate,deathRateICU,deathRateNoIcu)]=result

        [y_U95, y_UQ, y_LQ, y_L95, y_median] = self.generate_percentiles(sols)

        # standard run
        StandardSol = [self.run_model(T_stop=t_stop, beta=self.params.beta_list[1])]

        return sols_raw, StandardSol, [y_U95, y_UQ, y_LQ, y_L95, y_median], config_dict


    def simulate_over_parameter_range_parallel(self, numberOfIterations, t_stop, n_processes):
        logging.info(f"Running parallel simulation with {n_processes} processes")
        lazy_sols = []
        config_dict = []
        sols_raw = {}

        for ii in range(min(numberOfIterations,len(self.params.generated_disease_vectors))):
            latentRate  = 1/self.params.generated_disease_vectors.LatentPeriod[ii]
            removalRate = 1/self.params.generated_disease_vectors.RemovalPeriod[ii]

            beta        = removalRate*self.params.generated_disease_vectors.R0[ii]/self.params.largest_eigenvalue

            hospRate       = 1/self.params.generated_disease_vectors.HospPeriod[ii]
            deathRateICU   = 1/self.params.generated_disease_vectors.DeathICUPeriod[ii]
            deathRateNoIcu = 1/self.params.generated_disease_vectors.DeathNoICUPeriod[ii]

            lazy_result = dask.delayed(self.run_model)(T_stop=t_stop, beta=beta,
                                    latent_rate=latentRate,
                                    removal_rate=removalRate,
                                    hosp_rate=hospRate,
                                    death_rate_ICU=deathRateICU,
                                    death_rate_no_ICU=deathRateNoIcu
                                    )
            lazy_sols.append(lazy_result)
            #sols.append(result)

            Dict = dict(beta       = beta,
                    latentRate     = latentRate,
                    removalRate    = removalRate,
                    hospRate       = hospRate,
                    deathRateICU   = deathRateICU,
                    deathRateNoIcu = deathRateNoIcu
                    )
            config_dict.append(Dict)
            #TODO: sols_raw[(self.params.generated_disease_vectors.R0[ii],latentRate,removalRate,hospRate,deathRateICU,deathRateNoIcu)]=result

        with dask.config.set(scheduler='processes', num_workers=n_processes):
            with ProgressBar():
                sols = dask.compute(*lazy_sols)

        for ii in range(min(numberOfIterations, len(self.params.generated_disease_vectors))):
            dct = config_dict[ii]
            sols_raw[(self.params.generated_disease_vectors.R0[ii], dct['latentRate'], dct['removalRate'],
                      dct['hospRate'], dct['deathRateICU'], dct['deathRateNoIcu'])] = sols[ii]

        [y_U95, y_UQ, y_LQ, y_L95, y_median] = self.generate_percentiles(sols)

        # standard run
        StandardSol = [self.run_model(T_stop=t_stop, beta=self.params.beta_list[1])]

        return sols_raw, StandardSol, [y_U95, y_UQ, y_LQ, y_L95, y_median], config_dict


def generate_csv(data_to_save, params: Parameters,  input_type=None, time_vec=None) -> pd.DataFrame:
    population_frame = params.population_frame
    category_map = {}
    for key in params.categories.keys():
        category_map[str(params.categories[key]['index'])] = key

    if input_type=='percentile':
        csv_sol = np.transpose(data_to_save)

        solution_csv = pd.DataFrame(csv_sol)


        col_names = []
        for i in range(csv_sol.shape[1]):
            col_names.append(params.categories[category_map[str(i)]]['longname'])


        solution_csv.columns = col_names
        solution_csv['Time'] = time_vec
        # this is our dataframe to be saved

    elif input_type=='raw':

        final_frame=pd.DataFrame()

        for key, value in tqdm(data_to_save.items()):
            csv_sol = np.transpose(value['y']) # age structured

            solution_csv = pd.DataFrame(csv_sol)

            # setup column names
            col_names = []
            number_categories_with_age = csv_sol.shape[1]
            for i in range(number_categories_with_age):
                ii = i % params.number_compartments
                jj = floor(i/params.number_compartments)

                col_names.append(params.categories[category_map[str(ii)]]['longname'] +  ': ' + str(np.asarray(population_frame.Age)[jj]) )

            solution_csv.columns = col_names
            solution_csv['Time'] = value['t']

            for j in range(len(params.categories.keys())): # params.number_compartments
                solution_csv[params.categories[category_map[str(j)]]['longname']] = value['y_plot'][j] # summary/non age-structured

            (R0,latentRate,removalRate,hospRate,deathRateICU,deathRateNoIcu)=key
            solution_csv['R0']=[R0]*solution_csv.shape[0]
            solution_csv['latentRate']=[latentRate]*solution_csv.shape[0]
            solution_csv['removalRate']=[removalRate]*solution_csv.shape[0]
            solution_csv['hospRate']=[hospRate]*solution_csv.shape[0]
            solution_csv['deathRateICU']=[deathRateICU]*solution_csv.shape[0]
            solution_csv['deathRateNoIcu']=[deathRateNoIcu]*solution_csv.shape[0]
            final_frame=pd.concat([final_frame, solution_csv], ignore_index=True)

        solution_csv=final_frame
        #this is our dataframe to be saved

    elif input_type=='solution':
        csv_sol = np.transpose(data_to_save[0]['y']) # age structured

        solution_csv = pd.DataFrame(csv_sol)

        # setup column names
        col_names = []
        number_categories_with_age = csv_sol.shape[1]
        for i in range(number_categories_with_age):
            ii = i % params.number_compartments
            jj = floor(i/params.number_compartments)

            col_names.append(params.categories[category_map[str(ii)]]['longname'] +  ': ' + str(np.asarray(params.population_frame.Age)[jj]) )

        solution_csv.columns = col_names
        solution_csv['Time'] = data_to_save[0]['t']

        for j in range(len(params.categories.keys())): # params.number_compartments
            solution_csv[params.categories[category_map[str(j)]]['longname']] = data_to_save[0]['y_plot'][j] # summary/non age-structured
        # this is our dataframe to be saved


    # save it
    #solution_csv.to_csv(os.path.join(os.path.dirname(cwd),'CSV_output/' + filename + '.csv' ))

    return solution_csv
