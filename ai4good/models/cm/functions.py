#from ai4good.models.cm.initialise_parameters import params, control_data, categories, calculated_categories, change_in_categories

from ai4good.models.cm.initialise_parameters import Parameters
from math import exp, ceil, log, floor, sqrt
import numpy as np
from scipy.integrate import ode
from scipy.stats import norm, gamma
import pandas as pd
import statistics
import os
import pickle
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

    def ode_system(self, t, y,  # state of system
                   infection_matrix, age_categories, symptomatic_prob, hospital_prob, critical_prob, beta,  # params
                   latentRate, removalRate, hospRate, deathRateICU, deathRateNoIcu,  # more params
                   better_hygiene, remove_symptomatic, remove_high_risk, ICU_capacity  # control
                   ):
        ##
        params = self.params
        dydt = np.zeros(y.shape)

        I_vec = [y[params.I_ind + i * params.number_compartments] for i in range(age_categories)]
        # H_vec = [ y[params.H_ind+i*params.number_compartments] for i in range(age_categories)]
        C_vec = [y[params.C_ind + i * params.number_compartments] for i in range(age_categories)]

        A_vec = [y[params.A_ind + i * params.number_compartments] for i in range(age_categories)]

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
            S_removal += y[params.S_ind + i * params.number_compartments]  # add all old people to remove

        for i in range(age_categories):
            # removing symptomatic individuals
            # these are put into Q ('quarantine');
            quarantine_sick = remove_symptomatic_rate * y[
                params.I_ind + i * params.number_compartments] / total_I  # no age bias in who is moved

            # removing susceptible high risk individuals
            # these are moved into O ('offsite')
            if i in range(age_categories - remove_high_risk['n_categories_removed'],
                          age_categories) and timing_function(t, remove_high_risk['timing']):
                remove_high_risk_people = min(remove_high_risk['rate'],
                                              S_removal)  # only removing high risk (within time control window). Can't remove more than we have
            else:
                remove_high_risk_people = 0

            # ICU capacity
            if sum(C_vec) > 0:  # can't divide by 0
                ICU_for_this_age = ICU_capacity['value'] * y[params.C_ind + i * params.number_compartments] / sum(
                    C_vec)  # hospital beds allocated on a first come, first served basis
            else:
                ICU_for_this_age = ICU_capacity['value']

            # ODE system:
            # S
            dydt[params.S_ind + i * params.number_compartments] = (
                        - y[params.S_ind + i * params.number_compartments] * control_factor * beta * (
                            np.dot(infection_matrix[i, :], I_vec) + params.AsymptInfectiousFactor * np.dot(
                        infection_matrix[i, :], A_vec))
                        - remove_high_risk_people * y[params.S_ind + i * params.number_compartments] / S_removal)
            # E
            dydt[params.E_ind + i * params.number_compartments] = (
                        y[params.S_ind + i * params.number_compartments] * control_factor * beta * (
                            np.dot(infection_matrix[i, :], I_vec) + params.AsymptInfectiousFactor * np.dot(
                        infection_matrix[i, :], A_vec))
                        - latentRate * y[params.E_ind + i * params.number_compartments])
            # I
            dydt[params.I_ind + i * params.number_compartments] = (
                        latentRate * (1 - symptomatic_prob[i]) * y[params.E_ind + i * params.number_compartments]
                        - removalRate * y[params.I_ind + i * params.number_compartments]
                        - quarantine_sick
                        )
            # A
            dydt[params.A_ind + i * params.number_compartments] = (
                        latentRate * symptomatic_prob[i] * y[params.E_ind + i * params.number_compartments]
                        - removalRate * y[params.A_ind + i * params.number_compartments])
            # H
            dydt[params.H_ind + i * params.number_compartments] = (
                        removalRate * (hospital_prob[i]) * y[params.I_ind + i * params.number_compartments]
                        - hospRate * y[params.H_ind + i * params.number_compartments]
                        #  + deathRateNoIcu * (1 - params.death_prob) * max(0,y[params.C_ind + i*params.number_compartments] - ICU_for_this_age) # recovered despite no ICU (0, since now assume death_prob is 1)
                        + deathRateICU * (1 - params.death_prob_with_ICU) * min(
                    y[params.C_ind + i * params.number_compartments], ICU_for_this_age)  # recovered from ICU
                        + (hospital_prob[i]) * params.quarant_rate * y[params.Q_ind + i * params.number_compartments]
                        # proportion of removed people who were hospitalised once returned
                        )
            # Critical care (ICU)
            dydt[params.C_ind + i * params.number_compartments] = (
                        min(hospRate * (critical_prob[i]) * y[params.H_ind + i * params.number_compartments],
                            max(0,
                                ICU_for_this_age - y[params.C_ind + i * params.number_compartments]
                                + deathRateICU * y[params.C_ind + i * params.number_compartments]  # with ICU treatment
                                )
                            )  # amount entering is minimum of: amount of beds available**/number needing it
                        # **including those that will be made available by new deaths
                        - deathRateICU * y[params.C_ind + i * params.number_compartments]  # with ICU treatment
                        )

            # Uncared - no ICU
            dydt[params.U_ind + i * params.number_compartments] = (hospRate * (critical_prob[i]) * y[
                params.H_ind + i * params.number_compartments]  # number needing care
                                                                   - min(
                        hospRate * (critical_prob[i]) * y[params.H_ind + i * params.number_compartments],
                        max(0,
                            ICU_for_this_age - y[params.C_ind + i * params.number_compartments]
                            + deathRateICU * y[params.C_ind + i * params.number_compartments]
                            ))  # minus number who get it (these entered category C)
                                                                   - deathRateNoIcu * y[
                                                                       params.U_ind + i * params.number_compartments]
                                                                   # without ICU treatment
                                                                   )

            # R
            dydt[params.R_ind + i * params.number_compartments] = (
                        removalRate * (1 - hospital_prob[i]) * y[params.I_ind + i * params.number_compartments]
                        + removalRate * y[params.A_ind + i * params.number_compartments]
                        + hospRate * (1 - critical_prob[i]) * y[params.H_ind + i * params.number_compartments]
                        + (1 - hospital_prob[i]) * params.quarant_rate * y[
                            params.Q_ind + i * params.number_compartments]
                        # proportion of removed people who recovered once returned
                        )

            # D
            dydt[params.D_ind + i * params.number_compartments] = (deathRateNoIcu * y[
                params.U_ind + i * params.number_compartments]  # died without ICU treatment (all cases that don't get treatment die)
                                                                   + deathRateICU * (params.death_prob_with_ICU) * y[
                                                                       params.C_ind + i * params.number_compartments]
                                                                   # died despite attempted ICU treatment
                                                                   )
            # O
            dydt[params.O_ind + i * params.number_compartments] = remove_high_risk_people * y[
                params.S_ind + i * params.number_compartments] / S_removal

            # Q
            dydt[params.Q_ind + i * params.number_compartments] = quarantine_sick - params.quarant_rate * y[
                params.Q_ind + i * params.number_compartments]
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
        
        E0 = 0 # exposed
        I0 = 1/population # sympt
        A0 = 1/population # asympt
        R0 = 0 # recovered
        H0 = 0 # hospitalised/needing hospital care
        C0 = 0 # critical (cared)
        D0 = 0 # dead
        O0 = 0 # offsite
        Q0 = 0 # quarantined
        U0 = 0 # critical (uncared)



        S0 = 1 - I0 - R0 - C0 - H0 - D0 - O0 - Q0 - U0
        
        age_categories = int(population_frame.shape[0])

        y0 = np.zeros(self.params.number_compartments*age_categories)

        population_vector = np.asarray(population_frame.Population_structure)

        # initial conditions
        for i in range(age_categories):
            y0[self.params.S_ind + i * self.params.number_compartments] = (population_vector[i] / 100) * S0
            y0[self.params.E_ind + i * self.params.number_compartments] = (population_vector[i] / 100) * E0
            y0[self.params.I_ind + i * self.params.number_compartments] = (population_vector[i] / 100) * I0
            y0[self.params.A_ind + i * self.params.number_compartments] = (population_vector[i] / 100) * A0
            y0[self.params.R_ind + i * self.params.number_compartments] = (population_vector[i] / 100) * R0
            y0[self.params.H_ind + i * self.params.number_compartments] = (population_vector[i] / 100) * H0
            y0[self.params.C_ind + i * self.params.number_compartments] = (population_vector[i] / 100) * C0
            y0[self.params.D_ind + i * self.params.number_compartments] = (population_vector[i] / 100) * D0
            y0[self.params.O_ind + i * self.params.number_compartments] = (population_vector[i] / 100) * O0
            y0[self.params.Q_ind + i * self.params.number_compartments] = (population_vector[i] / 100) * Q0
            y0[self.params.U_ind + i * self.params.number_compartments] = (population_vector[i] / 100) * U0

        symptomatic_prob = np.asarray(population_frame.p_symptomatic)
        hospital_prob = np.asarray(population_frame.p_hospitalised)
        critical_prob = np.asarray(population_frame.p_critical)

        sol = ode(self.ode_system).set_f_params(
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

        with dask.config.set(scheduler='processes', n_processes=n_processes):
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
