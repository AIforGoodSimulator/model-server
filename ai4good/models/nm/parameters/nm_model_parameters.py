from ai4good.models.nm.parameters.camp_params import *

# SEIRS+ Model parameters
transmission_rate = [1.28]
progression_rate = [round(1/5.1, 3)]
recovery_rate_list = [0.056]  # Approx 1/18 -> Recovery occurs after 18 days
hosp_rate_list = [round(1/11.4, 3)]  # 1/6.3 # From Tucker Model
# crit_rate = 0.3 # From camp_params
crit_rate = list(
    (sample_pop["death_rate"] / sample_pop["prob_symptomatic"]) / sample_pop["prob_hospitalisation"])
death_rate_list = [0.75]

prob_global_contact = 1
prob_detected_global_contact = 1

# prob_hosp_to_critical = list(sample_pop["death_rate"]/sample_pop["prob_hospitalisation"])
prob_death = list(sample_pop["death_rate"])
prob_asymptomatic = list(1 - sample_pop["prob_symptomatic"])
prob_symp_to_hosp = list(sample_pop["prob_hospitalisation"])

init_symp_cases = 1
init_asymp_cases = 1

t_steps = 5
