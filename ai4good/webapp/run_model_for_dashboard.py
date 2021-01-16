from ai4good.webapp.model_results_config import model_profile_config
from ai4good.utils.logger_util import get_logger
from collections import defaultdict
import pandas as pd
import ai4good.utils.path_utils as pu
import pickle
import os

logger = get_logger(__name__)


# model name possibiltiies: ['compartmental-model', 'network-model', 'agent-based-model']


class ModelQueueItem:
    def __init__(self, model, profile):
        self.model = model
        self.profile = profile


def run_model_results_for_messages(model_runner, message_keys):
    run_config = defaultdict(list)
    for message_key in message_keys:
        for model in model_profile_config[message_key].keys():
            run_config[model] += (model_profile_config[message_key][model])
    logger.info(run_config)
    res = model_runner.batch_run_model(run_config)
    return res


def check_model_results_for_messages(model_runner, message_keys):
    logger.info("checking weather the model results are ready")
    results_ready = False
    for message_key in message_keys:
        for model in model_profile_config[message_key].keys():
            if len(model_profile_config[message_key][model])>0:
                for profile in model_profile_config[message_key][model]:
                    if not model_runner.results_exist(model, profile):
                        return results_ready
    results_ready = True
    return results_ready


def check_model_results_for_messages_unrun(model_runner, message_keys):
    logger.info("checking which messages haven't been run yet")
    unrun_model_profiles = defaultdict(list)
    for message_key in message_keys:
        for model in model_profile_config[message_key].keys():
            if len(model_profile_config[message_key][model])>0:
                for profile in model_profile_config[message_key][model]:
                    if not model_runner.results_exist(model, profile):
                        unrun_model_profiles[model].append(profile)
    return unrun_model_profiles


def load_report_cm(mr, total_population) -> pd.DataFrame:
    return normalize_report_cm(mr.get('report'), total_population)


def normalize_report_cm(df, total_population):
    df = df.copy()
    df.R0 = df.R0.apply(lambda x: round(complex(x).real, 1))
    df_temp = df.drop(['Time', 'R0', 'latentRate', 'removalRate', 'hospRate', 'deathRateICU', 'deathRateNoIcu'],
                      axis=1)
    df_temp = df_temp * total_population
    df.update(df_temp)
    return df


def collate_model_results_for_user(model_runner, message_keys, camp, total_population):
    user_result = defaultdict(dict)
    p = pu.user_results_path(f"{camp}_results_collage.pkl")
    if not os.path.exists(p):
        for message_key in message_keys:
            for model in model_profile_config[message_key].keys():
                if len(model_profile_config[message_key][model])>0:
                    if model == 'compartmental-model':
                        for profile in model_profile_config[message_key][model]:
                            mr = model_runner.get_result(model, profile)
                            report = load_report_cm(mr, total_population)
                            user_result[model][profile] = report
        # here we can write user_result to DB but we write to file system for now
        logger.info("writing user results collage to disk")
        with open(p, 'wb') as handle:
            pickle.dump(user_result, handle, protocol=pickle.HIGHEST_PROTOCOL)


