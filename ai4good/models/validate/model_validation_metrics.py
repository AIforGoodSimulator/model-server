import pandas as pd
import numpy as np
from datetime import date, timedelta
from ai4good.utils.logger_util import get_logger
from ai4good.models.validate.model_metrics import model_metrics

logger = get_logger(__name__)

def model_validation_metrics(population:int, model:str, age_categories:list, case_cols:list, df_baseline:pd.DataFrame, df_model:pd.DataFrame, save_output=""):

    # Initialise metrics
    if model.upper() == "CM":
        time_col = "Time"
        cols_results = ["age", "case"]
    elif model.upper() == "ABM":
        time_col = "DAY"
        cols_results = ["age", "case"]
    elif model.upper() == "NM":
        time_col = "Time"
        cols_results = ["case"]
    cols_overall = [time_col] + case_cols
    df_model_metrics = pd.DataFrame(columns=cols_results)

    # Process Baseline First;
    df_base_time = df_baseline[time_col]
    baseline_n_days = df_base_time.nunique()
    baseline_n_rows = df_base_time.shape[0]

    # Process Model Output and compare with baseline;
    df_time = df_model[time_col]
    n_days = df_time.nunique()
    n_rows = df_time.shape[0]

    # Process number of simulations
    if model.upper() == "CM":
        baseline_n_simul = df_base_time[df_base_time==0].count()
        n_simul = df_time[df_time==0].count()
    elif model.upper() == "ABM":
        baseline_n_simul = df_base_time[df_base_time==1].count()
        n_simul = df_time[df_time==1].count()
    elif model.upper() == "NM":
        baseline_n_simul = df_base_time[df_base_time==1].count()
        n_simul = df_time[df_time==1].count()

    # Log output information
    logger.info("Validation outputs: number of days (baseline/model): %s/%s, number of simulation (baseline/model): %s/%s", str(baseline_n_days), str(n_days), str(baseline_n_simul), str(n_simul))

    # Compute metrics
    if model.upper() =="CM":
        # Get df for population
        # Use this as the benchmark for the age group
        df_baseline_all_simul = df_baseline[cols_overall]
        df_baseline_all_sum=df_baseline_all_simul.groupby([time_col]).sum()*population
        df_baseline_all=df_baseline_all_sum/baseline_n_simul
        df_baseline_all_mean=df_baseline_all.mean()
        df_baseline_all_std=df_baseline_all.std()
    
        # Get df for population
        # Use this as the benchmark for the age group
        df_model_all_simul = df_model[cols_overall]
        df_model_all_sum=df_model_all_simul.groupby([time_col]).sum()*population
        df_model_all=df_model_all_sum/n_simul
        df_model_all_mean=df_model_all.mean()
        df_model_all_std=df_model_all.std()

        # Process for each age group:
        for age in age_categories:
    
            # get columns for the age group
            cols = [ col + ": " + age for col in case_cols]
            cols.append(time_col)
    
            #baseline
            df_baseline_age_simul = df_baseline[cols]
            #Calculate averages for all simulations
            df_baseline_age_sum=df_baseline_age_simul.groupby([time_col]).sum()*population
            df_baseline_age=df_baseline_age_sum/baseline_n_simul
            df_baseline_age_mean=df_baseline_age.mean()
            df_baseline_age_std=df_baseline_age.std()
    
            #Model
            df_model_age_simul = df_model[cols]
    
            #Calculate averages for all simulations
            df_model_age_sum=df_model_age_simul.groupby([time_col]).sum()*population
            df_model_age=df_model_age_sum/n_simul
            df_model_age_mean=df_model_age.mean()
            df_model_age_std=df_model_age.std()
    
            #Call Model Metrics for each case Col
            for col in case_cols:
                col_age = col + ": " + age
                y=df_baseline_age[col_age]
                pred=df_model_age[col_age]
    
                # filter out nan or zero values of y;
                rows = y > 0
                y = y[rows]
                pred = pred[rows]
                y = y.dropna()
                pred = pred.dropna()
                results=model_metrics(y,pred)
                results['age']= age
                results['case'] = col
                df_model_metrics = df_model_metrics.append(results, ignore_index=True)


    elif model.upper() == "ABM":
        # Get df for population
        # Use this as the benchmark for the age group
        df_baseline_all_simul = df_baseline[cols_overall]
        df_baseline_all_sum=df_baseline_all_simul.groupby([time_col]).sum()
        #df_baseline_all=df_baseline_all_sum/baseline_n_simul
        #df_baseline_all_mean=df_baseline_all.mean()
        #df_baseline_all_std=df_baseline_all.std()
        df_baseline_all_mean=df_baseline_all_sum/baseline_n_simul
        df_baseline_all_std=df_baseline_all_simul.groupby([time_col]).std()
        
       # Get df for population
        # Use this as the benchmark for the age group
        df_model_all_simul = df_model[cols_overall]
        df_model_all_sum=df_model_all_simul.groupby(['DAY']).sum()
        #df_model_all=df_model_all_sum/n_simul
        #df_model_all_mean=df_model_all.mean()
        #df_model_all_std=df_model_all.std()
        df_model_all_mean=df_model_all_sum/n_simul
        df_model_all_std=df_model_all_simul.groupby(['DAY']).std()

        # Process for each age group:
        for age in age_categories:
        
            # get columns for the age group
        
            cols = [col + "_AGE" + age for col in case_cols]
            cols.append(time_col)
        
            # baseline
            df_baseline_age_simul = df_baseline[cols]
            # Calculate averages for all simulations
            df_baseline_age_sum = df_baseline_age_simul.groupby(['DAY']).sum()
            df_baseline_age = df_baseline_age_sum / baseline_n_simul
            # df_baseline_age_mean=df_baseline_age.mean()
            # df_baseline_age_std=df_baseline_age.std()
            df_baseline_age_mean = df_baseline_age_sum / baseline_n_simul
            df_baseline_age_std = df_baseline_age_simul.groupby(['DAY']).std()
        
            # Model
            df_model_age_simul = df_model[cols]
            # Calculate averages for all simulations
            df_model_age_sum = df_model_age_simul.groupby(['DAY']).sum()
            df_model_age = df_model_age_sum / n_simul
            # df_model_age_mean=df_model_age.mean()
            # df_model_age_std=df_model_age.std()
            df_model_age_mean = df_model_age_sum / n_simul
            df_model_age_std = df_model_age_simul.groupby(['DAY']).std
        
            # Call Model Metrics for each case Col
            for col in case_cols:
                col_age = col + "_AGE" + age
                y = df_baseline_age[col_age]
                pred = df_model_age[col_age]
                # filter out nan or zero values of y;
                rows = y > 0
                y = y[rows]
                pred = pred[rows]
                y = y.dropna()
                pred = pred.dropna()
                ln = min(len(y), len(pred))
                if ln > 1:
                    results = model_metrics(y.iloc[0:ln], pred.iloc[0:ln])
                    results['age'] = age
                    results['case'] = col
                    df_model_metrics = df_model_metrics.append(results, ignore_index=True)


    elif model.upper() == "NM":
        # Get df for population
        # Use this as the benchmark for the age group
        df_baseline_all_simul = df_baseline[cols_overall]
        df_baseline_all_sum = df_baseline_all_simul.groupby(['Time']).sum()
        df_baseline_all = df_baseline_all_sum / baseline_n_simul
        df_baseline_all_mean = df_baseline_all.mean()
        df_baseline_all_std = df_baseline_all.std()

        # Get df for population
        # Use this as the benchmark for the age group
        df_model_all_simul = df_model[cols_overall]
        df_model_all_sum = df_model_all_simul.groupby(['Time']).sum()
        df_model_all = df_model_all_sum / n_simul
        df_model_all_mean = df_model_all.mean()
        df_model_all_std = df_model_all.std()

        # Call Model Metrics for each case Col
        for col in case_cols:
            y = df_baseline_all[col]
            pred = df_model_all[col]
            # filter out nan or zero values of y;
            rows = y > 0
            y = y[rows]
            pred = pred[rows]
            y = y.dropna()
            pred = pred.dropna()
            results = model_metrics(y, pred)
            results['case'] = col
            df_model_metrics = df_model_metrics.append(results, ignore_index=True)


    df_model_metrics.reset_index(drop=True, inplace=True)

    if save_output:
        df_model_metrics.to_csv(save_output)
        logger.info("Model Validation Metrics is saved in %s", str(save_output))
        
    return df_model_metrics