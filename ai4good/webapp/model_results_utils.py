import pandas as pd


def load_report(mr, params) -> pd.DataFrame:
    return normalize_report(mr.get('report'), params)


def normalize_report(df, params):
    df = df.copy()
    df.R0 = df.R0.apply(lambda x: round(complex(x).real, 1))
    df_temp = df.drop(['Time', 'R0', 'latentRate', 'removalRate', 'hospRate', 'deathRateICU', 'deathRateNoIcu'],
                      axis=1)
    df_temp = df_temp * params.population
    df.update(df_temp)
    return df
