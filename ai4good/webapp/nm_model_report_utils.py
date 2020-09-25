import pandas as pd


def load_report(mr, params) -> pd.DataFrame:
    return mr.get('report')
