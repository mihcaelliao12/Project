from src.utils import spark_utils

        
def preprocess(df):
    df = spark_utils.is_zero(df, 'newbalanceOrig', drop=True)
    df = spark_utils.log1p(df, ['amount', 'oldbalanceOrg', 'oldbalanceDest', 'newbalanceDest'], drop=True)
    df = spark_utils.first_str(df, ['nameOrig', 'nameDest'], drop=True)
    df = spark_utils.concat_cols(df, 'nameOrig_firstStr', 'nameDest_firstStr', new_col='nameOrig__X__nameDest', drop=True)
    df = spark_utils.concat_cols(df, 'nameOrig__X__nameDest', 'type', new_col='nameOrig__X__nameDest__X__type', drop=False)
    df = spark_utils.set_index(df)
    return df

