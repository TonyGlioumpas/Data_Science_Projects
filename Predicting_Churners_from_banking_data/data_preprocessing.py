import pandas as pd
import numpy as np


# allowable upper age limit
AGE_THRESH = 120
# calculate today's date
TODAY = pd.to_datetime('today').normalize()
# create a list of column names that are dates
COLUMN_DATES = ['customer_since_all', 'customer_since_bank', 'customer_birth_date']
COLUMN_BALANCES_CAPS = ['bal_insurance_21', 'bal_insurance_23', 'cap_life_insurance_fixed_cap', 
'cap_life_insurance_decreasing_cap', 'prem_fire_car_other_insurance', 'bal_personal_loan', 
'bal_mortgage_loan', 'bal_current_account', 'bal_pension_saving', 'bal_savings_account', 
'bal_savings_account_starter', 'bal_current_account_starter']

def calculate_age(x):
    if pd.isna(x):
        return np.nan
    else:
        # change birthdate to something more appropriate
        birthdate = pd.to_datetime(x, format="%Y-%m")
        age_delta = TODAY - birthdate
        age = round(age_delta / pd.Timedelta(365, 'd'))
        return age
        
# Convert all non-numerical values to int64 (scaled)
# NaNs will be interpolated according to argument
def remove_na_map_categorical(df: pd.DataFrame) -> pd.DataFrame:
    df.drop_duplicates(inplace=True)
    # drop duplicated based on client_ids
    df.drop_duplicates(subset=['client_id'])
    # drop duplicate columns (e.g if we have columns=['A', 'B', 'A'])
    # we drop the last 'A' 
    df = df.loc[:, ~df.columns.duplicated()]
    # drop the customer education column since >70% of the values are missing
    # df.drop(columns=['customer_education'], inplace=True)
    # remove client id column since it's not useful after dropping the duplicate values
    # duplicates based on client_id
    # drop if we have identical columns (based on column name or values)
    # df.drop(columns=['client_id'], index=1, inplace=True)
    
    for column_name in df:
        # print(f'column dtype: {df[column_name].dtype}, column_name: {column_name}')
        # If the column is an object, we need to map the string values to numerical values
        # or in the case of dates we need to convert them to age.
        if df[column_name].dtype == 'O' and column_name != 'client_id':
            if column_name in COLUMN_DATES:
                # replace irregular values as NaNs to interpolate them later
                # make sure that we replace irregularities in customer_since_bank
                # and customer_since_all as well.
                df[column_name] = df[column_name].apply(calculate_age)
                if column_name == 'customer_birth_date':
                    df.loc[df['customer_birth_date'] >= AGE_THRESH, 'customer_birth_date'] = np.nan
                continue
            unique_values = pd.Series(df[column_name].unique()).dropna()
            # print(unique_values)
            # string entries to numerical entries example ['male', 'female'] -> [0, 1]
            # create empty dict to put unique values with the values that we'll map them
            mapping_dict = {}
            for idx, val in unique_values.items():
                mapping_dict.update({val: idx})
            df[column_name] = df[column_name].map(mapping_dict)
    # interpolate and then forward and backward fill to fill values that were not interpolated
    # values that are not interpolated are the edge values 
    df = df.interpolate(method='nearest').ffill().bfill()
    return df

def bin_birthdate(df: pd.DataFrame) -> pd.DataFrame:
    bins = [0, 20, 30, 40, 50, 65, 130]
    df.loc[:, 'customer_birth_date'] = pd.cut(df['customer_birth_date'], bins=bins, labels=False)
    return df

def bin_customer_time(df: pd.DataFrame) -> pd.DataFrame:
    bins = [0, 2, 5, 10, 15, 25, 40, 80]
    df.loc[:, 'customer_since_all'] = pd.cut(df['customer_since_all'], bins=bins, labels=False)
    df.loc[:, 'customer_since_bank'] = pd.cut(df['customer_since_bank'], bins=bins, labels=False)
    return df

def bin_bal_cap(df: pd.DataFrame, nbins: int = 10) -> pd.DataFrame:
    for col in COLUMN_BALANCES_CAPS:
        df.loc[:, col] = pd.cut(df[col], bins=nbins, labels=False)
    return df

def bin_postcode(pcode: pd.Series):
    pcode = pcode.map(str)
    return  pcode.apply(lambda x: x.count('0'))


def calculate_iv(X, y):
    df = pd.concat([X, y], axis=1)

    iv = {'predictive_power': ['useless (iv < 0.02)', 'weak (iv >= 0.02 & iv < 0.1)', 'medium (iv >= 0.1 & iv < 0.3)', 
        'strong (iv >= 0.3 & iv < 0.5)', 'too good to be true (iv > 0.5)'], 'column_name': ['', '', '', '', '']}
    df_iv = pd.DataFrame(iv)
    target = 'target'
    iv_dict = {}
    for c in df.columns:
        feature = c

        df_woe_iv = pd.crosstab(df[feature],df[target], normalize='columns')
        df_woe_iv = df_woe_iv.assign(woe=lambda dfx: np.log(dfx[1] / dfx[0]+0.01))
        df_woe_iv = df_woe_iv.assign(iv=lambda dfx: np.sum(dfx['woe'] * (dfx[1] - dfx[0])))

        if df_woe_iv['iv'].iloc[0] < 0.02:
            df_iv.iloc[0].column_name += c + ', '
        elif df_woe_iv['iv'].iloc[0] >= 0.02 and df_woe_iv['iv'].iloc[0] < 0.1:
            df_iv.iloc[1].column_name += c + ', '
        elif df_woe_iv['iv'].iloc[0] >= 0.1 and df_woe_iv['iv'].iloc[0] < 0.3:
            df_iv.iloc[2].column_name += c + ', '
        elif df_woe_iv['iv'].iloc[0] >= 0.3 and df_woe_iv['iv'].iloc[0] < 0.5:
            df_iv.iloc[3].column_name += c + ', '
        else: 
            df_iv.iloc[4].column_name += c + ', '

        iv_dict.update({df_woe_iv.index.name : df_woe_iv['iv'].values[0]})

    iv_series = pd.Series(iv_dict)
    iv_series.drop(labels=['target'], inplace=True)
    return iv_series.sort_values(ascending=False)

if __name__ == '__main__':
    # path as in local repo
    file_path = r'data\train_month_3_with_target.csv'
    # in_case you want to save some information about the dataset
    # notes_path = 'notes_on_dataset\\'

    # read csv of 3d month with targets
    train_month_3_target = pd.read_csv(file_path)

    # df_dtypes = train_month_3_target.dtypes 
    interpolation_method = 'nearest'
    # calculate ages from dates, drop duplicates, map string values to numerical values,
    # interpolate nan values according to desired method.
    new_df = remove_na_map_categorical(train_month_3_target.copy())
    
    # save the preprocessed dataframe to .csv (optional)
    # new_df.to_csv(r'data\out.csv', index=False)