import pickle
import pandas as pd
import numpy as np

from inflection import underscore


class Rossmann(object):

    def __init__(self):
        self.state = 1

        # competition_distance
        self.rs_competition_distance = pickle.load(open('parameter/competition_distance_scaler.pkl', 'rb'))
        # competition_time_month
        self.rs_competition_time_month = pickle.load(open('parameter/competition_time_month_scaler.pkl', 'rb'))
        # year
        self.minmax_year = pickle.load(open('parameter/year_scaler.pkl', 'rb'))
        # promo_time_week
        self.minmax_promo_time_week = pickle.load(open('parameter/promo_time_week_scaler.pkl', 'rb'))
        #state_holiday - One Hot Encoding
        self.one_hot_state_holiday = pickle.load(open('parameter/state_holiday_encoded.pkl', 'rb'))
        #store_type - Label Encoding
        self.le_store_type = pickle.load(open('parameter/store_type_encoding.pkl', 'rb'))

    def data_cleaning(self, df01):

        # lower and separating by _
        df01.columns = df01.columns.to_series().apply(
            lambda x: underscore(x))

        # to date
        df01['date'] = pd.to_datetime(df01['date'])

        #competition_distance --> fill with a really big distante --> implies --> no competition
        df01['competition_distance'].fillna(75000 *
                                            3,
                                            inplace=True)

        #### The next 4 "since" attributes --> Filling the respective label ( month, year) of the sale date
        #### Because if since month = sale month --> the difference will be 0
        #competition_open_since_month --> fill with the month of the sale
        df01['competition_open_since_month'].fillna(
            df01.loc[df01['competition_open_since_month'].isna(),
                     'date'].dt.month,
            inplace=True)

        #competition_open_since_year --> fill with the year of the sale
        df01['competition_open_since_year'].fillna(
            df01.loc[df01['competition_open_since_year'].isna(),
                     'date'].dt.year,
            inplace=True)

        #promo2_since_week --> fill with the week of the sale
        df01['promo2_since_week'].fillna(
            df01.loc[df01['promo2_since_week'].isna(),
                     'date'].dt.isocalendar().week,
            inplace=True)

        #promo2_since_year --> fill with the year of the sale
        df01['promo2_since_year'].fillna(
            df01.loc[df01['promo2_since_year'].isna(), 'date'].dt.year,
            inplace=True)

        #promo_interval
        df01['promo_interval'] = df01['promo_interval'].str.replace(
            'Sept', 'Sep')  # en_US format
        df01['promo_interval'].fillna(
            0, inplace=True)  # there's no promo_interval

        #df01['sale_month'] = df01['date'].dt.strftime('%b') --> deletar
        # creating a column to know if the sale is in a promo month
        df01['is_promo'] = df01[['promo_interval', 'date']].apply(
            lambda x: 0 if x['promo_interval'] == 0 else 1
            if x['date'].strftime('%b') in x['promo_interval'] else 0,
            axis=1)

        df01['competition_open_since_month'] = df01[
            'competition_open_since_month'].astype('int64')
        df01['competition_open_since_year'] = df01[
            'competition_open_since_year'].astype('int64')

        df01['promo2_since_week'] = df01['promo2_since_week'].astype('int64')
        df01['promo2_since_year'] = df01['promo2_since_year'].astype('int64')

        return df01

    def feature_engineering(self, df02):

        # year
        df02['year'] = df02['date'].dt.year

        # month
        df02['month'] = df02['date'].dt.month

        # day
        df02['day'] = df02['date'].dt.day

        # week of year
        df02['week_of_year'] = df02['date'].dt.isocalendar().week

        # year week
        df02['year_week'] = df02['date'].dt.strftime('%Y-%V')

        # competition since
        df02['competition_since'] = pd.to_datetime(
            df02['competition_open_since_year'].astype(str) + '-' +
            df02['competition_open_since_month'].astype(str),
            format='%Y-%m')

        df02['competition_time_month'] = (
            (df02['date'] - df02['competition_since']) / 30).dt.days

        # promo since -- > needs a day of week to transform to date
        df02['promo_since'] = pd.to_datetime(
            df02['promo2_since_year'].astype(str) + '-' +
            df02['promo2_since_week'].astype(str) + '-0',
            format='%G-%V-%w')

        df02['promo_time_week'] = ((df02['date'] - df02['promo_since']) /
                                   7).dt.days  # how many weeks in promo

        # assorment
        dic_assorment = {'a': 'basic', 'b': 'extra', 'c': 'extended'}
        df02['assortment'] = df02['assortment'].replace(dic_assorment)

        # state holiday
        dic_holidays = {
            'a': 'public_holiday',
            'b': 'easter_holiday',
            'c': 'christmas',
            '0': 'regular_day'
        }
        df02['state_holiday'] = df02['state_holiday'].replace(dic_holidays)

        # removing promo_inverval
        df02 = df02.drop(columns=['promo_interval'])

        # filter line
        df02 = df02[df02['open'] != 0]

        # filter columns
        cols_drop = ['open']
        df02 = df02.drop(columns=cols_drop)

        return df02

    def data_prepatarion(self, df03):  
        
        # competition_distance --> outliers
        df03['competition_distance'] = self.rs_competition_distance.transform(df03[['competition_distance']].values)


        # competition_time_month
        df03['competition_time_month'] = self.rs_competition_time_month.transform(df03[['competition_time_month']].values)
        
        # year
        df03['year'] = self.minmax_year.transform(df03[['year']].values)

        # promo_time_week
        df03['promo_time_week'] = self.minmax_promo_time_week.transform(df03[['promo_time_week']].values)
        
        #state_holiday - One Hot Encoding
        state_holiday_encoded = self.one_hot_state_holiday.transform(df03['state_holiday'])
        columns_created = self.one_hot_state_holiday.get_feature_names()
        
        df03[columns_created] =  state_holiday_encoded.values
        df03 = df03.drop(columns=['state_holiday', columns_created[0]])
        
        
        #store_type - Label Encoding
        df03['store_type'] = self.le_store_type.transform(df03['store_type'])
        
        #assortment- Ordinal Encoding
        assortment_dict = {'basic':1,
                           'extra':2,
                           'extended':3}
        df03['assortment'] = df03['assortment'].map(assortment_dict)
        
        
        ########### Nature Transformation ###########
    
        # day_of_week
        len_day_of_week = 7
        df03['day_of_week_sin'] = df03['day_of_week'].apply(lambda x: np.sin(x*(2*np.pi/len_day_of_week)))
        df03['day_of_week_cos'] = df03['day_of_week'].apply(lambda x: np.cos(x*(2*np.pi/len_day_of_week)))

        # month
        len_month = 12
        df03['month_sin'] = df03['month'].apply(lambda x: np.sin(x*(2*np.pi/len_month)))
        df03['month_cos'] = df03['month'].apply(lambda x: np.cos(x*(2*np.pi/len_month)))

        # day
        len_day = 31
        df03['day_sin'] = df03['day'].apply(lambda x: np.sin(x*(2*np.pi/len_day)))
        df03['day_cos'] = df03['day'].apply(lambda x: np.cos(x*(2*np.pi/len_day)))

        # week_of_year
        len_week_of_year = 52
        df03['week_of_year_sin'] = df03['week_of_year'].apply(lambda x: np.sin(x*(2*np.pi/len_week_of_year)))
        df03['week_of_year_cos'] = df03['week_of_year'].apply(lambda x: np.cos(x*(2*np.pi/len_week_of_year)))

        df03 = df03.drop(columns=['day_of_week', 'month', 'day', 'week_of_year'])
        

        # cols selected by boruta
        cols_selected_boruta = [
            'store', 'promo', 'store_type', 'assortment',
            'competition_distance', 'competition_open_since_month',
            'competition_open_since_year', 'promo2', 'promo2_since_week',
            'promo2_since_year', 'competition_time_month', 'promo_time_week',
            'day_of_week_sin', 'day_of_week_cos', 'month_cos', 'day_sin',
            'day_cos', 'week_of_year_cos'
        ]

        #plus month sin and week of year sin
        cols_selected_boruta.extend(['month_sin', 'week_of_year_sin'])

        
        return df03[cols_selected_boruta]
    
    
    def get_prediction(self, model, original_data,test_data):
        # predictions
        pred = model.predict(test_data)
        
        original_data['prediction'] = np.expm1(pred)
        
        return original_data.to_json(orient='records', date_format='iso')