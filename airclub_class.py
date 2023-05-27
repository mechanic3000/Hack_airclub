import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder

import re
import datetime as dt
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
tqdm.pandas()

from catboost import CatBoostClassifier

class AeroclubPipeline:
    def __init__(self):
        #данные по аэропортам
        self.df_timezones = pd.read_pickle('./Data/timezones.pkl')
        self.cat_feats = ['SegmentCount', 'IsBaggage', 'isRefundPermitted', 
                          'isExchangePermitted', 'isDiscount', 'InTravelPolicy', 'SearchRouteFrom1', 
                          'SearchRouteTo1', 'SearchRouteFrom2', 'SearchRouteTo2', 'Airlanes_enc']
        
        self.num_feats = ['Amount', 'To_flight_dur', 'Return_flight_dur', 'diff_betw_dep_time', 'diff_betw_arr_time', 'One_sec_cost']

        self.ctb_clsf_model = CatBoostClassifier(random_seed=42, cat_features=self.cat_feats, silent=False,
                                   depth=2, l2_leaf_reg=3, learning_rate=0.1, iterations=200)

    
    # данные об аэропрте вылета и прилета и обратный маршрут. Нобходимо разбить на колонки
    @staticmethod
    def splited_searchroute_data(searchroute):
        return pd.Series([searchroute[:3], searchroute[3:6], searchroute[7:10], searchroute[10:13]])
    
    # получить таймзон по IATAcode
    def get_timezone(self, iata):
        return self.df_timezones.get(iata, default=9999)
        
    
    # вытащить название авиакомпании из FlightOptions
    @staticmethod
    def get_airlanes(flight_options):
        flights_list = [_[:2] for _ in re.findall(r'\w\S\d{4}', flight_options)]
        
        return '/'.join(set(flights_list))
    

    # получить период по времени
    @staticmethod
    def get_period(date):   
        if date == pd.to_datetime('1900-01-01 00:00:00.000'):
            return 0
        time = date.time()
        # 22-6
        if (dt.time(22) < time or dt.time(0) <= time <= dt.time(6)):
            period = 1
        # 6 - 12
        elif dt.time(6) < time <= dt.time(12):
            period = 2
        # 12 - 18
        elif dt.time(12) < time <= dt.time(18):
            period = 3
        # 18 - 22
        elif dt.time(18) < time <= dt.time(22):
            period = 4
            
        return period
    
    
    def _feature_engineering(self, df):
        df = df.copy()
         # заменим пропуски на 9999
        df.loc[df.TravellerGrade.isna(), 'TravellerGrade'] = '9999'
        df.TravellerGrade = df.TravellerGrade.astype(str)

        # разобъем маршрут на отдельные фичи
        df[['SearchRouteFrom1', 'SearchRouteTo1', 'SearchRouteFrom2', 'SearchRouteTo2']] = df.SearchRoute.progress_apply(
                                                                                            self.splited_searchroute_data)
        
        # добавим часовые пояся аэропортов
        for location in (pbar := tqdm(set(df.SearchRouteFrom1.to_list()))):
            pbar.set_description("Set timezones 1/4")
            df.loc[df.SearchRouteFrom1 == location, 'TimeZone_From1'] = self.df_timezones.get(location, default=9999)
        for location in (pbar := tqdm(set(df.SearchRouteTo1.to_list()))):
            pbar.set_description("Set timezones 2/4")
            df.loc[df.SearchRouteTo1 == location, 'TimeZone_To1'] = self.df_timezones.get(location, default=9999)
        for location in (pbar := tqdm(set(df.SearchRouteFrom2.to_list()))):
            pbar.set_description("Set timezones 3/4")
            df.loc[df.SearchRouteFrom2 == location, 'TimeZone_From2'] = self.df_timezones.get(location, default=9999)
        for location in (pbar := tqdm(set(df.SearchRouteTo2.to_list()))):
            pbar.set_description("Set timezones 4/4")
            df.loc[df.SearchRouteTo2 == location, 'TimeZone_To2'] = self.df_timezones.get(location, default=9999)

        # обработка дат
        date_columns = df.filter(like='Date').columns.to_list()
        # обозначим непроставленные даты минимальным значением
        for col in date_columns:
            df.loc[df[col].isna(), col] = '1900-01-01 00:00:00.000'
            
        # переведем даты в формат datetime64
        for col in date_columns:
            df[col] = pd.to_datetime(df[col])

        # укажем время перелета туда / обратно (если есть) в секундах
        df['To_flight_dur'] = pd.Series(df.ArrivalDate - pd.to_timedelta(df.TimeZone_To1, unit='h') - \
                                    (df.DepartureDate - pd.to_timedelta(df.TimeZone_From1, unit='h'))).astype(int) / 10**9

        df['Return_flight_dur'] = pd.Series(df.ReturnArrivalDate - pd.to_timedelta(df.TimeZone_To2, unit='h') - \
                                    (df.ReturnDepatrureDate - pd.to_timedelta(df.TimeZone_From2, unit='h'))).astype(int) / 10**9

        #Заменим NanN на 9999
        df.loc[df.isRefundPermitted.isna(), 'isRefundPermitted'] = 9999
        df.loc[df.isExchangePermitted.isna(), 'isExchangePermitted'] = 9999
        df.loc[df.IsBaggage.isna(), 'IsBaggage'] = 9999

        # Укажем авиакомпании и закодируем их
        df['Airlanes'] = df.FligtOption.apply(self.get_airlanes)
        encoder = LabelEncoder()
        flights_list_encode = encoder.fit(df.Airlanes.unique())
        df['Airlanes_enc'] = encoder.transform(df.Airlanes.values)

        # разница между запрошенным временем и фактическим
        df['diff_betw_dep_time'] = np.abs(df.RequestDepartureDate - df.DepartureDate)
        cond = df.SearchRouteTo2 != ''
        df.loc[cond, 'diff_betw_arr_time'] = np.abs(df.RequestReturnDate - df.ReturnArrivalDate)
        cond = df.SearchRouteTo2 == ''
        df.loc[cond, 'diff_betw_arr_time'] = np.abs(df.RequestReturnDate - df.ArrivalDate)
        # переведем время в секунды
        df.diff_betw_arr_time = df.diff_betw_arr_time.apply(lambda x: x.total_seconds())
        df.diff_betw_dep_time = df.diff_betw_dep_time.apply(lambda x: x.total_seconds())

        # разделим время вылета по периодам
        df['DepartureDate_per'] = df.DepartureDate.apply(self.get_period)
        df['ArrivalDate_per'] = df.ArrivalDate.apply(self.get_period)
        df['ReturnDepatrureDate_per'] = df.ReturnDepatrureDate.apply(self.get_period)
        df['ReturnArrivalDate_per'] = df.ReturnArrivalDate.apply(self.get_period)

        df['One_sec_cost'] = df.Amount / df.To_flight_dur

        df[self.cat_feats] = df[self.cat_feats].astype(str)
        df = df[self.num_feats + self.cat_feats]

        return df 

    def transform(self, df):
        df = self._feature_engineering(df)
        return df
    
    def save_model(self, file_path):
        self.ctb_clsf_model.save_model(file_path)

    def load_model(self, file_path):
        self.ctb_clsf_model.load_model(file_path)

    def fit(self, df, target):
        self.ctb_clsf_model.fit(df, target, plot=False)
        return self.ctb_clsf_model.get_params()

    def fit_transform(self, df, target):
        df = self.transform(df)
        self.fit(df, target)
        return self.ctb_clsf_model.get_params()

    def predict(self, df):
        return self.ctb_clsf_model.predict(df)
    
    def predict_proba(self, df):
        return self.ctb_clsf_model.predict_proba(df)
    
    # метод формирует результат сортируя строки в порядке понижения вероятности выбора
    def get_result(self, df, proba):
        df['proba'] = proba[:,1]
        requests = df.RequestID.unique()
        result_df = pd.DataFrame(columns=df.columns.to_list())

        for req in requests:
            n_rows = df.loc[df.RequestID == req].shape[0]
            sort_df = df.loc[df.RequestID == req].sort_values(['proba'], ascending=False).reset_index()
            nummeric = pd.Series([i for i in range(1, n_rows+1)])
            result_df = result_df.append(pd.concat([sort_df, nummeric], axis=1))

        result_df.drop(columns=['Position ( from 1 to n)', 'proba', 'index'], inplace=True)
        result_df.rename(columns={0: 'Position ( from 1 to n)'}, inplace=True)
        return result_df
    
    def get_xls_result(self, df, xls_file_path):
        df_trnsf = self.transform(df)
        predict_proba = self.predict_proba(df_trnsf)
        result_df = self.get_result(df, predict_proba)
        result_df.to_excel(xls_file_path)
