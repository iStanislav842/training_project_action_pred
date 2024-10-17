import datetime
import dill
import json
import pandas as pd
import random
from geopy.geocoders import Nominatim
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import FunctionTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.metrics import roc_auc_score
import time

def main(): 
    def drop_dublicat_hit_number(df_ga_hits):
        df_ga_hits.drop_duplicates(subset = ['session_id', 'hit_date', 'hit_number'], keep='first', inplace=True)
        return df_ga_hits

    def transform_event_action(df_ga_hits): # замена target_action_list в целевом столбце на 0,1
        target = []
        target_action_list = ['sub_car_claim_click', 'sub_car_claim_submit_click', 'sub_open_dialog_click', 'sub_custom_question_submit_click', 
                              'sub_call_number_click', 'sub_callback_submit_click', 'sub_submit_success', 'sub_car_request_submit_click']
        list_event_action = df_ga_hits.event_action.tolist()
        for elem in list_event_action:
            if elem in target_action_list:
                target.append(1)
            else:
                target.append(0)
        df_ga_hits['event_action'] = target  
        df_ga_hits['event_action'] = df_ga_hits['event_action'].astype(str)
        return df_ga_hits

    def drop_columns(df_ga_hits): #удаление столбцов, не учавствующих в финальном датасете
        df_ga_hits = df_ga_hits.drop(['session_id', 'hit_date', 'hit_time', 'hit_number', 'hit_type',
       'hit_referer', 'hit_page_path', 'event_category', 'event_label',
       'event_value', 'client_id', 'visit_date', 'visit_time', 'visit_number',], axis=1)
        return df_ga_hits
 
    def df_fillna_other(df):  #преобразование типов и замена данных
        df['utm_source'] = df['utm_source'].astype(str)
        df['utm_medium'] = df['utm_medium'].astype(str)
        df['utm_campaign'] = df['utm_campaign'].astype(str)
        df['utm_adcontent'] = df['utm_adcontent'].astype(str)
        df['utm_keyword'] = df['utm_keyword'].astype(str)
        df['device_category'] = df['device_category'].astype(str)
        df['device_os'] = df['device_os'].astype(str)
        df['device_brand'] = df['device_brand'].astype(str)
        df['device_model'] = df['device_model'].astype(str)
        df['device_screen_resolution'] = df['device_screen_resolution'].astype(str)
        df['device_browser'] = df['device_browser'].astype(str)
        df['geo_country'] = df['geo_country'].astype(str)
        df['geo_city'] = df['geo_city'].astype(str)
        df['utm_source'] = df['utm_source'].replace(['NaN', 'nan'], 'other')
        df['utm_medium'] = df['utm_medium'].replace(['NaN', 'nan'], 'other')
        df['utm_campaign'] = df['utm_campaign'].replace(['NaN', 'nan'], 'other')
        df['utm_adcontent'] = df['utm_adcontent'].replace(['NaN', 'nan'], 'other')
        df['utm_keyword'] = df['utm_keyword'].replace(['NaN', 'nan'], 'other')
        df['device_category'] = df['device_category'].replace(['NaN', 'nan'], 'other')
        df['device_os'] = df['device_os'].replace(['NaN', 'nan'], 'other')
        df['device_brand'] = df['device_brand'].replace(['NaN', 'nan'], 'other')
        df['device_model'] = df['device_model'].replace(['NaN', 'nan'], 'other')
        df['device_screen_resolution'] = df['device_screen_resolution'].replace(['NaN', 'nan'], 'other')
        df['device_browser'] = df['device_browser'].replace(['NaN', 'nan'], 'other')
        df['geo_country'] = df['geo_country'].replace(['NaN', 'nan'], 'other')
        df['geo_city'] = df['geo_city'].replace(['NaN', 'nan'], 'other')
        return df

    def geo_frame_generation(df_ga_hits): #преобразование названия города в координаты
        try:
            df_geo_city_loc = pd.read_csv('data/df_geo_city_loc.csv')
        except:
            df_geo_city_loc = pd.DataFrame()
            i = 0
            n = 0
            for elem in df_ga_hits['geo_city'].unique().tolist():
                geolocator = Nominatim(user_agent='gmail@gmail.com')
                location = geolocator.geocode(elem)
                df_geo_city_loc.loc[i, 'geo_city'] = elem
                try:
                    df_geo_city_loc.loc[i, 'latitude'] = location.latitude
                    df_geo_city_loc.loc[i, 'longitude'] = location.longitude
                except:
                    print(f'Ошибка в элементе {elem}')
                    df_geo_city_loc.loc[i, 'geo_city'] = elem
                    df_geo_city_loc.loc[i, 'latitude'] = 'нет данных в сервисе'
                    df_geo_city_loc.loc[i, 'longitude'] = 'нет данных в сервисе'
                    i += 1
                    continue
                print(elem, ',', i, 'Latitude: '+str(location.latitude)+', Longitude: '+str(location.longitude))
                n = random.uniform(0.2, 0.3)
                time.sleep(n)
                i += 1
                df_geo_city_loc.to_csv('data/df_geo_city_loc_with_na.csv', index=False)
            with open('data/missing_city.json', 'r') as missing_city:
                data = json.load(missing_city)
            set_geo = data
            for elem in set_geo.keys():
                i = df_geo_city_loc[(df_geo_city_loc.geo_city == elem)].index
                df_geo_city_loc.loc[i, 'latitude'] = set_geo[elem][0]
                df_geo_city_loc.loc[i, 'longitude'] = set_geo[elem][1]
            df_geo_city_loc.to_csv('data/df_geo_city_loc.csv', index=False)
        return df_ga_hits

    def p_categorical_var_frames_generation(df_ga_hits):
        try:
            categorical_columns = ['utm_source', 'utm_medium', 'utm_campaign', 'utm_adcontent', 'utm_keyword', 
                                   'device_category', 'device_os', 'device_brand', 'device_model','device_browser']
            for column in categorical_columns:
                dataframe = pd.DataFrame()
                dataframe = pd.read_csv(f'data/df_likelihood/df_likelihood_{column}.csv')
        except:
            df_ga_hits.rename(columns = {'event_action':'target'}, inplace = True) 
            categorical_columns = ['utm_source', 'utm_medium', 'utm_campaign', 'utm_adcontent', 'utm_keyword', 
                                   'device_category', 'device_os', 'device_brand', 'device_model','device_browser']
            for column in categorical_columns:
                print(f'в {datetime.datetime.now().time()} начало обработки столбца: {column}, количество значений {len(df_ga_hits[column].unique())}')
                i = 0
                dataframe = pd.DataFrame()
                values = [] 
                list_count_target0 = [] 
                list_count_target1 = [] 
                likelihood_values = [] 
                list_values_target0_only = [] 
                df_to_cvs = 'df_likelihood_' + column 
                df_target1 = df_ga_hits[(df_ga_hits.target == 1)]
                list_values_target0_only = list(set(df_ga_hits[column].value_counts().index) - set(df_target1[column].value_counts().index))
                list_values_target0_target1 = df_target1[column].value_counts().index
                for value in list_values_target0_only:
                    counts_target1 = 0
                    counts_target0 = df_ga_hits.target[(df_ga_hits[column] == value) & (df_ga_hits.target == 0)].shape[0]
                    df_likelihood_column = 0
                    values.append(value) 
                    list_count_target0.append(counts_target0) 
                    list_count_target1.append(counts_target1)
                    likelihood_values.append(df_likelihood_column)  
                    i += 1
                    if i % 100 == 0:
                        print(f'в {datetime.datetime.now().time()} всего обработано {i} переменных |target==0')
                print(f'в {datetime.datetime.now().time()} обработано {i} переменных, обработка переменных завершена, P(value)|target==0')
                for value in list_values_target0_target1:
                    counts_target1 = df_ga_hits.target[(df_ga_hits[column] == value) & (df_ga_hits.target == 1)].shape[0]
                    counts_target0 = df_ga_hits.target[(df_ga_hits[column] == value) & (df_ga_hits.target == 0)].shape[0]
                    df_likelihood_column = counts_target1 / (counts_target1+counts_target0)   
                    values.append(value) 
                    list_count_target0.append(counts_target0) 
                    list_count_target1.append(counts_target1)
                    likelihood_values.append(df_likelihood_column)  
                    i += 1
                    if i % 10 == 0:
                        print(f'в {datetime.datetime.now().time()} всего обработано {i} переменных |target==1')
                print(f'в {datetime.datetime.now().time()} обработано {i} переменных, обработка всех переменных завершена')
                dataframe[column] = values
                dataframe['counts_T1'] = list_count_target1
                dataframe['counts_T0'] = list_count_target0
                dataframe['likelihood'] = likelihood_values
                print(f'в {datetime.datetime.now().time()} старт записи файла: {df_to_cvs}')
                dataframe.to_csv(f'data/df_likelihood/{df_to_cvs}.csv', index=False)
                print(f'в {datetime.datetime.now().time()} файл {df_to_cvs} записан')
                print(f'проверка: counts_T1={dataframe.counts_T1.sum()}, counts_T1={dataframe.counts_T0.sum()}, counts_all={dataframe.counts_T1.sum()+dataframe.counts_T0.sum()}, df_ga_hits.shape[0]={df_ga_hits.shape[0]}')
            df_ga_hits.rename(columns = {'target': 'event_action'}, inplace = True) 
        return df_ga_hits

    def transform_geo(df_ga_hits):
        df_geo_city_loc = pd.read_csv('data/df_geo_city_loc.csv')
        df_ga_hits = df_ga_hits.merge(df_geo_city_loc, left_on='geo_city', right_on='geo_city', how='left')
        df_ga_hits = df_ga_hits.drop(['geo_country', 'geo_city'], axis=1)
        return df_ga_hits

    def transform_device_screen(df_ga_hits):
        list_index = df_ga_hits.index.tolist()
        list_screen_height = []
        list_screen_width = []
        for index in list_index:
            if df_ga_hits.loc[index, 'device_screen_resolution'] == '(not set)':
                screen_height = 0
                screen_width = 0
            else:
                screen_height = int(df_ga_hits.loc[index, 'device_screen_resolution'].split('x')[0])
                screen_width = int(df_ga_hits.loc[index, 'device_screen_resolution'].split('x')[1])
            list_screen_height.append(screen_height)
            list_screen_width.append(screen_width)
        df_ga_hits['screen_h_new'] = list_screen_height
        df_ga_hits['screen_w_new'] = list_screen_width
        df_ga_hits = df_ga_hits.drop(['device_screen_resolution'], axis=1)
        return df_ga_hits
    
    def p_transform_categorical_var(df_ga_hits):
        categorical_columns = ['utm_source', 'utm_medium', 'utm_campaign', 'utm_adcontent', 'utm_keyword', 
                               'device_category', 'device_os', 'device_brand', 'device_model','device_browser']
        for column in categorical_columns:
            dataframe = pd.DataFrame()
            dataframe = pd.read_csv(f'data/df_likelihood/df_likelihood_{column}.csv')
            dict_likelihood_value = {}
            for index in dataframe.index.tolist():
                dict_likelihood_value[dataframe.loc[index, column]] = dataframe.loc[index, 'likelihood']
            column_new = []
            list_value_from_ga_hits = df_ga_hits[column].tolist() 
            for value_from_ga_hits in list_value_from_ga_hits:
                column_new.append(float(dict_likelihood_value[value_from_ga_hits])) 
            df_ga_hits[f'{column}'] = column_new
        return df_ga_hits

    print(f'в {datetime.datetime.now().time()} начало предварительной обработки файлов') 
    df_ga_hits = pd.read_csv('data/ga_hits.csv')
    df_ga_hits = drop_dublicat_hit_number(df_ga_hits)                   # удаление дубликатов в первом фрейме
    df_ga_sessions = pd.read_csv('data/ga_sessions.csv')
    df_ga_hits = df_ga_hits.merge(df_ga_sessions, left_on='session_id', right_on='session_id', how='inner')
    df_ga_hits = transform_event_action(df_ga_hits)                     # замена в таргете на 0 и 1
    df_ga_hits = drop_columns(df_ga_hits)                               # удаление не учавствующих в модели колонок
    df_ga_hits = geo_frame_generation(df_ga_hits)                       # проверка наличия файлов данных и генерация при их отсутсвии
    df_ga_hits = p_categorical_var_frames_generation(df_ga_hits)        # проверка наличия файлов данных и генерация при их отсутсвии
    print(f'в {datetime.datetime.now().time()} окончание предварительной обработки файлов')

    X = df_ga_hits.drop(['event_action'], axis=1)
    y = df_ga_hits['event_action']
 
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ('numerical', numerical_transformer, make_column_selector(dtype_include=['int64', 'float64'])),
        ('categorical', categorical_transformer, make_column_selector(dtype_include=object))
    ])
    
    models = [
        RandomForestClassifier(),
        LogisticRegression(),
        MLPClassifier()
        ]

    best_score = .0
    best_model = None
    for model in models:
        pipe = Pipeline(steps=[
            ('df_fillna_other', FunctionTransformer(df_fillna_other)),
            ('transform_geo', FunctionTransformer(transform_geo)), 
            ('transform_device_screen', FunctionTransformer(transform_device_screen)),
            ('p_transform_categorical_var', FunctionTransformer(p_transform_categorical_var)),
            ('preprocessor', preprocessor),
            ('classifier', model)
        ])
        
        print(f'в {datetime.datetime.now().time()} старт pipeline')
        pipe.fit(X, y)
        print(f'в {datetime.datetime.now().time()} старт roc_auc_score')
        score = roc_auc_score(y, pipe.predict_proba(X)[:, 1])
        if score > best_score:
            best_score = score
            best_pipe = pipe
            best_model = model
            best_model_str = str(best_model)
        print(f'model:{model}: ROC-AUC={score}')
    print(f'best_model:{best_model}: best_ROC-AUC={best_score}')
    
    with open('data/hit_predict.pkl', 'wb') as file:
        dill.dump({
        'model': best_pipe,
        'metadata': {
            'name': 'intro in DS, final work',
            'author': 'iStanislav',
            'version': 1,
            'date': datetime.datetime.now(),
            'model': best_model_str,
            'ROC-AUC=': best_score
        }
    }, file, recurse=True)  
    print(f'в {datetime.datetime.now().time()} подготовка модели завершена')    
 
    
if __name__ == '__main__':
    main()