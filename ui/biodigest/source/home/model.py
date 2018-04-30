# -*- coding: utf-8 -*-

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import perceptron
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBClassifier

HAINAN_COLUMNS = [
    'Pig Manure',
    'Cassava',
    'Fish Waste Water',
    'Kitchen Food',
    'Municipial Fecal Residue',
    'Tea',
    'Chicken Litter',
    'Bagasse Feed',
    'Alcohol',
    'Chinese Medicine',
    'Energy Grass',
    'Banana Fruit Shafts',
    'Lemon',
    'Percolate',
    'Other'
]

SHENZHEN_COLUMNS = [
    'Kitchen Food',
    'Fruit and Vegetables',
    'Bread Paste',
    'Oil',
    'Diesel Waste Water',
    'Flour and Waste Oil',
    'Kitchen Food Paste',
    'Acid Feed',
    'Acid Discharge'
]

MULTITRAIN = True
NUM_TRAIN_ITERS = 30

class Model:
    def __init__(self):
        processed_data_hainan = self.get_hainan_data()
        self.hainan = self.hainan_model(processed_data_hainan)
        processed_data_shenzhen = self.get_shenzhen_data()
        self.shenzhen = self.shenzhen_model(processed_data_shenzhen)

    def get_hainan_data(self):
        excel = pd.ExcelFile(os.path.abspath('base/static/data/HainanClean_New.xlsx'))
        hainan = excel.parse("fulldf")
        hainan.columns = hainan.columns.str.replace('  ', '_')
        hainan.columns = hainan.columns.str.replace(' ', '_')
        hainan.columns = hainan.columns.str.replace('(', '')
        hainan.columns = hainan.columns.str.replace('（', '')
        hainan.columns = hainan.columns.str.replace(')', '')

        d = {'January':1, 'February':2, 'March':3, 'April':4, 'May':5, 'June':6,
             'July':7, 'August':8, 'September':9, 'October':10, 'November':11, 'December':12}
        hainan.Month = hainan.Month.map(d)
        hainan.BioCNG_Produced_Nm3 = hainan.BioCNG_Produced_m3.shift(-15)
        hainan.drop(hainan.tail(15).index,inplace=True)
        hainan = hainan[np.isfinite(hainan['Month'])]
        hainan = hainan[np.isfinite(hainan['Lemon_waste_t'])]
        hainan = hainan[np.isfinite(hainan['Percolate_t'])]
        hainan = hainan.replace(' ',0)
        hainan = hainan.replace('',0)
        hainan = hainan.replace('  ',0)
        hainan = hainan.drop(['Year', 'Month', 'Day', 'Month_#', 'Day_#', 'Raw_Biogas_Produced_m3', 'BioCNG_Sold_m3', 'Vehicle_use_m3',
               'Liquid_Fertilizer_Produced_t', 'Solid_fertilizer_produced_t',
               'Wastewater_flow_to_WWTP_unit?', 'Solid_residues_kg','50%_NaOH/kg', 'FeCl2/kg', 'PAM/kg',
               'Defoamer/kg', 'Project_electricity_use/kWh',
               'Office_space_electricity_use/kWh', 'Water/m3', 'Diesel/L'], axis=1)
        return hainan

    def get_shenzhen_data(self):
        shenzhen = pd.read_csv("base/static/data/Shenzhen_useful.csv")
        shenzhen['acid_feed'] = shenzhen['1_acidification_hydrolysis_tank_feed_'] \
                                                     + shenzhen['2_acidification_hydrolysis_tank_feed_']
        shenzhen['acid_discharge']  = shenzhen['1_acidification_hydrolysis_tank_discharge_']\
                                                     + shenzhen['2_acidification_hydrolysis_tank_discharge_']
        shenzhen['anaerobic_feed'] = shenzhen['1_Anaerobic_tank_slurry_feed_'] \
                                                     + shenzhen['2_Anaerobic_tank_slurry_feed_']
        shenzhen['anaerobic_cumuprod'] = shenzhen['1_Anaerobic_tank_biogas_cumulative_production_'] \
                                                     + shenzhen['2_anaerobic_tank_biogas_cumulative_production_']
        shenzhen['anaerobic_dailyoutput'] = shenzhen['1_anaerobic_tank_biogas_daily_output_'] \
                                                     + shenzhen['2_anaerobic_tank_biogas_daily_output_']
        shenzhen = shenzhen.drop(['1_acidification_hydrolysis_tank_feed_','2_acidification_hydrolysis_tank_feed_',\
                       '1_acidification_hydrolysis_tank_discharge_','2_acidification_hydrolysis_tank_discharge_',\
                       '1_Anaerobic_tank_slurry_feed_','2_Anaerobic_tank_slurry_feed_',\
                       '1_Anaerobic_tank_biogas_cumulative_production_','2_anaerobic_tank_biogas_cumulative_production_',\
                       '1_anaerobic_tank_biogas_daily_output_','2_anaerobic_tank_biogas_daily_output_'],axis = 1)

        shenzhen.acid_feed = shenzhen.acid_feed.shift(-15)
        shenzhen.acid_discharge = shenzhen.acid_discharge.shift(-15)
        shenzhen.anaerobic_feed = shenzhen.anaerobic_feed.shift(-15)
        shenzhen.anaerobic_cumuprod = shenzhen.anaerobic_cumuprod.shift(-15)
        shenzhen.anaerobic_dailyoutput = shenzhen.anaerobic_dailyoutput.shift(-15)
        shenzhen = shenzhen.drop(['Unnamed: 0'], axis=1)
        shenzhen = shenzhen[:-15]

        return shenzhen

    def hainan_model(self, data):
        for col in data.columns[1:]:
            data['1/'+col] = 1/(data[col])
        data.replace(float('inf'), 0, inplace=True)
        for col in data.columns[1:]:
            data[col+"**2"] = (data[col])**2

        curr_model = None
        curr_accuracy = 0.0

        if MULTITRAIN:
            for i in range(NUM_TRAIN_ITERS):
                print "Training Hainan model iteration", i + 1, "...",
                X_train, y_train, X_test, y_test = self.split_hainan_data(data)
                knn = KNeighborsRegressor(n_neighbors=7)
                knn.fit(X_train, y_train)
                acc = knn.score(X_test, y_test)
                print "Accuracy: ", acc
                if acc > curr_accuracy:
                    curr_model = knn
                    curr_accuracy = acc

        self.hainan_accuracy = curr_accuracy

        print "Selecting Hainan model with accuracy", curr_accuracy
        return curr_model

    def shenzhen_model(self, data):
        data['anaerobic_dailyoutput'] = pd.cut(data['anaerobic_dailyoutput'],bins = 3)
        data.anaerobic_dailyoutput = pd.factorize(data.anaerobic_dailyoutput)[0]

        for col in data.columns[:10]:
            data['1/'+col] = 1/(data[col])
        for col in data.columns[:10]:
            data[col+"**2"] = (data[col])**2
        for col in data.columns[:10]:
            data[col+"log"] = np.log(data[col])

        data.replace(float('inf'), 0, inplace = True)
        data.replace(float('-inf'), 0, inplace = True)

        curr_rf_model = None
        curr_rf_accuracy = 0.0
        curr_xgb_model = None
        curr_xgb_accuracy = 0.0

        if MULTITRAIN:
            for i in range(NUM_TRAIN_ITERS):
                print "Training Shenzhen random forest model iteration", i + 1, "...",
                X_traincla, y_traincla, X_testcla, y_testcla = self.split_shenzhen_data(data)
                random_forest = RandomForestClassifier(n_estimators = 50)
                random_forest.fit(X_traincla, y_traincla)
                rf_acc = random_forest.score(X_testcla, y_testcla)

                print "Accuracy: ", rf_acc
                if rf_acc > curr_rf_accuracy:
                    curr_rf_model = random_forest
                    curr_rf_accuracy = rf_acc

            # for i in range(NUM_TRAIN_ITERS):
            #     print "Training Shenzhen XGBoost model iteration", i + 1, "...",
            #     X_traincla, y_traincla, X_testcla, y_testcla = self.split_shenzhen_data(data)
            #     xgb1 = XGBClassifier(
            #         learning_rate = 0.1,
            #         n_estimators=1000,
            #         max_depth=3,
            #         min_child_weight=5,
            #         gamma=0.2,
            #         subsample=0.8,
            #         colsample_bytree=0.8,
            #         objective= 'binary:logistic',
            #         nthread=8,
            #         scale_pos_weight=8,
            #         seed=27)
            #     xgb1.fit(X_traincla, y_traincla)
            #     xgb_acc = xgb1.score(X_testcla, y_testcla)

            #     print "Accuracy: ", xgb_acc
            #     if xgb_acc > curr_xgb_accuracy:
            #         curr_xgb_model = xgb1
            #         curr_xgb_accuracy = xgb_acc

        if curr_xgb_accuracy > curr_rf_accuracy:
            print "Selecting Shenzhen XGB model with accuracy", curr_xgb_accuracy, "Random forest scored", curr_rf_accuracy
            return curr_xgb_model
        else:
            print "Selecting Shenzhen random forest model with accuracy", curr_rf_accuracy, "XGBoost scored", curr_xgb_accuracy
            return curr_rf_model

    def predict_hainan(self, x):
        xs = x[HAINAN_COLUMNS]
        for col in xs.columns:
            xs['1/'+col] = 1/(xs[col])
        xs.replace(float('inf'), 0, inplace=True)
        xs.replace(float('-inf'), 0, inplace=True)
        for col in xs.columns:
            xs[col+"**2"] = (xs[col])**2
        return self.hainan.predict(xs) / 5.0

    def predict_shenzhen(self, x):
        xs = x[SHENZHEN_COLUMNS]
        total_col = x['Kitchen Food'] + x['Bread Paste'] + x['Fruit and Vegetables']
        xs.insert(4, 'Total Waste', total_col)
        for col in xs.columns[:10]:
            xs['1/'+col] = 1/(xs[col])
        for col in xs.columns[:10]:
            xs[col+"**2"] = (xs[col])**2
        for col in xs.columns[:10]:
            xs[col+"log"] = np.log(xs[col])

        xs.replace(float('inf'), 0, inplace=True)
        xs.replace(float('-inf'), 0, inplace=True)

        replace = {
            0: 2089.5835 / 5,
            1: 6250.0 / 5,
            2: 10416.6665 / 5
        }

        pred = self.shenzhen.predict(xs)
        predCopy = np.copy(pred)
        for k, v in replace.iteritems(): predCopy[pred==k] = v

        return predCopy

    def split_hainan_data(self, data):
        hainan_train, hainan_test = train_test_split(data, test_size=0.2)
        X_train = hainan_train[['Pig_Manure_t', 'Cassava_t', 'Fish_waste_water_t',
           'Kitchen_food_waste_t', 'Municipal_fecal_residue_t', 'Tea_waste_t',
           'Chicken_litter_t', 'Bagasse_feed_t', 'Alcohol_waste_t',
           'Chinese_medicine_waste_t', 'Energy_grass_t', 'Banana_fruit_shafts_t',
           'Lemon_waste_t', 'Percolate_t', 'Other_waste_t', '1/Pig_Manure_t',
           '1/Cassava_t', '1/Fish_waste_water_t', '1/Kitchen_food_waste_t',
           '1/Municipal_fecal_residue_t', '1/Tea_waste_t', '1/Chicken_litter_t',
           '1/Bagasse_feed_t', '1/Alcohol_waste_t', '1/Chinese_medicine_waste_t',
           '1/Energy_grass_t', '1/Banana_fruit_shafts_t', '1/Lemon_waste_t',
           '1/Percolate_t', '1/Other_waste_t', 'Pig_Manure_t**2', 'Cassava_t**2',
           'Fish_waste_water_t**2', 'Kitchen_food_waste_t**2',
           'Municipal_fecal_residue_t**2', 'Tea_waste_t**2', 'Chicken_litter_t**2',
           'Bagasse_feed_t**2', 'Alcohol_waste_t**2',
           'Chinese_medicine_waste_t**2', 'Energy_grass_t**2',
           'Banana_fruit_shafts_t**2', 'Lemon_waste_t**2', 'Percolate_t**2',
           'Other_waste_t**2', '1/Pig_Manure_t**2', '1/Cassava_t**2',
           '1/Fish_waste_water_t**2', '1/Kitchen_food_waste_t**2',
           '1/Municipal_fecal_residue_t**2', '1/Tea_waste_t**2',
           '1/Chicken_litter_t**2', '1/Bagasse_feed_t**2', '1/Alcohol_waste_t**2',
           '1/Chinese_medicine_waste_t**2', '1/Energy_grass_t**2',
           '1/Banana_fruit_shafts_t**2', '1/Lemon_waste_t**2', '1/Percolate_t**2',
           '1/Other_waste_t**2']]

        y_train = hainan_train.BioCNG_Produced_m3

        # Predict on the test data
        X_test = hainan_test[['Pig_Manure_t', 'Cassava_t', 'Fish_waste_water_t',
               'Kitchen_food_waste_t', 'Municipal_fecal_residue_t', 'Tea_waste_t',
               'Chicken_litter_t', 'Bagasse_feed_t', 'Alcohol_waste_t',
               'Chinese_medicine_waste_t', 'Energy_grass_t', 'Banana_fruit_shafts_t',
               'Lemon_waste_t', 'Percolate_t', 'Other_waste_t', '1/Pig_Manure_t',
               '1/Cassava_t', '1/Fish_waste_water_t', '1/Kitchen_food_waste_t',
               '1/Municipal_fecal_residue_t', '1/Tea_waste_t', '1/Chicken_litter_t',
               '1/Bagasse_feed_t', '1/Alcohol_waste_t', '1/Chinese_medicine_waste_t',
               '1/Energy_grass_t', '1/Banana_fruit_shafts_t', '1/Lemon_waste_t',
               '1/Percolate_t', '1/Other_waste_t', 'Pig_Manure_t**2', 'Cassava_t**2',
               'Fish_waste_water_t**2', 'Kitchen_food_waste_t**2',
               'Municipal_fecal_residue_t**2', 'Tea_waste_t**2', 'Chicken_litter_t**2',
               'Bagasse_feed_t**2', 'Alcohol_waste_t**2',
               'Chinese_medicine_waste_t**2', 'Energy_grass_t**2',
               'Banana_fruit_shafts_t**2', 'Lemon_waste_t**2', 'Percolate_t**2',
               'Other_waste_t**2', '1/Pig_Manure_t**2', '1/Cassava_t**2',
               '1/Fish_waste_water_t**2', '1/Kitchen_food_waste_t**2',
               '1/Municipal_fecal_residue_t**2', '1/Tea_waste_t**2',
               '1/Chicken_litter_t**2', '1/Bagasse_feed_t**2', '1/Alcohol_waste_t**2',
               '1/Chinese_medicine_waste_t**2', '1/Energy_grass_t**2',
               '1/Banana_fruit_shafts_t**2', '1/Lemon_waste_t**2', '1/Percolate_t**2',
               '1/Other_waste_t**2']]
        y_test = hainan_test.BioCNG_Produced_m3

        return (X_train, y_train, X_test, y_test)

    def split_shenzhen_data(self, data):
        shenzhen_train, shenzhen_test = train_test_split(data, test_size=0.15)

        X_traincla = shenzhen_train[['Kitchen_waste_', 'Fruit_and_vegetable_waste_', 'Bread_Paste_',\
                                     'Waste_oil_', 'Total_Waste_', 'Diesel_waste_water_',\
                                     'Flour_and_waste_oil_', 'Kitchen_waste_paste_', 'acid_feed',\
                                     'acid_discharge','1/Kitchen_waste_', '1/Fruit_and_vegetable_waste_',\
                                     '1/Bread_Paste_', '1/Waste_oil_', '1/Total_Waste_',\
                                     '1/Diesel_waste_water_', '1/Flour_and_waste_oil_',\
                                     '1/Kitchen_waste_paste_', '1/acid_feed', '1/acid_discharge',\
                                     'Kitchen_waste_**2', 'Fruit_and_vegetable_waste_**2', \
                                     'Bread_Paste_**2','Waste_oil_**2', 'Total_Waste_**2', \
                                     'Diesel_waste_water_**2','Flour_and_waste_oil_**2', \
                                     'Kitchen_waste_paste_**2', 'acid_feed**2','acid_discharge**2',\
                                     'Kitchen_waste_log','Fruit_and_vegetable_waste_log', \
                                     'Bread_Paste_log', 'Waste_oil_log','Total_Waste_log', \
                                     'Diesel_waste_water_log', 'Flour_and_waste_oil_log',\
                                     'Kitchen_waste_paste_log', 'acid_feedlog', 'acid_dischargelog']]
        y_traincla = shenzhen_train.anaerobic_dailyoutput
        X_testcla = shenzhen_test[['Kitchen_waste_', 'Fruit_and_vegetable_waste_', 'Bread_Paste_',\
                                     'Waste_oil_', 'Total_Waste_', 'Diesel_waste_water_',\
                                     'Flour_and_waste_oil_', 'Kitchen_waste_paste_', 'acid_feed',\
                                     'acid_discharge','1/Kitchen_waste_', '1/Fruit_and_vegetable_waste_',\
                                     '1/Bread_Paste_', '1/Waste_oil_', '1/Total_Waste_',\
                                     '1/Diesel_waste_water_', '1/Flour_and_waste_oil_',\
                                     '1/Kitchen_waste_paste_', '1/acid_feed', '1/acid_discharge',\
                                     'Kitchen_waste_**2', 'Fruit_and_vegetable_waste_**2', \
                                     'Bread_Paste_**2','Waste_oil_**2', 'Total_Waste_**2', \
                                     'Diesel_waste_water_**2','Flour_and_waste_oil_**2', \
                                     'Kitchen_waste_paste_**2', 'acid_feed**2','acid_discharge**2',\
                                     'Kitchen_waste_log','Fruit_and_vegetable_waste_log', \
                                     'Bread_Paste_log', 'Waste_oil_log','Total_Waste_log', \
                                     'Diesel_waste_water_log', 'Flour_and_waste_oil_log',\
                                     'Kitchen_waste_paste_log', 'acid_feedlog', 'acid_dischargelog']]
        y_testcla = shenzhen_test.anaerobic_dailyoutput
        return (X_traincla, y_traincla, X_testcla, y_testcla)
