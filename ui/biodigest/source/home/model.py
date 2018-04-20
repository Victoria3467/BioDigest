# -*- coding: utf-8 -*-

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
import statsmodels.api as sm
from statsmodels.formula.api import ols
from sklearn.linear_model import perceptron
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsRegressor

class Model:
  def __init__(self):
    processed_data_hainan = self.get_hainan_data()
    self.hainan = self.hainan_model(processed_data_hainan)

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

  def hainan_model(self, data):
    for col in data.columns[1:]:
      data['1/'+col] = 1/(data[col])
    data.replace(float('inf'), 0, inplace=True)
    for col in data.columns[1:]:
      data[col+"**2"] = (data[col])**2
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

    knn = KNeighborsRegressor(n_neighbors=7)
    knn.fit(X_train, y_train)

    self.hainan_accuracy = knn.score(X_test, y_test)

    return knn

  def predict_hainan(self, x):
    for col in x.columns:
      x['1/'+col] = 1/(x[col])
    x.replace(float('inf'), 0, inplace=True)
    for col in x.columns:
      x[col+"**2"] = (x[col])**2
    return self.hainan.predict(x)
