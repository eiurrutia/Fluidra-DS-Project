import pickle
import numpy as np
import datetime
import pandas as pd
import xgboost as xgb
import warnings
from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings(action='ignore', category=UserWarning)

scaler = {}
classifier_model = {}

def parse_to_model(order, operators, line, plan_qty, theorical_time, production_date, all_operators, all_lines):
    with open('scaler.pkl', 'rb') as file:
        scaler = pickle.load(file)
    

    columns = ['good_qty_scaled', 'theorical_time_scaled']+['operator_' + str(i) for i in all_operators]+['line_' + str(i) for i in all_lines]+['grupo_qty_0', 'grupo_qty_1', 'grupo_qty_2', 'grupo_qty_3', 
               'grupo_qty_4', 'weekday_0', 'weekday_1', 'weekday_2', 'weekday_3', 'weekday_4', 'weekday_5', 'weekday_6', 'month_1', 'month_2', 'month_3', 'month_4', 
               'month_5', 'month_6', 'month_7', 'month_8', 'month_9', 'month_10', 'month_11', 'month_12', 'time_of']

    order_of = pd.DataFrame(np.zeros((1, len(columns))), columns=columns)

    scaled = scaler.transform([[plan_qty, theorical_time]])[0]
    
    order_of['good_qty_scaled'] = scaled[0]
    order_of['theorical_time_scaled'] = scaled[1]

    for operator in operators:
        order_of['operator_' + str(operator)] = 1

    order_of['line_' + line] = 1

    weekday = production_date.weekday()
    order_of['weekday_' + str(weekday)] = 1

    mes = production_date.month
    order_of['month_' + str(mes)] = 1

    # Definir los límites de los grupos
    grupo_1_limite = 0.2
    grupo_2_limite = 0.4
    grupo_3_limite = 0.6
    grupo_4_limite = 0.8

    order_of['grupo_qty_0'] = np.where(order_of['good_qty_scaled'] < grupo_1_limite, 1, 0)
    order_of['grupo_qty_1'] = np.where((order_of['good_qty_scaled'] > grupo_1_limite) & (order_of['good_qty_scaled'] <= grupo_2_limite), 1, 0)
    order_of['grupo_qty_2'] = np.where((order_of['good_qty_scaled'] > grupo_2_limite) & (order_of['good_qty_scaled'] <= grupo_3_limite), 1, 0)
    order_of['grupo_qty_3'] = np.where((order_of['good_qty_scaled'] > grupo_3_limite) & (order_of['good_qty_scaled'] <= grupo_4_limite), 1, 0)
    order_of['grupo_qty_4'] = np.where(order_of['good_qty_scaled'] > grupo_4_limite, 1, 0)

    order_of['time_of'] = order_of['good_qty_scaled'] * order_of['theorical_time_scaled']

    return order_of


with open('classifier_model.pickle', 'rb') as file:
        classifier_model = pickle.load(file)