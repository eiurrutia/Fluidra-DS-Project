import sys
import os
sys.path.append(".")
import re
import copy
import pandas as pd
import numpy as np
from datetime import timedelta


import pandas as pd
import unicodedata

def remove_accents(text):
    return ''.join(c for c in unicodedata.normalize('NFD', text) if unicodedata.category(c) != 'Mn')

def get_modified_dataframe():
    # Ruta completa del archivo CSV
    ruta_archivo = 'data/processed_csv/df_operators_participation.csv'

    # Leer el archivo CSV en un DataFrame
    df_operators_participation = pd.read_csv(ruta_archivo)

    df_operators_participation['line'] = df_operators_participation['line'].apply(lambda x: remove_accents(x))

    # Eliminar los registros que contienen 'PREFILTRO' en la columna 'line'
    df_operators_participation = df_operators_participation[~df_operators_participation['line'].str.contains('PREFILTRO')]

    return df_operators_participation

def get_active_operators_by_line(dataframe, available_operators, available_lines):
    lineas_operarios_disponibles = {}

    for index, row in dataframe.iterrows():
        operador = str(row['operator_id'])
        linea = row['line']

        if linea in available_lines and operador in available_operators:
            if linea in lineas_operarios_disponibles:
                if operador not in lineas_operarios_disponibles[linea]:
                    lineas_operarios_disponibles[linea].append(operador)
            else:
                lineas_operarios_disponibles[linea] = [operador]

    return lineas_operarios_disponibles


def get_active_lines_by_order(dataframe, available_lines, orders):
    def lineas_por_bomba(dataframe):
        bombas_lineas = {}

        for index, row in dataframe.iterrows():
            linea = row['line']
            bomba = row['bomb_type']

            if bomba in bombas_lineas:
                if linea not in bombas_lineas[bomba]:
                    bombas_lineas[bomba].append(linea)
            else:
                bombas_lineas[bomba] = [linea]

        return bombas_lineas
    
    df = dataframe.copy()
    df['line'] = df['line'].apply(lambda x: remove_accents(x))

    bombas_lineas = lineas_por_bomba(df)

    active_lines = {}

    for order in orders:
        order_id = order['id']
        bomba = order['bomb_type']
        valid_lines = [line for line in available_lines if line in bombas_lineas.get(bomba, [])]

        if valid_lines:
            active_lines[order_id] = valid_lines

    return active_lines


