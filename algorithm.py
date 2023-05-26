import sys
import os
sys.path.append(".")
import re
import copy
import random
import itertools
import pickle
import model
import pandas as pd
import numpy as np
from datetime import timedelta
from deap import creator, base, tools, algorithms
import warnings
from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings(action='ignore', category=UserWarning)


unique_individuals = set()
evaluations_count = 0
ORDERS = []
production_date = None
available_operators = []
available_lines = []
all_operators = []
all_lines = []

with open('classifier_model.pickle', 'rb') as file:
        model.classifier_model = pickle.load(file)

def run_algorithm():
    global evaluations_count, unique_individuals
    unique_individuals = set()
    evaluations_count = 0
    def individual_to_str(individual):
        return str(sorted((assignment['order'], tuple(sorted(assignment['operators'])), assignment['line']) for assignment in individual))

    def evaluate(individual):
        global evaluations_count, unique_individuals
        evaluations_count += 1
        # Agrega el individuo al conjunto de individuos únicos
        unique_individuals.add(individual_to_str(individual))

        df_order_list = [] 
        for assignment in individual:

            order_id = assignment['order']
            operators = assignment['operators']
            line = assignment['line']

            # Encuentra la orden correspondiente en la lista de órdenes
            order = next(order for order in ORDERS if order['id'] == order_id)
        
            order_df = model.parse_to_model(
                order_id, operators, line, order['plan_qty'], order['theorical_time'],
                production_date, all_operators, all_lines
            )
            df_order_list.append(order_df)
        
        df_orders = pd.concat(df_order_list, ignore_index=True)
        prediction = model.classifier_model.predict(df_orders[model.classifier_model.feature_names_in_])
        prediction_proba = model.classifier_model.predict_proba(df_orders[model.classifier_model.feature_names_in_])

        df_orders['prediction'] = prediction
        df_orders['prediction_proba'] = prediction_proba[:, 1]

        fitness = df_orders.prediction.sum() + (df_orders.prediction_proba.sum() / 10000) 
        return fitness,

    operator_combos = []
    max_operators_per_order = 4
    for r in range(1, max_operators_per_order + 1):
        operator_combos.extend(itertools.combinations(available_operators, r))
    operator_line_combos = list(itertools.product(operator_combos, available_lines))

    # Definimos los tipos para el problema de optimización
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))  # Maximizamos la función de fitness
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()

    # La función initIndividual crea un individuo, que es una lista de asignaciones de operadores y líneas a las órdenes
    def initIndividual():
        attempts = 0
        max_attempts = 10000  # ajusta este número según sea necesario
        while attempts < max_attempts:
            individual = []
            available_operators_for_this_individual = available_operators.copy()  # Crear una copia de la lista de operadores disponibles
            available_lines_for_this_individual = available_lines.copy()  # Crear una copia de la lista de líneas disponibles
            for order in ORDERS:
                # Elegir un conjunto de operadores que aún no se hayan asignado
                valid_operator_combos = [combo for combo in operator_combos if set(combo).issubset(available_operators_for_this_individual)]
                if not valid_operator_combos:  # Si no hay más operadores disponibles, no podemos generar un individuo válido
                    break
                operators = random.choice(valid_operator_combos)
                for operator in operators:
                    available_operators_for_this_individual.remove(operator)  # El operador ya no está disponible para las siguientes órdenes

                # Elegir una línea que aún no se haya asignado
                if not available_lines_for_this_individual:  # Si no hay más líneas disponibles, no podemos generar un individuo válido
                    break
                line = random.choice(available_lines_for_this_individual)
                available_lines_for_this_individual.remove(line)  # La línea ya no está disponible para las siguientes órdenes

                individual.append({'order': order['id'], 'operators': operators, 'line': line})

            if len(individual) == len(ORDERS):  # si el individuo es válido (tiene todas las órdenes)
                print(f'[INIT INDIVIDUAL]: {attempts}')
                print(individual)
                return creator.Individual(individual)

            attempts += 1

        # Si hemos llegado a este punto, es porque todos los intentos de generar un individuo válido han fallado.
        raise Exception(f"No se pudo generar un individuo válido después de {max_attempts} intentos.")


    toolbox.register("individual", initIndividual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # Definimos los operadores genéticos
    toolbox.register("evaluate", evaluate)


    # Definimos la función de crossovr
    def custom_crossover(ind1, ind2):
        # Crea copias de los individuos para no modificar los originales
        new_ind1 = copy.deepcopy(ind1)
        new_ind2 = copy.deepcopy(ind2)
        
        for order in ORDERS:
            order_id = order['id']

            # Encuentra las asignaciones para esta orden en ambos individuos
            assignment1 = next(assignment for assignment in new_ind1 if assignment['order'] == order_id)
            assignment2 = next(assignment for assignment in new_ind2 if assignment['order'] == order_id)
            
            # Intercambia las asignaciones de operadores y líneas
            assignment1['operators'], assignment2['operators'] = assignment2['operators'], assignment1['operators']
            assignment1['line'], assignment2['line'] = assignment2['line'], assignment1['line']


        # Verifica si las nuevas asignaciones violan las restricciones
        violation = False
        for new_ind in [new_ind1, new_ind2]:
            for i in range(len(new_ind1)):
                for j, assignment in enumerate(new_ind):
                    if j != i:
                        # Verifica si la nueva línea se repite en otras asignaciones
                        if new_ind[i]['line'] == assignment['line']:
                            violation = True
                            break

                        # Verifica si los nuevos operadores se repiten en otras asignaciones
                        if any(op in new_ind[i]['operators'] for op in assignment['operators']):
                            violation = True
                            break
        if violation:
            print('[ERROR-CROSSOVER VIOLATION]:')
            print(new_ind1)
            print(new_ind2)
            return ind1, ind2
        
        print('[CROSSOVER GENERATED]')
        return new_ind1, new_ind2

    toolbox.register("mate", custom_crossover)

    def custom_mutation(individual, indpb):
        attempts = 0
        for i in range(len(individual)):
            if random.random() < indpb:
                attempts = 0
                max_attempts = 10000  # ajusta este número según sea necesario
                while attempts < max_attempts:
                    # Selecciona una nueva línea que aún no se haya asignado
                    new_line = random.choice(available_lines)

                    # Verifica si la nueva asignación viola las restricciones
                    violation = False
                    new_operators = random.choice(operator_combos)
                    for j, assignment in enumerate(individual):
                        if j != i:
                            # Verifica si la nueva línea se repite en otras asignaciones
                            if new_line == assignment['line']:
                                violation = True
                                break

                            # Verifica si los nuevos operadores se repiten en otras asignaciones
                            if any(op in new_operators for op in assignment['operators']):
                                violation = True
                                break

                    if not violation:
                        # Si la asignación es válida, actualiza la asignación en el individuo
                        individual[i]['line'] = new_line
                        individual[i]['operators'] = tuple(new_operators)  # Convierte la lista en una tupla
                        break

                    attempts += 1
        print(f'[MUTATION GENERATED]: {attempts or 0} intentos')
        return individual,

    toolbox.register("mutate", custom_mutation, indpb=0.4)
    toolbox.register("select", tools.selTournament, tournsize=3)

    # Inicializamos la población y ejecutamos el algoritmo genético
    population = toolbox.population(n=50)
    result = algorithms.eaSimple(population, toolbox, cxpb=0.5, mutpb=0.2, ngen=40, verbose=False)
    best_individual = tools.selBest(result[0], 3)[0]
    best_fitness = best_individual.fitness.values

    print('···· RESULTADOS ····')
    print(f'Se han realizado {evaluations_count} evaluaciones de combinaciones distintas.')
    print(f'El mejor combinación es: {best_individual}')
    def selectUnique(individuals, k):
        unique_individuals = []
        for ind in individuals:
            if not any(ind2 == ind for ind2 in unique_individuals):
                unique_individuals.append(ind)
            if len(unique_individuals) == k:
                break
        return unique_individuals
    
    def selectUniqueAndHighFitness(individuals, k):
        unique_individuals = []
        last_fitness = None
        count_same_fitness = 0

        for ind in sorted(individuals, key=lambda x: x.fitness.values[0], reverse=True):
            if last_fitness is None or ind.fitness.values[0] < last_fitness:
                last_fitness = ind.fitness.values[0]
                count_same_fitness = 1
            elif ind.fitness.values[0] == last_fitness:
                count_same_fitness += 1
                if count_same_fitness > 2:
                    continue
            unique_individuals.append(ind)
            if len(unique_individuals) == k:
                break
        return unique_individuals

    #unique_individuals = selectUnique(tools.selNSGA2(result[0], len(result[0])), 10)
    unique_individuals = selectUniqueAndHighFitness(tools.selNSGA2(result[0], len(result[0])), 10)

    best_fitnesses = [ind.fitness.values for ind in unique_individuals]

    for i in range(len(unique_individuals)):
        print('########################')
        print(f'\nMejor combinación {i+1}:')
        print(f'Valor de fitness: {best_fitnesses[i]}')
        for assignment in unique_individuals[i]:
            order_id = assignment['order']
            operators = assignment['operators']
            line = assignment['line']
            print(f'  Orden: {order_id}, Operadores: {operators}, Línea: {line}')
    
    return unique_individuals, best_fitnesses, evaluations_count
