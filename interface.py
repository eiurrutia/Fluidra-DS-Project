import streamlit as st
import pandas as pd
import algorithm as alg
import model as m
import utils
import io
import time
import contextlib
pd.set_option('display.float_format', '{:.0f}'.format)

def get_logs(func, *args, **kwargs):
    log_output = io.StringIO()
    with contextlib.redirect_stdout(log_output):
        func_output = func(*args, **kwargs)
    return log_output.getvalue(), func_output

st.set_page_config(
    page_title="Fluidra App",
    page_icon="🌀",
    layout="centered",
    initial_sidebar_state="collapsed",
)

def get_best_combinations():
    file_users = file_lines = file_orders = None

    st.header('Asignación de órdenes de fabricación')

    if file_users is None or file_lines is None or file_orders is None:
        st.header('Cargar archivos excel')
        st.subheader('Operadores')
        file_users = st.file_uploader('Subir archivo de usuarios habilitados', type='xlsx')
        st.subheader('Líneas')
        file_lines = st.file_uploader('Subir archivo de líneas habilitadas', type='xlsx')
        file_orders = True
        st.subheader('Órdenes')
        file_orders = st.file_uploader('Subir archivo de órdenes de fabricación', type='xlsx')

    if file_users is not None and file_lines is not None and file_orders is not None:
        st.session_state.files_loaded = True
        st.header('Archivos cargados correctamente')
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Cargamos los archivos excel con pandas
        users_df = pd.read_excel(file_users, index_col=None).sort_values(by=['active', 'operator'], ascending=False).reset_index(drop=True)
        users_df['operator'] = users_df['operator'].astype(int)
        users_df['active'] = users_df['active'].astype(bool)
        lines_df = pd.read_excel(file_lines)
        orders_df = pd.read_excel(file_orders)
        orders_df['theorical_time'] = orders_df['theorical_time'].astype(float)
        orders_df['plan_qty'] = orders_df['plan_qty'].astype(int)


        ### OPERATORS
        st.subheader('Operadores Activos')
        # Generamos una lista de casillas de verificación para cada operador
        with st.expander("Modificar operadores activos"):
            active_operators = []
            for i in range(len(users_df)):
                is_active = st.checkbox(f"{users_df.loc[i, 'operator']}", value=bool(users_df.loc[i, 'active']))
                active_operators.append(is_active)

            # Actualizamos el DataFrame de operadores con la selección del usuario
            users_df['active'] = active_operators
                

        st.markdown(users_df[users_df.active == True][['operator']].to_html(index=False), unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        st.write(f"Total de operadores: {len(users_df)}")
        st.write(f":green[Total de operadores activos: {len(users_df[users_df.active == True])}]")


        ### LINES
        st.subheader('Líneas')
        # Generamos una lista de casillas de verificación para cada operador
        with st.expander("Modificar lineas activas"):
            active_lines = []
            for i in range(len(lines_df)):
                is_active = st.checkbox(f"{lines_df.loc[i, 'line']}", value=bool(lines_df.loc[i, 'active']))
                active_lines.append(is_active)

            # Actualizamos el DataFrame de operadores con la selección del usuario
            lines_df['active'] = active_lines

        st.markdown(lines_df[lines_df.active == True][['line']].to_html(index=False), unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        st.write(f"Total de líneas: {len(lines_df)}")
        st.write(f":green[Total de lineas activas: {len(lines_df[lines_df.active == True])}]")
        
        st.subheader('Órdenes de Fabricación')
        orders_df_copy = orders_df.copy()
        orders_df_copy['theorical_time'] = orders_df_copy['theorical_time'].apply(lambda x: f'{x:.2f}')
        st.markdown(orders_df_copy[['id', 'bomb_type', 'theorical_time', 'plan_qty']].to_html(index=False), unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        st.write(f"Total de órdenes de fabricación cargadas: {len(orders_df)}")
        

        st.subheader('Seleccionar fecha de producción')
        production_date = st.date_input('Fecha de producción', value=pd.to_datetime('today').date())
        
        st.markdown("<br>", unsafe_allow_html=True)
        ### Algotimo
        st.subheader('Algoritmo')
        if st.button("Ejecutar algoritmo"):
            
            # Configuramos variables globales del algoritmo
            alg.production_date = production_date
            alg.available_operators = users_df[users_df['active']]['operator'].tolist()
            alg.available_lines = lines_df[lines_df['active']]['line'].tolist()
            alg.all_operators = users_df['operator'].tolist()
            alg.all_lines = lines_df['line'].tolist()
            df_operators_participation = utils.get_modified_dataframe()
            alg.ORDERS = orders_df[['id', 'bomb_type', 'theorical_time', 'plan_qty']].to_dict('records')

            df_bombs_constraint = pd.read_excel('data/to_load/constraint/bombs_constraint.xlsx')
            df_bombs_constraint['Linea'] = df_bombs_constraint['Linea'].apply(lambda x: [i.strip() for i in x.split(',')])
            active_lines_by_bomb = df_bombs_constraint.set_index('Bomb_type')['Linea'].to_dict()
            alg.active_lines_by_order = {}
            for order in alg.ORDERS:
                bomb_type = str(order['bomb_type'])  # asegurarse de que bomb_type sea string
                order_id = order['id']
                if bomb_type in active_lines_by_bomb:
                    alg.active_lines_by_order[order_id] = active_lines_by_bomb[bomb_type]
                else:
                    alg.active_lines_by_order[order_id] = []  # si el bomb_type no está en el diccionario, añadir una lista vacía

            df_operators_constraint = pd.read_excel('data/to_load/constraint/operators_constraint.xlsx')
            df_operators_constraint['Lineas'] = df_operators_constraint['Lineas'].apply(lambda x: [i.strip() for i in x.split(',')])
            active_lines_by_operator = df_operators_constraint.set_index('Operario')['Lineas'].to_dict()
            alg.active_operators_by_line = {}
            for _, row in df_operators_constraint.iterrows():
                for line in row['Lineas']:
                    if line in alg.active_operators_by_line:
                        alg.active_operators_by_line[line].append(row['Operario'])
                    else:
                        alg.active_operators_by_line[line] = [row['Operario']]

            print('available_operators', alg.available_operators)
            print('available_lines', alg.available_lines)
            print('all_operators', alg.all_operators)
            print('all_lines', alg.all_lines)
            print('ORDERS', alg.ORDERS)
            print('active_operators_by_line', alg.active_operators_by_line)
            print('active_lines_by_order', alg.active_lines_by_order)
            logs = result = None
            st.write("Ejecutando algoritmo...")
            
            progress_bar = st.progress(0)
            for i in range(100):
                time.sleep(0.1) 
                progress_bar.progress(i + 1)
            
            start_time = time.time()
            logs, (result_unique_individuals, result_best_fitnesses, evaluations_count) = get_logs(alg.run_algorithm)
            end_time = time.time()
            execution_time = (end_time - start_time) / 60.0
           
            custom_css = """
            <style>
                .result-header {
                    color: white;
                    background-color: #3F51B5;
                    padding: 20px;
                    border-radius: 5px;
                }
                
                .log-header {
                    color: white;
                    background-color: #FFA500;
                    padding: 20px;
                    border-radius: 5px;
                }
                
                .table {
                    margin-top: 20px;
                }
                
                .table th {
                    font-weight: bold;
                }
                
                .table td, .table th {
                    border: 1px solid #dee2e6;
                    padding: 8px;
                }
            </style>
            """

            st.markdown(custom_css, unsafe_allow_html=True)

            st.markdown(f'<div class="log-header"><h4>Tiempo de ejecución: {execution_time:.2f} minutos</h4></div>', unsafe_allow_html=True)
            st.markdown(f'<div class="log-header"><h4>Total de evaluaciones: {evaluations_count}</h4></div>', unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)
            st.title(':orange[◉ ◉ ◉ Resultados ◉ ◉ ◉]')
            st.markdown("<br><br>", unsafe_allow_html=True)

            col1, col2 = st.columns(2)
            with col1:
                for i in range(len(result_unique_individuals)): 
                    st.markdown(f'''<div class="result-header"><h3>Mejor Combinación {i+1}</h3>
                                <h4>Cumplen {int(result_best_fitnesses[i][0])} órdenes de fabricación con el criterio de performance definido.</h4>
                                <p>Valor de fitness: {result_best_fitnesses[i][0]}</p></div>
                                ''', unsafe_allow_html=True)
                    df_result = pd.concat([pd.DataFrame([j]) for j in result_unique_individuals[i]], ignore_index=True)
                    df_result['operators'] = df_result['operators'].apply(lambda x: str(x).replace('(', '').replace(')', ''))
                    st.markdown(df_result.to_html(index=False, classes=["table"]), unsafe_allow_html=True)

            with col2:
                st.markdown("<br><br><br>", unsafe_allow_html=True)
                with st.expander("Ver logs"):
                    st.text(logs)


    else:
        st.warning('Por favor, carga todos los archivos excel para proceder.')


def evaluate_combination():
    if 'new_bomb_type' not in st.session_state:
        st.session_state.new_bomb_type = False

    st.header('Evaluar combinación')
    df_bombs_constraint = pd.read_excel('data/to_load/constraint/bombs_constraint.xlsx')
    df_lines = pd.read_excel('data/to_load/lines.xlsx').sort_values(by='line')
    df_operators = pd.read_excel('data/to_load/operators.xlsx').sort_values(by='operator')
    st.subheader('Complete la información de la orden de fabricación a evaluar')
    production_date = st.date_input('Fecha de producción', value=pd.to_datetime('today').date())
    order_id = st.text_input('Ingrese número de Ordern de Fabricación:')
    st.checkbox('Tipo de bomba nuevo', key='new_bomb_type')
    bomb_type = st.selectbox(
        'Seleccionar tipo de bomba:',
        df_bombs_constraint.Bomb_type.tolist(),
        disabled=st.session_state.new_bomb_type
    )
    theorical_time = st.number_input(
        'Define un tiempo teórico:',
        min_value=0.00001,
        disabled=not st.session_state.new_bomb_type
    )
    plan_qty = st.number_input(
        'Cantidad de bombas a producir:',
        min_value=1,
        step=int(1)
    )
    line = st.selectbox(
        'Seleccionar línea:',
        df_lines.line.tolist()
    )
    operators = st.multiselect(
        'Seleccionar operadores:',
        df_operators.operator.tolist()
    )
    st.write(f'Total de operadores: {len(operators)}')
    if st.button('Realizar predicción',
                 disabled=(not operators or not line or not (bomb_type or theorical_time))):
        # Create a CSS string
        custom_css = """
        <style>
            .prediction-success {
                color: white;
                background-color: #4CAF50;
                padding: 20px;
                border-radius: 5px;
            }
            
            .prediction-fail {
                color: white;
                background-color: #f44336;
                padding: 20px;
                border-radius: 5px;
            }

            .prediction-header {
                color: white;
                background-color: #3F51B5;
                padding: 20px;
                border-radius: 5px;
            }
        </style>
        """

        st.markdown(custom_css, unsafe_allow_html=True)
        
        df_order = m.parse_to_model(
            order_id, operators, line, plan_qty, theorical_time, production_date,
            df_operators.operator.tolist(), df_lines.line.tolist()
        )
        st.markdown('<br>', unsafe_allow_html=True)

        prediction = m.classifier_model.predict(df_order[m.classifier_model.feature_names_in_])
        prediction_proba = m.classifier_model.predict_proba(df_order[m.classifier_model.feature_names_in_])
        
        st.markdown(f'<div class="prediction-header"><h2>Orden:<br>{order_id}</h2></div>', unsafe_allow_html=True)
        st.markdown('<br>', unsafe_allow_html=True)
        col1, col2 = st.columns([1,1])

        with col1:
            st.markdown(f'<div class="prediction-header"><h2>Operadores:<br> {operators}</h2></div>', unsafe_allow_html=True)
        with col2:
            st.markdown(f'<div class="prediction-header"><h2>Linea:<br> {line}</h2></div>', unsafe_allow_html=True)
        st.markdown('<br>', unsafe_allow_html=True)

        prob = prediction_proba[0][prediction[0]]
        prediction_status = 'CUMPLE' if prediction[0] else 'NO CUMPLE'
        prediction_status = 'NO CUMPLE'
        prediction_prob = round(prob*100, 2) if prob > 0.5 else round((1-prob)*100, 2)
        prediction_color = 'prediction-success' if prediction_status == 'CUMPLE' else 'prediction-fail'
        
        st.markdown(f'<div class="{prediction_color}"><h2>Predicción: {prediction_status}</h2>'
                    f'<h2>Probabilidad: {prediction_prob}%</h2></div>', unsafe_allow_html=True)
        

        



def configure_restrictions():
    st.session_state['operator_lines'] = {} if 'operator_lines' not in st.session_state else st.session_state['operator_lines']
    st.session_state['bomb_lines'] = {} if 'bomb_lines' not in st.session_state else st.session_state['bomb_lines']
    st.session_state['search_operator_id'] = '' if 'search_operator_id' not in st.session_state else st.session_state['search_operator_id']
    st.session_state['search_bomb_id'] = '' if 'search_bomb_id' not in st.session_state else st.session_state['search_bomb_id']
    df_lines = pd.read_excel('data/to_load/lines.xlsx').sort_values(by='line')
    st.header('Configuración de restricciones')
    # Operators constraint
    df_operators_constraint = pd.read_excel('data/to_load/constraint/operators_constraint.xlsx')
    df_operators_constraint['Operario'] = df_operators_constraint['Operario'].astype(str)
    df_operators_constraint_original = pd.read_excel('data/to_load/constraint/operators_constraint.xlsx')

    # Bombs constraint
    df_bombs_constraint = pd.read_excel('data/to_load/constraint/bombs_constraint.xlsx')
    #df_bombs_constraint['Bomba'] = df_bombs_constraint['Bomba'].astype(str)
    df_bombs_constraint_original = pd.read_excel('data/to_load/constraint/bombs_constraint.xlsx')


    #st.markdown(df_operators_constraint.to_html(index=False), unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    with st.expander('Modificar restricciones de lineas para operadores'):
        # Añadimos un cuadro de texto para buscar un operador
        operator_id = st.text_input('Ingrese el ID del operador a buscar', value=st.session_state.search_operator_id)
        #operator_id = st.selectbox('Ingrese el ID del operador a buscar',
        #                           df_operators_constraint['Operario'].tolist())
        st.session_state.search_operator_id = operator_id

        if operator_id:
            df_operators_constraint = df_operators_constraint[df_operators_constraint['Operario'].str.startswith(operator_id)]
        
            for i in range(df_operators_constraint.shape[0]):
                # Extraer operario y líneas
                operario = df_operators_constraint.iloc[i, 0]
                lineas = df_operators_constraint.iloc[i, 1].split(', ')

                # Crear una sección para cada operario
                st.markdown(f"#### Operario: {operario}")
                
                # Crear columnas para cada línea
                cols = st.columns(len(df_lines))
                for i in range(len(df_lines)):
                    line = df_lines.iloc[i, 0]  # Asume que el nombre de la línea está en la primera columna
                    st.session_state.operator_lines[f"{operario}_{line}"] = \
                        cols[i].checkbox(line, value=(line in lineas), key=f"{operario}_{line}")

        # Aplicar los cambios a df_operators_constraint_original
        for i in range(df_operators_constraint.shape[0]):
            operario = df_operators_constraint.iloc[i, 0]
            lineas = []
            for j in range(len(df_lines['line'])):
                line = df_lines.iloc[j, 0]
                if st.session_state.operator_lines.get(f"{operario}_{line}"):
                    lineas.append(line)
            df_operators_constraint_original.loc[
                df_operators_constraint_original['Operario'] == operario, 'Lineas'
            ] = ', '.join(lineas)
        
        if st.button('Guardar cambios de restricciones de operadores'):
            df_operators_constraint_original.to_excel('data/to_load/constraint/operators_constraint.xlsx', index=False)
        print(df_operators_constraint_original)
        st.markdown(
            df_operators_constraint_original.sort_values(
            by='Operario', ascending=False).to_html(index=False),
            unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)


    with st.expander('Modificar restricciones de lineas para bombas'):
        # Añadimos un cuadro de texto para buscar una bomba
        #bomb_id = st.text_input('Ingrese el ID de la bomba a buscar', value=st.session_state.search_bomb_id)
        bomb_id = st.selectbox('Ingrese el ID de la bomba a buscarr',
                                   df_bombs_constraint['Bomb_type'].tolist())
        st.session_state.search_bomb_id = bomb_id

        if bomb_id:
            df_bombs_constraint = df_bombs_constraint[df_bombs_constraint['Bomb_type'].str.startswith(bomb_id)]
        
            for i in range(df_bombs_constraint.shape[0]):
                # Extraer operario y líneas
                bomba = df_bombs_constraint.iloc[i, 0]
                lineas = df_bombs_constraint.iloc[i, 1].split(', ')

                # Crear una sección para cada bomba
                st.markdown(f"#### Tipo de Bomba: {bomba}")
                
                # Crear columnas para cada línea
                cols = st.columns(len(df_lines))
                for i in range(len(df_lines)):
                    line = df_lines.iloc[i, 0]
                    st.session_state.bomb_lines[f"{bomba}_{line}"] = \
                        cols[i].checkbox(line, value=(line in lineas), key=f"{bomba}_{line}")
        
        # Aplicar los cambios a df_bombs_constraint_original
        for i in range(df_bombs_constraint.shape[0]):
            bomba = df_bombs_constraint.iloc[i, 0]
            lineas = []
            for j in range(len(df_lines['line'])):
                line = df_lines.iloc[j, 0]
                if st.session_state.bomb_lines.get(f"{bomba}_{line}"):
                    lineas.append(line)
            df_bombs_constraint_original.loc[
                df_bombs_constraint_original['Bomb_type'] == bomba, 'Linea'
            ] = ', '.join(lineas)
        
        if st.button('Guardar cambios de restricciones de bombas'):
            df_bombs_constraint_original.to_excel('data/to_load/constraint/bombs_constraint.xlsx', index=False)
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(
            df_bombs_constraint_original.sort_values(
            by='Bomb_type', ascending=False).to_html(index=False),
            unsafe_allow_html=True)
                    

def main():
    st.title(':blue[◉ ◉ ◉ Fluidra ◉ ◉ ◉]')
    st.markdown("<br><br>", unsafe_allow_html=True)
    page = st.sidebar.radio('Navegación', ['Buscar mejor combinación', 'Evaluar combinación', 'Configuración de restricciones'])

    if page == 'Buscar mejor combinación':
        get_best_combinations()
    elif page == 'Evaluar combinación':
        evaluate_combination()
    elif page == 'Configuración de restricciones':
        configure_restrictions()

if __name__ == "__main__":
    main()