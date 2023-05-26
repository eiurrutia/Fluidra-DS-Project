import streamlit as st
import pandas as pd
import algorithm as alg
import io
import time
import contextlib
pd.set_option('display.float_format', '{:.0f}'.format)

def get_logs(func, *args, **kwargs):
    log_output = io.StringIO()
    with contextlib.redirect_stdout(log_output):
        func_output = func(*args, **kwargs)
    return log_output.getvalue(), func_output

def app():
    file_users = file_lines = file_orders = None

    st.title(':blue[◉ ◉ ◉ Fluidra ◉ ◉ ◉]')
    st.subheader('Asignación de órdenes de fabricación')

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
            
            # Aquí puedes hacer la transición a otra vista o ejecutar tu algoritmo
            alg.unique_individuals = set()
            alg.evaluations_count = 0
            alg.ORDERS = [
                {'id': 5160396, 'good_qty': 100, 'theorical_time': 2.5, 'registers_qty': 30, 'operators_distinct_qty': 2, 'days_accumulated_experience': 102, 'OFs_accumulated_experience': 50},
                {'id': 5169247, 'good_qty': 200, 'theorical_time': 3.0, 'registers_qty': 10, 'operators_distinct_qty': 3, 'days_accumulated_experience': 102, 'OFs_accumulated_experience': 50},
                {'id': 5171973, 'good_qty': 150, 'theorical_time': 2.0, 'registers_qty': 50, 'operators_distinct_qty': 4, 'days_accumulated_experience': 102, 'OFs_accumulated_experience': 50},
            ]
            #alg.available_operators = [1, 2, 3, 4, 5, 6, 7, 8, 9]
            #alg.available_lines = ['LINEA_1', 'LINEA_2', 'LINEA_3', 'LINEA_4']

            alg.production_date = production_date
            alg.available_operators = users_df[users_df['active']]['operator'].tolist()
            alg.available_lines = lines_df[lines_df['active']]['line'].tolist()
            alg.all_operators = users_df['operator'].tolist()
            alg.all_lines = lines_df['line'].tolist()
            alg.ORDERS = orders_df[['id', 'bomb_type', 'theorical_time', 'plan_qty']].to_dict('records')

            print('available_operators', alg.available_operators)
            print('available_lines', alg.available_lines)
            print('all_operators', alg.all_operators)
            print('all_lines', alg.all_lines)
            print('ORDERS', alg.ORDERS)
            logs = result = None
            st.write("Ejecutando algoritmo...")
            
            progress_bar = st.progress(0)
            for i in range(100):
                time.sleep(0.1)  # Si asumimos que el algoritmo tarda 10 segundos en ejecutarse
                progress_bar.progress(i + 1)
            
            logs, (result_unique_individuals, result_best_fitnesses, evaluations_count) = get_logs(alg.run_algorithm)
            st.write(f"Total de evaluaciones: {evaluations_count}")
            st.markdown("<br>", unsafe_allow_html=True)
            st.title('◉ ◉ ◉ Resultados ◉ ◉ ◉')
            st.markdown("<br><br>", unsafe_allow_html=True)

            col1, col2 = st.columns(2)
            with col1:
                for i in range(len(result_unique_individuals)): 
                    st.header(f'Mejor Combinación {i+1}')
                    st.subheader(f'Valor de fitness: {result_best_fitnesses[i][0]}')
                    st.text(f'Cumplen {int(result_best_fitnesses[i][0])} órdenes de fabricación con el criterio de performance definido.')
                    df_result = pd.concat([pd.DataFrame([j]) for j in result_unique_individuals[i]], ignore_index=True)
                    df_result['operators'] = df_result['operators'].apply(lambda x: str(x).replace('(', '').replace(')', ''))
                    st.markdown(df_result.to_html(index=False), unsafe_allow_html=True)
            
            with col2:
                st.markdown("<br><br><br>", unsafe_allow_html=True)
                with st.expander("Ver logs"):
                    st.text(logs)


    else:
        st.warning('Por favor, carga todos los archivos excel para proceder.')

if __name__ == "__main__":
    app()