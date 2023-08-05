import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import sklearn
import requests
import numpy as np
import requests

# Configurar opciones de la página
st.set_page_config(
    page_title="Clasificación de cardiopatía",
    page_icon="🏥",  
    layout="wide", 
    initial_sidebar_state="collapsed"
)

def user_input_parameters(c):
    inputs = {}
    cols = st.columns(2)  # Dividir en 2 columnas
    for i, feature in enumerate(c):
        inputs[feature] = cols[i % 2].text_input(feature)
    return inputs

def solicitud_API(muestra:list):
    urlApi = 'http://127.0.0.1:8000/predict'

    data = {
        "data": muestra
    }
    response = requests.post(urlApi, json=data)
    if response.status_code == 200:
        result = response.json()
        prediction = result["prediction"]
        return prediction
    else:
        return none

def main():
    folder = 'data'
    archivo_data = 'heart.csv'
    data = pd.read_csv(folder + '/' + archivo_data, sep=',')
    # Diccionario para mapear clases a etiquetas
    clase_etiqueta = {0: "No presenta enfermedad cardíaca", 1: "Presenta enfermedad cardíaca"}
    d = data.copy()
    d['target'] = d['target'].replace(clase_etiqueta)
    caracteristicas = d.drop(['target'], axis=1)

    # Barra lateral a la izquierda
    st.write("# ¡Bienvenido al Clasificador Cardiopático! 💉🩺🏨")
    #st.info("Bienvenido al Clasificador cardiopático. ¿Puedes mejorar?")
    st.sidebar.title("🎯 Sistema de Predicción")
    option = st.sidebar.selectbox("Seleccionar Opción", ["Home", "Modelo de Negocio","Descriptiva", "Predicciones"])
    
     # Mostrar contenido según la opción seleccionada en la barra lateral
    if option == "Home":
        st.markdown("## Contexto:")
        st.markdown("Disponemos de datos que clasifican si los pacientes padecen o no una cardiopatía en función de las características que contienen. Intentaremos utilizar estos datos para crear un modelo que intente predecir si un paciente tiene esta enfermedad o no. Utilizaremos el algoritmo de RandomForest (clasificación).\n- Fuente: https://www.kaggle.com/code/cdabakoglu/heart-disease-classifications-machine-learning/notebook")
        st.markdown("## Descripción de las Variables:")
        st.markdown("""
            * **edad:** edad en años
            * **sexo:** (1 = masculino; 0 = femenino)
            * **cp:** tipo de dolor en el pecho
            * **trestbps:** presión arterial en reposo (en mm Hg al ingresar al hospital)
            * **col:** colesterol sérico en mg/dl
            * **fbs:** (azúcar en sangre en ayunas > 120 mg/dl) (1 = verdadero; 0 = falso)
            * **restecg:** resultados electrocardiográficos en reposo
            * **thalach:** frecuencia cardíaca máxima alcanzada
            * **exang:** angina inducida por ejercicio (1 = sí; 0 = no)
            * **oldpeak:** depresión del segmento ST inducida por el ejercicio en relación al reposo
            * **slope:** la pendiente del segmento ST del ejercicio máximo
            * **ca:** número de vasos principales (0-3) coloreados por fluoroscopia
            * **thal:** 3 = normal; 6 = defecto fijo; 7 = defecto reversible
            * **target:** tiene enfermedad o no (1=sí, 0=no)
            """)
    elif option == "Modelo de Negocio":
         # Incrustar una imagen desde el sistema local
        st.image("Images/ML II.png", caption="Canvas Machine Learning")
    elif option == "Descriptiva":
        st.write("¡Top 5!")
        st.dataframe(d.head())
        st.markdown("<hr style='margin-top: 2px; margin-bottom: 15px;'>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            conteo = d['target'].value_counts()
            st.write("¡Distribución general de los datos!")
            st.dataframe(conteo)
            st.write("")
            # Configurar el estilo de la gráfica
            sns.set(style="whitegrid")
            # Crear una figura y un eje usando Matplotlib
            fig2, ax = plt.subplots(figsize=(6, 6))
            # Graficar la gráfica de barras usando Seaborn
            sns.countplot(x='sex', data=d, palette="mako_r", ax=ax)
            # Etiquetas y título
            ax.set_xlabel("Sexo (0 = Femenino, 1 = Masculino)")
            ax.set_ylabel("Cantidad")
            ax.set_title("Distribución de género")
            # Mostrar la figura en Streamlit
            st.pyplot(fig2)
        with col2:    
            # Graficar la distribución de los datos como una gráfica de torta
            st.write("")
            plt.rcParams['font.size'] = 20
            inidices = conteo.index.tolist()
            fig, ax = plt.subplots()
            ax.pie(list(conteo.values),labels = inidices, autopct='%1.1f%%', startangle=90)
            ax.axis('equal')  
            st.pyplot(fig)
            st.write("")
    elif option == "Predicciones":
        with open ('Models/modelo.pkl' , 'rb') as m: # En modo lectura
            modelo = pickle.load(m)
        st.subheader("¡Predicciones ML! 🤖")
        st.markdown("<hr style='margin-top: 2px; margin-bottom: 15px;'>", unsafe_allow_html=True)
        st.write("Por favor ingrese los valores de las características del paciente: ")  
        ContenedorDataFrame = user_input_parameters(caracteristicas)
        if st.button("Predecir"):
            # Convertir el diccionario ContenedorDataFrame en una matriz NumPy
            input_data = np.array(list(ContenedorDataFrame.values()))
            # Convertir el array 1D en una matriz 2D
            input_data_2d = input_data.reshape(1, -1)
            # Realizar la predicción
            #prediction = modelo.predict(input_data_2d)
            prediction = solicitud_API(list(ContenedorDataFrame.values()))
            # Crear un diccionario para asociar las predicciones
            prediction_descriptions = {
                0: '✅ Negativo: Enfermedad cardíaca ausente, sin embargo es recomendable realizar chequeos regulares.',
                1: '❌ Positivo: Enfermedad cardíaca presente, por favor consultar con un especialista.'
            }
            
            #if prediction[0] == 0:
            #    st.success(prediction_descriptions[prediction[0]])
            #elif prediction[0] == 1:
            #    st.error(prediction_descriptions[prediction[0]])
            
            prediction = int(prediction)
            if prediction == 0:
                st.success(prediction_descriptions[prediction])
            elif prediction == 1:
                st.error(prediction_descriptions[prediction])

if __name__ == "__main__":
    main()


