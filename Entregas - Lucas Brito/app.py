import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# --- CONFIGURAÇÃO INICIAL ---
st.set_page_config(page_title="Preditor de Diabetes", page_icon="🩺", layout="centered")

st.title("🩺 Predição de Diabetes")
st.write("Preencha os dados abaixo para prever o risco de diabetes usando um modelo de aprendizado de máquina.")

# --- CARREGAR OU CRIAR DADOS ---
@st.cache_data
def load_data():
    # Dataset público de diabetes
    url = "https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv"
    data = pd.read_csv(url)
    return data

data = load_data()

# --- TREINAMENTO DO MODELO ---
X = data.drop("Outcome", axis=1)
y = data["Outcome"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

# --- INTERFACE DE ENTRADA ---
st.subheader("📋 Insira os dados do paciente:")

col1, col2 = st.columns(2)
with col1:
    pregnancies = st.number_input("Número de gestações", min_value=0, max_value=20, value=1)
    glucose = st.number_input("Nível de Glicose", min_value=0, max_value=300, value=120)
    blood_pressure = st.number_input("Pressão arterial", min_value=0, max_value=200, value=70)
    skin_thickness = st.number_input("Espessura da pele", min_value=0, max_value=100, value=20)

with col2:
    insulin = st.number_input("Nível de Insulina", min_value=0, max_value=900, value=80)
    bmi = st.number_input("IMC", min_value=0.0, max_value=70.0, value=25.0)
    dpf = st.number_input("Histórico familiar de diabetes (DPF)", min_value=0.0, max_value=3.0, value=0.5)
    age = st.number_input("Idade", min_value=10, max_value=100, value=30)

# --- PREDIÇÃO ---
if st.button("🔍 Prever"):
    input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness,
                            insulin, bmi, dpf, age]])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]

    st.markdown("---")
    if prediction == 1:
        st.error("⚠️ Alto risco de diabetes detectado!")
    else:
        st.success("✅ Baixo risco de diabetes.")
    
    st.markdown(f"**Acurácia do modelo:** {acc:.2%}")

# --- EXPLORAR DADOS ---
with st.expander("📊 Visualizar dados do conjunto de treino"):
    st.dataframe(data.head())
