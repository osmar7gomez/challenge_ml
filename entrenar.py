"""
entrenar.py
Entrena el modelo de Mantenimiento Predictivo y lo exporta a modelo_mantenimiento.pkl
Estados: 0 = Normal, 1 = Mantenimiento Preventivo, 2 = Fallo Crítico
"""
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

ARCHIVO_CSV = "sensores_planta.csv"
ARCHIVO_MODELO = "modelo_mantenimiento.pkl"
FEATURES = ["temperatura_c", "vibracion_mms", "horas_operacion", "presion_psi"]
TARGET = "estado"

# 1. Cargar y limpiar datos
df = pd.read_csv(ARCHIVO_CSV)
filas_antes = len(df)
df = df.dropna()
print(f"📄 Registros leídos: {filas_antes} | Registros tras dropna(): {len(df)}")
print(f"📊 Distribución de estados:\n{df[TARGET].value_counts().sort_index()}\n")

X = df[FEATURES]
y = df[TARGET].astype(int)

# 2. División 80% / 20% (stratify mantiene la proporción de cada estado)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3. Entrenamiento
modelo = RandomForestClassifier(n_estimators=100, random_state=42)
modelo.fit(X_train, y_train)

# 4. Evaluación
y_pred = modelo.predict(X_test)
precision = accuracy_score(y_test, y_pred)
print(f"🎯 Precisión del modelo (Accuracy): {precision * 100:.2f}%")

if precision >= 0.80:
    print("✅ El modelo cumple el requisito (>= 80%)")
else:
    print("⚠️ El modelo NO alcanza el 80%. Revisa o amplía el dataset.")

# 5. Exportar modelo
joblib.dump(modelo, ARCHIVO_MODELO)
print(f"💾 Modelo exportado como {ARCHIVO_MODELO}")
