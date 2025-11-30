import pandas as pd
import joblib
import matplotlib.pyplot as plt
import numpy as np

# ==========================================
# 1. CONFIGURACIÓN
# ==========================================
# Carga tu CSV (el que generó tu modelo .pt)
archivo_csv = 'resumen_final_peces_v2.csv' 
df = pd.read_csv(archivo_csv)

# Carga los modelos externos
scaler = joblib.load('WeightPredictor_scaler_model.pkl')
regressor = joblib.load('WeightPredictor_regression_model.pkl')

# ==========================================
# 2. ADAPTACIÓN DE DATOS (CRUCIAL)
# ==========================================

# A. Identifica tus columnas
# Cambia 'largo_cm' y 'alto_cm' por los nombres reales que tienes en tu CSV
columna_largo_usuario = 'Largo_Prom_cm' 
columna_alto_usuario = 'Ancho_Prom_cm'
columna_peso_usuario  = 'Peso_Prom_g' # modelo predice en gramos

# B. Crear el DataFrame EXACTO que pide la imagen
# La imagen muestra que requieren columnas: ['Est. Length', 'Est. Height']
X_pkl = pd.DataFrame()
X_pkl['Est. Length'] = df[columna_largo_usuario]
X_pkl['Est. Height'] = df[columna_alto_usuario]

# ==========================================
# 3. PREDICCIÓN DEL MODELO EXTERNO (.pkl)
# ==========================================
print(">> Escalando datos...")
X_scaled = scaler.transform(X_pkl)

print(">> Prediciendo pesos con modelo .pkl...")
prediccion_kg = regressor.predict(X_scaled)

# C. Conversión de unidades
# Como la imagen dice "kg" y asumimos que tú tienes "g", pasamos todo a gramos para comparar
df['peso_pkl_calculado_g'] = prediccion_kg * 1000  

# ==========================================
# 4. COMPARACIÓN VISUAL
# ==========================================
plt.figure(figsize=(12, 6))

# Gráfico de dispersión
plt.scatter(df.index, df[columna_peso_usuario], 
            color='blue', label='Tu Modelo .pt (Original)', alpha=0.7)
plt.scatter(df.index, df['peso_pkl_calculado_g'], 
            color='red', label='Modelo .pkl (Externo)', marker='x', alpha=0.7)

# Dibujar líneas grises conectando las predicciones para el mismo pez
for i in df.index:
    p1 = df.loc[i, columna_peso_usuario]
    p2 = df.loc[i, 'peso_pkl_calculado_g']
    plt.plot([i, i], [p1, p2], color='gray', alpha=0.3)

plt.title('Comparativa: Tu Modelo vs Modelo Externo (.pkl)')
plt.xlabel('Índice del Salmón')
plt.ylabel('Peso (gramos)')
plt.legend()
plt.grid(True, alpha=0.3)

# Calcular diferencia promedio
df['diferencia_g'] = df[columna_peso_usuario] - df['peso_pkl_calculado_g']
mae_diff = np.mean(np.abs(df['diferencia_g']))

print(f"\n--- RESULTADOS ---")
print(f"Diferencia media entre modelos: {mae_diff:.2f} gramos")
print("Si la diferencia es baja (<100g), ambos modelos 'piensan' igual.")
print("Si es alta, revisa si tus 'cm' son realmente cm y no píxeles.")

plt.show()