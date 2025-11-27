import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest

def generar_analisis_completo(archivo_csv):
    # 1. CARGAR DATOS
    print(f"Cargando datos desde {archivo_csv}...")
    df = pd.read_csv(archivo_csv)
    total_inicial = len(df)

    # 2. LIMPIEZA AUTOMÁTICA (Isolation Forest)
    # Detecta anomalías basándose en la coherencia Largo/Ancho/Peso
    print("Detectando anomalías...")
    X_iso = df[['Largo_Prom_cm', 'Ancho_Prom_cm', 'Peso_Prom_g']]
    iso = IsolationForest(contamination=0.1, random_state=42) # 10% estimado de ruido
    df['Es_Valido'] = iso.fit_predict(X_iso)

    # Filtrar: Nos quedamos solo con los datos válidos (1)
    df_clean = df[df['Es_Valido'] == 1].copy()
    eliminados = total_inicial - len(df_clean)
    print(f"-> Limpieza completada: Se eliminaron {eliminados} datos anómalos.")

    # 3. ANÁLISIS DESCRIPTIVO: PERFIL DE BIOMASA
    plt.figure(figsize=(10, 6))
    sns.histplot(df_clean['Peso_Prom_g'], kde=True, bins=15, color='#2ecc71')
    plt.title('Perfil de Biomasa (Datos Validados)')
    plt.xlabel('Peso (g)')
    plt.ylabel('Frecuencia')
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.savefig('grafico_biomasa_limpio.png')
    print("-> Gráfico 'grafico_biomasa_limpio.png' guardado.")
    plt.show()

    # 4. ANÁLISIS PREDICTIVO: SEGMENTACIÓN (K-Means)
    # Agrupamos los peces limpios en 3 categorías de talla
    X_kmeans = df_clean[['Largo_Prom_cm', 'Peso_Prom_g']]
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    df_clean['Cluster'] = kmeans.fit_predict(X_kmeans)

    # Visualizar Segmentación
    plt.figure(figsize=(10, 6))
    scatter = sns.scatterplot(data=df_clean, x='Largo_Prom_cm', y='Peso_Prom_g', 
                              hue='Cluster', palette='viridis', s=100, style='Cluster')
    plt.title('Segmentación Automática de Tallas (K-Means)')
    plt.xlabel('Largo (cm)')
    plt.ylabel('Peso (g)')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(title='Grupo Talla')
    plt.savefig('grafico_segmentacion_limpio.png')
    print("-> Gráfico 'grafico_segmentacion_limpio.png' guardado.")
    plt.show()

    # 5. REPORTE FINAL EN CONSOLA
    print("\n--- RESUMEN DE GRUPOS DETECTADOS ---")
    resumen = df_clean.groupby('Cluster')[['Largo_Prom_cm', 'Peso_Prom_g']].mean()
    resumen['Conteo'] = df_clean['Cluster'].value_counts()
    print(resumen.sort_values(by='Peso_Prom_g'))

# EJECUCIÓN
if __name__ == "__main__":
    # Asegúrate de poner el nombre correcto de tu archivo aquí
    archivo = 'resumen_final_peces.csv' 
    try:
        generar_analisis_completo(archivo)
    except FileNotFoundError:
        print(f"Error: No se encontró el archivo '{archivo}'. Verifica la ruta.")