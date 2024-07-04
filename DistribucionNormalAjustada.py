import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import norm

# Paso 1: Cargar los datos desde el archivo CSV
file_path = 'C:/Users/Camil/Downloads/costodevida/cost-of-living_v2.csv'  # Asegúrate de proporcionar la ruta correcta
data = pd.read_csv(file_path)

# Paso 2: Seleccionar la columna 'x48' para generar la distribución normal
x48_data = data['x48'].dropna()  # Eliminar NaN si los hay

# Paso 3: Ajustar una distribución normal a los datos de 'x48'
mu, std = norm.fit(x48_data)

# Paso 4: Generar puntos para la distribución normal ajustada
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = norm.pdf(x, mu, std)

# Paso 5: Visualizar el histograma y la distribución normal ajustada
plt.figure(figsize=(10, 6))
sns.histplot(x48_data, kde=True, bins=30, color='blue', alpha=0.7, label='Datos reales')
plt.plot(x, p, 'k', linewidth=2, label='Distribución Normal Ajustada')
plt.title('Ajuste de Distribución Normal a x48')
plt.xlabel('Valor de x48')
plt.ylabel('Frecuencia')
plt.legend()
plt.show()

print(f'Media = {mu:.2f}, Desviación Estándar = {std:.2f}')
