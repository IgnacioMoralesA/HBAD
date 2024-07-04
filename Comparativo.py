import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import norm

# Establecer backend de Matplotlib
import matplotlib
matplotlib.use('Agg')  # Usar el backend 'Agg' para evitar problemas con la interfaz gráfica

# Paso 1: Cargar los datos desde el archivo CSV
file_path = 'C:/Users/Camil/Downloads/costodevida/cost-of-living_v2.csv'
data = pd.read_csv(file_path)

# Verificar la carga de datos
print("Primeras filas del conjunto de datos:")
print(data.head())

# Paso 2: Crear gráfico de densidad e histograma para la variable 'x48'
x50_data = data['x50'].dropna()  # Eliminar NaN si los hay

# Reducir el tamaño del conjunto de datos si es muy grande (opcional)
if len(x50_data) > 1000:
    x50_data = x50_data.sample(1000)

plt.figure(figsize=(10, 6))
sns.histplot(x50_data, kde=True, bins=30, color='blue', alpha=0.7, label='Histograma y KDE')
plt.title('Distribución de Precios de Departamentos (x50)')
plt.xlabel('Valor de x50')
plt.ylabel('Frecuencia')
plt.legend()
plt.savefig('C:/Users/Camil/Downloads/costodevida/x48_distribution2.png')  # Guardar el gráfico

# Ajustar una distribución normal a los datos de 'x50'
mu, std = norm.fit(x50_data)

# Generar puntos para la distribución normal ajustada
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = norm.pdf(x, mu, std)

plt.plot(x, p, 'k', linewidth=2, label='Distribución Normal Ajustada')
plt.legend()
plt.savefig('C:/Users/Camil/Downloads/costodevida/x50_distribution_with_normal2.png')  # Guardar el gráfico

print(f'Media = {mu:.2f}, Desviación Estándar = {std:.2f}')

# Paso 3: Crear gráfico de densidad e histograma para la variable 'x51'
x51_data = data['x51'].dropna()  # Eliminar NaN si los hay

# Reducir el tamaño del conjunto de datos si es muy grande (opcional)
if len(x51_data) > 1000:
    x51_data = x51_data.sample(1000)

plt.figure(figsize=(10, 6))
sns.histplot(x51_data, kde=True, bins=30, color='green', alpha=0.7, label='Histograma y KDE')
plt.title('Distribución de Precios de Departamentos (x51)')
plt.xlabel('Valor de x51')
plt.ylabel('Frecuencia')
plt.legend()
plt.savefig('C:/Users/Camil/Downloads/costodevida/x51_distribution.png')  # Guardar el gráfico

# Ajustar una distribución normal a los datos de 'x49'
mu51, std51 = norm.fit(x51_data)

# Generar puntos para la distribución normal ajustada
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = norm.pdf(x, mu51, std51)

plt.plot(x, p, 'k', linewidth=2, label='Distribución Normal Ajustada')
plt.legend()
plt.savefig('C:/Users/Camil/Downloads/costodevida/x51_distribution_with_normal.png')  # Guardar el gráfico

print(f'Media = {mu51:.2f}, Desviación Estándar = {std51:.2f}')
