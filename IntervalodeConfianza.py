import pandas as pd
import numpy as np
from scipy.stats import t, chi2

def intervalo_confianza_media(data, alpha=0.05):
    """Calcula el intervalo de confianza para la media de una muestra."""
    data_clean = data.dropna()  # Eliminar filas con NaN
    n = len(data_clean)
    
    if n < 2:
        raise ValueError("No hay suficientes datos para calcular el intervalo de confianza.")
    
    media = np.mean(data_clean)
    desviacion_estandar = np.std(data_clean, ddof=1)  # Usamos ddof=1 para calcular la desviación estándar muestral
    
    t_valor = t.ppf(1 - alpha / 2, df=n - 1)  # Valor crítico de t para el intervalo de confianza
    error_estandar = desviacion_estandar / np.sqrt(n)
    
    intervalo_inf = media - t_valor * error_estandar
    intervalo_sup = media + t_valor * error_estandar
    
    return intervalo_inf, intervalo_sup

def intervalo_confianza_varianza(data, alpha=0.05):
    """
    Calcula el intervalo de confianza para la varianza de una muestra.

    Args:
        data (array-like): Arreglo de datos de la muestra.
        alpha (float): Nivel de significancia del intervalo de confianza.

    Returns:
        tuple: Límites inferior y superior del intervalo de confianza.
    """
    data_clean = data.dropna()  # Eliminar filas con NaN
    n = len(data_clean)
    
    if n < 2:
        raise ValueError("No hay suficientes datos para calcular el intervalo de confianza.")
    
    varianza_muestral = np.var(data_clean, ddof=1)  # Varianza muestral
    chi2_left = chi2.ppf(alpha / 2, df=n - 1)
    chi2_right = chi2.ppf(1 - alpha / 2, df=n - 1)
    ci_lower = (n - 1) * varianza_muestral / chi2_right
    ci_upper = (n - 1) * varianza_muestral / chi2_left
    
    return ci_lower, ci_upper

# Cargar el archivo CSV
file_path = 'C:/Users/Camil/Downloads/costodevida/cost-of-living_v2.csv'
df = pd.read_csv(file_path)

# Seleccionar la columna x48 y calcular el intervalo de confianza para la media
intervalo_inf_media, intervalo_sup_media = intervalo_confianza_media(df['x48'])

# Calcular el intervalo de confianza para la varianza de x48
intervalo_inf_var, intervalo_sup_var = intervalo_confianza_varianza(df['x48'])

# Imprimir los resultados
print(f'Intervalo de confianza para la media de x48: [{intervalo_inf_media:.2f}, {intervalo_sup_media:.2f}]')
print(f'Intervalo de confianza para la varianza de x48: [{intervalo_inf_var:.2f}, {intervalo_sup_var:.2f}]')import pandas as pd
import numpy as np
from scipy.stats import t, chi2

def intervalo_confianza_media(data, alpha=0.05):
    """Calcula el intervalo de confianza para la media de una muestra."""
    data_clean = data.dropna()  # Eliminar filas con NaN
    n = len(data_clean)
    
    if n < 2:
        raise ValueError("No hay suficientes datos para calcular el intervalo de confianza.")
    
    media = np.mean(data_clean)
    desviacion_estandar = np.std(data_clean, ddof=1)  # Usamos ddof=1 para calcular la desviación estándar muestral
    
    t_valor = t.ppf(1 - alpha / 2, df=n - 1)  # Valor crítico de t para el intervalo de confianza
    error_estandar = desviacion_estandar / np.sqrt(n)
    
    intervalo_inf = media - t_valor * error_estandar
    intervalo_sup = media + t_valor * error_estandar
    
    return intervalo_inf, intervalo_sup

def intervalo_confianza_varianza(data, alpha=0.05):
    """
    Calcula el intervalo de confianza para la varianza de una muestra.

    Args:
        data (array-like): Arreglo de datos de la muestra.
        alpha (float): Nivel de significancia del intervalo de confianza.

    Returns:
        tuple: Límites inferior y superior del intervalo de confianza.
    """
    data_clean = data.dropna()  # Eliminar filas con NaN
    n = len(data_clean)
    
    if n < 2:
        raise ValueError("No hay suficientes datos para calcular el intervalo de confianza.")
    
    varianza_muestral = np.var(data_clean, ddof=1)  # Varianza muestral
    chi2_left = chi2.ppf(alpha / 2, df=n - 1)
    chi2_right = chi2.ppf(1 - alpha / 2, df=n - 1)
    ci_lower = (n - 1) * varianza_muestral / chi2_right
    ci_upper = (n - 1) * varianza_muestral / chi2_left
    
    return ci_lower, ci_upper

# Cargar el archivo CSV
file_path = 'C:/Users/Camil/Downloads/costodevida/cost-of-living_v2.csv'
df = pd.read_csv(file_path)

# Seleccionar la columna x48 y calcular el intervalo de confianza para la media
intervalo_inf_media, intervalo_sup_media = intervalo_confianza_media(df['x48'])

# Calcular el intervalo de confianza para la varianza de x48
intervalo_inf_var, intervalo_sup_var = intervalo_confianza_varianza(df['x48'])

# Imprimir los resultados
print(f'Intervalo de confianza para la media de x48: [{intervalo_inf_media:.2f}, {intervalo_sup_media:.2f}]')
print(f'Intervalo de confianza para la varianza de x48: [{intervalo_inf_var:.2f}, {intervalo_sup_var:.2f}]')
