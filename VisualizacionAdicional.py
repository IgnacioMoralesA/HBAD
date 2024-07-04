#Gráfico del intervalo de confianza para la media
sns.set_style('whitegrid')
sns.displot(df['x48'].dropna(), kind='kde', fill=True)

plt.axvline(intervalo_inf_media, color='blue', linestyle='dashed', linewidth=1, label=f'Intervalo media ({100*(1-alpha):.0f}%)')
plt.axvline(intervalo_inf_media, color='blue', linestyle='dashed', linewidth=1)
plt.axvline(np.mean(df['x48].dropna()), color='red', linestyle='dashed', label=f'Media muestral ({np.mean(df['x48'].dropna()):.2f})')

plt.title('Intervalo de confianza para la media de x48')
plt.legend()
plt.show()
