#predecir la probabilidad de incumplimiento de pago de un préstamo por medio de la variable incumplimiento.
# posibles valores -> incumplimiento, niveles "no","si"

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns

# Cargar los datos
datos = pd.read_csv('credito.csv')

# Preprocesamiento de Datos
# Rellenar valores nulos con la mediana (si existieran)
datos.fillna(datos.median(), inplace=True)

# Separación de Variables Independientes y Dependientes
X = datos.drop('incumplimiento', axis=1)  # Variables independientes
y = datos['incumplimiento']  # Variable dependiente

print("Variables independientes (X):")
print(X.head())
print("\nVariable dependiente (y):")
print(y.head())

# División del Conjunto de Datos en Entrenamiento y Prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Estandarización de los Datos
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Mostrar una muestra de los datos estandarizados
print("Datos estandarizados (X_train):")
print(pd.DataFrame(X_train, columns=X.columns).head())

"""
# distribución de la variable 'Edad'
plt.figure(figsize=(8, 6))
sns.histplot(datos['edad'], kde=True, color='green', bins=30)
plt.title('Distribución de la variable "Edad"')
plt.xlabel('Edad')
plt.ylabel('Frecuencia')
plt.show()

# Diagrama de caja de la variable 'monto'
plt.figure(figsize=(8, 6))
sns.boxplot(x='monto', data=datos, color='green')
plt.title('Diagrama de Caja de Monto del Préstamo')
plt.xlabel('Monto')
plt.show()

# Gráfico de violín de la variable 'duracion_prestamo'
plt.figure(figsize=(8, 6))
sns.violinplot(x='duracion_prestamo', data=datos, color='purple')
plt.title('Gráfico de Violín de Duración del Préstamo')
plt.xlabel('Duración del Préstamo')
plt.show()

# Gráfico de distribución conjunta (densidad) entre 'edad' y 'monto'
plt.figure(figsize=(8, 6))
sns.jointplot(x='edad', y='monto', data=datos, kind='kde', color='red')
plt.title('Gráfico de Distribución Conjunta (Densidad)')
plt.xlabel('Edad')
plt.ylabel('Monto')
plt.show()

# Gráfico de distribución acumulativa (ECDF) de la variable 'edad'
plt.figure(figsize=(8, 6))
sns.ecdfplot(data=datos, x='edad')
plt.title('Gráfico de Distribución Acumulativa de Edad')
plt.xlabel('Edad')
plt.ylabel('Probabilidad Acumulativa')
plt.show()

"""
# Diagrama de caja de incumplimiento por edad
plt.figure(figsize=(8, 6))
sns.boxplot(x='incumplimiento', y='edad', data=datos)
plt.title('Diagrama de Caja de Incumplimiento por Edad')
plt.xlabel('Incumplimiento')
plt.ylabel('Edad')
plt.show()

# Diagrama de caja de incumplimiento por monto del préstamo
plt.figure(figsize=(8, 6))
sns.boxplot(x='incumplimiento', y='monto', data=datos)
plt.title('Diagrama de Caja de Incumplimiento por Monto del Préstamo')
plt.xlabel('Incumplimiento')
plt.ylabel('Monto del Préstamo')
plt.show()

# Gráfico de violín de incumplimiento por duración del empleo
plt.figure(figsize=(8, 6))
sns.violinplot(x='incumplimiento', y='longitud_empleo', data=datos)
plt.title('Gráfico de Violín de Incumplimiento por Duración del Empleo')
plt.xlabel('Incumplimiento')
plt.ylabel('Duración del Empleo')
plt.show()

# Gráfico de barras de incumplimiento por estado civil
plt.figure(figsize=(8, 6))
sns.countplot(x='estado_personal', hue='incumplimiento', data=datos)
plt.title('Gráfico de Incumplimiento por Estado Civil')
plt.xlabel('Estado Civil')
plt.ylabel('Conteo')
plt.show()

# Matriz de correlación entre todas las variables numéricas
corr_matrix = datos.corr()

plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Matriz de Correlación')
plt.show()

# Gráfico de distribución conjunta de incumplimiento por edad y monto del préstamo
plt.figure(figsize=(8, 6))
sns.jointplot(x='edad', y='monto', data=datos, hue='incumplimiento', kind='scatter')
plt.title('Gráfico de Distribución Conjunta de Incumplimiento por Edad y Monto del Préstamo')
plt.xlabel('Edad')
plt.ylabel('Monto del Préstamo')
plt.show()