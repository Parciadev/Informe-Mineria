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

# Creación y Entrenamiento del Modelo KNN
knn = KNeighborsClassifier(n_neighbors=5)  # Número de vecinos (k)
knn.fit(X_train, y_train)

# Evaluación del Modelo KNN
y_pred_knn = knn.predict(X_test)
conf_matrix_knn = confusion_matrix(y_test, y_pred_knn)
class_report_knn = classification_report(y_test, y_pred_knn)
accuracy_knn = accuracy_score(y_test, y_pred_knn)
y_prob_knn = knn.predict_proba(X_test)[:, 1]
fpr_knn, tpr_knn, _ = roc_curve(y_test, y_prob_knn, pos_label=2)
roc_auc_knn = auc(fpr_knn, tpr_knn)

# Creación y Entrenamiento del Modelo Random Forest
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)

# Evaluación del Modelo Random Forest
y_pred_rf = rf.predict(X_test)
conf_matrix_rf = confusion_matrix(y_test, y_pred_rf)
class_report_rf = classification_report(y_test, y_pred_rf)
accuracy_rf = accuracy_score(y_test, y_pred_rf)
y_prob_rf = rf.predict_proba(X_test)[:, 1]
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_prob_rf, pos_label=2)
roc_auc_rf = auc(fpr_rf, tpr_rf)

# Creación y Entrenamiento del Modelo de Regresión Logística
lr = LogisticRegression(random_state=42, max_iter=1000)
lr.fit(X_train, y_train)

# Evaluación del Modelo de Regresión Logística
y_pred_lr = lr.predict(X_test)
conf_matrix_lr = confusion_matrix(y_test, y_pred_lr)
class_report_lr = classification_report(y_test, y_pred_lr)
accuracy_lr = accuracy_score(y_test, y_pred_lr)
y_prob_lr = lr.predict_proba(X_test)[:, 1]
fpr_lr, tpr_lr, _ = roc_curve(y_test, y_prob_lr, pos_label=2)
roc_auc_lr = auc(fpr_lr, tpr_lr)

# Creación y Entrenamiento del Modelo de Árbol de Decisión
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)

# Evaluación del Modelo de Árbol de Decisión
y_pred_dt = dt.predict(X_test)
conf_matrix_dt = confusion_matrix(y_test, y_pred_dt)
class_report_dt = classification_report(y_test, y_pred_dt)
accuracy_dt = accuracy_score(y_test, y_pred_dt)
y_prob_dt = dt.predict_proba(X_test)[:, 1]
fpr_dt, tpr_dt, _ = roc_curve(y_test, y_prob_dt, pos_label=2)
roc_auc_dt = auc(fpr_dt, tpr_dt)

# Función para mostrar curva ROC y matriz de confusión
def plot_roc_and_confusion_matrix(model_name, fpr, tpr, roc_auc, conf_matrix):
    plt.figure(figsize=(12, 5))

    # Curva ROC
    plt.subplot(1, 2, 1)
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'AUC = {roc_auc:.2f}')
    plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {model_name}')
    plt.legend(loc='lower right')

    # Matriz de confusión
    plt.subplot(1, 2, 2)
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix - {model_name}')

    plt.tight_layout()
    plt.show()

# KNN
plot_roc_and_confusion_matrix('KNN', fpr_knn, tpr_knn, roc_auc_knn, conf_matrix_knn)

# Random Forest
plot_roc_and_confusion_matrix('Random Forest', fpr_rf, tpr_rf, roc_auc_rf, conf_matrix_rf)

# Logistic Regression
plot_roc_and_confusion_matrix('Logistic Regression', fpr_lr, tpr_lr, roc_auc_lr, conf_matrix_lr)

# Decision Tree
plot_roc_and_confusion_matrix('Decision Tree', fpr_dt, tpr_dt, roc_auc_dt, conf_matrix_dt)

# Mostrar el box plot para la variable 'monto', categorizado por 'incumplimiento'
plt.figure(figsize=(10, 6))
sns.boxplot(x='incumplimiento', y='monto', data=datos)
plt.title('Box Plot del Monto del Préstamo por Incumplimiento')
plt.xlabel('Incumplimiento')
plt.ylabel('Monto del Préstamo (DM)')
plt.show()