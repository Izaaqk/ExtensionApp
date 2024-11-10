#Carga de Librerias para el Proyecto
#pip install pandas numpy seaborn matplotlib
import pandas as pd
pd.set_option('display.max_columns', None)
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
from wordcloud import WordCloud
import re
from urllib.parse import urlparse
from googlesearch import search
from urllib.parse import urlparse
from tld import get_tld
import os.path
from sklearn.metrics import roc_curve,confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, auc
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
import xgboost as xgb
from tabulate import tabulate
import joblib
import pickle

#########################################
#########################################
#Pre-Procesamiento de Datos
#########################################
#########################################

#Cargar Datos de los Datasets

#PhishStorm
df_PS = pd.read_csv("./Dataset/PhishStorm/urlset.csv", on_bad_lines='skip', encoding='ISO-8859-1', delimiter=",", low_memory=False)
df_PS = df_PS[['domain', 'label']]

#Phishtank
df_PT = pd.read_csv("./Dataset/Phishtank/verified_online.csv", on_bad_lines='skip', encoding='ISO-8859-1', delimiter=",", low_memory=False)
df_PT = df_PT[['url']]
df_PT['Peligroso'] = 1

#Kaggle
df_Kg = pd.read_csv("./Dataset/Kaggle/new_data_urls.csv", on_bad_lines='skip', encoding='ISO-8859-1', delimiter=",", low_memory=False)
df_Kg = df_Kg[['url', 'status']]

#Corrección de nombre para cabeceras

#PhishStorm
columnas_renombrar = {
    'domain':'URL',
    'label':'Peligroso'
    }
df_PS.rename(columns = columnas_renombrar, inplace = True)

#Phishtank
columnas_renombrar = {
    'url':'URL',
    'Peligroso':'Peligroso'
    }
df_PT.rename(columns = columnas_renombrar, inplace = True)

#Kaggle
columnas_renombrar = {
    'url':'URL',
    'status':'Peligroso'
    }
df_Kg.rename(columns = columnas_renombrar, inplace = True)

"""
#Verificar Datos duplicados

#PhishStorm
#print(df_PS[df_PS.duplicated()].count())

#Phishtank
#print(df_PT[df_PT.duplicated()].count())
"""

#Eliminar duplicados

#PhishStorm
df_PS.drop_duplicates(inplace=True)

#Phishtank
df_PT.drop_duplicates(inplace=True)

#Kaggle
df_Kg.drop_duplicates(inplace=True)

#Detección de valores nulos

#PhishStorm
#print(df_PS.isnull().sum(axis = 1).sort_values(ascending = False))

#Phishtank
#print(df_PT.isnull().sum(axis = 1).sort_values(ascending = False))

#Cambiar 

#PhishStorm
df_PS.loc[df_PS['Peligroso'] == 1.0, 'Peligroso'] = 1
df_PS.loc[df_PS['Peligroso'].isna(), 'Peligroso'] = 1
df_PS.loc[df_PS['Peligroso'] == 0.0, 'Peligroso'] = 0

#Cambiar Tipo de Dato
columnas_castear = {
    'URL':'object',
    'Peligroso' : 'int64'
    }

#PhishStorm
df_PS = df_PS.astype(columnas_castear)

#Phishtank
df_PT = df_PT.astype(columnas_castear)

#Kaggle
df_Kg = df_Kg.astype(columnas_castear)
df_Kg['Peligroso'] = df_Kg['Peligroso'].apply(lambda x: 0 if x == 1 else 1)

"""
#Gráfico de barras 
ax = df_PS['Peligroso'].value_counts().plot(kind='bar')
ax.set_ylabel("Frecuencia")
ax.set_xlabel("Peligroso")

# Mostrar el gráfico
plt.title("Frecuencia de Peligroso")
plt.show()

#Gráfico de barras 
ax = df_PT['Peligroso'].value_counts().plot(kind='bar')
ax.set_ylabel("Frecuencia")
ax.set_xlabel("Peligroso")

# Mostrar el gráfico
plt.title("Frecuencia de Peligroso")
plt.show()
"""

#Combinar los Datasets en un solo df
df_DatasetURLs = pd.concat([df_PS,df_PT,df_Kg])
#df_DatasetURLs = pd.concat([df_PS,df_PT])
df_DatasetURLs = df_DatasetURLs.sample(frac=1).reset_index(drop=True)

"""
#Gráfico de barras 
ax = df_DatasetURLs['Peligroso'].value_counts().plot(kind='bar')
ax.set_ylabel("Frecuencia")
ax.set_xlabel("Peligroso")

# Mostrar el gráfico
#plt.title("Frecuencia de Peligroso")
#plt.show()

#print(df_DatasetURLs.info())
"""

#Plotting Wordcloud
#Filtramos los Registros Peligrosos
df_Peligroso=df_DatasetURLs[df_DatasetURLs.Peligroso==1]
#print(df_PT.Peligroso.value_counts())


#Generar una Nube de Palabras Detectadas en el df
"""
URL_Peligroso = " ".join(i for i in df_Peligroso.URL)
wordcloud = WordCloud(width=1600, height=800,colormap='Paired').generate(URL_Peligroso)
plt.figure( figsize=(12,14),facecolor='k')
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.tight_layout(pad=0)
plt.show()
"""

#Feature Engineering
#Validar si el URL usa una IP
def Dominio_IP(url):
    match = re.search(
        '(([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\.([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\.([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\.'
        '([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\/)|'  # IPv4
        '((0x[0-9a-fA-F]{1,2})\\.(0x[0-9a-fA-F]{1,2})\\.(0x[0-9a-fA-F]{1,2})\\.(0x[0-9a-fA-F]{1,2})\\/)' # IPv4 in hexadecimal
        '(?:[a-fA-F0-9]{1,4}:){7}[a-fA-F0-9]{1,4}', url)  # Ipv6
    if match:
        return 1
    else:
        return 0
df_DatasetURLs['usa_IP'] = df_DatasetURLs['URL'].apply(lambda i: Dominio_IP(i))

#Validar si el URL es Anormal
def anormal_URL(url):
    try:
        hostname = urlparse(url).hostname
        if hostname is None:
            return 0  # Retorna 0 si no se puede extraer un hostname válido
        match = re.search(hostname, url)
        return 1 if match else 0
    except Exception:
        return 0  # Retorna 0 si la URL es inválida


df_DatasetURLs['anormal_URL'] = df_DatasetURLs['URL'].apply(lambda i: anormal_URL(i))

#Validar si el URL se encuentra indexado en Google
def indexado_Google(url):
    site = search(url, 5)
    return 1 if site else 0
df_DatasetURLs['indexado_Google'] = df_DatasetURLs['URL'].apply(lambda i: indexado_Google(i))

#Contar el número de "." en el URL
def count_punto(url):
    count_punto = url.count('.')
    return count_punto

df_DatasetURLs['count_punto'] = df_DatasetURLs['URL'].apply(lambda i: count_punto(i))
#print(df_DatasetURLs.head())

#Contar el número de "www" en el URL
def count_www(url):
    url.count('www')
    return url.count('www')

df_DatasetURLs['count_www'] = df_DatasetURLs['URL'].apply(lambda i: count_www(i))

#Contar el número de "@" en el URL
def count_a(url):
    return url.count('@')

df_DatasetURLs['count_@'] = df_DatasetURLs['URL'].apply(lambda i: count_a(i))

#Contar el número de "/" en el path del URL
def count_dir(url):
    """Cuenta el número de directorios en el path de la URL."""
    try:
        urldir = urlparse(url).path
        return urldir.count('/')
    except Exception:
        return 0  # Retorna 0 si la URL es mal formada

df_DatasetURLs['count_dir'] = df_DatasetURLs['URL'].apply(count_dir)
#df_DatasetURLs['count_dir'] = df_DatasetURLs['URL'].apply(lambda i: count_dir(i))

#Contar el número de "//" en el URL
def count_embed(url):
    """Cuenta el número de embeds '//' en el path de la URL."""
    try:
        urldir = urlparse(url).path
        return urldir.count('//')
    except Exception:
        return 0  # Retorna 0 si la URL es mal formada

df_DatasetURLs['count_embed'] = df_DatasetURLs['URL'].apply(count_embed)
#df_DatasetURLs['count_embed'] = df_DatasetURLs['URL'].apply(lambda i: count_embed(i))

#Valida si la url esta acortada
def servicio_acortamiento(url):
    match = re.search(
        r'bit\.ly|goo\.gl|shorte\.st|go2l\.ink|x\.co|ow\.ly|t\.co|tinyurl|tr\.im|is\.gd|cli\.gs|'
        r'yfrog\.com|migre\.me|ff\.im|tiny\.cc|url4\.eu|twit\.ac|su\.pr|twurl\.nl|snipurl\.com|'
        r'short\.to|BudURL\.com|ping\.fm|post\.ly|Just\.as|bkite\.com|snipr\.com|fic\.kr|loopt\.us|'
        r'doiop\.com|short\.ie|kl\.am|wp\.me|rubyurl\.com|om\.ly|to\.ly|bit\.do|t\.co|lnkd\.in|'
        r'db\.tt|qr\.ae|adf\.ly|goo\.gl|bitly\.com|cur\.lv|tinyurl\.com|ow\.ly|bit\.ly|ity\.im|'
        r'q\.gs|is\.gd|po\.st|bc\.vc|twitthis\.com|u\.to|j\.mp|buzurl\.com|cutt\.us|u\.bb|yourls\.org|'
        r'x\.co|prettylinkpro\.com|scrnch\.me|filoops\.info|vzturl\.com|qr\.net|1url\.com|tweez\.me|v\.gd|'
        r'tr\.im|link\.zip\.net', url
    )
    return 1 if match else 0

df_DatasetURLs['short_url'] = df_DatasetURLs['URL'].apply(lambda i: servicio_acortamiento(i))

#Contar el número de "https" en el URL
def count_https(url):
    return url.count('https')
df_DatasetURLs['count_https'] = df_DatasetURLs['URL'].apply(lambda i : count_https(i))

#Contar el número de "http" en el URL
def count_http(url):
    return url.count('http')

df_DatasetURLs['count_http'] = df_DatasetURLs['URL'].apply(lambda i : count_http(i))

#Contar el número de "%" en el URL
def count_per(url):
    return url.count('%')
df_DatasetURLs['count_%'] = df_DatasetURLs['URL'].apply(lambda i : count_per(i))

#Contar el número de "?" en el URL
def count_ques(url):
    return url.count('?')
df_DatasetURLs['count_?'] = df_DatasetURLs['URL'].apply(lambda i: count_ques(i))

#Contar el número de "-" en el URL
def count_hyphen(url):
    return url.count('-')
df_DatasetURLs['count_-'] = df_DatasetURLs['URL'].apply(lambda i: count_hyphen(i))

#Contar el número de "=" en el URL
def count_equal(url):
    return url.count('=')
df_DatasetURLs['count_='] = df_DatasetURLs['URL'].apply(lambda i: count_equal(i))

#Cuenta los caracteres de la URL
def tamaño_URL(url):
    return len(str(url))
df_DatasetURLs['url_length'] = df_DatasetURLs['URL'].apply(lambda i: tamaño_URL(i))

#Cuenta la cantidad de caracteres del hostname
def tamaño_hostname(url):
    try:
        hostname = urlparse(url).hostname
        if hostname:
            return len(hostname)
        else:
            return 0  # Si no hay hostname, retornar 0
    except Exception:
        return 0  # Si falla la operación, retornar 0
#print(df_DatasetURLs.head())
df_DatasetURLs['hostname_length'] = df_DatasetURLs['URL'].apply(tamaño_hostname)

#Valida si en la URL se encuentran palabras sospechosas
def suspicious_words(url):
    #'PayPal|login|signin|bank|account|update|free|lucky|service|bonus|ebayisapi|webscr'
    match = re.search('PayPal|login|signin|bank|account|update|free|lucky|service|bonus|ebayisapi|webscr',
                      url)
    if match:
        return 1
    else:
        return 0
df_DatasetURLs['sus_url'] = df_DatasetURLs['URL'].apply(lambda i: suspicious_words(i))

#Cuenta la cantidad de caracteres numéricos de la URL
def count_num(url):
    digits = 0
    for i in url:
        if i.isnumeric():
            digits = digits + 1
    return digits
df_DatasetURLs['count_num']= df_DatasetURLs['URL'].apply(lambda i: count_num(i))

#Cuenta la cantidad de letras de la URL
def letter_count(url):
    letters = 0
    for i in url:
        if i.isalpha():
            letters = letters + 1
    return letters
df_DatasetURLs['count_letras']= df_DatasetURLs['URL'].apply(lambda i: letter_count(i))

#Cuenta la longitud de los caracteres del primer directorio
def fd_length(url):
    try:
        urlpath = urlparse(url).path
        parts = urlpath.split('/')
        if len(parts) > 1:
            return len(parts[1])  # Longitud del primer directorio
        else:
            return 0  # No hay directorios en el path
    except Exception:
        return 0  # Manejar errores para entradas inválidas
df_DatasetURLs['fd_length'] = df_DatasetURLs['URL'].apply(fd_length)


#Extrae el dominio superior de la URL
df_DatasetURLs['tld'] = df_DatasetURLs['URL'].apply(lambda i: get_tld(i,fail_silently=True))



#Cuenta la longitud de los caracteres del dominio superior
def tld_length(tld):
    try:
        return len(tld)
    except:
        return -1
df_DatasetURLs['tld_length'] = df_DatasetURLs['tld'].apply(lambda i: tld_length(i))

#Eliminamos la Columna "tld"
df_DatasetURLs = df_DatasetURLs.drop('tld', axis=1)
#print(df_DatasetURLs.columns)


#print(df_DatasetURLs['Peligroso'].value_counts())
#print(df_DatasetURLs.head(1))

#########################################
#########################################
#Analisis EDA
#########################################
#########################################

#Gráfico IP en al URL
"""
sns.set_theme(style="darkgrid")
ax = sns.countplot(y="Peligroso", data=df_DatasetURLs,hue="usa_IP")
plt.show()
"""

#Gráfico Distribución Anormal en la URL
"""
sns.set_theme(style="darkgrid")
ax = sns.countplot(y="Peligroso", data=df_DatasetURLs,hue="anormal_URL")
plt.show()
"""

#Gráfico de URL Indexada en Google
"""
sns.set_theme(style="darkgrid")
ax = sns.countplot(y="Peligroso", data=df_DatasetURLs,hue="indexado_Google")
plt.show()
"""

#Gráfico de URL Acortada
"""
sns.set_theme(style="darkgrid")
ax = sns.countplot(y="Peligroso", data=df_DatasetURLs,hue="short_url")
plt.show()
"""

#Gráfico de URL Sospechosa
"""
sns.set_theme(style="darkgrid")
ax = sns.countplot(y="Peligroso", data=df_DatasetURLs,hue="sus_url")
plt.show()
"""

#Gráfico de Distribución de Conteo de "." en la URL
"""
sns.set_theme(style="darkgrid")
ax = sns.catplot(x="Peligroso", y="count_punto", kind="box", data=df_DatasetURLs)
plt.show()
"""

#Gráfico de Distribución de Conteo de "www" en la URL
"""
sns.set_theme(style="darkgrid")
ax = sns.catplot(x="Peligroso", y="count_www", kind="box", data=df_DatasetURLs)
plt.show()
"""

#Gráfico de Distribución de conteo de "@" en la URL
"""
sns.set_theme(style="darkgrid")
ax = sns.catplot(x="Peligroso", y="count_@", kind="box", data=df_DatasetURLs)
plt.show()
"""

#Gráfico de Distribución de Conteo de "/" en el path del URL
"""
sns.set_theme(style="darkgrid")
ax = sns.catplot(x="Peligroso", y="count_dir", kind="box", data=df_DatasetURLs)
plt.show()
"""

#Gráfico de Distribución de la Longitud de la URL
"""
sns.set_theme(style="darkgrid")
ax = sns.catplot(x="Peligroso", y="hostname_length", kind="box", data=df_DatasetURLs)
plt.show()
"""

#Gráfico de Distribución de la Longitud del Primer Directorio
"""
sns.set_theme(style="darkgrid")
ax = sns.catplot(x="Peligroso", y="fd_length", kind="box", data=df_DatasetURLs)
plt.show()
"""

#Gráfico de Distribución de la Longitud del Dominio Superior
"""
sns.set_theme(style="darkgrid")
ax = sns.catplot(x="Peligroso", y="tld_length", kind="box", data=df_DatasetURLs)
plt.show()
"""

#########################################
#########################################
#Creation of Feature & Target
#########################################
#########################################
df_DatasetURLs_sampled = df_DatasetURLs.sample(frac=0.1, random_state=42)
#Creación de Variables
x = df_DatasetURLs[['usa_IP','anormal_URL', 'count_punto', 'count_www', 'count_@',
       'count_dir', 'count_embed', 'short_url', 'count_https',
       'count_http', 'count_%', 'count_?', 'count_-', 'count_=', 'url_length',
        'sus_url', 'fd_length',  'count_num', 'tld_length', 'hostname_length',
       'count_letras']]

y = df_DatasetURLs['Peligroso']

#Separación del conjunto de datos de train y test 200 -4200 bajo 0.1
X_train, X_test, y_train, y_test = train_test_split(x, y, stratify=y, test_size=0.10,shuffle=True, random_state=4200)

#Creación del Modelo
#1. Clasificador Random Forest
#rf = RandomForestClassifier(n_estimators=100,max_features='sqrt')
rf = RandomForestClassifier(n_estimators=50, max_depth=10, max_features='sqrt')
rf.fit(X_train,y_train)
y_pred = rf.predict(X_test)
y_pred_proba = rf.predict_proba(X_test)[:, 1]

# Matriz de confusión
conf_matrix = confusion_matrix(y_test, y_pred)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Matriz de confusión para calcular la especificidad
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
specificity = tn / (tn + fp)

# Crear un DataFrame con los resultados
metrics_table = pd.DataFrame({
    'Metric': ['Accuracy', 'Precision', 'Recall', 'Specificity', 'F1 Score'],
    'Value': [accuracy, precision, recall, specificity, f1]
})

# Mostrar la tabla
print(tabulate(metrics_table, headers='keys', tablefmt='pretty'))

#2. Light GBM
#lgb = LGBMClassifier(objective='binary',boosting_type= 'gbdt',n_jobs = 42, random_state=42)
lgb = LGBMClassifier(n_estimators=50, max_depth=10, boosting_type='gbdt', n_jobs=4, random_state=42)
LGB_C = lgb.fit(X_train, y_train)
y_pred_proba = LGB_C.predict_proba(X_test)[:, 1]

y_pred_lgb = LGB_C.predict(X_test)
# Matriz de confusión
conf_matrix = confusion_matrix(y_test, y_pred_lgb)

accuracy = accuracy_score(y_test, y_pred_lgb)
precision = precision_score(y_test, y_pred_lgb)
recall = recall_score(y_test, y_pred_lgb)
f1 = f1_score(y_test, y_pred_lgb)

# Matriz de confusión para calcular la especificidad
tn, fp, fn, tp = confusion_matrix(y_test, y_pred_lgb).ravel()
specificity = tn / (tn + fp)

# Crear un DataFrame con los resultados
metrics_table = pd.DataFrame({
    'Metric': ['Accuracy', 'Precision', 'Recall', 'Specificity', 'F1 Score'],
    'Value': [accuracy, precision, recall, specificity, f1]
})

# Mostrar la tabla
print(tabulate(metrics_table, headers='keys', tablefmt='pretty'))

#3. XGboost
#xgb_c = xgb.XGBClassifier(n_estimators= 100)
xgb_c = xgb.XGBClassifier(n_estimators=50, max_depth=10)
xgb_c.fit(X_train,y_train)
y_pred_x = xgb_c.predict(X_test)
y_pred_proba = xgb_c.predict_proba(X_test)[:, 1]
# Matriz de confusión
conf_matrix = confusion_matrix(y_test, y_pred_x)

accuracy = accuracy_score(y_test, y_pred_x)
precision = precision_score(y_test, y_pred_x)
recall = recall_score(y_test, y_pred_x)
f1 = f1_score(y_test, y_pred_x)

# Matriz de confusión para calcular la especificidad
tn, fp, fn, tp = confusion_matrix(y_test, y_pred_x).ravel()
specificity = tn / (tn + fp)

# Crear un DataFrame con los resultados
metrics_table = pd.DataFrame({
    'Metric': ['Accuracy', 'Precision', 'Recall', 'Specificity', 'F1 Score'],
    'Value': [accuracy, precision, recall, specificity, f1]
})

# Mostrar la tabla
print(tabulate(metrics_table, headers='keys', tablefmt='pretty'))

#Obtener las probabilidades de predicción para la clase positiva
rf_probs = rf.predict_proba(X_test)[:, 1]
lgb_probs = LGB_C.predict_proba(X_test)[:, 1]
xgb_probs = xgb_c.predict_proba(X_test)[:, 1]

#Calcular la curva ROC y el AUC para cada modelo
rf_fpr, rf_tpr, _ = roc_curve(y_test, rf_probs)
xgb_fpr, xgb_tpr, _ = roc_curve(y_test, xgb_probs)
lgb_fpr, lgb_tpr, _ = roc_curve(y_test, lgb_probs)

rf_auc = auc(rf_fpr, rf_tpr)
xgb_auc = auc(xgb_fpr, xgb_tpr)
lgb_auc = auc(lgb_fpr, lgb_tpr)

#Graficar las curvas ROC
plt.figure(figsize=(10, 6))

plt.plot(rf_fpr, rf_tpr, label=f'Random Forest (AUC = {rf_auc:.2f})', color='blue')
plt.plot(xgb_fpr, xgb_tpr, label=f'XGBoost (AUC = {xgb_auc:.2f})', color='green')
plt.plot(lgb_fpr, lgb_tpr, label=f'LightGBM (AUC = {lgb_auc:.2f})', color='red')

# Línea de referencia
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')

# Títulos y etiquetas
plt.title('Curva ROC')
plt.xlabel('Tasa de Falsos Positivos')
plt.ylabel('Tasa de Verdaderos Positivos')
plt.legend(loc='lower right')

#plt.show()
####################################

# Definir todas las funciones de extracción de características
def Dominio_IP2(url):
    match = re.search(r'(([01]?\d\d?|2[0-4]\d|25[0-5])\.){3}([01]?\d\d?|2[0-4]\d|25[0-5])|([a-fA-F0-9]{1,4}:){7}[a-fA-F0-9]{1,4}', url)
    return 1 if match else 0

def anormal_URL2(url):
    hostname = urlparse(url).hostname
    return 1 if re.search(str(hostname), url) else 0

def count_punto2(url):
    return url.count('.')

def count_www2(url):
    return url.count('www')

def count_a2(url):
    return url.count('@')

def count_dir2(url):
    return urlparse(url).path.count('/')

def count_embed2(url):
    return urlparse(url).path.count('//')

def servicio_acortamiento2(url):
    match = re.search(r'bit\.ly|goo\.gl|shorte\.st|go2l\.ink|x\.co|ow\.ly|t\.co|tinyurl|tr\.im|is\.gd|cli\.gs', url)
    return 1 if match else 0

def count_https2(url):
    return url.count('https')

def count_http2(url):
    return url.count('http')

def count_per2(url):
    return url.count('%')

def count_ques2(url):
    return url.count('?')

def count_hyphen2(url):
    return url.count('-')

def count_equal2(url):
    return url.count('=')

def tamaño_URL2(url):
    return len(str(url))

def tamaño_hostname2(url):
    return len(urlparse(url).netloc)

def suspicious_words2(url):
    match = re.search(r'PayPal|login|signin|bank|account|update|free|lucky|service|bonus|ebayisapi|webscr', url)
    return 1 if match else 0

def count_num2(url):
    return sum(1 for char in url if char.isnumeric())

def letter_count2(url):
    return sum(1 for char in url if char.isalpha())

def fd_length2(url):
    urlpath = urlparse(url).path
    return len(urlpath.split('/')[1]) if '/' in urlpath else 0

def tld_length2(tld):
    return len(tld) if tld else -1

# Función de procesamiento para preparar todas las características de una URL
#def procesar_url(url):
    tld = get_tld(url, fail_silently=True)
    features = [
        Dominio_IP2(url),
        anormal_URL2(url),
        count_punto2(url),
        count_www2(url),
        count_a2(url),
        count_dir2(url),
        count_embed2(url),
        servicio_acortamiento2(url),
        count_https2(url),
        count_http2(url),
        count_per2(url),
        count_ques2(url),
        count_hyphen2(url),
        count_equal2(url),
        tamaño_URL2(url),
        tamaño_hostname2(url),
        suspicious_words2(url),
        count_num2(url),
        letter_count2(url),
        fd_length2(url),
        tld_length2(tld)
    ]
    return np.array(features).reshape(1, -1)


def extract_features(url):
    features = {}
    # Ejemplo de extracción de características
    features['usa_IP'] = int(bool(re.search(r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b', url)))
    features['anormal_URL'] = int('@' in url)
    features['count_punto'] = url.count('.')
    features['count_www'] = url.count('www')
    features['count_@'] = url.count('@')
    features['count_dir'] = url.count('/')
    features['count_embed'] = url.count('embed')
    features['short_url'] = int(len(url) < 15)
    features['count_https'] = url.count('https')
    features['count_http'] = url.count('http')
    features['count_%'] = url.count('%')
    features['count_?'] = url.count('?')
    features['count_-'] = url.count('-')
    features['count_='] = url.count('=')
    features['url_length'] = len(url)
    features['sus_url'] = int(any(keyword in url for keyword in ['login', 'secure', 'account', 'update', 'verify']))
    features['fd_length'] = len(url.split('/')[2].split('.')[0]) if '//' in url else 0
    features['count_num'] = sum(c.isdigit() for c in url)
    features['tld_length'] = len(url.split('.')[-1]) if '.' in url else 0
    features['hostname_length'] = len(url.split('/')[2]) if '//' in url else len(url)
    features['count_letras'] = sum(c.isalpha() for c in url)
    
    
    return list(features.values())

# Función de predicción
def predecir_url1(url):
    features = extract_features(url)
    prediccion = LGB_C.predict([features])
    probabilidad = LGB_C.predict_proba([features])[:, 1]
    resultado = 'Peligrosa' if prediccion[0] == 0.80 else 'Benigna'
    return resultado, probabilidad

def predecir_url2(url):
    features = extract_features(url)
    prediccion = rf.predict([features])
    probabilidad = rf.predict_proba([features])[:, 1]
    resultado = 'Peligrosa' if probabilidad[0] >= 0.80 else 'Benigna'
    return resultado, probabilidad

def predecir_url3(url):
    features = extract_features(url)
    prediccion = xgb_c.predict([features])
    probabilidad = xgb_c.predict_proba([features])[:, 1]
    resultado = 'Peligrosa' if prediccion[0] >= 0.80 else 'Benigna'
    return resultado, probabilidad

# Solicitar la URL 3
"""
url = "https://www.google.com/"
resultado, probabilidad = predecir_url2(url)
print(f"La URL es {resultado} con una probabilidad de peligro de {probabilidad[0]:.2f} y {url}")

url = "news.bbc.co.uk/1/hi/technology/7445956.stm"
resultado, probabilidad = predecir_url2(url)
print(f"La URL es {resultado} con una probabilidad de peligro de {probabilidad[0]:.2f} y {url}")
"""
#url = "www.dghjdgf.com/paypal.co.uk/cycgi-bin/webscrcmd=_home-customer&nav=1/loading.php"
#resultado, probabilidad = predecir_url3(url)
#print(f"La URL es {resultado} con una probabilidad de peligro de {probabilidad[0]:.2f} y {url}")
#joblib.dump(xgb_c, 'xgb_model.pkl')
#print("Modelo XGBoost guardado como 'xgb_model.pkl'")