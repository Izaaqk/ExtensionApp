#Carga de Librerias para el Proyecto
#pip install pandas numpy seaborn matplotlib
import pandas as pd
import itertools
#from sklearn.metrics import classification_report,confusion_matrix, accuracy_score
#from sklearn.model_selection import train_test_split
import pandas as pd
pd.set_option('display.max_columns', None)
import numpy as np
import matplotlib.pyplot as plt
#import xgboost as xgb
#from lightgbm import LGBMClassifier
import os
import seaborn as sns
from wordcloud import WordCloud

#######################################################

#Carga de Datos (Git Faizan)
df_PS = pd.read_csv("./Dataset/PhishStorm/urlset.csv", on_bad_lines='skip', encoding='ISO-8859-1', delimiter=",")
df_PS = df_PS[['domain', 'label']]
#print(df_PS.head(5))


#Corrección de Nombre para Cabeceras
columnas_renombrar = {
    'domain':'URL',
    'label':'Peligroso'
    }
df_PS.rename(columns = columnas_renombrar, inplace = True)
#Estructura del Dataset
#print(df_PS.info())
#print(df_PS['Peligroso'].value_counts())
#Valores Duplicados
#print(df_PS[df_PS.duplicated()])

df_PS.drop_duplicates(inplace=True)
#print(df_GF.shape)

#Detección de Valores Nulos
#print(df_GF.isnull().sum(axis = 1).sort_values(ascending = False))

#Cambiar Valores
df_PS.loc[df_PS['Peligroso'] == 1.0, 'Peligroso'] = 1
df_PS.loc[df_PS['Peligroso'].isna(), 'Peligroso'] = 1
df_PS.loc[df_PS['Peligroso'] == 0.0, 'Peligroso'] = 0
#print(df_GF.tail(5))

#Cambiar Tipo de Dato
columnas_castear = {
    'URL':'object',
    'Peligroso' : 'int64'
    }
df_PS = df_PS.astype(columnas_castear)

#Gráfico de barras 
ax = df_PS['Peligroso'].value_counts().plot(kind='bar')
ax.set_ylabel("Frecuencia")
ax.set_xlabel("Peligroso")

# Mostrar el gráfico
#plt.title("Frecuencia de Peligroso")
#plt.show()

#Plotting Wordcloud
df_Legitimo=df_PS[df_PS.Peligroso==0]
df_Peligroso=df_PS[df_PS.Peligroso==1]
#print(df_Peligroso.Peligroso.value_counts())
#print(df_Legitimo.Peligroso.value_counts())

URL_Peligroso = " ".join(i for i in df_Peligroso.URL)
wordcloud = WordCloud(width=1600, height=800,colormap='Paired').generate(URL_Peligroso)
plt.figure( figsize=(12,14),facecolor='k')
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.tight_layout(pad=0)
#plt.show()

URL_Legitimo = " ".join(i for i in df_Legitimo.URL)
wordcloud = WordCloud(width=1600, height=800,colormap='Paired').generate(URL_Legitimo)
plt.figure( figsize=(12,14),facecolor='k')
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.tight_layout(pad=0)
#plt.show()

#Feature Engineering
import re
#Use of IP or not in domain
def having_ip_address(url):
    match = re.search(
        '(([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\.([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\.([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\.'
        '([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\/)|'  # IPv4
        '((0x[0-9a-fA-F]{1,2})\\.(0x[0-9a-fA-F]{1,2})\\.(0x[0-9a-fA-F]{1,2})\\.(0x[0-9a-fA-F]{1,2})\\/)' # IPv4 in hexadecimal
        '(?:[a-fA-F0-9]{1,4}:){7}[a-fA-F0-9]{1,4}', url)  # Ipv6
    if match:
        # print match.group()
        return 1
    else:
        # print 'No matching pattern found'
        return 0
df_Peligroso['use_of_ip'] = df_Peligroso['URL'].apply(lambda i: having_ip_address(i))

from urllib.parse import urlparse

def abnormal_url(url):
    hostname = urlparse(url).hostname
    hostname = str(hostname)
    match = re.search(hostname, url)
    if match:
        # print match.group()
        return 1
    else:
        # print 'No matching pattern found'
        return 0


df_Peligroso['abnormal_url'] = df_Peligroso['URL'].apply(lambda i: abnormal_url(i))
from googlesearch import search
def google_index(url):
    site = search(url, 5)
    return 1 if site else 0
df_Peligroso['google_index'] = df_Peligroso['URL'].apply(lambda i: google_index(i))

def count_dot(url):
    count_dot = url.count('.')
    return count_dot

df_Peligroso['count.'] = df_Peligroso['URL'].apply(lambda i: count_dot(i))
print(df_Peligroso.head())
