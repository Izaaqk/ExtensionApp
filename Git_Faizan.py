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
df_GF1 = pd.read_csv("./Dataset/Git Faizan/data.csv", encoding='ISO-8859-1', delimiter=",")
df_GF2 = pd.read_csv("./Dataset/Git Faizan/data2.csv", header=None, names=['url', 'label'], encoding='ISO-8859-1', delimiter=",")

#Combinar los CSV
df_GF = pd.concat([df_GF1, df_GF2])
print(df_GF.info())

"""
#Corrección de Nombre para Cabeceras
columnas_renombrar = {'url':'URL',
                      'label':'Phishing'
                      }
df_GF.rename(columns = columnas_renombrar, inplace = True)

#Estructura del Dataset
#print(df_GF.info())

#Valores Duplicados
#print(df_GF[df_GF.duplicated()])

df_GF.drop_duplicates(inplace=True)
#print(df_GF.shape)

#Detección de Valores Nulos
#print(df_GF.isnull().sum(axis = 1).sort_values(ascending = False))

#Cambiar Valores
df_GF.loc[df_GF['Phishing'] == 'bad', 'Phishing'] = 1
df_GF.loc[df_GF['Phishing'] == 'good', 'Phishing'] = 0
#print(df_GF.tail(5))

#Cambiar Tipo de Dato
columnas_castear = {
                    'URL' : 'object',
                    'Phishing' : 'int64',
                    }
df_GF = df_GF.astype(columnas_castear)
#print(df_GF.dtypes)

#Gráfico de barras 
ax = df_GF['Phishing'].value_counts().plot(kind='bar')
ax.set_ylabel("Frecuencia")
ax.set_xlabel("Phishing")

# Mostrar el gráfico
plt.title("Frecuencia de Phishing")
#plt.show()

#Plotting Wordcloud
df_Legitimo=df_GF[df_GF.Phishing==0]
df_Phishing=df_GF[df_GF.Phishing==1]
#print(df_Phishing.Phishing.value_counts())
#print(df_Legitimo.Phishing.value_counts())

URL_Phishing = " ".join(i for i in df_Phishing.URL)
wordcloud = WordCloud(width=1600, height=800,colormap='Paired').generate(URL_Phishing)
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
df_Phishing['use_of_ip'] = df_Phishing['URL'].apply(lambda i: having_ip_address(i))

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


df_Phishing['abnormal_url'] = df_Phishing['URL'].apply(lambda i: abnormal_url(i))
from googlesearch import search
def google_index(url):
    site = search(url, 5)
    return 1 if site else 0
df_Phishing['google_index'] = df_Phishing['URL'].apply(lambda i: google_index(i))

def count_dot(url):
    count_dot = url.count('.')
    return count_dot

df_Phishing['count.'] = df_Phishing['URL'].apply(lambda i: count_dot(i))
print(df_Phishing.head())
"""