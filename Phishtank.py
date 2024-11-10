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
df_PT = pd.read_csv("./Dataset/Phishtank/verified_online.csv", on_bad_lines='skip', encoding='ISO-8859-1', delimiter=",")
df_PT = df_PT[['url']]
df_PT['Peligroso'] = 1
#print(df_PT.head(5))


#Corrección de Nombre para Cabeceras
columnas_renombrar = {
    'url':'URL',
    'Peligroso':'Peligroso'
    }
df_PT.rename(columns = columnas_renombrar, inplace = True)
#Estructura del Dataset
#print(df_PT.info())
#print(df_PT['Peligroso'].value_counts())
#Valores Duplicados
#print(df_PT[df_PT.duplicated()])

df_PT.drop_duplicates(inplace=True)
#print(df_PT.shape)

#Detección de Valores Nulos
#print(df_ PT.isnull().sum(axis = 1).sort_values(ascending = False))

#Cambiar Tipo de Dato
columnas_castear = {
    'URL':'object',
    'Peligroso' : 'int64'
    }
df_PT = df_PT.astype(columnas_castear)

#Gráfico de barras 
ax = df_PT['Peligroso'].value_counts().plot(kind='bar')
ax.set_ylabel("Frecuencia")
ax.set_xlabel("Peligroso")

# Mostrar el gráfico
plt.title("Frecuencia de Peligroso")
#plt.show()

#Plotting Wordcloud
df_PT=df_PT[df_PT.Peligroso==1]
#print(df_PT.Peligroso.value_counts())

URL_Peligroso = " ".join(i for i in df_PT.URL)
wordcloud = WordCloud(width=1600, height=800,colormap='Paired').generate(URL_Peligroso)
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
df_PT['use_of_ip'] = df_PT['URL'].apply(lambda i: having_ip_address(i))

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


df_PT['abnormal_url'] = df_PT['URL'].apply(lambda i: abnormal_url(i))
from googlesearch import search
def google_index(url):
    site = search(url, 5)
    return 1 if site else 0
df_PT['google_index'] = df_PT['URL'].apply(lambda i: google_index(i))

def count_dot(url):
    count_dot = url.count('.')
    return count_dot

df_PT['count.'] = df_PT['URL'].apply(lambda i: count_dot(i))
#print(df_PT.head())

def count_www(url):
    url.count('www')
    return url.count('www')

df_PT['count-www'] = df_PT['URL'].apply(lambda i: count_www(i))

def count_atrate(url):
     
    return url.count('@')

df_PT['count@'] = df_PT['URL'].apply(lambda i: count_atrate(i))


def no_of_dir(url):
    urldir = urlparse(url).path
    return urldir.count('/')

df_PT['count_dir'] = df_PT['URL'].apply(lambda i: no_of_dir(i))

def no_of_embed(url):
    urldir = urlparse(url).path
    return urldir.count('//')

df_PT['count_embed_domian'] = df_PT['URL'].apply(lambda i: no_of_embed(i))


def shortening_service(url):
    match = re.search('bit\.ly|goo\.gl|shorte\.st|go2l\.ink|x\.co|ow\.ly|t\.co|tinyurl|tr\.im|is\.gd|cli\.gs|'
                      'yfrog\.com|migre\.me|ff\.im|tiny\.cc|url4\.eu|twit\.ac|su\.pr|twurl\.nl|snipurl\.com|'
                      'short\.to|BudURL\.com|ping\.fm|post\.ly|Just\.as|bkite\.com|snipr\.com|fic\.kr|loopt\.us|'
                      'doiop\.com|short\.ie|kl\.am|wp\.me|rubyurl\.com|om\.ly|to\.ly|bit\.do|t\.co|lnkd\.in|'
                      'db\.tt|qr\.ae|adf\.ly|goo\.gl|bitly\.com|cur\.lv|tinyurl\.com|ow\.ly|bit\.ly|ity\.im|'
                      'q\.gs|is\.gd|po\.st|bc\.vc|twitthis\.com|u\.to|j\.mp|buzurl\.com|cutt\.us|u\.bb|yourls\.org|'
                      'x\.co|prettylinkpro\.com|scrnch\.me|filoops\.info|vzturl\.com|qr\.net|1url\.com|tweez\.me|v\.gd|'
                      'tr\.im|link\.zip\.net',
                      url)
    if match:
        return 1
    else:
        return 0
    
    
df_PT['short_url'] = df_PT['URL'].apply(lambda i: shortening_service(i))

def count_https(url):
    return url.count('https')

df_PT['count-https'] = df_PT['URL'].apply(lambda i : count_https(i))

def count_http(url):
    return url.count('http')

df_PT['count-http'] = df_PT['URL'].apply(lambda i : count_http(i))

def count_per(url):
    return url.count('%')

df_PT['count%'] = df_PT['URL'].apply(lambda i : count_per(i))

def count_ques(url):
    return url.count('?')

df_PT['count?'] = df_PT['URL'].apply(lambda i: count_ques(i))

def count_hyphen(url):
    return url.count('-')

df_PT['count-'] = df_PT['URL'].apply(lambda i: count_hyphen(i))

def count_equal(url):
    return url.count('=')

df_PT['count='] = df_PT['URL'].apply(lambda i: count_equal(i))

def url_length(url):
    return len(str(url))


#Length of URL
df_PT['url_length'] = df_PT['URL'].apply(lambda i: url_length(i))
#Hostname Length

def hostname_length(url):
    return len(urlparse(url).netloc)

df_PT['hostname_length'] = df_PT['URL'].apply(lambda i: hostname_length(i))

print(df_PT.head())

def suspicious_words(url):
    match = re.search('PayPal|login|signin|bank|account|update|free|lucky|service|bonus|ebayisapi|webscr',
                      url)
    if match:
        return 1
    else:
        return 0
df_PT['sus_url'] = df_PT['URL'].apply(lambda i: suspicious_words(i))


def digit_count(url):
    digits = 0
    for i in url:
        if i.isnumeric():
            digits = digits + 1
    return digits


df_PT['count-digits']= df_PT['URL'].apply(lambda i: digit_count(i))


def letter_count(url):
    letters = 0
    for i in url:
        if i.isalpha():
            letters = letters + 1
    return letters


df_PT['count-letters']= df_PT['URL'].apply(lambda i: letter_count(i))

df_PT.head()

from urllib.parse import urlparse
from tld import get_tld
import os.path

#First Directory Length
def fd_length(url):
    urlpath= urlparse(url).path
    try:
        return len(urlpath.split('/')[1])
    except:
        return 0

df_PT['fd_length'] = df_PT['URL'].apply(lambda i: fd_length(i))

#Length of Top Level Domain
df_PT['tld'] = df_PT['URL'].apply(lambda i: get_tld(i,fail_silently=True))


def tld_length(tld):
    try:
        return len(tld)
    except:
        return -1

df_PT['tld_length'] = df_PT['tld'].apply(lambda i: tld_length(i))

df_PT = df_PT.drop('tld', axis=1)

print(df_PT.columns)

print(df_PT['Peligroso'].value_counts())