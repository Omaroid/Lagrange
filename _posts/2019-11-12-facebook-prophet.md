---
layout: post
title: "Facebook Prophet"
author: "Omar LARAQUI"
categories: journal
tags: [ai,machineLearning,timeSeries]
image: facebookProphet.jpg
---

> Une série temporelle, ou série chronologique, est une suite de valeurs numériques représentant l'évolution d'une quantité spécifique au cours du temps. De telles suites de variables aléatoires peuvent être exprimées mathématiquement afin d'en analyser le comportement, généralement pour comprendre son évolution passée et pour en prévoir le comportement futur. Une telle transposition mathématique utilise le plus souvent des concepts de probabilités et de statistique.

## Présentation

Il y a 3 ans, l'équipe de Facebook Core Data Science team à sorti en open source un puissant outil de prévision pour les séries temporelles appelé
Prophet.

Prophet se base sur un modèle additif qui s’adapte par rapport à la saisonnalité des données mais pas que, il est assez robuste pour des jeux de
données qui représentent des données manquantes ou comportant des valeurs aberrantes (ou outliers).

Pour les non statisticiens ou data scientistes, il est assez délicat de partir sur une connaissance des principes d’analyse et de prévision temporelles
mais Prophet est venu répondre à ce besoin de la plus simple des façon.

L’objectif de l’article est de faire une introduction de Prophet sous Python 🐍. Les données et le code seront disponibles sur le Github de
Ingeniance ainsi que sur Google Colaboratory en fin du blog.

## Pourquoi Prophet?

A travers son article scientifique, Prophet vient répondre à une problématique de scalabilité sous 3 volets:
Le grand nombre de personnes qui travaillent sur les prévisions temporelles et qui ont une faible connaissance mathématiques du
domaine
Le grand nombre de problèmes de prévisions qui peuvent parfois venir d’un caractère idiomatique
Le grand nombre de sujets sur lesquels il est possible d’appliquer des prévisions temporelles qui demandent des configurations différentes
Prophet vient répondre à ce problème avec:

### Une intégration sous R et Python

Prophet propose sa bibliothèque sous **CRAN** pour les utilisateurs de R ainsi que sur **PyPi** pour les adeptes de Python.

### Une saisonnalité additive et multiplicative

La saisonnalité est une caractéristique commune des séries chronologiques. Il peut apparaître sous deux formes:

- Additive
- Multiplicative

Dans le premier cas, l'amplitude de la variation saisonnière est indépendante du niveau, tandis que dans le second, elle est liée. Comme le montre 
la figure suivante:

![additive_vs_multiplicative_seasonality](assets/img/facebook_prophet/additive_multiplicative_seasonality.png "Saisonnalité additive vs multiplicative")

Par défaut, Prophet adapte son modèle à travers un modèle additif, mais supporte également le modèle multiplicatif.[(Documentation)](https://facebook.github.io/prophet/docs/multiplicative_seasonality.html)

```python
# Modèle additif
additiveModel = Prophet()
additiveModel = Prophet(seasonality_mode='additive')

# Modèle multiplicatif
MultiplicativeModel = Prophet(seasonality_mode='multiplicative')
```

### Un support des données mensuelles et sous quotidiennes

Prophet supporte les séries non quotidiennes de type mensuel (si on a un enregistrement par mois) ou sous quotidien (si on a un enregistrement toutes les 6 ou 12 heures par exemple). **Prophet** s’adapte automatiquement à ce type de données sans lui indiquer ceci en analysant l'écart entre les lignes du jeu de données d’entrée.[(Documentation)](https://facebook.github.io/prophet/docs/non-daily_data.html)

### Un support des événements spéciaux, jours fériés et vacances

Un point très fort de Prophet est la possibilité de personnaliser une saisonnalité dépendamment de plusieurs événements spéciaux, fêtes et vacances. Il est possible d’utiliser ses fonctions pour créer de façon automatique une liste d'évènements par **pays**, **région**, **département** et **année** et les passer au modèle pour apprentissage.
Si tout de même vous n'êtes pas satisfaits par le nombre de pays assez restreint, il est possible d’utiliser la bibliothèque **holidays** ou même en important son propre jeu de données de ces événements. Prophet s’adaptera automatiquement pour en tenir compte.

> Sous Python, ces événements sont disponibles pour n’importe quelle date de n’importe quelle année.

> Sous R, ces événements sont disponibles entre 1995 et jusqu'en 2044 et pouvant être étendues en lançant un script qui vient avec Prophet pour en générer d’avantage.

### Une gestion des valeurs aberrantes

Prophet peut gérer les individus aberrants de façon automatique en ne tenant pas compte de leurs présence lors du calcul des prévisions. Or, des fois il trouvera des difficultés pour le faire et si un grand nombre de valeurs aberrantes est présent à un certain moment, ceci peut biaiser les prédictions futures. Pour ceci, Facebook Prophet demande à les supprimer de la série du moment que Prophet supporte les valeurs manquantes qui seront remplacés par les prévisions de son modèle.[(Documentation)](https://facebook.github.io/prophet/docs/outliers.html)

### Une gestion des changements de tendance

Prophet propose une gestion des changements sur la courbe, en positionnant sur les 80% de la courbe et de façon uniforme des points de changement (25 point pour être précis) et réduira ce nombre de points au fur et à mesure qu’il paramètre son modèle pour ne garder que le minimum. Il est même possible d’indiquer de façon manuelle les points de changement sur la courbe.
La flexibilité de la courbe de prédiction est automatiquement initialisée, mais on peut également la modifier en spécifiant le paramètre de flexibilité à la création du modèle.[(Documentation)](https://facebook.github.io/prophet/docs/trend_changepoints.html#automatic-changepoint-detection-in-prophet)

### Et bien plus...

Outre les forces déjà citées, [l’article scientifique](https://peerj.com/preprints/3190/) de Facebook Data Science team et la [documentation officielle](https://facebook.github.io/prophet/docs/quick_start.html) peuvent vous convaincre d’avantage d’utiliser Prophet et commencer à faire vos prévisions temporelles.

## POC

L’exemple sur lequel on va travailler est un exemple dans le domaine du retail, qui est l’exemple Retail Sales. Ce jeu de données concerne le nombre mensuel des ventes réalisés entre Janvier 1992 et Mai 2016 ce qui équivaut à 293 lignes de données représentées sur 2 colonnes:

* Une colonne de type **datetime** dont le label est **ds**
* Une colonne de type **entier** dont le label est **y**

> Si les colonnes du jeu de données ne sont pas ['ds', 'y'], il est nécessaire d’enlever les colonnes non pertinentes et de renommer les entêtes de colonnes.

On commence par installer et importer les bibliothèques nécessaires.

```python
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as dates
from datetime import datetime

# Facebook Prophet
!pip install fbprophet
from fbprophet import Prophet
from fbprophet.diagnostics import cross_validation
from fbprophet.diagnostics import performance_metrics
from fbprophet.plot import plot_cross_validation_metric
from fbprophet import hdays

!pip install holidays
import holidays
```

On importe par la suite le jeu de données, on reformate le type des colonnes et on mets l’index sur la colonnes **ds**.

```python
RetailSalesDataframe = pd.read_csv("./RetailSales.csv", delimiter=",")
RetailSalesDataframe["ds"] = pd.to_datetime(RetailSalesDataframe["ds"], infer_datetime_format=True)
RetailSalesDataframe["y"] = pd.to_numeric(RetailSalesDataframe["y"])
indexedDataframe = RetailSalesDataframe.set_index(["ds"])
```

Pour visualiser et comprendre les données, on calcule la moyenne mobile et l'écart type et on les affiche.

```python
Roolmean = indexedDataframe.rolling(window = 12).mean()
Roolstd = indexedDataframe.rolling(window = 12).std()
orig = plt.plot(indexedDataframe, color = "blue", label = "Original")
mean = plt.plot(Roolmean, color = "red", label = "Moyenne mobile")
std = plt.plot(Roolstd, color = "black", label = "Ecart Type")
plt.legend(loc = "best")
plt.title("Tendance des ventes par date avec la moyenne mobile et la 
variance")
plt.rcParams["figure.figsize"] = [15,9]
plt.show(block = False)
```

On remarque que la courbe présente une tendance à la hausse durant toute la période étudiée (sauf en 2008 période de la crise financière) mais  également un effet saisonnier avec une hausse pour certaines périodes de l’année et une forte baisse pour d’autres. Pour mieux voir ceci, on affiche l’allure de la courbe pendant les années de 1992 jusqu'à 1995.

```python
ye1 = plt.plot(indexedDataframe[0:11], color = "blue", label = "1992")
ye2 = plt.plot(indexedDataframe[12:23], color = "orange", label = "1993")
ye3 = plt.plot(indexedDataframe[24:35], color = "green", label = "1994")
ye4 = plt.plot(indexedDataframe[36:47], color = "red", label = "1995")
plt.legend(loc = "best")
plt.title("Tendance des ventes par date sur 4 années")
plt.show(block = False)
```

## image


Courbe dont l’allure est non stationnaire et suit une tendance saisonnière avec peu de ventes sur les 1 ers mois de l'année. Les ventes augmentent de façon significative pendant l'été et retombent après le mois d’Août et reprennent leurs hausses pendant la période de noël.
Pour comprendre l'évolution des ventes en 2008, on analysera les quelques années qui ont suivi 2008.


```python
crisis08 = plt.plot(indexedDataframe.loc['2008-01-01':'2008-12-30'], color = "blue", label = "2008")
crisis09 = plt.plot(indexedDataframe.loc['2009-01-01':'2009-12-30'], color = "red", label = "2009")
crisis09 = plt.plot(indexedDataframe.loc['2010-01-01':'2010-12-30'], color = "green", label = "2010")
plt.legend(loc = "best")
plt.title("Tendance des ventes par date sur la période après crise de 2008")
plt.show(block = False)
```

## image

Le graphe précédent relate l'impact de la crise financière de 2008 sur le nombre de ventes. En effet, les ventes ont baissé durant l'année 2009 pour reprendre la tendance positive progressive à partir de 2010.
Passons aux prédictions, on commence par créer un objet Prophet en lui passant en paramètre le type de saisonnalité car on est face à une saisonnalité multiplicative.

```python
prophetModel = Prophet(seasonality_mode="multiplicative")
```

On entraîne le modèle sur notre jeu de données.

```python
prophetModel = prophetModel.fit(RetailSalesDataframe)
```

Une fois que c’est fait, on lance les prévisions sur la fréquence ainsi que le nombre de période voulus, pour mon cas, j’ai choisi une prévision sur 60 
mois (5ans) et donc jusqu'en Avril 2021:

```python
future = prophetModel.make_future_dataframe(periods=60, freq='M')
```

L’argument _freq_ peut être n’importe quel argument de type pd.date_range, comme **D** pour **Daily**  ou **M** pour **Monthly**.

Pour voir les ventes prévus dans 5ans (à partir de 2016):

```python
forecast = prophetModel.predict(future)
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
```

## Image

Le yhat ou y prevu est le résultat prévu avec un degré de certitude de 95% que notre résultat sera forcement entre le yhat_lower et le yhat_upper. 
Enfin on affiche le graphe qui résume ces résultats:

```python
fig1 = prophetModel.plot(forecast)
```

## Image

```python
fig2 = prophetModel.plot_components(forecast)
```

## Image

## L'évaluation du modèle

Pour diagnostiquer nos résultats, il faudra bien comparer sur les données en entrées à celles que Prophet à prévu. Pour ceci, Prophet propose d’effectuer une validation croisée sur notre modèle sur un intervalle de temps. La période d’entrainement est 3 fois l’horizon et les coupures ont lieux chaque mi-horizon mais c’est possible de personnaliser ces paramètres lors de l’appel. Pour mon cas, j’ai donné 21 ans comme période d’entrainement initial, une période d’un mois et 6 mois d’horizon.

```python
df_cv = cross_validation(prophetModel, initial = '7665 days', period = 
'30 days', horizon = '180 days')
df_p = performance_metrics(df_cv)
df_p
```

## Image

L’erreur absolue moyenne en pourcentage (Mean Absolute Percentage Error, alias MAPE) est la moyenne des écarts en valeur absolue par rapport aux valeurs observées. C’est donc un pourcentage et par conséquent un indicateur pratique de comparaison.

```python
fig = plot_cross_validation_metric(df_cv, metric='mape')
```

## Image

Pour notre cas, le MAPE ne dépasse pas les 2.5% et donc on peut juger que nos prédictions sont excellentes pour avoir une vision des ventes sur 
la période des 5 années à venir.

> Pour récompenser vos efforts pour arriver à la fin, un bonus relatif à expérimentation des jours fériés vous attends au niveau du code sur le repository Github et le notebook Google Collaboratory.

## Conclusion

L’analyse des séries temporelles a depuis toujours été une branche très étudiée en analyse de données et en machine learning et la création de telles librairies par l’une des plus grandes entreprises du monde en est la preuve. Ces librairies viennent cacher la complexité d’une connaissance poussée des statistiques en simplifiant la réalisation des prédictions, leur visualisation et leur exportation de la plus simple des façons.

![EverythingIsWellPredicted](https://media3.giphy.com/media/l3fZW5FcwQqxkz5uw/giphy.gif?cid=790b76118c65251890096b5b5b5d771a98ab098a391c1a3a&rid=giphy.gif&ct=g)

## Liens utiles

* [Lien du notebook](https://colab.research.google.com/drive/1cERRgwnulScgj3bzHeqoet4SYN6f1Sza?usp=sharing)
* [Documentation officielle](https://facebook.github.io/prophet/)
* [Bibliothèque holidays](https://github.com/dr-prodigy/python-holidays)
* [Article de recherche: Forecasting at scale](https://peerj.com/preprints/3190.pdf)