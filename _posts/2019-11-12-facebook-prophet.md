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

* Une intégration sous R et Python

Prophet propose sa bibliothèque sous **CRAN** pour les utilisateurs de R ainsi que sur **PyPi** pour les adeptes de Python.

* Une saisonnalité additive et multiplicative

La saisonnalité est une caractéristique commune des séries chronologiques. Il peut apparaître sous deux formes:

- Additive
- Multiplicative

Dans le premier cas, l'amplitude de la variation saisonnière est indépendante du niveau, tandis que dans le second, elle est liée. Comme le montre 
la figure suivante:

![additive_vs_multiplicative_seasonality](assets/img/facebook_prophet/additive_multiplicative_seasonality.png "Saisonnalité additive vs multiplicative")

Par défaut, Prophet adapte son modèle à travers un modèle additif, mais supporte également le modèle multiplicatif.

```python
# Modèle additif
additiveModel = Prophet()
additiveModel = Prophet(seasonality_mode='additive')

# Modèle multiplicatif
MultiplicativeModel = Prophet(seasonality_mode='multiplicative')
```