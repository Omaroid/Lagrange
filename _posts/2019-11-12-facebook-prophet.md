---
layout: post
title: "Facebook Prophet"
author: "Omar LARAQUI"
categories: journal
tags: [ai,machineLearning,timeSeries]
image: facebookProphet.jpg
---

Une série temporelle, ou série chronologique, est une suite de valeurs numériques représentant l'évolution d'une quantité spécifique au
cours du temps. De telles suites de variables aléatoires peuvent être exprimées mathématiquement afin d'en analyser le comportement,
généralement pour comprendre son évolution passée et pour en prévoir le comportement futur. Une telle transposition mathématique
utilise le plus souvent des concepts de probabilités et de statistique.

## Présentation

Il y a 3 ans, l'équipe de Facebook Core Data Science team à sorti en open source un puissant outil de prévision pour les séries temporelles appelé
Prophet.

Prophet se base sur un modèle additif qui s’adapte par rapport à la saisonnalité des données mais pas que, il est assez robuste pour des jeux de
données qui représentent des données manquantes ou comportant des valeurs aberrantes (ou outliers).

Pour les non statisticiens ou data scientistes, il est assez délicat de partir sur une connaissance des principes d’analyse et de prévision temporelles
mais Prophet est venu répondre à ce besoin de la plus simple des façon.

L’objectif de l’article est de faire une introduction de Prophet sous Python :snake:. Les données et le code seront disponibles sur le Github de
Ingeniance ainsi que sur Google Colaboratory en fin du blog.