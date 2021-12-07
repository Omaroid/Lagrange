---
layout: post
title: "Comment Spark peut être lancé à travers Python ? 💫"
author: "Omar LARAQUI"
categories: journal
tags: [bigdata,spark,pyspark,py4j]
image: pyspark.jpg
---

## Qu\'est ce que PySpark

PySpark est une API Python pour Spark publiée par la communauté Apache Spark pour prendre en charge Python avec Spark. En utilisant PySpark, on peut facilement intégrer et travailler avec des RDD dans le langage de programmation Python. Il existe de nombreuses fonctionnalités qui font de PySpark un cadre aussi incroyable lorsqu\'il s\'agit de travailler avec d\'énormes ensembles de données. Que ce soit pour effectuer des calculs sur de grands ensembles de données ou simplement pour les analyser.

## Pourquoi Python?

Les data engineers se tournent de plus en plus vers PySpark pour la simplicité du langage Python mais ce n\'est pas la seule raison. En effet, la majorité des data scientists utilisent Python comme premier langage de programmation. Ce passage à l\'échelle sur les données qu\'ils utilisent, préparent, formatent et sauvegardent demande l\'utilisation du calcul distribué proposé notamment par PySpark. Sans oublier que Spark comporte une brique pour entrainer des modèles Machine Learning appelée MLlib.

## Comment Python peut-il lancer et gérer des jobs Spark?

En étant une librairie tournant sur une JVM (Java Virtual Machine), on peut facilement se demander comment est ce possible de le lancer des jobs Spark via Python. Ce dernier étant un langage interprété.

En consultant le code source de PySpark, on retrouve le fichier **java_gateway.py** et c\'est ce fichier la qui nous montera comment Python peut tourner Spark, ou plutôt interagir avec la JVM du noeud maitre de Spark.

Concrètement, toute la partie traitement des données est gérée par Python et toute la partie persistance et échange de données entre les éxécuteurs (worker nodes) et le maitre (master node) sont gérés par la JVM des process Spark.

En initialisant la session Spark, on crée un contexte Spark. Ce contexte est utilisé pour créer une socket TCP avec un process Py4J.

> Py4J est une bibliothèque Java intégrée à PySpark et permettant à Python de s\'interfacer dynamiquement avec des objets JVM.

Le process Py4J étant responsable de communiquer les différentes transformations et actions appliquées à des RDDs (ou à un plus haut niveau des DataFrames ou DataSets) en les sérialisant. La sérialisation passe par le biais du fichier **cloudpickle.py** utilisant la bibliothèque pickle de Python. Il les envoi alors vers le process Py4J qui les désérialise puis les envoie à son tour au noeud maitre du job Spark en tant qu\'objets PythonRDD, le plan d\'éxécution Spark est donc modifié pour en tenir compte. Il est exécuté sur les worker nodes en cas d\'opérations de transformation.

![how_python_runs_spark](assets/img/how-python-spark/pyspark_architecture_overview.png "Comment Python peut-il lancer Spark?")

## Et en cas d\'une opération de type action?

Une opération de type action sur un PythonRDD déclenche en plus le chemin inverse c\'est à dire l\'envoi des résultats sous forme de PythonRDD au noeud maitre qui les envoie à son tour au process Py4J. Ce dernier sérialise l\'objet PythonRDD au sein de sa JVM et l\'envoie au driver PySpark via la même socket TCP créée précédemment. A son arrivée, il est désérialisé et redevient un PythonRDD.

## Aurons nous les mêmes performances que de lancer Spark sur Scala ou Java?

Du moment que toute la partie persistance des données est gérée sur la JVM des noeuds maitre et executeurs, on sera sur des performances égales à des runs sur Scala ou Java. La différence sur Python est principalement due au travail effectué par le process de sérialisation entre le driver PySpark et le process Py4J. Tout dépendra donc de la taille des données que l\'on est entrain de sérialiser ou désérialiser. Néanmoins, avec la puissance de calcul en augmentation continue, on peut juger que cette différence de l\'ordre du millisecondes près peut être parfois négligeable devant la possibilité d\'utiliser un langage aussi simple et intuitif que Python pour lancer des jobs Spark. 🙂

Pour pousser vos recherches sur les sujets abordés lors de ce blog, je vous invite à consulter:

- [La page officielle de Py4J](https://www.py4j.org/) pour comprendre comment un interpréteur Python échange avec une JVM
- [La bibliothèque pickle](https://docs.python.org/3/library/pickle.html) de Python pour la serialisation et deserialisation d\'objets
- [Le code source officiel de PySpark](https://github.com/apache/spark/tree/0cf59fcbe3799dd3c4469cbf8cd842d668a76f34/python/pyspark) où le code est remarquablement simple à comprendre nottament les fichiers [cloudpickle.py](https://github.com/apache/spark/blob/0cf59fcbe3799dd3c4469cbf8cd842d668a76f34/python/pyspark/cloudpickle.py) et [java_gateway.py](https://github.com/apache/spark/blob/0cf59fcbe3799dd3c4469cbf8cd842d668a76f34/python/pyspark/java_gateway.py) pour les explication de ce blog
- S\'abonner à la newsletter et à la page Instagram du blog pour etre informé de toutes nos nouvelles publications 😁