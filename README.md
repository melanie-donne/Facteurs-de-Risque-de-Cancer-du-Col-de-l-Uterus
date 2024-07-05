# Facteurs de Risque de Cancer du Col de l'Utérus

Ce script montre comment préparer des données, entraîner un modèle de machine learning, évaluer ses performances et visualiser les résultats. Il suit une approche systématique pour la classification des risques de cancer du col de l'utérus en utilisant un RandomForest.

## Dataset sur les Facteurs de Risque de Cancer du Col de l'Utérus

Ce dataset, obtenu depuis le référentiel UCI, contient une liste de facteurs de risque associés au cancer du col de l'utérus, menant à des examens de biopsie.

## Points Clés :

- **Incidence et Mortalité** : Aux États-Unis, environ 11 000 nouveaux cas de cancer invasif du col de l'utérus sont diagnostiqués chaque année, avec environ 4 000 décès. Cependant, les taux de mortalité ont diminué grâce au dépistage accru.
  
- **Facteurs de Risque** :
  - **Âge** : Le risque augmente significativement entre 35 et 54 ans.
  - **Activité Sexuelle** : Les partenaires multiples et des rapports sexuels précoces augmentent le risque.
  - **Antécédents Familiaux** : Un parent ayant eu un cancer du col de l'utérus accroît le risque.
  - **Contraceptifs Oraux** : Une utilisation prolongée est associée à un risque accru.
  - **Tabagisme** : Augmente le risque de développement de dysplasies et de cancer invasif.
  - **Immunsuppression** : Les systèmes immunitaires affaiblis, comme chez les personnes atteintes du VIH/SIDA, sont plus vulnérables.
  - **Diethylstilbestrol (DES)** : Les filles exposées in utero ont un risque plus élevé.
  
## Importance Clinique :
- Le dépistage régulier et la prévention sont cruciaux pour réduire le risque.
- Des disparités socio-économiques et ethniques influencent les taux de cancer du col de l'utérus.


# Explication du code
Ce script Python utilise plusieurs bibliothèques courantes pour la manipulation de données, la visualisation et la modélisation de machine learning. Voici une explication détaillée du code :

## Import des bibliothèques
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, roc_auc_score, recall_score, confusion_matrix
```

Les bibliothèques sont importées pour le traitement de données (NumPy, Pandas), la visualisation (Matplotlib, Seaborn) et les outils de machine learning (scikit-learn).

## Chargement des données
```python
file_path = './facteurs_de_risque_de_cancer_du_col_de_l_uterus.csv'
df = pd.read_csv(file_path)
```

Le dataset est chargé à partir du fichier CSV spécifié dans file_path et stocké dans un DataFrame df.

## Renommage des colonnes
```python
df.columns = ['Âge', 'Nombre de partenaires sexuels', 'Première relation sexuelle', ...]
```
Les colonnes du DataFrame sont renommées en français pour plus de clarté.

## Inspection des données
```python
print(df.head(10))
print(df.shape)
print(df.info())
``` 

Affiche les premières lignes du dataset, sa forme (nombre de lignes et de colonnes) et des informations sur les types de données et les valeurs manquantes éventuelles.

## Traitement des valeurs manquantes
```python
df = df.apply(pd.to_numeric, errors='coerce')
# Remplissage des valeurs manquantes avec des stratégies spécifiques
df['Nombre de partenaires sexuels'] = df['Nombre de partenaires sexuels'].fillna(df['Nombre de partenaires sexuels'].median())
# (d'autres colonnes sont également traitées de la même manière)
```

Les valeurs manquantes sont converties en NaN (si non numériques) puis remplacées par la médiane ou d'autres stratégies spécifiques pour chaque colonne.

## Encodage des variables catégorielles
```python
categorical_columns = ['Fume', 'Contraceptifs hormonaux', 'DIU', 'MST', 'Dx: Cancer', ...]
df = pd.get_dummies(data=df, columns=categorical_columns)
```

Les variables catégorielles sont transformées en variables binaires à l'aide de variables fictives (dummy variables).

## Visualisation des données
```python
sns.countplot(x='Âge', data=df, ax=axes[0])
# (d'autres visualisations similaires pour différentes caractéristiques)
plt.show()
```

Des graphiques sont créés pour visualiser la distribution des données par caractéristique.

## Préparation des données pour l'apprentissage automatique
```python
scaler = MinMaxScaler()
train_feature = scaler.fit_transform(df_train_feature)
test_feature = scaler.transform(df_test_feature)
``` 

Les caractéristiques sont normalisées en utilisant MinMaxScaler pour mettre à l'échelle les valeurs entre 0 et 1.

## Entraînement et évaluation du modèle
```python
random_forest = RandomForestClassifier(n_estimators=100, random_state=42)
random_forest.fit(train_feature, train_label)
predictions = random_forest.predict(test_feature)
```

Un modèle de classification RandomForest est entraîné et évalué en utilisant différentes métriques comme le F1-score, l'aire sous la courbe ROC (ROC AUC), et la matrice de confusion.

## Sélection de caractéristiques et ré-entraînement du modèle

```python
selected_features = ['Hinselmann_0', 'Hinselmann_1', 'Cytologie_0', ...]
df_train_feature_selected = df_train[selected_features]
# (normalisation et ré-entraînement du modèle avec les caractéristiques sélectionnées)
```

Le modèle est ré-entraîné en utilisant uniquement les caractéristiques sélectionnées, avec une évaluation similaire des performances.

## Visualisation des résultats

```python
sns.heatmap(conf_matrix_selected, annot=True, cmap='Blues', fmt='g', xticklabels=class_labels, yticklabels=class_labels)
plt.xlabel('Étiquette Prédite')
plt.ylabel('Étiquette Réelle')
plt.title('Matrice de Confusion (Caractéristiques Sélectionnées)')
plt.show()
```
La matrice de confusion est visualisée pour évaluer les performances du modèle sur les données de test, en particulier avec les caractéristiques sélectionnées.

