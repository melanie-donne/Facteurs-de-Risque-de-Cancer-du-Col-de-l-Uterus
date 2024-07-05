import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, roc_auc_score, recall_score, confusion_matrix

# Charger le dataset
file_path = '.\kag_risk_factors_cervical_cancer.csv'
df = pd.read_csv(file_path)

# Afficher les premières lignes du dataset pour inspection
print(df.head(10))

# Afficher la forme et les informations du dataset
print(df.shape)
print(df.info())

# Vérifier le nombre de valeurs manquantes par colonne
print(df.isnull().sum())

# Convertir toutes les colonnes en numériques, les erreurs sont converties en NaN
df = df.apply(pd.to_numeric, errors='coerce')

# Remplir les valeurs manquantes avec des stratégies spécifiques
df['Number of sexual partners'] = df['Number of sexual partners'].fillna(df['Number of sexual partners'].median())
df['First sexual intercourse'] = df['First sexual intercourse'].fillna(df['First sexual intercourse'].median())
df['Num of pregnancies'] = df['Num of pregnancies'].fillna(df['Num of pregnancies'].median())
df['Smokes'] = df['Smokes'].fillna(1)
df['Smokes (years)'] = df['Smokes (years)'].fillna(df['Smokes (years)'].median())
df['Smokes (packs/year)'] = df['Smokes (packs/year)'].fillna(df['Smokes (packs/year)'].median())
df['Hormonal Contraceptives'] = df['Hormonal Contraceptives'].fillna(1)
df['Hormonal Contraceptives (years)'] = df['Hormonal Contraceptives (years)'].fillna(df['Hormonal Contraceptives (years)'].median())
df['IUD'] = df['IUD'].fillna(0)  # Sous suggestion
df['IUD (years)'] = df['IUD (years)'].fillna(0)  # Sous suggestion
df['STDs'] = df['STDs'].fillna(1)
df['STDs (number)'] = df['STDs (number)'].fillna(df['STDs (number)'].median())
# Remplir les autres colonnes STDs de la même manière
stds_columns = ['STDs:condylomatosis', 'STDs:cervical condylomatosis', 'STDs:vaginal condylomatosis',
                'STDs:vulvo-perineal condylomatosis', 'STDs:syphilis', 'STDs:pelvic inflammatory disease',
                'STDs:genital herpes', 'STDs:molluscum contagiosum', 'STDs:AIDS', 'STDs:HIV',
                'STDs:Hepatitis B', 'STDs:HPV', 'STDs: Time since first diagnosis', 'STDs: Time since last diagnosis']
for col in stds_columns:
    df[col] = df[col].fillna(df[col].median())

# Vérifier les valeurs manquantes après remplissage
print(df.isnull().sum())

# Créer des variables dummy pour les colonnes catégorielles
categorical_columns = ['Smokes','Hormonal Contraceptives','IUD','STDs', 'Dx:Cancer','Dx:CIN','Dx:HPV','Dx','Hinselmann','Citology','Schiller']
df = pd.get_dummies(data=df, columns=categorical_columns)

# Afficher les premières lignes du dataset après transformation
print(df.head(13))

# Visualisation des données pour certaines caractéristiques
fig, axes = plt.subplots(7, 1, figsize=(20, 40))
sns.countplot(x='Age', data=df, ax=axes[0])
sns.countplot(x='Number of sexual partners', data=df, ax=axes[1])
sns.countplot(x='Num of pregnancies', data=df, ax=axes[2])
sns.countplot(x='Smokes (years)', data=df, ax=axes[3])
sns.countplot(x='Hormonal Contraceptives (years)', data=df, ax=axes[4])
sns.countplot(x='IUD (years)', data=df, ax=axes[5])
sns.countplot(x='STDs (number)', data=df, ax=axes[6])
plt.show()

# Mélanger les données de manière aléatoire
df_data_shuffle = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Séparer les données en ensembles d'entraînement et de test
train_size = int(0.8 * len(df_data_shuffle))
df_train = df_data_shuffle[:train_size]
df_test = df_data_shuffle[train_size:]

# Sélectionner les caractéristiques et les étiquettes
features = df.columns.difference(['Biopsy'])
df_train_feature = df_train[features]
train_label = df_train['Biopsy']
df_test_feature = df_test[features]
test_label = df_test['Biopsy']

# Normaliser les caractéristiques
scaler = MinMaxScaler()
train_feature = scaler.fit_transform(df_train_feature)
test_feature = scaler.transform(df_test_feature)

# Entraîner un modèle de Random Forest
random_forest = RandomForestClassifier(n_estimators=100, random_state=42)
random_forest.fit(train_feature, train_label)

# Prédire sur les données de test
predictions = random_forest.predict(test_feature)

# Évaluer le modèle
f1 = f1_score(test_label, predictions)
auc_roc = roc_auc_score(test_label, predictions)
recall = recall_score(test_label, predictions)
conf_matrix = confusion_matrix(test_label, predictions)

print(f"F1 Score: {f1}")
print(f"ROC AUC Score: {auc_roc}")
print(f"Recall Score: {recall}")

# Visualiser la matrice de confusion
class_labels = ['Negatif', 'Positif']
sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='g', xticklabels=class_labels, yticklabels=class_labels)
plt.xlabel('Étiquette Prédite')
plt.ylabel('Étiquette Réelle')
plt.title('Matrice de Confusion')
plt.show()

# Visualiser la matrice de corrélation
corrmat = df.corr()
plt.figure(figsize=(12, 9))
sns.heatmap(corrmat, vmax=1, square=True, cmap='rainbow')
plt.show()

# Réentraîner le modèle avec des caractéristiques sélectionnées
selected_features = ['Hinselmann_0', 'Hinselmann_1', 'Citology_0', 'Citology_1', 'Schiller_0', 'Schiller_1']
df_train_feature_selected = df_train[selected_features]
df_test_feature_selected = df_test[selected_features]

train_feature_selected = scaler.fit_transform(df_train_feature_selected)
test_feature_selected = scaler.transform(df_test_feature_selected)

random_forest.fit(train_feature_selected, train_label)
predictions_selected = random_forest.predict(test_feature_selected)

f1_selected = f1_score(test_label, predictions_selected)
auc_roc_selected = roc_auc_score(test_label, predictions_selected)
recall_selected = recall_score(test_label, predictions_selected)
conf_matrix_selected = confusion_matrix(test_label, predictions_selected)

print(f"F1 Score avec caractéristiques sélectionnées: {f1_selected}")
print(f"ROC AUC Score avec caractéristiques sélectionnées: {auc_roc_selected}")
print(f"Recall Score avec caractéristiques sélectionnées: {recall_selected}")

# Visualiser la matrice de confusion pour les caractéristiques sélectionnées
sns.heatmap(conf_matrix_selected, annot=True, cmap='Blues', fmt='g', xticklabels=class_labels, yticklabels=class_labels)
plt.xlabel('Étiquette Prédite')
plt.ylabel('Étiquette Réelle')
plt.title('Matrice de Confusion (Caractéristiques Sélectionnées)')
plt.show()
