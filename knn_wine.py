# -*- coding: utf-8 -*-
"""
Created on Fri Nov 20 01:33:53 2026

@author: Mohamed Kontao
"""

# ==============================================================
# PROJET 1 : CLASSIFICATION DES VINS AVEC KNN
# Machine Learning supervisé - Scikit-learn
# ==============================================================

# %% 1. IMPORTATION DES BIBLIOTHÈQUES

import pandas as pd                                         # Manipulation des données
import matplotlib.pyplot as plt                             # Création des graphiques
import seaborn as sns                                        # Visualisation statistique
import joblib                                               # Sauvegarde des modèles

from sklearn.datasets import load_wine                      # Jeu de données Wine
from sklearn.model_selection import train_test_split        # Séparation des données
from sklearn.model_selection import StratifiedKFold          # Création des plis stratifiés
from sklearn.model_selection import cross_val_score          # Validation croisée
from sklearn.preprocessing import StandardScaler            # Standardisation
from sklearn.neighbors import KNeighborsClassifier          # Algorithme KNN
from sklearn.pipeline import make_pipeline                  # Pipeline Machine Learning
from sklearn.metrics import accuracy_score                  # Calcul de l'accuracy
from sklearn.metrics import classification_report           # Précision, rappel et F1
from sklearn.metrics import confusion_matrix                # Matrice de confusion


# %% 2. CHARGEMENT DES DONNÉES

wine = load_wine()                                          # Charge les 178 vins

X = pd.DataFrame(wine.data, columns=wine.feature_names)     # Crée les caractéristiques
y = pd.Series(wine.target, name="classe")                   # Crée les classes réelles

print("Dimensions :", X.shape)                              # Affiche (178, 13)
print(X.head())                                             # Affiche les premiers vins


# %% 3. EXPLORATION DES DONNÉES

print(X.describe())                                         # Statistiques descriptives
print(X.isnull().sum())                                     # Recherche les valeurs manquantes
print(y.value_counts())                                     # Compte les vins par classe


# %% 4. SÉPARATION ENTRAÎNEMENT / TEST

X_train, X_test, y_train, y_test = train_test_split(
    X,                                                      # Caractéristiques
    y,                                                      # Classes réelles
    test_size=0.20,                                         # Réserve 20 % pour le test
    random_state=42,                                        # Fixe l'aléatoire
    stratify=y                                              # Préserve les proportions
)

print("Entraînement :", X_train.shape)                       # 142 vins, 13 variables
print("Test :", X_test.shape)                               # 36 vins, 13 variables


# %% 5. VALIDATION CROISÉE ET CHOIX DE K

valeurs_k = [1, 3, 5, 7, 9]                                 # Hyperparamètres à comparer

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)  # Cinq plis stratifiés

resultats_cv = []                                           # Liste des performances

for k in valeurs_k:                                        # Essaie chaque valeur de K

    modele = make_pipeline(                                # Crée un Pipeline
        StandardScaler(),                                  # Standardise dans chaque pli
        KNeighborsClassifier(n_neighbors=k)                # Utilise K voisins
    )

    scores = cross_val_score(                              # Effectue la validation croisée
        modele, X_train, y_train,                          # Utilise uniquement l'entraînement
        cv=cv, scoring="accuracy"                           # Mesure les accuracies
    )

    resultats_cv.append({                                  # Enregistre les performances
        "K": k,                                             # Nombre de voisins
        "Accuracy moyenne": scores.mean(),                  # Moyenne des cinq scores
        "Ecart-type": scores.std()                           # Dispersion des scores
    })

df_cv = pd.DataFrame(resultats_cv)                         # Crée le tableau des résultats

print(df_cv.round(4))                                      # Affiche les performances


# %% 6. SÉLECTION AUTOMATIQUE DU MEILLEUR K

index_meilleur = df_cv["Accuracy moyenne"].idxmax()        # Trouve le meilleur index

meilleur_k = int(df_cv.loc[index_meilleur, "K"])            # Récupère la valeur de K

print("Meilleur K :", meilleur_k)                          # Affiche K sélectionné


# %% 7. ENTRAÎNEMENT DU MODÈLE FINAL

modele_final = make_pipeline(                              # Crée le Pipeline définitif
    StandardScaler(),                                      # Standardisation automatique
    KNeighborsClassifier(n_neighbors=meilleur_k)           # Utilise le meilleur K
)

modele_final.fit(X_train, y_train)                          # Entraîne sur les 142 vins


# %% 8. PRÉDICTIONS FINALES

y_pred_final = modele_final.predict(X_test)                 # Prédit les 36 vins

print("Classes réelles :", y_test.to_numpy())               # Affiche les vraies classes
print("Prédictions :", y_pred_final)                         # Affiche les prédictions


# %% 9. ÉVALUATION DU MODÈLE

accuracy_finale = accuracy_score(y_test, y_pred_final)      # Calcule l'accuracy

print(f"Accuracy finale : {accuracy_finale:.2%}")           # Affiche le pourcentage

print(classification_report(y_test, y_pred_final))          # Rapport complet

cm = confusion_matrix(y_test, y_pred_final)                 # Calcule la matrice


# %% 10. VISUALISATION DE LA MATRICE DE CONFUSION

plt.figure(figsize=(7, 5))                                 # Crée une figure

sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")           # Affiche la matrice

plt.title("Matrice de confusion - KNN Wine")                 # Titre du graphique
plt.xlabel("Classes prédites")                              # Axe horizontal
plt.ylabel("Classes réelles")                               # Axe vertical

plt.show()                                                  # Affiche la figure


# %% 11. SAUVEGARDE DU MODÈLE

joblib.dump(modele_final, "modele_knn_wine.joblib")          # Sauvegarde le Pipeline


# %% 12. RECHARGEMENT ET PRÉDICTION

modele_charge = joblib.load("modele_knn_wine.joblib")        # Recharge notre propre fichier

nouveau_vin = X_test.iloc[[0]]                              # Sélectionne un vin de test

prediction = modele_charge.predict(nouveau_vin)             # Prédit sa classe

print("Classe prédite :", prediction[0])                    # Affiche la prédiction
print("Classe réelle :", y_test.iloc[0])                    # Affiche la réponse connue


# %% VÉRIFICATION FINALE DES PERFORMANCES

from sklearn.metrics import accuracy_score  # Importe la métrique

print("Nombre de vins testés :", len(y_test))  # Nombre d'observations du test

print("Bonnes prédictions :", (y_test.to_numpy() == y_pred_final).sum())  # Compte les bonnes réponses

print("Accuracy finale :", accuracy_score(y_test, y_pred_final) * 100)  # Calcule le pourcentage


from sklearn.metrics import confusion_matrix  # Importe la fonction

cm_finale = confusion_matrix(y_test, y_pred_final)  # Calcule la matrice finale

print(cm_finale)  # Affiche les résultats

# ==============================================================
# RÉSUMÉ - PROJET MACHINE LEARNING KNN WINE
# ==============================================================
#
# 1. APPRENTISSAGE SUPERVISÉ
# X : caractéristiques utilisées pour prédire
# y : classes correctes que le modèle doit apprendre
#
# 2. TRAIN_TEST_SPLIT
# Sépare les données en entraînement et test
# test_size=0.20 : réserve 20 % pour le test
# random_state=42 : rend la séparation reproductible
# stratify=y : conserve les proportions des classes
#
# 3. STANDARDISATION
# StandardScaler : moyenne proche de 0, écart-type proche de 1
# fit_transform() : apprend les paramètres et transforme
# transform() : transforme avec les paramètres déjà appris
# Important : ne pas apprendre la standardisation sur le test
#
# 4. KNN
# KNeighborsClassifier : algorithme de classification
# n_neighbors : nombre de voisins utilisés
# fit(X, y) : entraîne le modèle
# predict(X) : prédit les catégories
#
# 5. VALIDATION CROISÉE
# cross_val_score() : évalue sur plusieurs plis
# StratifiedKFold : conserve les proportions des classes
# idxmax() : retrouve l'index de la meilleure accuracy
# loc[] : récupère la valeur de K correspondante
#
# 6. ÉVALUATION
# Accuracy : proportion de prédictions correctes
# Précision = VP / (VP + FP)
# Rappel = VP / (VP + FN)
# F1 = 2 * (Précision * Rappel) / (Précision + Rappel)
#
# 7. MATRICE DE CONFUSION
# Lignes : classes réelles
# Colonnes : classes prédites
# Diagonale : prédictions correctes
# FP : colonne étudiée, hors diagonale
# FN : ligne étudiée, hors diagonale
#
# 8. PIPELINE
# Combine StandardScaler et KNN
# Évite de standardiser deux fois les données
# Permet de conserver toute la préparation avec le modèle
#
# 9. SAUVEGARDE
# joblib.dump() : enregistre le modèle
# joblib.load() : recharge un modèle provenant d'une source fiable
#
# CONCLUSION
# Nous avons construit un classificateur KNN sur 178 vins.
# Nous avons sélectionné K par validation croisée.
# Nous avons réservé 36 vins pour le test final.
#
# À RETENIR :
# Séparer -> Préparer -> Valider -> Entraîner -> Tester -> Sauvegarder
# ==============================================================
