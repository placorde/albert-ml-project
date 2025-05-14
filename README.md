# Bookmatch project 

Bookmatch est une interface ayant pour but que chaque personne trouve son prochain livre. 

En effet, on constate sur ces dernières années que le nombre de lecteur en France est en baisse. Une des difficultés majeures pour commencer un livre, c'est d'y trouver son compte parmi la multitude de choix. 
C'est pourquoi Bookmatch permet à son utilisateur, en fonction de critères personnalisés de trouver LE livre qui lui correspond. 


## Google Drive 
https://drive.google.com/drive/folders/18Xn_keZcIyG6ocghZPzwIJewXfVGiIqB?usp=sharing
## Lien de présentation
https://www.canva.com/design/DAGm9HhlnnU/aaZf7GlApQ8pg0cyp5kOpw/view?utm_content=DAGm9HhlnnU&utm_campaign=designshare&utm_medium=link2&utm_source=uniquelinks&utlId=h0e27116f10

## Configuration du dataset 
Pour obtenir un dataset exploitable, nous avons collecté des critiques d’utilisateurs sur différents livres en utilisant ParseHub, un outil de scraping. Mais pour affiner l’analyse, il nous fallait intégrer les genres et sous-genres des livres. Nous avons donc scrappé Amazon afin d’ajouter ces informations et fusionné les différentes sources pour obtenir un dataset final, enrichi et structuré.

## Data cleaning 
L'étape de nettoyage des données a été essentielle pour garantir la qualité et la fiabilité des analyses. Voici les principales opérations effectuées :

### Chargement des données
Les fichiers suivants ont été importés depuis le jeu de données Book-Crossing et un jeu complémentaire de genres :

* BX_Books.csv
* BX-Book-Ratings.csv
* BX-Users.csv
* Goodreads_books_with_genres.csv
  
### Prétraitement des ratings

* **Filtrage des utilisateurs** ayant évalué au moins **5 livres** et des livres ayant reçu au moins **50 évaluations**, pour se concentrer sur les interactions significatives.
* **Remplacement des notes à 0** (considérées comme absentes) :

  * Regroupement des utilisateurs par tranche d'âge.
  * Calcul de la **moyenne des notes** pour chaque couple *(ISBN, tranche d'âge)* lorsque ≥ 5 notes sont disponibles.
  * Substitution des notes nulles uniquement si une moyenne de groupe fiable était disponible.

> Résultat : 32.6 % des évaluations originales ont été conservées après nettoyage.

### Nettoyage des utilisateurs

* Suppression des utilisateurs avec une **valeur d'âge invalide ou manquante** (hors de l’intervalle 0–120).
* Création de **catégories d'âge** (AgeGroup) pour faciliter les analyses démographiques :

  * Enfant (1–12)
  * Adolescent (13–17)
  * Jeune adulte (18–25)
  * Adulte (26–35), etc.
### Traitement des genres

* Les genres issus de Goodreads ont été nettoyés :

  * Séparation en deux colonnes : **genre principal** + **genres secondaires**.
  * Remplissage des valeurs manquantes avec "Inconnu".

### Fusion des tables

Les données suivantes ont été jointes :

* books + ratings via ISBN
* Ajout des utilisateurs via User-ID
* Ajout des genres via ISBN (après harmonisation avec isbn de Goodreads)
  
### Nettoyage du dataset final

* Filtrage des années de publication valides (1450 <= année <= 2025), conversion en format date.
* Suppression des lignes où l’**âge** est manquant.
* Conservation des colonnes pertinentes : titre, auteur, année, éditeur, utilisateur, âge, groupe d’âge, note, genres, etc.
### Visualisation de la distribution

Des histogrammes ont été générés pour observer la distribution :

* des **notes de livres**
* de l’**âge des utilisateurs**
* des **années de publication**

> Le dataset final contient environ **22 400 interactions** propres et enrichies, prêtes pour l’analyse ou le développement de systèmes de recommandation.

## Feature Engineering : Représentation vectorielle des genres

Afin de mieux exploiter les genres littéraires dans un modèle de recommandation ou une analyse sémantique, des embeddings vectoriels ont été générés à l'aide du modèle paraphrase-MiniLM-L6-v2 de sentence-transformers.

---

### 1. Réduction des genres secondaires

Chaque livre pouvant avoir plusieurs genres secondaires, une étape de simplification a été mise en place :

* Extraction de **tous les genres secondaires uniques** dans le dataset.
* Génération d’un **embedding vectoriel pour chaque genre secondaire**.
* Pour chaque livre, calcul du **vecteur moyen** de ses genres secondaires.
* Sélection du **genre secondaire le plus proche** du vecteur moyen (cosine similarity).
* Création d'une nouvelle colonne : `Genre_Sec_Privege`, représentant le **genre secondaire dominant**.

Cette approche permet de simplifier la représentation tout en conservant la richesse sémantique des genres.

---

### 2. Embedding des genres

Pour préparer les données à des modèles basés sur la similarité ou l’apprentissage supervisé :

* **Genres principaux** et **genres secondaires privilégiés** ont été encodés en vecteurs numériques de taille 384 via `SentenceTransformer`.
* Création de deux colonnes :
  * Embedding_Genre_Principal
  * Embedding_Genre_Secondaire

> Ces vecteurs serviront ultérieurement à des calculs de proximité entre livres, à la création de clusters thématiques, ou à l’alimentation d’un système de recommandation.

# Réduction de dimension par PCA sur les embeddings

Pour optimiser la représentation vectorielle des genres littéraires tout en réduisant la complexité, une réduction de dimension par PCA a été réalisée :

- **Fusion pondérée** des embeddings de genre principal et secondaire (70% / 30%).
- Construction d'une **matrice de vecteurs combinés**.
- Application de la PCA pour analyser la **variance expliquée cumulée**.
- Détermination du **nombre optimal de composantes** pour capturer 90% de la variance.
- Réduction finale des vecteurs à ce nombre de dimensions.
- Intégration des nouvelles composantes principales (PC1, PC2, ...) dans le DataFrame.

Cette réduction permet d'alléger les calculs tout en conservant l'information essentielle pour les modèles de recommandation.


## Prédiction de la note utilisateur

L’objectif est de prédire la note qu’un utilisateur attribuerait à un livre, en combinant :
- les **embeddings** des genres (principal et secondaire),
- des **données utilisateur** (âge, ancienneté, etc.).

### 1. Préparation des données
- Nettoyage des embeddings et conversion en vecteurs.
- Fusion pondérée : 70% genre principal + 30% genre secondaire.
- Ajout de variables : âge, ancienneté du livre, encodage utilisateur, tranche d’âge.
- Création d’une matrice finale `X` avec vecteurs (384 dim) + données tabulaires.

### 2. Entraînement du modèle
- Modèle : `RandomForestRegressor` optimisé via `GridSearchCV`.
- Split 80/20 pour l'entraînement/test.

### 3. Résultats

| Métrique   | Valeur  |
|------------|---------|
| RMSE       | 1.322   |
| MAE        | 0.924   |
| R²         | 0.074   |
| MAPE       | 14.56%  |


## Entraînement du modèle avec XGBoost

- Utilisation du modèle `XGBRegressor` (XGBoost) pour la prédiction des notes (`Book-Rating`)
- Optimisation via `GridSearchCV` :
  - `max_depth`, `learning_rate`, `n_estimators`, `subsample`, etc.
- Séparation des données en jeu d’entraînement et de test via `train_test_split`.

### 3. Évaluation du modèle

- **RMSE** 
- **MAE** 
- **R² Score** 
- **MAPE** 

### Résultats observés
- L’erreur moyenne reste raisonnable mais le **R²** suggère qu’il reste des marges d’amélioration.
- Ce modèle peut servir de **baseline fiable** dans le système de recommandation.

## **Test avec LightGBM**

Dans cette étape, nous avons utilisé le modèle **LightGBM (Light Gradient Boosting Machine)**, un algorithme de machine learning efficace pour les tâches de régression. Voici les étapes suivies :

1. **Préparation des données :**
   - Les données ont été séparées en ensembles d'entraînement (80%) et de test (20%) à l'aide de `train_test_split`.

2. **Modélisation :**
   - Un modèle **LGBMRegressor** a été défini pour effectuer une régression, avec des paramètres d'objectif réglés sur "regression" et une graine aléatoire pour la reproductibilité.
   
3. **Optimisation des hyperparamètres :**
   - Un **GridSearchCV** a été utilisé pour tester plusieurs combinaisons de paramètres, notamment :
     - Nombre d'estimateurs (`n_estimators`)
     - Taux d'apprentissage (`learning_rate`)
     - Profondeur maximale de l'arbre (`max_depth`)
     - Nombre de feuilles (`num_leaves`)
     - Sous-échantillonnage (`subsample`)

4. **Prédiction et évaluation :**
   - Le modèle a été évalué sur les données de test en calculant plusieurs métriques :
     - **RMSE (Root Mean Squared Error)** : 1.329
     - **MAE (Mean Absolute Error)** : 0.940
     - **R2 Score** : 0.065
     - **MAPE (Mean Absolute Percentage Error)** : 14.81%

### Résultats

Les meilleurs paramètres trouvés via GridSearchCV sont les suivants :
- `n_estimators`: 200
- `learning_rate`: 0.1
- `max_depth`: 10
- `num_leaves`: 50
- `subsample`: 1.0

### Interprétation des résultats
- Le modèle a montré des performances correctes mais peut être amélioré. L'erreur moyenne absolue est proche de 1 point, mais le coefficient de détermination (R²) reste faible, indiquant qu'il existe encore de la marge pour améliorer la précision du modèle.

  
### Interface 
## Utilisation de Streamlit 




## Finalité 
Notre objectif final est de rendre Bookmatch accessible à tous. Nous voulons que chaque lecteur puisse découvrir des ouvrages adaptés à ses goûts grâce à une combinaison intelligente d’analyses de critiques, d’informations sur les genres et d’algorithmes de machine learning. Avec ce système, nous espérons transformer le choix d’un livre en une expérience fluide, intuitive et enrichissante.

#  Interface utilisateur - Recommandation avec Streamlit

Une interface a été développée avec **Streamlit** pour permettre aux utilisateurs d’obtenir des recommandations de livres personnalisées.

##  Fonctionnalités
- Sélection de l'âge de l'utilisateur.
- Choix du genre principal préféré.
- Choix du ou des genres secondaires.
- Génération d'une liste de recommandations basée sur les préférences saisies.

## Principe de fonctionnement

### Préparation des entrées utilisateur :
- Les genres sélectionnés sont convertis en vecteurs d'**embeddings** via *sentence-transformers*.
- L'âge et le groupe d'âge sont pris en compte pour enrichir le profil utilisateur.

### Construction du vecteur utilisateur :
- Fusion des embeddings des genres avec les données tabulaires (âge, groupe d’âge...).
- Génération d’un vecteur de caractéristiques similaire à ceux utilisés lors de l’entraînement du modèle.

### Prédiction et filtrage :
- Le vecteur est passé au modèle **XGBoost** (ou autre modèle entraîné).
- Une note prédite est générée pour chaque livre du dataset.
- Les livres sont triés par note pour afficher les meilleures suggestions.

### Affichage des recommandations :
- Les résultats sont présentés dans l’interface avec les informations clés : **titre, auteur, année**, etc.

