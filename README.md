# Bookmatch project 

Bookmatch est une interface ayant pour but que chaque personne trouve son prochain livre. 

En effet, on constate sur ces dernières années que le nombre de lecteur en France est en baisse. Une des difficultés majeures pour commencer un livre, c'est d'y trouver son compte parmi la multitude de choix. 
C'est pourquoi Bookmatch permet à son utilisateur, en fonction de critères personnalisés de trouver LE livre qui lui correspond. 


## Google Drive 
https://drive.google.com/drive/folders/18Xn_keZcIyG6ocghZPzwIJewXfVGiIqB?usp=sharing

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


## Modèle de ML utilisé 
Pour le Machine Learning, nous avons adopté une approche progressive. Nous avons commencé par tester une régression linéaire afin d’évaluer la relation entre les caractéristiques des livres et des utilisateurs ainsi que les notes attribuées. Ce premier modèle nous a aidés à identifier des tendances avant d’explorer des méthodes plus avancées, comme le filtrage collaboratif.

## Finalité 
Notre objectif final est de rendre Bookmatch accessible à tous. Nous voulons que chaque lecteur puisse découvrir des ouvrages adaptés à ses goûts grâce à une combinaison intelligente d’analyses de critiques, d’informations sur les genres et d’algorithmes de machine learning. Avec ce système, nous espérons transformer le choix d’un livre en une expérience fluide, intuitive et enrichissante.

## Foobar

Foobar is a Python library for dealing with word pluralization.

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install foobar.

```bash
pip install foobar
```

## Usage

```python
import foobar

# returns 'words'
foobar.pluralize('word')

# returns 'geese'
foobar.pluralize('goose')

# returns 'phenomenon'
foobar.singularize('phenomena')
```

## Contributing

Pull requests are welcome. For major changes, please open an issue first
to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License

[MIT](https://choosealicense.com/licenses/mit/)
