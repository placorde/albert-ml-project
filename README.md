# Bookmatch project 

Bookmatch est une interface ayant pour but que chaque personne trouve son prochain livre. 

En effet, on constate sur ces dernières années que le nombre de lecteur en France est en baisse. Une des difficultés majeures pour commencer un livre, c'est d'y trouver son compte parmi la multitude de choix. 
C'est pourquoi Bookmatch permet à son utilisateur, en fonction de critères personnalisés de trouver LE livre qui lui correspond. 


## Google Drive 
https://drive.google.com/drive/folders/18Xn_keZcIyG6ocghZPzwIJewXfVGiIqB?usp=sharing

## Configuration du dataset 
Pour obtenir un dataset exploitable, nous avons collecté des critiques d’utilisateurs sur différents livres en utilisant ParseHub, un outil de scraping. Mais pour affiner l’analyse, il nous fallait intégrer les genres et sous-genres des livres. Nous avons donc scrappé Amazon afin d’ajouter ces informations et fusionné les différentes sources pour obtenir un dataset final, enrichi et structuré.

## Data cleaning 
Une fois le dataset en notre possession, nous avons entrepris un nettoyage des données rigoureux. Nous avons remarqué un nombre anormalement élevé de notes égales à zéro, ce qui pouvait biaiser nos analyses. Afin de limiter cet impact, nous avons filtré ces valeurs en fonction du nombre d’avis par utilisateur. Nous avons également traité le problème des éditions multiples d’un même livre ainsi que les critiques répétées, en consolidant les données pour améliorer leur cohérence.

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
