# Bookmatch project 

Bookmatch est une interface ayant pour but que chaque personne trouve son prochain livre. 

En effet, on constate sur ces dernières années que le nombre de lecteur en France est en baisse. Une des difficultés majeures pour commencer un livre, c'est d'y trouver son compte parmi la multitude de choix. 
C'est pourquoi Bookmatch permet à son utilisateur, en fonction de critères personnalisés de trouver LE livre qui lui correspond. 





## Google Drive 
https://drive.google.com/drive/folders/18Xn_keZcIyG6ocghZPzwIJewXfVGiIqB?usp=sharing

## Configuration du dataset 
Nous avons obtenu un dataset permettant d'avoir un référencement de critiques faites par des utilisateurs sur des livres. Néanmoins, pour permettre à BookMatch d'être précis, nous avions besoin des genres des livres. Ainsi, nous avons récupéré les informations sur le genre et le sous-genre des livres grâce à un scrapping fait des livres d'Amazon.

Nous avons donc merge les colonnes nécessaires pour avoir notre dataset final.

## Data cleaning 
On peut remarquer dans la distribution des ratings, un fort nombre de notes égale à O. Ainsi en fonction du nombre d'avis par user, nous avons fait une sélection. 
Egalement, une problématique qui se posait était pour un même livre des éditions différentes ou plusieurs critiques pour un même livre.

## Modèle de ML utilisé 
Nous avons dans un premier temps utilisé une régression linéaire pour simplement comprendre la relation entre les caractéristiques utilisateurs/livre et les notes données.

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
