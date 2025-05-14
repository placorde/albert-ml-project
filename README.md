# Réajustement du Dataset

Dans cette branche `model`, nous avons réajusté la structure du dataset afin de permettre la création de davantage de features exploitables pour l'entraînement des modèles.

## Objectif

L’objectif de cette refonte était de mieux représenter les interactions entre les utilisateurs, les livres et leurs caractéristiques (genres, âge, etc.), en enrichissant chaque ligne avec des informations combinées issues des différentes sources de données.

## Modifications apportées

- Fusion et harmonisation des données utilisateurs, livres, évaluations et genres.
- Ajout de nouvelles colonnes telles que :
  - Embeddings des genres principaux et secondaires.
  - Groupe d’âge de l’utilisateur.
  - Encodage numérique de certaines variables catégorielles.
- Calculs de moyennes de rating par groupe d’âge.
- Réduction de la granularité des données pour passer à une ligne par livre enrichi, plutôt qu'une ligne par review brute.

Ces ajustements nous ont permis d’améliorer le feature engineering et de tester des modèles plus adaptés à cette nouvelle structure.

