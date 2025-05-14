# Modèles testés sur cette branche

Les modèles développés ici ont été entraînés à partir d’un dataset structuré de façon à ce que chaque ligne représente un livre, contrairement à la structure classique avec une ligne par review.

## Approche

- Une colonne de rating par tranche d’âge a été ajoutée.
- Lorsqu’un utilisateur renseigne son âge, le modèle utilise la note moyenne associée à sa tranche pour la prédiction.

## Modèles testés

- K-Nearest Neighbors (KNN)
- Random Forest
- Régression Linéaire

## Limites observées

Les performances globales (MAE, RMSE, R²) sont restées faibles malgré ces approches.  
Une amélioration du feature engineering est donc nécessaire afin de mieux représenter les données et d’obtenir des prédictions plus pertinentes.

Prochaine étape : retravailler les variables du dataset de base pour enrichir l'information disponible par ligne et améliorer les performances.

