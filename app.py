import pandas as pd 
import os
import joblib
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
import streamlit as st
import time
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error


st.set_page_config(page_title="📚 Recommandation de livres", page_icon="📚", layout="wide")

# Load the dataset
df = pd.read_csv("book_match_clean.csv", sep=";")

# Nettoyage de la colonne des genres secondaires (retirer les NaN, splitter par ',')
genres_list = df['Genres Secondaires'].dropna().str.split(',')

# Aplatir, nettoyer les genres et enlever les doublons
flattened_genres = set(
    genre.strip()
    for sublist in genres_list
    for genre in sublist
    if genre.strip()
)

# Trier les genres secondaires pour un affichage stable
unique_genres = sorted(flattened_genres)

# Charger un modèle de phrase multilingue basé sur des transformers
model = SentenceTransformer('all-MiniLM-L6-v2')

# Encoder tous les genres secondaires en vecteurs
genre_embeddings = {
    genre: model.encode(genre)
    for genre in unique_genres
}

# Nettoyage de la colonne des genres secondaires (retirer les NaN, splitter par ',')
genres_list = df['Genres Secondaires'].dropna().str.split(',')

# Étape 3 : Trouver le genre secondaire le plus représentatif par ligne
reduced_genres = []

for raw in df['Genres Secondaires']:
    if pd.isna(raw):
        reduced_genres.append(None)
        continue

    genres = [g.strip() for g in raw.split(',') if g.strip() in genre_embeddings]
    if not genres:
        reduced_genres.append(None)
        continue

    vectors = [genre_embeddings[g] for g in genres]
    mean_vector = sum(vectors) / len(vectors)
    similarities = {
        g: cosine_similarity([genre_embeddings[g]], [mean_vector])[0][0] for g in genres
    }
    best_match = max(similarities, key=similarities.get)
    reduced_genres.append(best_match)

# Ajouter la colonne au DataFrame
df['Genre_Sec_Privege'] = reduced_genres

# === Étape 4 : Embedding des genres principaux et secondaires ===

# Fonction utilitaire pour encoder un genre (texte) en vecteur
def encode_genre(genre):
    if pd.isna(genre):
        return np.zeros(384)  # vecteur nul si valeur manquante
    return model.encode(genre.strip())

# Encodage du genre principal
df['Embedding_Genre_Principal'] = df['Genre Principal'].apply(encode_genre)

# Encodage du genre secondaire privilégié
df['Embedding_Genre_Secondaire'] = df['Genre_Sec_Privege'].apply(encode_genre)


# ---------------------------
# ÉTAPE 1 : PRÉPARATION DES DONNÉES
# ---------------------------

# Nettoyage des embeddings

def parse_embedding(emb):
    try:
        # Gestion des strings ou listes
        if isinstance(emb, str):
            return np.array(eval(emb))
        elif isinstance(emb, list):
            return np.array(emb)
        elif isinstance(emb, np.ndarray):
            return emb
        else:
            return np.zeros(384)
    except:
        return np.zeros(384)

# Application de la fonction
print("Parsing des embeddings...")
df["Emb_Princ"] = df["Embedding_Genre_Principal"].apply(parse_embedding)
df["Emb_Sec"] = df["Embedding_Genre_Secondaire"].apply(parse_embedding)

# Pondération (le principal compte plus)
df["Final_Emb"] = df.apply(lambda row: 0.7 * row["Emb_Princ"] + 0.3 * row["Emb_Sec"], axis=1)

# Encodage du User ID
le_user = LabelEncoder()
df["User_ID_Encoded"] = le_user.fit_transform(df["User-ID"])

# Calcul de l'ancienneté de publication
current_year = 2025
df["Anciennete"] = current_year - pd.to_datetime(df["Year-Of-Publication"], errors='coerce').dt.year.fillna(current_year).astype(int)

# Features tabulaires
X_other = df[["Age", "Anciennete", "User_ID_Encoded"]].copy()
X_other = pd.get_dummies(pd.concat([X_other, df[["AgeGroup"]]], axis=1), columns=["AgeGroup"], drop_first=True)

# Embedding vector (384 dim)
X_emb = np.vstack(df["Final_Emb"].values)

# Matrice finale
X = np.concatenate([X_emb, X_other.values], axis=1)
y = df["Book-Rating"].values

# Split train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

MODEL_PATH = "random_forest_model.pkl"

if os.path.exists(MODEL_PATH):
    st.info("📦 Chargement du modèle Random Forest existant...")
    best_model = joblib.load(MODEL_PATH)
else:
    st.info("🛠️ Entraînement du modèle Random Forest (peut prendre un moment)...")

    params = {
        "n_estimators": [100, 200],
        "max_depth": [10, 20],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 2]
    }

    grid = GridSearchCV(RandomForestRegressor(random_state=42),
                        params,
                        cv=3,
                        scoring="neg_mean_squared_error",
                        n_jobs=-1)

    grid.fit(X_train, y_train)
    best_model = grid.best_estimator_

    # Sauvegarde du modèle
    joblib.dump(best_model, MODEL_PATH)
    st.success("✅ Modèle entraîné et sauvegardé avec succès.")

# Entraînement d'un RandomForest optimisé
params = {
    "n_estimators": [100, 200],
    "max_depth": [10, 20],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 2]
}

grid = GridSearchCV(RandomForestRegressor(random_state=42),
                    params,
                    cv=3,
                    scoring="neg_mean_squared_error",
                    n_jobs=-1)

grid.fit(X_train, y_train)

# Meilleur modèle
best_model = grid.best_estimator_
y_pred = best_model.predict(X_test)

# Évaluation
rmse = mean_squared_error(y_test, y_pred)**0.5  # Évite l'argument squared
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
mape = mean_absolute_percentage_error(y_test, y_pred)



# Titre de l'application
st.title("📚 Recommandation de livres avec Llama 2")

# 📋 Bloc de questions
with st.container():
    st.subheader("📝 Répondez aux questions suivantes")

    # Âge (curseur)
    age = st.slider("Quel âge as-tu ?", min_value=10, max_value=80, value=25)

    # Boutons de sélection pour choisir **jusqu'à 3 genres**
    genres_disponibles = [
        "Literature & Fiction", "Sports", "Medicine & Health Sciences",
        "Biographies, Diaries & True Accounts", "Society & Social Sciences",
        "Crafts, Home & Lifestyle", "Children's Books", "Crime, Thriller & Mystery",
        "Reference", "Sciences, Technology & Medicine", "Arts, Film & Photography",
        "Fantasy, Horror & Science Fiction", "History", "Computing, Internet & Digital Media",
        "Religion", "Science & Mathematics", "Politics", "Business & Economics",
        "Higher Education Textbooks", "Travel"
    ]
    
    genres_selectionnes = st.multiselect(
        "Choisissez **jusqu'à 3 genres de livres**",
        genres_disponibles,
        max_selections=3
    )
    
    # Boutons de sélection pour choisir **jusqu'à 3 genres**
    sous_genres_disponibles = df["Genres Secondaires"].dropna().unique().tolist()
    
    sous_genres_selectionnes = st.multiselect(
        "Choisissez **jusqu'à 3 sous genres de livres**",
        sous_genres_disponibles,
        max_selections=3
    ) 


def get_age_group(age):
    if age <= 12:
        return 'Moyenne_Enfant 1-12'
    elif age <= 17:
        return 'Moyenne_Adolescent 12-17'
    elif age <= 25:
        return 'Moyenne_Jeuone adulte 18-25'
    elif age <= 35:
        return 'Moyenne_Adulte 26-35'
    elif age <= 50:
        return 'Moyenne_Adulte 36-50'
    else:
        return 'Moyenne_Senior 50+'

def recommander_livres_rf(user_age, user_genres, user_sous_genres, top_n=10):
    start_time = time.time()

    age_group_col = get_age_group(user_age)
    mask = ~df[age_group_col].isna()

    # On encode les genres sélectionnés
    user_genres_vecs = [model.encode(g.strip()) for g in user_genres if g.strip() in genre_embeddings]
    user_sous_genres_vecs = [model.encode(g.strip()) for g in user_sous_genres if g.strip() in genre_embeddings]

    # Moyenne pondérée
    if not user_genres_vecs:
        user_genres_vecs = [np.zeros(384)]
    if not user_sous_genres_vecs:
        user_sous_genres_vecs = [np.zeros(384)]

    genre_vec = np.mean(user_genres_vecs, axis=0)
    sous_genre_vec = np.mean(user_sous_genres_vecs, axis=0)
    user_vector = 0.7 * genre_vec + 0.3 * sous_genre_vec

    # On prépare X et y
    X_features = df.loc[mask].copy()
    X_emb = np.vstack(X_features["Final_Emb"].values)
    user_emb_matrix = np.tile(user_vector, (X_emb.shape[0], 1))

    X_input = np.concatenate([user_emb_matrix, X_other.loc[mask].values], axis=1)
    y_target = df.loc[mask, age_group_col].values

    # Prediction avec le meilleur modèle Random Forest déjà entraîné
    y_pred = best_model.predict(X_input)

    # Ajout des prédictions au DataFrame
    X_features = X_features.copy()
    X_features["Predicted_Rating"] = y_pred

    # Top recommandations
    top_books = X_features.sort_values("Predicted_Rating", ascending=False).head(top_n)[["Book-Title", "Book-Author", "Predicted_Rating"]]

    elapsed = time.time() - start_time
    st.info(f"⏱️ Temps d'exécution : {elapsed:.2f} secondes")
    
    return top_books


# 🔍 Résumé + recommandations
if st.button("🔎 Obtenir des recommandations"):

    if not genres_selectionnes:
        st.warning("❗ Veuillez sélectionner au moins un genre pour obtenir des recommandations.")
    else:
        st.subheader("✨ Résumé de tes préférences")
        st.write(f"**Âge** : {age} ans")
        st.write(f"**Genres choisis** : {', '.join(genres_selectionnes)}")
        st.write(f"**Sous-genres sélectonnés** : {', '.join(genres_selectionnes)}")
        
        st.success("✅ Tes réponses sont enregistrées ! Voici les livres recommandés :")

    recommandations = recommander_livres_rf(user_age=age, user_genres=genres_selectionnes, user_sous_genres=sous_genres_selectionnes)
    st.dataframe(recommandations.style.set_properties(**{'text-align': 'left'}).hide(axis="index"))


