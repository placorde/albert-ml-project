import pandas as pd
import os
import joblib
import numpy as np
import streamlit as st
import time
from xgboost import XGBRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split

# Configuration de la page Streamlit
st.set_page_config(page_title="📚 Recommandation de livres", page_icon="📚", layout="wide")

# Chemins des fichiers
MODEL_PATH = "xgboost_model.pkl"
PCA_PATH = "pca_model.pkl"

# Charger le dataset
@st.cache_data
def load_data():
    return pd.read_csv("book_match_clean.csv", sep=";")

df = load_data()

# Chargement ou création du modèle d'embedding
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer('sentence-transformers/paraphrase-MiniLM-L6-v2')

model_embedding = load_embedding_model()

# Extraction et préparation des genres secondaires
@st.cache_data
def prepare_genres():
    # Nettoyage de la colonne des genres secondaires
    genres_list = df['Genres Secondaires'].dropna().str.split(',')
    
    # Aplatir, nettoyer les genres et enlever les doublons
    flattened_genres = set(
        genre.strip()
        for sublist in genres_list
        for genre in sublist
        if genre.strip()
    )
    
    # Trier les genres secondaires pour un affichage stable
    return sorted(flattened_genres)

unique_genres = prepare_genres()

# Création des embeddings de genres
@st.cache_data
def create_genre_embeddings(genres, model):
    return {genre: model.encode(genre) for genre in genres}

genre_embeddings = create_genre_embeddings(unique_genres, model_embedding)

# Fonction pour encoder un genre (texte) en vecteur
def encode_genre(genre, model):
    if pd.isna(genre):
        return np.zeros(384)  # vecteur nul si valeur manquante
    return model.encode(genre.strip())

# Traitement des données - à exécuter une seule fois lors du premier chargement
@st.cache_data
def process_data(df, model_embedding):
    # Trouvons le genre secondaire le plus représentatif par ligne
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
    df_processed = df.copy()
    df_processed['Genre_Sec_Privege'] = reduced_genres
    
    # Encodage des genres principaux et secondaires
    df_processed['Embedding_Genre_Principal'] = df_processed['Genre Principal'].apply(
        lambda x: encode_genre(x, model_embedding)
    )
    
    df_processed['Embedding_Genre_Secondaire'] = df_processed['Genre_Sec_Privege'].apply(
        lambda x: encode_genre(x, model_embedding)
    )
    
    # Ajout de features utilisateur globales
    user_stats = df_processed.groupby('User-ID')['Book-Rating'].agg(
        User_Mean_Rating='mean',
        User_Rating_Count='count',
        User_Rating_Std='std'
    ).reset_index()
    
    df_processed = df_processed.merge(user_stats, on='User-ID', how='left')
    
    # Combinaison pondérée des embeddings
    df_processed['Combined_Embedding'] = df_processed.apply(
        lambda row: 0.7 * row['Embedding_Genre_Principal'] + 0.3 * row['Embedding_Genre_Secondaire'], 
        axis=1
    )
    
    # Suppression des utilisateurs avec moins de 5 notes
    user_counts = df_processed['User-ID'].value_counts()
    valid_users = user_counts[user_counts >= 5].index
    df_processed = df_processed[df_processed['User-ID'].isin(valid_users)].copy()
    
    # Encodage de l'ID utilisateur
    user_encoder = LabelEncoder()
    df_processed['User_ID_Encoded'] = user_encoder.fit_transform(df_processed['User-ID'])
    
    return df_processed

df_processed = process_data(df, model_embedding)

# Création et entraînement du modèle PCA
def fit_pca_model(df_processed):
    if os.path.exists(PCA_PATH):
        st.info("📦 Chargement du modèle PCA existant...")
        pca = joblib.load(PCA_PATH)
    else:
        st.info("🛠️ Entraînement du modèle PCA (peut prendre un moment)...")
        # Construction de la matrice d'embeddings combinés
        embedding_matrix = np.vstack(df_processed['Combined_Embedding'].values)
        
        # PCA sans réduction pour calculer la variance cumulée
        pca = PCA()
        pca.fit(embedding_matrix)
        
        cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
        
        # Nombre optimal de composantes pour 90% de variance
        optimal_components = np.argmax(cumulative_variance >= 0.90) + 1
        
        # Réduction finale avec le bon nombre de composantes
        pca_final = PCA(n_components=optimal_components, random_state=42)
        pca_final.fit(embedding_matrix)
        
        # Sauvegarde du modèle PCA
        joblib.dump(pca_final, PCA_PATH)
        st.success(f"✅ Modèle PCA entraîné et sauvegardé avec {optimal_components} composantes.")
        
        pca = pca_final
    
    return pca

pca = fit_pca_model(df_processed)

# Application de la PCA aux embeddings
def apply_pca_to_data(df_processed, pca):
    embedding_matrix = np.vstack(df_processed['Combined_Embedding'].values)
    reduced_embeddings = pca.transform(embedding_matrix)
    
    # Intégration des composantes dans le DataFrame
    df_with_pca = df_processed.copy()
    for i in range(pca.n_components_):
        df_with_pca[f'Genre_PC_{i}'] = reduced_embeddings[:, i]
        
    return df_with_pca

df_with_pca = apply_pca_to_data(df_processed, pca)

# Préparation des données pour l'entraînement
def prepare_training_data(df_with_pca):
    # Colonnes PCA (embeddings réduits)
    embedding_cols = [col for col in df_with_pca.columns if col.startswith('Genre_PC_')]
    
    # Colonnes numériques utilisateur
    user_features = ['Age', 'User_ID_Encoded', 'User_Mean_Rating', 'User_Rating_Count', 'User_Rating_Std']
    
    # Encodage AgeGroup (si présente)
    if 'AgeGroup' in df_with_pca.columns:
        agegroup_dummies = pd.get_dummies(df_with_pca['AgeGroup'], prefix='AgeGroup')
        X = pd.concat([df_with_pca[embedding_cols + user_features], agegroup_dummies], axis=1)
    else:
        X = df_with_pca[embedding_cols + user_features]
    
    # Target
    y = df_with_pca['Book-Rating']
    
    # Suppression des lignes avec NaN
    valid_indices = ~X.isna().any(axis=1)
    X = X.loc[valid_indices]
    y = y.loc[valid_indices]
    
    return X, y

X, y = prepare_training_data(df_with_pca)

# Création et entraînement du modèle XGBoost
def fit_xgboost_model(X, y):
    if os.path.exists(MODEL_PATH):
        st.info("📦 Chargement du modèle XGBoost existant...")
        best_model = joblib.load(MODEL_PATH)
    else:
        st.info("🛠️ Entraînement du modèle XGBoost (peut prendre un moment)...")
        
        # Découpage train/test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Initialisation du modèle avec les meilleurs hyperparamètres
        best_model = XGBRegressor(
            objective='reg:squarederror',
            colsample_bytree=1.0,
            learning_rate=0.05,
            max_depth=3,
            n_estimators=200,
            subsample=0.8,
            random_state=42
        )
        
        # Entraînement
        best_model.fit(X_train, y_train)
        
        # Sauvegarde du modèle
        joblib.dump(best_model, MODEL_PATH)
        st.success("✅ Modèle XGBoost entraîné et sauvegardé avec succès.")
    
    return best_model

best_xgb = fit_xgboost_model(X, y)

# Fonction pour déterminer le groupe d'âge
def get_age_group(age):
    if age <= 12:
        return 'Enfant 1-12'
    elif age <= 17:
        return 'Adolescent 12-17'
    elif age <= 25:
        return 'Jeune adulte 18-25'
    elif age <= 35:
        return 'Adulte 26-35'
    elif age <= 50:
        return 'Adulte 36-50'
    else:
        return 'Senior 50+'

# Fonction de recommandation de livres avec XGBoost
def recommander_livres_xgboost(user_age, user_genres, user_sous_genres, df, pca, model_embedding, model_xgb, top_n=10):
    start_time = time.time()
    
    # Encodage des genres sélectionnés
    user_genres_vecs = [model_embedding.encode(g.strip()) for g in user_genres if g.strip()]
    user_sous_genres_vecs = [model_embedding.encode(g.strip()) for g in user_sous_genres if g.strip()]
    
    # Si aucun genre n'est sélectionné, utiliser un vecteur vide
    if not user_genres_vecs:
        user_genres_vecs = [np.zeros(384)]
    if not user_sous_genres_vecs:
        user_sous_genres_vecs = [np.zeros(384)]
    
    # Moyenne pondérée des vecteurs
    genre_vec = np.mean(user_genres_vecs, axis=0)
    sous_genre_vec = np.mean(user_sous_genres_vecs, axis=0)
    user_vector = 0.7 * genre_vec + 0.3 * sous_genre_vec
    
    # Réduction PCA
    reduced_user_vector = pca.transform([user_vector])[0]
    
    # Préparation des données pour la prédiction
    # On utilise tous les livres uniques du DataFrame
    unique_books = df.drop_duplicates(subset=['ISBN']).copy()
    
    # Ajout des composantes PCA
    for i in range(len(reduced_user_vector)):
        unique_books[f'Genre_PC_{i}'] = reduced_user_vector[i]
    
    # Ajout des informations utilisateur
    unique_books['Age'] = user_age
    unique_books['User_ID_Encoded'] = 0  # Valeur arbitraire
    unique_books['User_Mean_Rating'] = df['Book-Rating'].mean()  # Moyenne globale
    unique_books['User_Rating_Count'] = 5  # Valeur arbitraire minimale
    unique_books['User_Rating_Std'] = df['Book-Rating'].std()  # Écart-type global
    
    # Ajout des variables dummy pour AgeGroup
    age_group = get_age_group(user_age)
    age_groups = ['Enfant 1-12', 'Adolescent 12-17', 'Jeune adulte 18-25', 
                 'Adulte 26-35', 'Adulte 36-50', 'Senior 50+']
    
    for group in age_groups:
        unique_books[f'AgeGroup_{"_".join(group.split())}'] = 1 if group == age_group else 0
    
    # S'assurer que les colonnes correspondent aux features du modèle
    model_features = model_xgb.get_booster().feature_names
    missing_cols = set(model_features) - set(unique_books.columns)
    
    # Ajouter les colonnes manquantes avec des valeurs par défaut
    for col in missing_cols:
        unique_books[col] = 0
    
    # Ne sélectionner que les colonnes nécessaires pour la prédiction
    X_pred = unique_books[model_features]
    
    # Prédiction avec le modèle XGBoost
    unique_books['Predicted_Rating'] = model_xgb.predict(X_pred)
    
    # Sélection des meilleurs livres
    top_books = unique_books.sort_values('Predicted_Rating', ascending=False).head(top_n)[
        ['Book-Title', 'Book-Author', 'Image-URL-L', 'Predicted_Rating']
    ]
    
    elapsed = time.time() - start_time
    st.info(f"⏱️ Temps d'exécution : {elapsed:.2f} secondes")
    
    return top_books

# Titre de l'application
st.title("📚 Recommandation de livres")

# 📋 Bloc de questions
with st.container():
    st.subheader("📝 Répondez aux questions suivantes")

    # Âge (curseur)
    age = st.slider("Quel âge as-tu ?", min_value=10, max_value=80, value=25)

    # Liste des genres principaux disponibles
    genres_disponibles = df['Genre Principal'].dropna().unique().tolist()
    genres_disponibles.sort()
    
    # Boutons de sélection pour choisir jusqu'à 3 genres
    genres_selectionnes = st.multiselect(
        "Choisissez **jusqu'à 3 genres de livres principaux**",
        genres_disponibles,
        max_selections=3
    )
    
    # Boutons de sélection pour choisir jusqu'à 3 sous-genres
    sous_genres_disponibles = unique_genres
    
    sous_genres_selectionnes = st.multiselect(
        "Choisissez **jusqu'à 3 sous-genres de livres**",
        sous_genres_disponibles,
        max_selections=3
    )

# 🔍 Résumé + recommandations
if st.button("🔎 Obtenir des recommandations"):
    if not genres_selectionnes:
        st.warning("❗ Veuillez sélectionner au moins un genre pour obtenir des recommandations.")
    else:
        st.subheader("✨ Résumé de tes préférences")
        st.write(f"**Âge** : {age} ans")
        st.write(f"**Genres choisis** : {', '.join(genres_selectionnes)}")
        if sous_genres_selectionnes:
            st.write(f"**Sous-genres sélectionnés** : {', '.join(sous_genres_selectionnes)}")
        else:
            st.write("**Sous-genres sélectionnés** : Aucun")
        
        st.success("✅ Tes réponses sont enregistrées ! Voici les livres recommandés :")
        
        # Obtenir les recommandations
        recommandations = recommander_livres_xgboost(
            user_age=age, 
            user_genres=genres_selectionnes, 
            user_sous_genres=sous_genres_selectionnes,
            df=df,
            pca=pca,
            model_embedding=model_embedding,
            model_xgb=best_xgb
        )
        
        # Affichage des recommandations avec images
        cols = st.columns(5)  # 5 colonnes pour afficher les résultats
        
        for i, (index, row) in enumerate(recommandations.iterrows()):
            col_idx = i % 5
            with cols[col_idx]:
                st.image(row['Image-URL-L'], width=150)
                st.write(f"**{row['Book-Title']}**")
                st.write(f"*{row['Book-Author']}*")
                st.write(f"Score prédit: {row['Predicted_Rating']:.2f}")
                st.write("---")
