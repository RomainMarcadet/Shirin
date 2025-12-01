# Shirin
Projet end-to-end Master NLP

Construire un pipeline complet : 

1 - Entraîner un modèle de classification d'avis (positif / négatif) et générer des réponses personnalisé aux avis négatifs via un modèle de Mistral.
2 - Créer un dataframe final contenant : avis client, étiquette (positif / négatif), adresse mail, réponse générée.
3 - En utilisant les bonnes pratiques pour découper le projet en 4 fichiers Python, puis de créer un repositorie GitHub.
4 - Déployer une API FastAPI sur Azure avec CI/CD

## Architecture du projet

Le projet est organisé autour de 4 fichiers Python principaux :

1. `data_preprocessing.py`  
   - Charger les données brutes (`SetFit/amazon_reviews_multi_fr`).  
   - Supprimer les doublons / valeurs manquantes.  
   - Définir et appliquer la fonction `clean_review(text: str) -> str` pour nettoyer les avis (lowercase, ponctuation, stopwords, lemmatisation, etc.).  
   - Sauvegarder les données nettoyées dans `data/processed/`.

2. `train_classifier.py`  
   - Charger les données prétraitées depuis `data/processed/`.  
   - Séparer en jeux d’entraînement / validation / test.  
   - Entraîner un modèle de classification d’avis (positif / négatif).  
   - Sauvegarder le modèle et les objets nécessaires (vectorizer, label encoder…) dans `models/`.

3. `generate_responses.py`  
   - Charger le modèle entraîné et les données (nouvelles reviews ou jeu de test).  
   - Identifier les avis négatifs à traiter.  
   - Appeler l’API Mistral pour générer une réponse personnalisée à chaque avis négatif.  
   - Construire le dataframe final : **avis client**, **étiquette (positif / négatif)**, **adresse mail**, **réponse générée**.  
   - Sauvegarder ce dataframe final dans `data/final/` (par ex. en `.parquet` ou `.csv`).

4. `api.py`  
   - Exposer une API FastAPI permettant :  
     - Un endpoint de santé (`/health`).  
     - Un endpoint pour classifier un nouvel avis (`/predict`).  
     - Un endpoint pour générer une réponse à un avis négatif (`/respond`).  
   - Charger le modèle et la logique définie dans les fichiers précédents.  
   - Préparer le fichier pour un déploiement sur Azure avec CI/CD (Dockerfile + config).

---

## Plan détaillé du pipeline

### 1. Préparation des données

- Charger le dataset `SetFit/amazon_reviews_multi_fr` (échantillon de 100k reviews) dans un DataFrame `df`.  
- Analyser la distribution des labels et la qualité des données (EDA basique).  
- Supprimer les doublons sur la colonne `text` et gérer les éventuelles valeurs manquantes.  
- Définir la fonction `clean_review(text)` dans `data_preprocessing.py` :  
  - Mettre en minuscules.  
  - Retirer ponctuation, chiffres, caractères spéciaux.  
  - Tokeniser (NLTK), retirer les stopwords, lemmatiser.  
  - Rejoindre les tokens nettoyés en une chaîne.  
- Créer une colonne `clean_text` et sauvegarder les données nettoyées.

### 2. Entraînement du modèle de classification

- Définir les features (par ex. TF-IDF sur `clean_text`).  
- Split train/validation/test (par ex. 80/10/10).  
- Entraîner un modèle simple (Logistic Regression / Linear SVM / SetFit).  
- Évaluer les performances (accuracy, F1, confusion matrix).  
- Sauvegarder :  
  - Le modèle entraîné.  
  - Le vectorizer / tokenizer.  
  - Les mappings de labels.

### 3. Génération de réponses aux avis négatifs

- Charger les avis (test ou nouveaux avis) et leur prédiction de sentiment.  
- Filtrer les avis négatifs.  
- Pour chaque avis négatif, appeler l’API Mistral (clé lue depuis `.env`) avec un prompt adapté (politesse, ton professionnel, français).  
- Stocker :  
  - `review_id`, `text`, `clean_text`, `label`, `label_text`, `email`, `generated_response`.  
- Sauvegarder le dataframe final pour exploitation.

### 4. API FastAPI & déploiement

- Créer une app FastAPI dans `api.py` :  
  - Charger le modèle, le vectorizer et la fonction `clean_review`.  
  - Endpoint `/predict` : renvoie la polarité d’un avis envoyé en entrée.  
  - Endpoint `/respond` : renvoie une réponse générée pour un avis négatif.  
- Préparer les artefacts de déploiement (Dockerfile, config Azure Web App / Container, workflow CI/CD GitHub Actions).

---

## Fonction `clean_review(text)`

La fonction `clean_review` sera définie dans `data_preprocessing.py` et réutilisée :

- Lors de la préparation des données (création de `clean_text`).  
- À l’inférence dans l’API FastAPI (nettoyer les avis reçus avant classification).

Signature envisagée :

```python
def clean_review(text: str) -> str:
    """
    Nettoie une review texte (lowercase, ponctuation, stopwords, lemmatisation, etc.)
    et renvoie une version prête pour la vectorisation.
    """
    ...
