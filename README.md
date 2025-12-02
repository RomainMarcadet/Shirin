# Shirin
Projet end-to-end Master NLP

Construire un pipeline complet :

1. Entraîner un modèle de classification d'avis (positif / négatif) et générer des réponses personnalisées aux avis négatifs via un modèle de Mistral.  
2. Créer un DataFrame final contenant : avis client, étiquette (positif / négatif), adresse mail, réponse générée.  
3. Utiliser de bonnes pratiques de structuration (4 fichiers Python principaux) et une API propre.  
4. Déployer une API FastAPI sur Azure avec CI/CD.

---

## Architecture du projet

Le projet est organisé autour de 4 fichiers Python principaux :

1. `notebooks/data_preprocessing.py`  
   - Charge les données brutes (`SetFit/amazon_reviews_multi_fr`).  
   - Supprime les doublons / valeurs manquantes sur la colonne `text`.  
   - Définit et applique la fonction `clean_review(text: str) -> str` pour nettoyer les avis (lowercase, ponctuation, stopwords, lemmatisation simple, normalisation, etc.).  
   - Crée les colonnes `clean_text`, `sentiment`, `sentiment_id` (0 = négatif, 1 = neutre, 2 = positif).  
   - Sauvegarde les données nettoyées dans `data/processed/`.

2. `notebooks/train_classifier.py`  
   - Charge les données prétraitées depuis `data/processed/`.  
   - Sépare en jeux d'entraînement / validation / test.  
   - Entraîne un modèle de classification d'avis (régression logistique + TF‑IDF sur `clean_text`).  
   - Évalue les performances (accuracy, F1, matrices de confusion) sur les 3 classes (négatif / neutre / positif).  
   - Sauvegarde le modèle et le vectorizer dans `models/` :
     - `amazon_reviews_multi_fr_logreg.joblib`  
     - `amazon_reviews_multi_fr_tfidf.joblib`

3. `notebooks/generate_responses.py` et `notebooks/generate_responses_sarcasm.py`  
   - Chargent le modèle entraîné et les données (fichier prétraité).  
   - Appliquent le classifieur pour obtenir `pred_sentiment_id` / `pred_sentiment`.  
   - Filtrent les avis négatifs prédits.  
   - Appellent l'API Mistral pour générer une réponse personnalisée à chaque avis négatif :  
     - `generate_responses.py` : ton poli / professionnel.  
     - `generate_responses_sarcasm.py` : ton sarcastique contrôlé.  
   - Sauvegardent un DataFrame final dans `data/final/` :
     - `amazon_reviews_multi_fr_with_responses.csv`  
     - `amazon_reviews_multi_fr_with_responses_sarcasm.csv`

4. `api.py`  
   - Expose une API FastAPI permettant :  
     - `GET /health` : endpoint de santé.  
     - `POST /predict` : classifier un nouvel avis.  
     - `POST /respond` : générer une réponse à un avis (ton `polite` ou `sarcasm`).  
   - Charge le modèle TF‑IDF, la fonction `clean_review` et le client Mistral.  
   - Prépare l'application pour un déploiement ultérieur (Docker / Azure).

Un script Gradio (`notebooks/gradio_ui.py`) permet de tester facilement l'API.

---

## Installation et exécution

### 1. Cloner le projet et créer l'environnement

```bash
git clone https://github.com/<ton_compte>/Shirin.git
cd Shirin
python -m venv .venv
.\.venv\Scripts\activate  # sous Windows
pip install -r requirements.txt
```

### 2. Configurer la clé Mistral

Créer un fichier `.env` à la racine du projet avec ta clé Mistral :

```env
MISTRAL_SECRET_KEY=ta_cle_api_mistral_ici
```

ou bien :

```env
MISTRAL_API_KEY=ta_cle_api_mistral_ici
```

Les scripts (`generate_responses.py`, `generate_responses_sarcasm.py`, `api.py`) chargent automatiquement cette clé.

---

## Pipeline : étape par étape

### 1. Préparation des données

Depuis la racine du projet :

```bash
.\.venv\Scripts\activate
cd notebooks
python data_preprocessing.py
```

Ce script :
- charge le dataset `SetFit/amazon_reviews_multi_fr` (échantillon de 200k reviews),  
- supprime doublons et valeurs manquantes sur `text`,  
- nettoie les avis avec `clean_review`,  
- crée des labels regroupés :
  - `sentiment` : `"negatif"`, `"neutre"`, `"positif"`  
  - `sentiment_id` : 0 (négatif), 1 (neutre), 2 (positif)  
- sauvegarde dans : `data/processed/amazon_reviews_multi_fr_processed.csv`.

### 2. Entraînement du modèle de classification (TF‑IDF)

Toujours dans `notebooks` :

```bash
python train_classifier.py
```

- lit `data/processed/amazon_reviews_multi_fr_processed.csv`,  
- effectue un split train / validation / test (stratifié),  
- transforme `clean_text` en TF‑IDF (unigrammes/bigrammes),  
- entraîne une régression logistique (`scikit-learn`),  
- affiche :
  - un rapport de classification (precision, recall, F1),  
  - matrices de confusion (validation & test),  
- sauvegarde :
  - `models/amazon_reviews_multi_fr_logreg.joblib`  
  - `models/amazon_reviews_multi_fr_tfidf.joblib`.

### 3. Génération de réponses Mistral (batch)

**Version classique (polite) :**

```bash
python generate_responses.py
```

**Version sarcastique :**

```bash
python generate_responses_sarcasm.py
```

Ces scripts :
- chargent le fichier prétraité et le modèle TF‑IDF,  
- prédisent `pred_sentiment_id` pour chaque avis,  
- sélectionnent les avis négatifs (id 0),  
- appellent l'API de Mistral (`mistral-small-latest` par défaut) pour générer une réponse adaptée,  
- construisent un DataFrame final avec les colonnes :
  - `id`, `text`, `sentiment`, `pred_sentiment`,  
  - `email` (placeholder : `client_<id>@example.com`),  
  - `generated_response`,  
- sauvegardent dans `data/final/` :
  - `amazon_reviews_multi_fr_with_responses.csv`  
  - `amazon_reviews_multi_fr_with_responses_sarcasm.csv`.

---

## API FastAPI

Le fichier `api.py` expose une API pour utiliser le modèle et Mistral en temps réel.

### Lancer l'API

Depuis la racine du projet :

```bash
.\.venv\Scripts\activate
uvicorn api:app --reload
```

L'API est accessible sur `http://localhost:8000`.

### Endpoints principaux

- `GET /health`  
  Vérifie que l'API est en ligne :
  ```json
  { "status": "ok" }
  ```

- `POST /predict`  
  Corps de requête :
  ```json
  { "text": "Je suis très déçu de ce produit." }
  ```
  Réponse :
  ```json
  {
    "sentiment_id": 0,
    "sentiment_label": "negatif",
    "proba": 0.82
  }
  ```

- `POST /respond`  
  Corps de requête :
  ```json
  {
    "text": "Je suis très déçu de ce produit.",
    "tone": "polite",
    "force": false,
    "model": "mistral-small-latest"
  }
  ```
  - `tone` : `"polite"` ou `"sarcasm"`.  
  - `force` :
    - `false` → ne génère une réponse que si l'avis est prédit négatif,  
    - `true` → génère une réponse quel que soit le sentiment.  
  - `model` : permet de choisir dynamiquement le modèle Mistral (ex. `mistral-medium-latest`).

  Exemple de réponse :
  ```json
  {
    "sentiment_id": 0,
    "sentiment_label": "negatif",
    "generated": true,
    "response_text": "… réponse Mistral …"
  }
  ```

---

## Interface Gradio

Pour une démo rapide, `notebooks/gradio_ui.py` fournit une interface web simple.

1. Lancer l'API dans un premier terminal :

```bash
cd Shirin
.\.venv\Scripts\activate
uvicorn api:app --reload
```

2. Lancer l'interface Gradio dans un second terminal :

```bash
cd Shirin\notebooks
.\.venv\Scripts\activate
python gradio_ui.py
```

L'interface permet :
- de saisir un avis client en français,  
- de choisir le ton de la réponse (`polite` ou `sarcasm`),  
- d'afficher le sentiment prédit (`negatif / neutre / positif`),  
- si l'avis est négatif, de voir la réponse générée par Mistral dans un grand champ texte.

---

## Fonction `clean_review(text)`

La fonction `clean_review` (dans `notebooks/data_preprocessing.py`) est utilisée :

- lors de la préparation des données (création de `clean_text`),  
- lors de l'inférence dans l'API FastAPI (nettoyage des avis reçus avant classification).

Signature :

```python
def clean_review(text: str, stop_words=None, lemmatizer=None) -> str:
    """
    Nettoie une review texte (lowercase, suppression de ponctuation,
    stopwords adaptés au français, lemmatisation simple, normalisation).
    Renvoie une version prête pour la vectorisation.
    """
```

Cette fonction est le cœur de la phase de prétraitement et doit rester cohérente entre entraînement et inference.

