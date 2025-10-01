---
title: "Argument Mining Backend"
emoji: "🧠"
colorFrom: "blue"
colorTo: "green"
sdk: "python"
sdk_version: "3.10"
app_file: app.py
pinned: false
---

# 🧠 Argument Mining Backend (FastAPI)

Ce backend expose une API REST pour prédire les relations argumentatives (Support / Attack) entre des phrases.
Il est conçu pour être utilisé avec un frontend (par exemple un site en Next.js).

---

## 🚀 Fonctionnalités
- Endpoint `/predict-text` : prédiction sur deux arguments donnés à la main.  
- Endpoint `/predict-csv` : prédiction sur un fichier CSV contenant des paires d’arguments.  
- Prétraitement basique du texte avant passage au modèle.  
- Chargement d’un modèle sauvegardé (`model.pkl`, `.pt`, etc.).

---

## 📦 Installation

1. Cloner le dépôt :

```bash
git clone https://github.com/<ton-user>/argument-backend.git
cd argument-backend
```

### Option 1 : via pip (classique)

```bash
python -m venv venv
source venv/bin/activate   # Linux / Mac
venv\Scripts\activate      # Windows

pip install -r requirements.txt
```

### Option 2 : via Conda (recommandé)

Créer l’environnement à partir du fichier environment.yml :

```bash
conda env create -f environment.yml
conda activate argument-backend
```

3. Installer les dépendances :

```bash
pip install -r requirements.txt
```

## ▶️ Lancer en local

```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

L’API sera dispo sur : http://127.0.0.1:8000
Et la doc Swagger automatique ici : http://127.0.0.1:8000/docs

## 📂 Structure du projet

argument-backend/
│── app.py             # API FastAPI
│── model_utils.py     # Chargement du modèle + prédiction
│── requirements.txt   # Dépendances Python
│── model.pkl          # (à ajouter) modèle sauvegardé

## 🌍 Déploiement sur Render

Pousser ce repo sur GitHub.

Sur Render
, créer un New Web Service → connecter le repo.

Paramètres Render :

- Environment: Python 3.x
- Build Command: pip install -r requirements.txt
- Start Command:
uvicorn app:app --host 0.0.0.0 --port 10000

Une fois déployé, Render donnera une URL publique du type :
https://argument-backend.onrender.com

