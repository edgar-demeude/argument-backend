---
title: "Argument Mining Backend"
emoji: "🧠"
colorFrom: "blue"
colorTo: "green"
sdk: "docker"
sdk_version: "3.10"
app_file: app.py
pinned: false
---

# 🧠 Argument Mining Backend (FastAPI)

Argument Mining Backend is a REST API designed to predict argumentative relations (Support / Attack) between sentences and analyze text using the ABA (Assumption-Based Argumentation) framework.
It is intended to be used alongside a frontend application (e.g., Next.js or React).

---

## 🚀 Features
- Endpoint `/predict-text` : Predict the relation between two manually provided arguments.
- Endpoint `/predict-csv` : Predict relations from a CSV file containing argument pairs.
- Endpoint `/aba-upload` : Upload a text file to generate and analyze an ABA+ framework.
- Endpoint `/aba-example` : Run ABA analysis on predefined example text.
- Endpoint `/aba-exemple/{filename}` : Analyze ABA with a predefined text file.
- Text preprocessing before feeding data into the model.  
- Chargement d’un modèle sauvegardé (`model.pkl`, `.pt`, etc.).
- Load saved machine learning models (.pt, .pkl, etc.) for inference.
- Automatic Swagger documentation for easy API exploration.
- CORS middleware for cross-origin requests.

---

## 📦 Installation

## Clone the repository

```bash
git clone https://github.com/<ton-user>/argument-backend.git
cd argument-backend
```

## Setup environment

### Option 1: Using Conda (recommended)

```bash
conda env create -f environment.yml
conda activate argument-backend
```

### Option 2: Using Python venv

```bash
python -m venv venv
source venv/bin/activate   # Linux / Mac
venv\Scripts\activate      # Windows

pip install -r requirements.txt
```

## ▶️ Running Locally

```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

- API will be available at: http://127.0.0.1:8000
- Swagger UI documentation: http://127.0.0.1:8000/docs

## 📂 Project Structure

```bash
argument-backend/
│── app.py      # FastAPI application entrypoint
│── aba         # ABA framework modules
│── relations   # Argument relation prediction modules
│── models      # Saved ML models (.pth, .pkl, etc.)
```

## ⚡Notes

- Designed for seamless integration with a frontend application for visualizing argument graphs.
- Supports batch prediction with CSV files (limited to 100 rows per request).
- ABA+ framework generation supports assumptions, arguments, attacks, and reverse attacks.

## 🌐 Live Demo

Check the live frontend here: [Arguments Visualization](https://arguments-visualisation.vercel.app/)