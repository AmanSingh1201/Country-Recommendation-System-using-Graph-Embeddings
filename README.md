# 🌍 Country Recommendation System using Graph Embeddings

![Python](https://img.shields.io/badge/Python-3.10-blue)
![ML](https://img.shields.io/badge/Machine%20Learning-Graph%20ML-green)
![Framework](https://img.shields.io/badge/UI-Streamlit-red)
![API](https://img.shields.io/badge/API-FastAPI-orange)
![License](https://img.shields.io/badge/Status-Academic%20Project-lightgrey)

---

## 🚀 Overview

This project implements an **end-to-end Graph Machine Learning pipeline** to recommend similar countries based on LinkedIn user adoption patterns.

Countries are modeled as nodes in a graph, where relationships are learned using **cosine similarity, graph features, and embeddings**.

---

## 🧠 Problem Statement

Given a country, recommend the most similar countries based on:

- User adoption behavior  
- Growth trends  
- Graph structural relationships  

---

## 🏗️ System Architecture
Raw Data → Feature Engineering → Graph Construction → Graph Metrics
→ k-hop Features → PCA → Clustering → FAISS Index → API/UI


---

## ⚙️ Key Features

- 🌐 Graph construction using cosine similarity  
- 📊 20-dimensional feature engineering  
- 📉 PCA dimensionality reduction (94% variance retained)  
- 🌳 Random Forest classifier  
- ⚡ FAISS similarity search (<5ms latency)  
- 🔌 FastAPI for backend API  
- 🖥️ Streamlit interactive UI  

---

## 📊 Results

| Metric | Value |
|------|------|
| Accuracy | 98.25% |
| Cross Validation | 99.41% |
| F1 Score | 0.982 |
| PCA Variance Retained | 94.31% |
| Query Time | < 5ms |

---

## 🛠️ Tech Stack

- **Language:** Python  
- **Libraries:** pandas, numpy, NetworkX, scikit-learn  
- **ML:** Random Forest, PCA  
- **Graph ML:** NetworkX  
- **Search:** FAISS  
- **Backend:** FastAPI  
- **Frontend:** Streamlit  

---

## 📂 Project Structure
├── app.py # Streamlit UI

├── linkedin_graph_ml_bda.py # Main pipeline

├── linkedin-users-by-country-2024.csv

├── report.pdf # Full report

├── requirements.txt

└── README.md


---

## ▶️ How to Run

### 1️⃣ Clone Repository
git clone https://github.com/your-username/country-recommendation-graph-ml.git

cd country-recommendation-graph-ml

### 2️⃣ Install Dependencies

### 3️⃣ Run App

### 4️⃣ Upload dataset in UI

---

## 📦 requirements.txt

Create a file named `requirements.txt` and paste this:
pandas
numpy
networkx
scikit-learn
matplotlib
faiss-cpu
fastapi
uvicorn
streamlit

---

## 🔍 Sample Output

| Input Country | Recommended Countries |
|--------------|----------------------|
| India | China, USA, Brazil |
| Germany | France, UK, Italy |
| Nigeria | Kenya, South Africa, Egypt |

---

## 📌 Applications

- Market expansion strategy  
- Social network analysis  
- Recommendation systems  
- Graph-based analytics  

---

## ⚠️ Limitations

- Small dataset (227 countries)  
- Static graph (no temporal modeling)  
- No deep learning (GNNs not implemented)  

---

## 🚀 Future Improvements

- Add **Node2Vec / Graph Neural Networks (GNNs)**  
- Use large-scale real-world datasets  
- Add real-time streaming data  
- Deploy using Docker + Cloud  
- Improve recommendation ranking  

---

## 👨‍💻 Author

**Aman**  
CSE 
IIT Kharagpur  

---

## 🔗 Project Links

- GitHub Repo: *(Add your link here)*  
- Live Demo: *(Streamlit Cloud link if deployed)*  

---

## ⭐ Support

If you found this project useful, please give it a ⭐ on GitHub!

---


