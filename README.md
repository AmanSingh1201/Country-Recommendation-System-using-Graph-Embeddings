🌍 Country Recommendation System using Graph Embeddings

A Graph Machine Learning project that models countries as nodes in a similarity graph and recommends similar countries based on LinkedIn user adoption patterns.

🚀 Overview

This project builds an end-to-end recommendation system using graph-based feature engineering, dimensionality reduction, and machine learning.

227 countries modeled as graph nodes
Edges created using cosine similarity
20-dimensional feature vectors combining graph + statistical features
PCA for dimensionality reduction
Random Forest classifier for behavior prediction
FAISS for fast similarity search
FastAPI + Streamlit UI for real-time interaction
🧠 Problem Statement

Given a country, recommend the most similar countries based on:

User adoption patterns
Growth trends
Graph structural relationships
🏗️ System Architecture

Pipeline:

Raw Data → Feature Engineering → Graph Construction → Graph Metrics
→ k-hop Features → PCA → Clustering → FAISS Index → API/UI
📊 Key Features
📌 Graph construction using cosine similarity
📌 20 engineered features (node + graph + neighborhood + interaction)
📌 PCA (up to 94% variance retained)
📌 Random Forest classification
📌 FAISS-based nearest neighbor search (<5ms latency)
📌 REST API using FastAPI
📌 Interactive UI using Streamlit
📈 Results
✅ Accuracy: 98.25%
✅ Cross-validation: 99.41%
✅ F1 Score: 0.982
✅ PCA Variance Retained: 94.31%
✅ Query Time: < 5ms
🛠️ Tech Stack
Python
pandas, numpy
NetworkX
scikit-learn
FAISS
FastAPI
Streamlit
matplotlib
📂 Project Structure
├── app.py                          # Streamlit UI
├── linkedin_graph_ml_bda.py        # Main pipeline
├── linkedin-users-by-country.csv   # Dataset
├── report.pdf                      # Project report
├── README.md
▶️ How to Run
1. Clone the repository
git clone https://github.com/your-username/country-recommendation-graph-ml.git
cd country-recommendation-graph-ml
2. Install dependencies
pip install -r requirements.txt
3. Run the Streamlit app
streamlit run app.py
4. Upload dataset in UI and explore
🔍 Sample Output
Input: India
Output: China, USA, Brazil
📌 Applications
Market expansion strategy
Social network analysis
Recommendation systems
Graph-based analytics
⚠️ Limitations
Small dataset (227 nodes)
Static graph (no temporal modeling)
No deep learning (GNNs not used)
🚀 Future Improvements
Add Node2Vec / Graph Neural Networks (GNNs)
Use larger real-world graph datasets
Add real-time streaming data
Deploy using Docker + Cloud
👨‍💻 Author

Aman (25MA60R31)
IIT Kharagpur – Big Data Analytics Project

⭐ If you like this project

Give it a star ⭐ on GitHub!

If you want, I can also:

Add badges (GitHub, Streamlit live link)
Write requirements.txt
Make this README FAANG-level polished (next level)
