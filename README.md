# 🛡️ Cyberbullying Detection System

An end-to-end, multi-generation Natural Language Processing (NLP) framework and Machine Learning pipeline engineered to automatically parse text streams, expand noisy internet colloquialisms, and precisely classify online toxicity and harassment.

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg?style=for-the-badge&logo=python)](https://www.python.org/)
[![Machine Learning](https://img.shields.io/badge/ML%2FDL-Scikit--learn%20%2F%20NLTK-orange.svg?style=for-the-badge&logo=scikit-learn)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)

---

## 📌 Overview

Online harassment presents an ongoing challenge to digital communication platforms. This repository houses an evolving NLP workflow that handles everything from raw dataset ingestion and programmatic token clean-up to serialization checkpointing and live user inference. 

Rather than deploying a static model, the project records an engineered evolutionary progression across **four dedicated system iterations**—moving from standard word chops to dynamic synthetic data generation modules.

---

## ⚡ Key Features & Version Evolution

The system architecture spans four progressive development increments, each advancing structural feature clean-up:

* **NLP Version 1.0 (Baseline Implementation)**: Sets automated validation checks to fetch missing NLTK requirements (`stopwords`, `porter`, `punkt`). Leverages standard alphanumeric string cleaning, lowercase balancing, and a baseline **Porter Stemmer** connected to a 5,000 max-feature **TF-IDF Vectorizer** and **Logistic Regression**.
* **NLP Version 2.0 (Slang Parsing & Sentiment Nuance)**: Appends complex Unicode regular expression masks targeting specific emoji and pictograph ranges. Maps common shorthand net-slang and toxic acronyms (e.g., `gtfo` $\rightarrow$ `get out`, `smh` $\rightarrow$ `shaking my head`) to build an expansive textual training matrix.
* **NLP Version 3.0 (Advanced Lemmatization Pipeline)**: Upgrades text roots by replacing crude stemming with contextual **WordNet Lemmatization**. Strips explicit Web URLs (`http`/`https`), user profile handles (`@username`), platform tags (`#hashtag`), and maps detailed contraction dictionaries (e.g., `can't've` $\rightarrow$ `cannot have`).
* **NLP Version 4.0 (Synthetic Factories & Data Ingestion Adapters)**: 
    * *Real-World CSV Loader (`load_real_data_from_csv`)*: Production-ready data adapters handling custom label mapping, datatype casting, and file safety diagnostics.
    * *Parametric String Synthesis Pool (`generate_synthetic_data`)*: Programmatic token factory that randomly generates balanced threat/neutral test blocks using localized template matrices to mock larger datasets.
    * *Deep Learning Blueprints*: Outlines configuration boundaries for deep learning migrations including pre-trained embeddings (`Word2Vec`/`GloVe`), recurrent layers (`LSTM`), and contextual transformers (`BERT`).

---

## 🛠️ Tech Stack & Dependencies

- **Core Language Engine**: Python 3.8+
- **Data Architecture Tools**: `Pandas`, `NumPy`
- **Text Processing & ML Frameworks**: `Scikit-learn`, `NLTK`, `Joblib`
- **Underlying Conceptual Models**: Logistic Regression, Term Frequency-Inverse Document Frequency (TF-IDF)

---

## 📂 Project Directory Structure

```text
├── .vscode/                                             # Local workspace configurations
├── Cyberbullying_Detection_NLP_Project_Version_1.0.py   # Core baseline engine
├── Cyberbullying_Detection_NLP_Project_Version_2.0.py   # Slang expansion layer
├── Cyberbullying_Detection_NLP_Project_Version_3.0.py   # Lemmatizer optimization block
├── Cyberbullying_Detection_NLP_Project_Version_4.0.py   # Advanced synthetic & data pipelines
├── cyberbullying_model.joblib                           # Serialized model weights 
├── tfidf_vectorizer.joblib                              # Serialized TF-IDF feature vocabulary
└── README.md                                            # Project documentation

```

---

## 🚀 Local Deployment Lifecycle

### 📋 Prerequisites

Verify your local environment has Python and package management managers configured:

```bash
python --version
pip --version

```

### ⚙️ Installation & Workspace Setup

1. **Clone the Repository**:
```bash
git clone [https://github.com/meetpotdar777/Cyberbullying-Detection.git](https://github.com/meetpotdar777/Cyberbullying-Detection.git)
cd Cyberbullying-Detection

```


2. **Set Up a Virtual Environment**:
```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# Linux/macOS
python3 -m venv venv
source venv/bin/activate

```


3. **Install Package Requirements**:
```bash
pip install pandas numpy scikit-learn joblib nltk

```



---

## 🖥️ Running the Application Pipelines

Each script can execute standalone. The underlying codebase incorporates smart data checkpoint lookups to eliminate repetitive text vector re-computations.

```bash
# Execute the latest ingestion pipeline featuring synthetic template generation:
python Cyberbullying_Detection_NLP_Project_Version_4.0.py

```

### 💾 Persistence Caching Checkpoint

When initializing execution blocks, the main runtime engine checks if serialized weights already exist locally before triggering data loads or retraining routines:

```python
if os.path.exists(MODEL_FILE) and os.path.exists(VECTORIZER_FILE):
    print(f"Loading existing model from {MODEL_FILE} and vectorizer from {VECTORIZER_FILE}...")
    trained_model = joblib.load(MODEL_FILE)
    tfidf_vectorizer = joblib.load(VECTORIZER_FILE)

```

---

## 📊 Processing Workflow Lifecycle

1. **Ingestion Pool**: Selects real-world CSV file targets or defaults back onto structural mock-text generation loops.
2. **Sanitizer Sweep**: Drops active web links, clears profile handles (`@\w+`), and cleans out trending tag components (`#\w+`).
3. **Contraction & Slang Translation**: Iterates across raw sentence words to expand complex grammar markers and map internet shorthand text.
4. **Context Lemmatization**: Splits clean vectors into tokens and implements strict **WordNet Lemmatizer** iterations to find root structural definitions.
5. **Matrix Transformation**: Translates clean arrays using a 5,000 max-feature sparse array configuration (`TfidfVectorizer`).
6. **Inference Classification**: Splits data arrays using balanced stratification (`stratify=y`) and evaluates through Logistic Regression, returning probability confidence bounds (`model.predict_proba`).

---

## 🤝 Contributing

Contributions to scale classification accuracy, append comprehensive slang dictionaries, or integrate advanced transformer layers are welcome!

1. **Fork** the project repository.
2. **Create** your feature branch (`git checkout -b feature/AmazingFeature`).
3. **Commit** your refinements (`git commit -m 'Add some AmazingFeature'`).
4. **Push** to the origin branch (`git push origin feature/AmazingFeature`).
5. Open a **Pull Request**.

---

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.

---
