
# 🧠 Emotion AI

A desktop-based **Emotion Intelligence AI** built with **Tkinter** and **AI/NLP models**.  
It analyzes user input text, detects emotional states (joy, sadness, anger, fear, etc.), provides **psychological insights**, and visualizes emotional trends over time.  

---

## 🚀 Features
- **Emotion Detection**  
  - Uses HuggingFace **DistilRoBERTa emotion classifier** (deep learning).  
  - Fallback system with **TF-IDF + sentiment analysis** when offline.  
- **Interactive GUI (Tkinter)**  
  - User-friendly input box and visualization panels.  
  - Dark/Light mode toggle.  
- **Psychological Insights**  
  - Displays coping strategies and insights for detected emotions.  
- **Visualization**  
  - Real-time bar charts for emotion intensity.  
  - Historical trends over time (line charts).  
- **Emotion History**  
  - Logs all past analyses with timestamp, emotion, intensity, and generated wisdom.  

---

## 🛠️ Tech Stack
- **Frontend (GUI):** Tkinter (Python standard GUI lib)  
- **Data & Viz:** Pandas, Matplotlib, NumPy  
- **NLP & ML:** TextBlob, Scikit-learn (TF-IDF), NLTK (lemmatizer), Transformers (HuggingFace), PyTorch  
- **Deep Learning Model:** [`j-hartmann/emotion-english-distilroberta-base`](https://huggingface.co/j-hartmann/emotion-english-distilroberta-base)  

---

## 📂 Project Structure
```

ProfessionalEmotionAI/
│
├── main.py                 # Main application code
├── requirements.txt        # Dependencies
├── .gitignore              # Ignore venv, cache, etc.
└── README.md               # Project documentation

````

---

## 🔧 Installation & Setup

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/your-username/ProfessionalEmotionAI.git
cd ProfessionalEmotionAI
````

### 2️⃣ Create a Virtual Environment

#### Windows (PowerShell):

```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

#### macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Running the App

```bash
python main.py
```

---

## 📊 Screenshots / Demo

*(You can add images later, e.g., emotion analysis window, bar chart, history log.)*

---

## 🤖 How It Works

1. **Input Preprocessing** – Cleans text (lowercasing, punctuation removal, lemmatization).
2. **Deep Learning Model** – If available, uses HuggingFace emotion classifier.
3. **Fallback Method** – If DL model not available, uses TF-IDF keyword matching + sentiment polarity.
4. **Visualization** – Displays bar chart for top 5 emotions and confidence scores.
5. **Psychological Insights** – Provides contextual coping strategies and motivational messages.
6. **History & Trends** – Stores logs in a Pandas DataFrame and visualizes trends over time.

---

## 📈 Future Enhancements

* 🌐 Add **speech-to-text** for spoken emotion analysis.
* 📡 Deploy as a **web app** (Flask/Streamlit).
* 💾 Export history to CSV/Excel for further analysis.
* 🔥 Add more robust models (e.g., BERT-based multi-lingual emotion detection).

---

## 📜 License

MIT License © 2025 Harish S

---

## 🙌 Acknowledgments

* [HuggingFace Transformers](https://huggingface.co/transformers/)
* [NLTK](https://www.nltk.org/)
* [TextBlob](https://textblob.readthedocs.io/)
* [Scikit-learn](https://scikit-learn.org/)

```

---

```
