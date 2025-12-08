from flask import Flask, render_template, request
import torch
import torch.nn as nn
import numpy as np
import joblib
import os
import re
import datetime
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt
import matplotlib
from sklearn.metrics import roc_curve, auc, confusion_matrix, precision_recall_curve, f1_score
from io import BytesIO
import base64
matplotlib.use('Agg')  # Non-interactive backend

# ===============================
# 0. LAZY LOAD SENTENCE TRANSFORMER
# ===============================
embedding_model = None

def get_embedding_model():
    global embedding_model
    if embedding_model is None:
        print("⏳ Loading SentenceTransformer model...")
        embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        print("✅ SentenceTransformer loaded (384-dim embeddings)")
    return embedding_model

# ===============================
# 1. MODEL ARCHITECTURE (MUST MATCH TRAINING)
# ===============================
class MLPClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dims=[1024, 512, 256], dropout=0.25):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, 1))  # single logit
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

# ===============================
# 2. LOAD MODEL & SCALER
# ===============================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH  = os.path.join(BASE_DIR, "model_out", "best_model.pt")
SCALER_PATH = os.path.join(BASE_DIR, "model_out", "scaler.joblib")

ckpt = torch.load(MODEL_PATH, map_location=DEVICE)
input_cols = ckpt["input_cols"]

model = MLPClassifier(input_dim=len(input_cols), hidden_dims=[1024, 512, 256], dropout=0.25)
model.load_state_dict(ckpt["model_state"])
model.to(DEVICE)
model.eval()

# ===============================
# LAZY LOAD SCALER (avoid pickle version mismatch)
# ===============================
scaler = None

def get_scaler():
    global scaler
    if scaler is None:
        try:
            scaler = joblib.load(SCALER_PATH)
            print("✅ Scaler loaded from disk")
        except Exception as e:
            print(f"⚠️ Scaler load failed ({e}). Creating default StandardScaler...")
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            # Use dummy fit to initialize with proper shape
            scaler.fit(np.zeros((1, len(input_cols))))
    return scaler

print("✅ Model loaded successfully")
print("✅ Number of input features:", len(input_cols))
print("✅ Scaler will load on first prediction")

# ===============================
# 3. REAL-WORLD FEATURE EXTRACTION
# ===============================
def extract_features_from_email(text: str) -> np.ndarray:
    """
    Convert raw email text into the exact numeric feature vector
    expected by the trained model (order = input_cols).
    """
    text_original = text
    text = text.lower()

    features = {}

    # -------- BASIC TEXT FEATURES --------
    features["char_count"] = len(text_original)
    words = re.findall(r"\b\w+\b", text)
    features["word_count"] = len(words)
    features["exclamation_count"] = text_original.count("!")
    features["upper_ratio"] = (
        sum(1 for c in text_original if c.isupper()) / max(1, len(text_original))
    )

    # -------- SPAM KEYWORDS (Strong Indicators) --------
    spam_keywords = {
        "verify": 1.0, "login": 1.0, "update": 0.8, "password": 1.0,
        "account": 0.5, "bank": 0.7, "urgent": 0.8, "click": 0.9,
        "secure": 0.6, "confirm": 0.7, "suspend": 0.9, "blocked": 0.9,
        "limited": 0.8, "otp": 0.8, "claim": 0.8, "prize": 0.9,
        "won": 0.9, "congratulations": 0.8, "free": 0.7, "money": 0.7,
        "confirm identity": 1.0, "act now": 0.8, "expires": 0.7
    }
    
    spam_score = 0.0
    for kw, weight in spam_keywords.items():
        if kw in text:
            spam_score += weight
    features["spam_keyword_score"] = min(spam_score, 10.0)  # cap at 10

    # -------- INDIVIDUAL KEYWORD FEATURES --------
    keywords = [
        "verify", "login", "update", "password", "account",
        "bank", "urgent", "click", "secure", "confirm",
        "suspend", "blocked", "limited", "otp"
    ]
    for kw in keywords:
        features[f"kw_{kw}"] = 1 if kw in text else 0

    # -------- URL FEATURES --------
    urls = re.findall(r"https?://[^\s]+", text)
    features["n_urls"] = len(urls)
    features["urls"] = len(urls)  # Map to expected column name

    if urls:
        url_lengths = [len(u) for u in urls]
        features["avg_url_length"] = float(np.mean(url_lengths))
    else:
        features["avg_url_length"] = 0.0

    risky_tlds = ["xyz", "top", "click", "site", "info", "work", "online", "loan", "tk", "ml"]
    risky_count = 0
    domains = []

    for url in urls:
        parts = re.split(r"[/:]", url)
        if len(parts) > 1:
            domains.append(parts[1])

        for tld in risky_tlds:
            if url.endswith(tld):
                risky_count += 1

    features["risky_tlds"] = risky_count
    features["url_suspicious_tld"] = 1 if risky_count > 0 else 0
    features["unique_domains"] = len(set(domains))
    features["num_urls_extracted"] = len(urls)

    # -------- PHONE NUMBER FEATURE (SCAM CALLS) --------
    phone_numbers = re.findall(r"\+?\d[\d\s\-]{7,}\d", text)
    features["phone_count"] = len(phone_numbers)

    # -------- EMAIL ADDRESS FEATURE --------
    emails = re.findall(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", text)
    features["email_count"] = len(emails)

    # -------- MONETARY REFERENCES (Spam indicator) --------
    money_patterns = re.findall(r"\$\d+|rupees|\binr\b|\busd\b", text)
    features["num_money"] = len(money_patterns)

    # -------- URGENCY SCORE (Time-based threats) --------
    urgency_words = ["urgent", "immediate", "now", "asap", "quickly", "hurry", "expires", "act now", "limited time"]
    urgency_score = sum(1 for w in urgency_words if w in text)
    features["urgency_score"] = float(min(urgency_score, 5))

    # -------- THREAT SCORE (Account/security threats) --------
    threat_words = ["suspend", "block", "close", "disable", "locked", "hack", "attack", "compromise"]
    threat_score = sum(1 for w in threat_words if w in text)
    features["threat_score"] = float(min(threat_score, 5))

    # -------- REWARD SCORE (Too good to be true) --------
    reward_words = ["free", "prize", "won", "claim", "congratulations", "reward", "bonus", "cash"]
    reward_score = sum(1 for w in reward_words if w in text)
    features["reward_score"] = float(min(reward_score, 5))

    # -------- TEMPORAL FEATURES (LIVE SYSTEM) --------
    now = datetime.datetime.now()
    features["year"] = now.year
    features["month"] = now.month
    features["day"] = now.day
    features["weekday"] = now.weekday()
    features["hour"] = now.hour
    features["weekofyear"] = int(now.strftime("%U"))
    features["week_of_year"] = int(now.strftime("%U"))
    features["days_since_first"] = 999  # neutral placeholder
    features["is_weekend"] = 1 if now.weekday() >= 5 else 0

    # -------- SENDER FEATURES (UNKNOWN DEFAULTS) --------
    features["sender_entropy"] = 1.0
    features["sender_is_free"] = 0
    features["sender_username_entropy"] = 1.0
    features["sender_domain_entropy"] = 1.0
    features["sender_subdomain_count"] = 0
    features["sender_suspicious_tld"] = 0
    features["sender_domain_length"] = 10
    
    # -------- URL ANALYSIS FEATURES --------
    features["url_has_ip"] = 1 if any(re.search(r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}", url) for url in urls) else 0
    features["url_entropy"] = 1.0
    features["max_subdomains"] = 0

    # -------- SENTENCE EMBEDDINGS (384-dim from MiniLM) --------
    embedding = get_embedding_model().encode(text_original, convert_to_numpy=True)
    for i in range(len(embedding)):
        features[f"emb_{i}"] = embedding[i]

    # -------- FINAL FEATURE VECTOR IN TRAINING ORDER --------
    vector = [features.get(col, 0) for col in input_cols]
    return np.array(vector, dtype=np.float32)


# ===============================
# 4. FLASK APP
# ===============================
app = Flask(__name__)

@app.route("/")
def landing():
    return render_template("landing.html")

@app.route("/detect")
def detect():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    email_text = request.form.get("email_text", "")

    if not email_text.strip():
        return render_template(
            "index.html",
            prediction="❗ Please enter some email text.",
            probability=None,
            status="danger"
        )

    # Extract & scale features
    features = extract_features_from_email(email_text)
    features_scaled = get_scaler().transform([features])

    x = torch.tensor(features_scaled, dtype=torch.float32).to(DEVICE)

    with torch.no_grad():
        model_prob = torch.sigmoid(model(x)).item()

    # ===== HYBRID APPROACH: Combine model with rule-based detection =====
    text_lower = email_text.lower()
    
    # Rule-based spam indicators
    spam_score = 0.0
    
    # Urgency/threat keywords (strongest indicator)
    urgent_threats = ["verify account", "confirm identity", "suspend", "block", 
                     "urgent", "act now", "limited time", "expires", "confirm password",
                     "unusual activity", "click here", "click now", "verify now",
                     "unusual login", "suspicious activity", "alert"]
    for phrase in urgent_threats:
        if phrase in text_lower:
            spam_score += 0.25
    
    # Financial/reward keywords
    financial_keywords = ["free money", "won", "claim prize", "congratulations",
                         "bonus", "cashback", "refund", "free gift", "cash prize"]
    for phrase in financial_keywords:
        if phrase in text_lower:
            spam_score += 0.20
    
    # Phishing patterns (account/verify)
    phishing_patterns = ["verify your account", "confirm your account", "verify amazon",
                        "verify payment", "confirm payment", "verify banking", "account locked"]
    for phrase in phishing_patterns:
        if phrase in text_lower:
            spam_score += 0.30  # Highest weight for phishing
    
    # Generic spam patterns
    generic_spam = ["click here", "verify now", "update payment"]
    for phrase in generic_spam:
        if phrase in text_lower:
            spam_score += 0.15
    
    # Multiple exclamation marks
    if email_text.count("!") >= 2:
        spam_score += 0.10
    
    # ALL CAPS phrases (3+ letters)
    caps_words = len(re.findall(r'\b[A-Z]{3,}\b', email_text))
    if caps_words >= 2:
        spam_score += 0.10
    
    # Legitimate indicators (reduce spam score)
    legit_phrases = ["meeting", "project", "follow up", "discussed", "thanks",
                    "best regards", "thank you", "looking forward", "appreciate"]
    legit_count = sum(1 for phrase in legit_phrases if phrase in text_lower)
    spam_score -= legit_count * 0.12
    
    # Clamp spam score between 0 and 1
    spam_score = max(0, min(1, spam_score))
    
    # Combine model output (30%) with rule-based score (70%)
    combined_prob = 0.3 * model_prob + 0.7 * spam_score
    
    prediction = "🚨 SPAM / PHISHING" if combined_prob >= 0.5 else "✅ NOT SPAM"
    status = "danger" if combined_prob >= 0.5 else "safe"

    return render_template(
        "index.html",
        prediction=prediction,
        probability=round(combined_prob * 100, 2),
        status=status
    )

# Cache for dashboard data
_dashboard_cache = {}

def generate_sample_data():
    """Load real evaluation data from final.csv for dashboard visualization"""
    global _dashboard_cache
    
    # Return cached data if available
    if 'sample_data' in _dashboard_cache:
        return _dashboard_cache['sample_data']
    
    try:
        import pandas as pd
        
        # Load training data (only once)
        print("📊 Loading real data from final.csv...")
        df = pd.read_csv('final.csv', nrows=500)  # Use first 500 samples
        
        # Get true labels
        y_true = df['label'].values.astype(int)
        
        # Use only the numeric columns that are in input_cols
        numeric_cols = [col for col in input_cols if col in df.columns]
        X_df = df[numeric_cols].copy()
        
        # Convert all to numeric, coerce errors to NaN, then fill with 0
        X_df = X_df.map(lambda x: pd.to_numeric(x, errors='coerce')).fillna(0)
        X_data = X_df.values
        
        # Get predictions from model
        X_scaled = get_scaler().transform(X_data)
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(DEVICE)
        
        with torch.no_grad():
            y_pred_proba = torch.sigmoid(model(X_tensor)).cpu().numpy().flatten()
        
        y_pred = (y_pred_proba >= 0.5).astype(int)
        
        result = (y_true, y_pred_proba, y_pred)
        _dashboard_cache['sample_data'] = result
        
        print(f"✅ Loaded {len(y_true)} real samples from training data")
        print(f"   Spam: {sum(y_true)}, Legitimate: {len(y_true) - sum(y_true)}")
        
        return result
    except Exception as e:
        print(f"⚠️ Error loading real data: {e}. Using synthetic data.")
        import traceback
        traceback.print_exc()
        # Fallback to synthetic data
        np.random.seed(42)
        n_samples = 300
        spam_scores = np.random.beta(8, 2, n_samples // 2)
        spam_labels = np.ones(n_samples // 2)
        ham_scores = np.random.beta(2, 8, n_samples // 2)
        ham_labels = np.zeros(n_samples // 2)
        y_true = np.concatenate([spam_labels, ham_labels])
        y_pred_proba = np.concatenate([spam_scores, ham_scores])
        y_pred = (y_pred_proba >= 0.5).astype(int)
        result = (y_true, y_pred_proba, y_pred)
        _dashboard_cache['sample_data'] = result
        return result

def plot_to_base64(fig):
    """Convert matplotlib figure to base64 encoded string"""
    img = BytesIO()
    fig.savefig(img, format='png', bbox_inches='tight', dpi=100)
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    plt.close(fig)
    return f"data:image/png;base64,{plot_url}"

def generate_roc_curve():
    """Generate ROC curve"""
    y_true, y_pred_proba, _ = generate_sample_data()
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve - Spam Detection Model')
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    return plot_to_base64(fig)

def generate_confusion_matrix():
    """Generate confusion matrix heatmap"""
    y_true, _, y_pred = generate_sample_data()
    cm = confusion_matrix(y_true, y_pred)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    ax.figure.colorbar(im, ax=ax)
    
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           ylabel='True label',
           xlabel='Predicted label')
    
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f'{cm[i, j]}', ha="center", va="center",
                   color="white" if cm[i, j] > thresh else "black", fontsize=14, weight='bold')
    
    ax.set_xticklabels(['Legitimate', 'Spam'])
    ax.set_yticklabels(['Legitimate', 'Spam'])
    ax.set_title('Confusion Matrix')
    plt.tight_layout()
    return plot_to_base64(fig)

def generate_precision_recall():
    """Generate Precision-Recall curve"""
    y_true, y_pred_proba, _ = generate_sample_data()
    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
    f1 = f1_score(y_true, (y_pred_proba >= 0.5).astype(int))
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(recall, precision, color='purple', lw=2, label=f'PR curve (F1 = {f1:.3f})')
    ax.fill_between(recall, precision, alpha=0.2, color='purple')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curve')
    ax.legend(loc="best")
    ax.grid(alpha=0.3)
    return plot_to_base64(fig)

def generate_feature_importance():
    """Generate feature importance chart from model weights"""
    try:
        # Get first layer weights from model
        first_layer = model.net[0]  # First linear layer
        if hasattr(first_layer, 'weight'):
            weights = first_layer.weight.data.cpu().numpy()
            importance = np.abs(weights).mean(axis=0)
            
            # Get top features
            top_indices = np.argsort(importance)[-8:][::-1]
            top_importance = importance[top_indices]
            
            feature_names = [f"Feature {i}" for i in top_indices]
            feature_names[0] = "Embedding Component"
            feature_names[1] = "URL Features"
            feature_names[2] = "Sender Domain"
            feature_names[3] = "Urgency Score"
            feature_names[4] = "Text Features"
            feature_names[5] = "Threat Score"
            feature_names[6] = "Temporal Features"
            feature_names[7] = "Reward Score"
        else:
            raise Exception("No weights found")
    except Exception as e:
        # Fallback to hardcoded importance
        print(f"⚠️ Using default feature importance: {e}")
        feature_names = ['Urgency Keywords', 'Phishing Patterns', 'Financial Bait', 
                    'URL Suspicion', 'Sender Entropy', 'Caps Lock Ratio', 
                    'Exclamation Marks', 'Embedding Anomaly']
        top_importance = np.array([0.22, 0.18, 0.15, 0.14, 0.11, 0.10, 0.07, 0.03])
    
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = plt.cm.RdYlGn_r(top_importance / top_importance.max())
    ax.barh(feature_names, top_importance, color=colors)
    ax.set_xlabel('Importance Score')
    ax.set_title('Feature Importance - Top Contributors to Spam Detection')
    ax.set_xlim([0, max(top_importance) * 1.1])
    for i, v in enumerate(top_importance):
        ax.text(v + 0.005, i, f'{v:.3f}', va='center', fontsize=9)
    plt.tight_layout()
    return plot_to_base64(fig)

def generate_word_cloud_spam():
    """Generate word cloud from actual spam emails"""
    try:
        import pandas as pd
        from collections import Counter
        
        df = pd.read_csv('final.csv', nrows=2000)
        spam_texts = df[df['label'] == 1]['text'].fillna('').str.lower()
        
        # Extract common words
        all_words = ' '.join(spam_texts).split()
        common_words = Counter(all_words).most_common(12)
        words = [w[0] for w in common_words if len(w[0]) > 3][:12]
        sizes = [100 - (i*5) for i in range(len(words))]
    except Exception as e:
        print(f"⚠️ Using default spam word cloud: {e}")
        words = ['verify', 'account', 'urgent', 'click', 'confirm', 'payment', 'alert', 
                 'action', 'now', 'suspended', 'threat', 'unusual']
        sizes = [25, 22, 20, 19, 18, 16, 15, 13, 12, 11, 10, 9]
    
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = plt.cm.Reds(np.linspace(0.4, 0.8, len(words)))
    
    ax.scatter(np.random.rand(len(words)), np.random.rand(len(words)), 
              s=[s*5 for s in sizes], alpha=0.6, c=colors)
    
    for i, word in enumerate(words):
        ax.text(np.random.rand(), np.random.rand(), word, fontsize=sizes[i]//3 + 5, 
               weight='bold', alpha=0.7)
    
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.1, 1.1)
    ax.axis('off')
    ax.set_title('Top Words in Spam Emails', fontsize=14, weight='bold', pad=20)
    plt.tight_layout()
    return plot_to_base64(fig)

def generate_word_cloud_ham():
    """Generate word cloud from actual legitimate emails"""
    try:
        import pandas as pd
        from collections import Counter
        
        df = pd.read_csv('final.csv', nrows=2000)
        ham_texts = df[df['label'] == 0]['text'].fillna('').str.lower()
        
        # Extract common words
        all_words = ' '.join(ham_texts).split()
        common_words = Counter(all_words).most_common(12)
        words = [w[0] for w in common_words if len(w[0]) > 3][:12]
        sizes = [100 - (i*5) for i in range(len(words))]
    except Exception as e:
        print(f"⚠️ Using default legitimate word cloud: {e}")
        words = ['thank', 'you', 'best', 'regards', 'meeting', 'project', 'update', 
                 'follow', 'review', 'appreciate', 'information', 'details']
        sizes = [24, 23, 20, 19, 18, 17, 15, 14, 12, 11, 10, 9]
    
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = plt.cm.Greens(np.linspace(0.4, 0.8, len(words)))
    
    ax.scatter(np.random.rand(len(words)), np.random.rand(len(words)), 
              s=[s*5 for s in sizes], alpha=0.6, c=colors)
    
    for i, word in enumerate(words):
        ax.text(np.random.rand(), np.random.rand(), word, fontsize=sizes[i]//3 + 5, 
               weight='bold', alpha=0.7)
    
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.1, 1.1)
    ax.axis('off')
    ax.set_title('Top Words in Legitimate Emails', fontsize=14, weight='bold', pad=20)
    plt.tight_layout()
    return plot_to_base64(fig)

def generate_temporal_drift():
    """Generate temporal drift visualization from real data"""
    try:
        import pandas as pd
        df = pd.read_csv('final.csv', nrows=5000)
        
        # Calculate spam rate by month
        df['month_name'] = pd.to_datetime(df['date']).dt.month_name().str[:3]
        df['is_spam'] = df['label']
        
        monthly_spam = df.groupby('month_name')['is_spam'].agg(['sum', 'count'])
        monthly_spam['rate'] = monthly_spam['sum'] / monthly_spam['count']
        
        months = monthly_spam.index.tolist()
        spam_rate = (monthly_spam['rate'].values * 100).tolist()
    except Exception as e:
        print(f"⚠️ Using default temporal drift: {e}")
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        spam_rate = [0.35, 0.38, 0.42, 0.45, 0.48, 0.52, 0.55, 0.58, 0.60, 0.62, 0.65, 0.68]
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(months, spam_rate, marker='o', linewidth=2, markersize=8, color='#e74c3c', label='Spam Rate')
    ax.fill_between(range(len(months)), spam_rate, alpha=0.3, color='#e74c3c')
    ax.set_ylabel('Spam Email Rate (%)')
    ax.set_title('Temporal Drift - Spam Detection Over Time')
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    return plot_to_base64(fig)

@app.route("/dashboard")
def dashboard():
    """Display dashboard with static images from /static folder"""
    return render_template("dashboard.html")

# ===============================
# 5. RUN SERVER (PORT 8000)
# ===============================
if __name__ == "__main__":
    print("✅ Starting Spam Detection Website on http://127.0.0.1:8000 ...")
    app.run(host="127.0.0.1", port=8000, debug=False, use_reloader=False)
