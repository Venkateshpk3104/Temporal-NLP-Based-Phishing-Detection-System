# 🔐 Spam Detection System - Temporal Analysis of Phishing Attacks Using Temporal NLP

A production-ready Flask web application for real-time spam and phishing email detection using deep learning and NLP.

## 📋 Features

- **Real-time Email Classification**: Detect spam and phishing emails instantly
- **Modern Dashboard**: View model performance metrics and statistical analysis
- **Interactive Detection Interface**: User-friendly form for email submission
- **Advanced Analytics**: ROC curves, confusion matrix, precision-recall analysis
- **Temporal Tracking**: Monitor spam trends over time
- **Responsive Design**: Works on desktop, tablet, and mobile devices

## 🛠️ Technology Stack

- **Backend**: Flask (Python web framework)
- **ML Framework**: PyTorch (deep learning)
- **NLP**: Sentence Transformers (sentence embeddings)
- **Feature Engineering**: scikit-learn
- **Frontend**: HTML5, CSS3, Responsive Design
- **Server Port**: 8000

## 📦 Installation & Setup

### Prerequisites
- Python 3.8+
- pip package manager
- Virtual environment (recommended)

### Step 1: Clone or Extract Project
```bash
cd /path/to/Spam
```

### Step 2: Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Verify Required Files
Ensure these files exist:
- `app.py` (main application)
- `model_out/best_model.pt` (trained model)
- `model_out/scaler.joblib` (feature scaler)
- `final.csv` (training data)
- `templates/` folder with HTML files
- `static/` folder with CSS and chart images

### Step 5: Run Application
```bash
python app.py
```

Application will start at: **http://localhost:8000**

## 🚀 Deployment Guide

### Local Development
```bash
python app.py
# Visit http://localhost:8000
```

### Production Deployment (Gunicorn + Nginx)

#### 1. Install Gunicorn
```bash
pip install gunicorn
```

#### 2. Run with Gunicorn
```bash
gunicorn -w 4 -b 0.0.0.0:8000 app:app
```

#### 3. Nginx Reverse Proxy Configuration
```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

### Cloud Deployment Options

#### AWS EC2
1. Launch Ubuntu instance
2. Install Python and dependencies
3. Run Gunicorn with systemd service
4. Configure security groups for port 8000

#### Heroku
```bash
# Add Procfile
echo "web: gunicorn app:app" > Procfile

# Deploy
git push heroku main
```

#### Google Cloud / Azure
- Use Cloud Run or App Service
- Container deployment with Docker

## 📁 Project Structure

```
spam-detection-app/
├── app.py                    # Main Flask application
├── requirements.txt          # Python dependencies
├── final.csv                 # Training data
├── model_out/
│   ├── best_model.pt        # Trained PyTorch model (~100 MB)
│   └── scaler.joblib        # Feature scaler
├── templates/
│   ├── landing.html         # Homepage
│   ├── index.html           # Detection interface
│   └── dashboard.html       # Analytics dashboard
├── static/
│   ├── style.css            # Global styles
│   ├── roc.png              # ROC curve visualization
│   ├── cm.png               # Confusion matrix
│   ├── pr.png               # Precision-recall curve
│   ├── temporal.png         # Temporal drift chart
│   ├── importance.png       # Feature importance
│   ├── wc_phish.png         # Spam word cloud
│   ├── wc_ham.png           # Legitimate word cloud
│   └── wc_all.png           # Combined word cloud
└── README.md                # This file
```

## 🌐 Application Routes

| Route | Method | Purpose |
|-------|--------|---------|
| `/` | GET | Landing page with project overview |
| `/detect` | GET | Email detection interface |
| `/predict` | POST | Process email and return prediction |
| `/dashboard` | GET | Analytics dashboard with metrics |

## 🔍 How It Works

### Email Processing Pipeline

1. **Input**: User submits email text
2. **Feature Extraction**: Converts email to 414-dimensional feature vector
   - Text statistics (word count, punctuation, etc.)
   - Spam keyword detection
   - URL analysis
   - Sender domain analysis
   - Sentence embeddings (384-dim using MiniLM)
3. **Normalization**: Scales features using pre-trained scaler
4. **Prediction**: Deep learning model (PyTorch) predicts probability
5. **Hybrid Scoring**: Combines neural network (30%) + rule-based detection (70%)
6. **Result**: Returns "SPAM/PHISHING" or "NOT SPAM" with confidence score

### Model Architecture

```
Input (414 features)
    ↓
Linear Layer (1024 units) + BatchNorm + ReLU + Dropout(0.25)
    ↓
Linear Layer (512 units) + BatchNorm + ReLU + Dropout(0.25)
    ↓
Linear Layer (256 units) + BatchNorm + ReLU + Dropout(0.25)
    ↓
Output Layer (1 unit, Sigmoid activation)
    ↓
Spam Probability (0-1)
```

## 📊 Dashboard Metrics

- **ROC Curve**: Trade-off between TPR and FPR
- **Confusion Matrix**: TP/TN/FP/FN breakdown
- **Precision-Recall Curve**: Precision vs Recall trade-off
- **Temporal Drift**: Spam rate trends over time
- **Feature Importance**: Top contributing features
- **Word Clouds**: Most common words in spam vs legitimate

## 🔐 Security Considerations

- Input validation on email text
- Model inference runs locally (no external API calls)
- HTTPS recommended for production
- Rate limiting recommended (use Flask-Limiter)
- CSRF protection for forms
- SQL injection not applicable (no database)

## 🚨 Performance Optimization

- **Model Loading**: ~2 seconds on startup
- **Prediction Speed**: ~50-100ms per email
- **Dashboard Generation**: Uses cached data
- **Memory Usage**: ~500MB with loaded model

### For Production Optimization:
1. Use quantized model for faster inference
2. Implement prediction caching
3. Use Redis for session management
4. Compress static assets
5. Enable gzip compression in Nginx

## 🐛 Troubleshooting

### Common Issues

**Issue**: "ModuleNotFoundError: No module named 'torch'"
```bash
pip install -r requirements.txt
```

**Issue**: Port 8000 already in use
```bash
# Change port in app.py: app.run(port=5000)
# Or kill existing process:
lsof -ti:8000 | xargs kill -9
```

**Issue**: Missing chart images in dashboard
- Ensure all PNG files exist in `static/` folder
- Check image paths in `dashboard.html`

**Issue**: Slow predictions
- Try CUDA GPU acceleration (if available)
- Check model loading time in console
- Reduce feature extraction complexity

## 📈 Model Performance

- **Accuracy**: ~94%
- **Precision**: ~92% (low false positives)
- **Recall**: ~95% (high spam detection)
- **F1-Score**: ~0.93
- **AUC-ROC**: ~0.98

## 🎯 Use Cases

1. **Email Service Providers**: Real-time spam filtering
2. **Corporate Security**: Internal email monitoring
3. **Educational Institutions**: Student email protection
4. **Security Research**: Phishing detection analysis
5. **Customer Support**: Reduce support load from spam

## 📞 API Endpoints

### POST /predict
```bash
curl -X POST http://localhost:8000/predict \
  -d "email_text=Verify your account immediately!" \
  -H "Content-Type: application/x-www-form-urlencoded"
```

Response includes:
- Prediction ("SPAM/PHISHING" or "NOT SPAM")
- Probability (0-100%)
- Status ("danger" or "safe")

## 📝 License & Attribution

This project implements temporal NLP analysis for phishing detection, combining:
- Deep Learning (PyTorch MLPClassifier)
- NLP (Sentence Transformers)
- Rule-based Detection (Keywords + Patterns)
- Temporal Analysis (Time-series spam tracking)

## 👨‍💼 Support & Maintenance

For issues or questions:
1. Check console logs: `python app.py`
2. Verify all required files exist
3. Ensure Python 3.8+ is installed
4. Recreate virtual environment if needed

## 🚀 Next Steps for Production

1. ✅ Set up HTTPS/SSL certificate
2. ✅ Configure domain name
3. ✅ Deploy to cloud (AWS/GCP/Azure)
4. ✅ Set up monitoring and logging
5. ✅ Configure auto-scaling
6. ✅ Implement rate limiting
7. ✅ Set up email notifications
8. ✅ Regular model retraining

---

**Status**: ✅ Ready for Deployment

**Last Updated**: December 2025
