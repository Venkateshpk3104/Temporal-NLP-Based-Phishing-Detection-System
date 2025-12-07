# Spam Detection Deployment - File Structure Guide

## Required Files for Deployment ✅

### Core Application Files
- `app.py` - Main Flask application
- `templates/landing.html` - Landing page
- `templates/index.html` - Spam detection interface
- `templates/dashboard.html` - Analytics dashboard
- `static/style.css` - Global styling

### Model & Data Files
- `model_out/best_model.pt` - Trained PyTorch model
- `model_out/scaler.joblib` - Feature scaler
- `final.csv` - Training data (for dashboard visualizations)

### Static Assets (Chart Images)
- `static/roc.png` - ROC curve visualization
- `static/cm.png` - Confusion matrix
- `static/pr.png` - Precision-recall curve
- `static/temporal.png` - Temporal drift chart
- `static/importance.png` - Feature importance chart
- `static/wc_phish.png` - Spam word cloud
- `static/wc_ham.png` - Legitimate word cloud
- `static/wc_all.png` - Combined word cloud

### Configuration Files
- `requirements.txt` - Python dependencies (TO CREATE)
- `.gitignore` - Git ignore rules (OPTIONAL)

## Files to Delete (Not Required) ❌

### Temporary/Cache Files
- `__pycache__/` - Python cache directory
- `.DS_Store` - macOS system file
- `server.log` - Log file
- `Preprocessing'.ipynb` - Old notebook file

### Documentation Files (Keep if needed)
- `LANDING_PAGE_DOCUMENTATION.md` - Can delete
- `PREPROCESSING_ANALYSIS.md` - Can delete
- `PREPROCESSING_QUICK_REFERENCE.md` - Can delete
- `sample_emails.txt` - Can delete

### Development Files
- `.venv/` - Virtual environment (DON'T delete locally, recreate on server)

---

## Recommended Folder Structure for Deployment

```
spam-detection-app/
├── app.py                    # Main Flask app
├── requirements.txt          # Dependencies
├── .gitignore               # Git ignore
├── model_out/               # Models
│   ├── best_model.pt
│   └── scaler.joblib
├── static/                  # Static assets
│   ├── style.css
│   ├── roc.png
│   ├── cm.png
│   ├── pr.png
│   ├── temporal.png
│   ├── importance.png
│   ├── wc_phish.png
│   ├── wc_ham.png
│   └── wc_all.png
├── templates/               # HTML templates
│   ├── landing.html
│   ├── index.html
│   └── dashboard.html
├── final.csv               # Training data
└── README.md               # Deployment instructions
```

---

## Deployment Steps

### 1. Create requirements.txt
```bash
Flask==2.3.3
torch==2.0.1
scikit-learn==1.3.0
sentence-transformers==2.2.2
joblib==1.3.1
numpy==1.24.3
pandas==2.0.3
matplotlib==3.7.2
```

### 2. Prepare files
- Remove unnecessary files
- Create README with setup instructions
- Ensure all chart images are in `/static/`

### 3. Deploy to Server
- Push to Git/GitHub
- Clone on server
- Install dependencies: `pip install -r requirements.txt`
- Run: `python app.py`

### 4. Production Deployment (with Gunicorn)
```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:8000 app:app
```

### 5. Server Configuration (Nginx/Apache)
- Setup reverse proxy to Flask app
- Configure SSL/HTTPS
- Set up domain DNS

---

## Storage Requirements

- **Models**: ~100 MB (best_model.pt)
- **Scaler**: ~1 MB (scaler.joblib)
- **Training Data**: ~50-100 MB (final.csv)
- **Static Assets**: ~10 MB (chart images)
- **Code**: ~2 MB
- **Total**: ~160-200 MB minimum

---

## Dependencies Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run application
python app.py
```

Application will be available at: `http://localhost:8000`
