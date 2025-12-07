#!/bin/bash

# Spam Detection - Cleanup Script for Deployment
# This script removes unnecessary files and keeps only production-required files

echo "🧹 Cleaning up project directory for deployment..."
echo ""

# Files to delete
TO_DELETE=(
    "LANDING_PAGE_DOCUMENTATION.md"
    "PREPROCESSING_ANALYSIS.md"
    "PREPROCESSING_QUICK_REFERENCE.md"
    "Preprocessing'.ipynb"
    "sample_emails.txt"
    "server.log"
    ".DS_Store"
)

echo "❌ Deleting unnecessary files:"
for file in "${TO_DELETE[@]}"; do
    if [ -f "$file" ]; then
        rm -v "$file"
        echo "   ✓ Deleted: $file"
    fi
done

# Cleanup cache
echo ""
echo "🔧 Cleaning Python cache..."
find . -type d -name "__pycache__" -exec rm -rv {} + 2>/dev/null || true
find . -type f -name "*.pyc" -delete
find . -type f -name ".DS_Store" -delete
echo "   ✓ Cache cleaned"

echo ""
echo "📁 Keeping required files:"
echo "   ✓ app.py"
echo "   ✓ requirements.txt"
echo "   ✓ final.csv"
echo "   ✓ README.md"
echo "   ✓ DEPLOYMENT_STRUCTURE.md"
echo "   ✓ .gitignore"
echo "   ✓ templates/ (landing.html, index.html, dashboard.html)"
echo "   ✓ static/ (style.css + chart images)"
echo "   ✓ model_out/ (best_model.pt, scaler.joblib)"

echo ""
echo "✅ Cleanup complete! Project ready for deployment."
echo ""
echo "📊 Remaining files:"
ls -lah | grep -v "^d" | grep -v "^total"
