# CIFAR-10 CNN Model Deployment

This folder contains all the files needed to deploy the CIFAR-10 image classification model.

## Files Included

### Core Application Files
- **`app.py`** - Flask web application with prediction endpoints
- **`requirements.txt`** - Python dependencies
- **`best_model_cifar10.pth`** - Trained CNN model for CIFAR-10 classification

### Deployment Configuration
- **`Procfile`** - Heroku deployment configuration
- **`railway.toml`** - Railway deployment configuration
- **`model_stride.pth`** - Alternative model file
- **`.gitignore`** - Git ignore rules for deployment

## API Endpoints

### Health Check
- **GET** `/health` - Returns server status

### Prediction
- **POST** `/predict` - Upload an image file to get classification prediction
  - Accepts image files (PNG, JPG, etc.)
  - Returns JSON with prediction, probability, and all class probabilities

## CIFAR-10 Classes
The model classifies images into one of these 10 classes:
- airplane
- automobile  
- bird
- cat
- deer
- dog
- frog
- horse
- ship
- truck

## Deployment Instructions

### Railway Deployment
1. Push this folder to a Git repository
2. Connect the repository to Railway
3. Railway will automatically detect the `railway.toml` configuration
4. The service will be available at the provided Railway URL

### Heroku Deployment
1. Install Heroku CLI
2. Login to Heroku: `heroku login`
3. Create a new app: `heroku create your-app-name`
4. Deploy: `git push heroku main`

### Local Testing
```bash
# Install dependencies
pip install -r requirements.txt

# Run the application
python app.py

# Test with curl
curl -X POST -F "file=@your_image.jpg" http://localhost:8000/predict
```

## Environment Variables
- `MODEL_PATH` - Path to the model file (default: "best_model_cifar10.pth")
- `PORT` - Port to run the server (default: 8000)

## Model Information
- Input size: 32x32 RGB images
- Architecture: Custom CNN with fallback support
- Normalization: CIFAR-10 standard normalization values