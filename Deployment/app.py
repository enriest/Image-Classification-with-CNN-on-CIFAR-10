import io
import os
from typing import List, Any
from PIL import Image
import torch
import torchvision.transforms as T
from flask import Flask, request, jsonify

MODEL_PATH = os.environ.get("MODEL_PATH", "best_model_cifar10.pth")
CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"
]

app = Flask(__name__)

# Lazy-load model so startup is fast
_model = None
_transform = T.Compose([
    T.Resize((32, 32)),
    T.ToTensor(),
    T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

def _build_fallback_model(num_classes: int = 10) -> torch.nn.Module:
    # Minimal CNN fallback (only used if a state_dict is provided without full model object)
    return torch.nn.Sequential(
        torch.nn.Conv2d(3, 32, 3, padding=1),
        torch.nn.ReLU(),
        torch.nn.MaxPool2d(2),
        torch.nn.Conv2d(32, 64, 3, padding=1),
        torch.nn.ReLU(),
        torch.nn.MaxPool2d(2),
        torch.nn.Flatten(),
        torch.nn.Linear(64 * 8 * 8, 256),
        torch.nn.ReLU(),
        torch.nn.Linear(256, num_classes)
    )


def load_model() -> Any:
    global _model
    if _model is not None:
        return _model
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaded = torch.load(MODEL_PATH, map_location=device)
    # Case 1: directly a torch.nn.Module
    if isinstance(loaded, torch.nn.Module):
        _model = loaded.to(device).eval()
        return _model
    # Case 2: dictionary that might contain 'model_state' or direct state_dict
    if isinstance(loaded, dict):
        # Heuristics
        possible_keys = ["state_dict", "model_state", "model_state_dict"]
        state = None
        for k in possible_keys:
            if k in loaded:
                state = loaded[k]
                break
        if state is None:
            # maybe it's already a state_dict
            if all(isinstance(v, torch.Tensor) for v in loaded.values()):
                state = loaded
        if state is not None:
            model = _build_fallback_model(len(CLASS_NAMES))
            missing, unexpected = model.load_state_dict(state, strict=False)
            if missing:
                print(f"[warn] Missing keys when loading model: {missing}")
            if unexpected:
                print(f"[warn] Unexpected keys when loading model: {unexpected}")
            _model = model.to(device).eval()
            return _model
    raise RuntimeError("Unsupported model file format. Save the whole model with torch.save(model) or a dict with a 'state_dict' key.")


def prepare_image(img: Image.Image):
    return _transform(img.convert("RGB")).unsqueeze(0)


@app.route("/health", methods=["GET"])  # simple health check
def health():
    return {"status": "ok"}


@app.route("/predict", methods=["POST"])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "Empty filename"}), 400
    try:
        img_bytes = file.read()
        img = Image.open(io.BytesIO(img_bytes))
    except Exception as e:
        return jsonify({"error": f"Invalid image: {e}"}), 400

    tensor = prepare_image(img)
    model = load_model()
    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]
    top_idx = int(probs.argmax())

    return jsonify({
        "prediction": CLASS_NAMES[top_idx],
        "probability": float(probs[top_idx]),
        "class_probabilities": {CLASS_NAMES[i]: float(p) for i, p in enumerate(probs)}
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port)
