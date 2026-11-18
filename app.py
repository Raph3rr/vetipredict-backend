from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import pandas as pd
import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import classification_report
import traceback
from werkzeug.utils import secure_filename

# ---------- CONFIG ----------
CROP_CSV = "livestock_diseases.csv"
FOOD_CSV = "food_nutrients.csv"
MODEL_FILE = "livestock_model.joblib"
ENCODER_FILE = "label_encoder.joblib"
ANIMAL_MODEL_FILES = {
    animal: (
        f"livestock_model_{animal.lower().replace(' ', '_')}.joblib",
        f"label_encoder_{animal.lower().replace(' ', '_')}.joblib"
    )
    for animal in (
        "Cattle", "Goat", "Sheep", "Pig", "Chicken", "Duck", "Turkey",
        "Guinea fowl", "Rabbit", "Grasscutter"
    )
}
SYMPTOM_COLUMNS = [
    "coughing", "diarrhea", "lethargy", "loss_of_appetite",
    "nasal_discharge", "lesions", "lameness", "labored_breathing",
    "ocular_discharge", "swelling", "weight_loss", "neurological_signs",
    "bloody_stool", "excessive_salivation", "reduced_production"
]
FEATURE_COLUMNS = SYMPTOM_COLUMNS

DATA_DIR = os.path.join(os.getcwd(), "data")
os.makedirs(DATA_DIR, exist_ok=True)

ALLOWED_EXTENSIONS = {"csv", "txt"}

app = Flask(__name__)
CORS(app)


# ---------- Utilities: file helpers ----------
def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def safe_path_for(filename: str) -> str:
    filename = secure_filename(filename)
    return os.path.join(DATA_DIR, filename)


# ---------- Utilities: Load nutrition DB ----------
def load_food_db():
    candidates = [
        os.path.join(DATA_DIR, FOOD_CSV),
        FOOD_CSV
    ]

    for path in candidates:
        if os.path.exists(path):
            df = pd.read_csv(path, sep=None, engine="python")
            df.columns = [c.strip() for c in df.columns]
            return df

    raise FileNotFoundError(
        f"{FOOD_CSV} not found in data/ or working directory."
    )


try:
    food_df = load_food_db()
except Exception:
    food_df = pd.DataFrame()


# ---------- Model training / loading ----------
def find_crop_csv_path():
    for path in [os.path.join(DATA_DIR, CROP_CSV), CROP_CSV]:
        if os.path.exists(path):
            return path
    return None


def load_crop_data():
    path = find_crop_csv_path()

    if path is None:
        raise FileNotFoundError(
            f"{CROP_CSV} not found in data/ or working directory."
        )

    df = pd.read_csv(path, sep=None, engine="python")
    df.columns = [c.strip() for c in df.columns]
    return df


def models_match_dataset():
    """Check that saved model files match the current species, labels, and features."""
    df = load_crop_data()
    if not all(column in df.columns for column in FEATURE_COLUMNS + ["animal_type", "label"]):
        return False

    for animal, (model_file, encoder_file) in ANIMAL_MODEL_FILES.items():
        if not os.path.exists(model_file) or not os.path.exists(encoder_file):
            return False
        rows = df[df["animal_type"].astype(str).str.strip().str.casefold() == animal.casefold()]
        if rows.empty:
            return False
        model = joblib.load(model_file)
        encoder = joblib.load(encoder_file)
        if model.n_features_in_ != len(FEATURE_COLUMNS):
            return False
        if set(encoder.classes_) != set(rows["label"].astype(str).unique()):
            return False
    return True


def train_and_save_model():
    df = load_crop_data()

    expected_cols = ["animal_type"] + FEATURE_COLUMNS + ["label"]

    cols = [c for c in df.columns]
    col_map = {}

    for expected_col in expected_cols:
        found = None

        for col in cols:
            if col.lower() == expected_col.lower():
                found = col
                break

        if found is None:
            raise ValueError(
                f"Expected column '{expected_col}' not found in CSV. "
                f"Found columns: {cols}"
            )

        col_map[expected_col] = found

    df = df[[col_map[c] for c in expected_cols]].rename(
        columns={v: k for k, v in col_map.items()}
    )

    feature_cols = FEATURE_COLUMNS
    reports = {}
    trained_models = []
    for animal, (model_file, encoder_file) in ANIMAL_MODEL_FILES.items():
        animal_df = df[df["animal_type"].astype(str).str.strip().str.casefold() == animal.casefold()].copy()
        if animal_df.empty:
            raise ValueError(f"No training rows found for animal type '{animal}'.")

        X = animal_df[feature_cols].astype(float)
        y = animal_df["label"].astype(str)
        le = LabelEncoder()
        y_enc = le.fit_transform(y)
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("rf", RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1))
        ])

        groups = X.astype(int).astype(str).agg("|".join, axis=1)
        splitter = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
        train_indices, test_indices = next(splitter.split(X, y_enc, groups=groups))
        X_train, X_test = X.iloc[train_indices], X.iloc[test_indices]
        y_train, y_test = y_enc[train_indices], y_enc[test_indices]

        pipeline.fit(X_train, y_train)
        preds = pipeline.predict(X_test)
        report = classification_report(
            y_test, preds, labels=range(len(le.classes_)),
            target_names=le.classes_, zero_division=0
        )
        # Use every available row for the final model after held-out evaluation.
        pipeline.fit(X, y_enc)
        trained_models.append((model_file, encoder_file, pipeline, le))
        reports[animal] = {"report": report, "classes": list(le.classes_), "rows": len(animal_df)}

    # Save only after all animals trained successfully, avoiding a partial model set.
    for model_file, encoder_file, pipeline, encoder in trained_models:
        joblib.dump(pipeline, model_file)
        joblib.dump(encoder, encoder_file)

    return {"animals": reports}


def load_model(animal_type):
    if animal_type not in ANIMAL_MODEL_FILES:
        raise ValueError(f"Unsupported animal type: {animal_type}")
    model_file, encoder_file = ANIMAL_MODEL_FILES[animal_type]
    df = load_crop_data()
    animal_rows = df[
        df["animal_type"].astype(str).str.strip().str.casefold()
        == animal_type.casefold()
    ]
    if not os.path.exists(model_file) or not os.path.exists(encoder_file):
        raise RuntimeError("Models are missing or stale. Retrain them from the current CSV first.")
    model = joblib.load(model_file)
    encoder = joblib.load(encoder_file)
    if (
        animal_rows.empty
        or model.n_features_in_ != len(FEATURE_COLUMNS)
        or set(encoder.classes_) != set(animal_rows["label"].astype(str).unique())
    ):
        raise RuntimeError("Saved model does not match this CSV. Retrain it before predicting.")
    return model, encoder


try:
    if models_match_dataset():
        print(f"{len(ANIMAL_MODEL_FILES)} animal-specific livestock models ready.")
    else:
        print("Saved livestock models do not match the current CSV; retrain from the Train Model page.")
except Exception as e:
    print("Model load/train failed at startup:", e)
    traceback.print_exc()


# ---------- Helper predictions ----------
def predict_disease_single(features_dict, top_k=3):
    animal_type_val = features_dict.get("animal_type", "Cattle")

    if animal_type_val not in ANIMAL_MODEL_FILES:
        raise ValueError(f"Unsupported animal type: {animal_type_val}")
    model_pipeline, label_encoder = load_model(animal_type_val)

    values = {
        column: float(features_dict.get(column, 0))
        for column in FEATURE_COLUMNS
    }
    X = pd.DataFrame([values], columns=FEATURE_COLUMNS)

    probs = model_pipeline.predict_proba(X)[0]
    classes = label_encoder.classes_

    eligible_results = [
        {"disease": disease, "probability": float(probs[i])}
        for i, disease in enumerate(classes)
    ]

    eligible_results.sort(
        key=lambda item: item["probability"],
        reverse=True
    )

    # Return every eligible class. The frontend separates Healthy
    # and chooses how many non-Healthy results to display.
    return eligible_results


# ---------- Flask endpoints ----------
@app.route("/")
def index():
    return jsonify({
        "message": "Livestock Disease Prediction & Feed Nutrition API",
        "routes": {
            "POST /predict": "Predict disease from animal type and symptoms (JSON).",
            "GET /train": "Retrain model from CSV.",
            "POST /nutrition/summary": "Compute nutrient totals from feed items list.",
            "GET /datasets/list": "List available dataset files",
            "GET /datasets/download?name=...": "Download CSV by name",
            "GET /datasets/preview?name=...": "Preview first N lines (json)",
            "POST /datasets/upload": "Upload CSV file"
        }
    })


@app.route("/train", methods=["GET"])
def train_route():
    try:
        info = train_and_save_model()

        return jsonify({
            "status": "trained",
            "animals": info["animals"]
        })

    except Exception as e:
        return jsonify({
            "status": "error",
            "error": str(e),
            "trace": traceback.format_exc()
        }), 500


@app.route("/predict", methods=["POST"])
def predict_route():
    try:
        data = request.get_json(force=True)

        required = ["animal_type"]

        for field in required:
            if field not in data:
                return jsonify({"error": f"Missing field {field}"}), 400

        # Retained for compatibility with the existing frontend.
        top_k = int(data.get("top_k", 3))

        features = {"animal_type": data["animal_type"]}

        for column in SYMPTOM_COLUMNS:
            features[column] = float(data.get(column, 0))

        if not any(features[column] > 0 for column in SYMPTOM_COLUMNS):
            return jsonify({
                "error": "Select at least one symptom to generate a prediction."
            }), 400

        preds = predict_disease_single(features, top_k=top_k)

        return jsonify({
            "input": features,
            "predictions": preds
        })

    except Exception as e:
        return jsonify({
            "error": str(e),
            "trace": traceback.format_exc()
        }), 500


# ---------- Nutrition summary route ----------
@app.route("/nutrition/summary", methods=["POST"])
def nutrition_summary():
    try:
        req = request.get_json(force=True)
        items = req.get("items")

        if not items or not isinstance(items, list):
            return jsonify({
                "error": "items must be a list of {food_id, amount_g}"
            }), 400

        global food_df
        food_df = load_food_db()

        nutrient_cols = [
            "calories_kcal", "carbs_g", "protein_g", "fat_g",
            "vitamin_a_mcg", "vitamin_c_mg", "iron_mg",
            "calcium_mg", "zinc_mg"
        ]

        fd = food_df.copy()
        fd.columns = [c.strip() for c in fd.columns]

        total = {col: 0.0 for col in nutrient_cols}
        per_item = []

        for item in items:
            if "food_id" not in item or "amount_g" not in item:
                return jsonify({
                    "error": "each item must have food_id and amount_g"
                }), 400

            fid = int(item["food_id"])
            amount = float(item["amount_g"])

            row = fd[fd["food_id"] == fid]

            if row.empty:
                return jsonify({
                    "error": f"food_id {fid} not found in {FOOD_CSV}"
                }), 400

            row = row.iloc[0]
            serving = float(row.get("serving_g", 100))
            scale = amount / serving

            contribution = {
                "food_id": int(fid),
                "food_name": str(row.get("food_name")),
                "amount_g": amount
            }

            for col in nutrient_cols:
                val = row.get(col, None)

                if val is None:
                    val = row.get(col.replace("_", " "), 0)

                try:
                    val_f = float(val)
                except Exception:
                    val_f = 0.0

                contribution[col] = round(val_f * scale, 3)
                total[col] += val_f * scale

            per_item.append(contribution)

        total = {
            key: round(value, 3)
            for key, value in total.items()
        }

        return jsonify({
            "items": per_item,
            "totals": total,
            "note": (
                "Nutrient totals are computed from the provided "
                "food_nutrients.csv values. These are estimates and "
                "for guidance only."
            )
        })

    except Exception as e:
        return jsonify({
            "error": str(e),
            "trace": traceback.format_exc()
        }), 500


# ---------- Dataset routes ----------
@app.route("/datasets/list", methods=["GET"])
def datasets_list():
    try:
        files = []

        for filename in os.listdir(DATA_DIR):
            path = os.path.join(DATA_DIR, filename)

            if (
                os.path.isfile(path)
                and filename.rsplit(".", 1)[-1].lower() in ALLOWED_EXTENSIONS
            ):
                files.append(filename)

        for candidate in [CROP_CSV, FOOD_CSV]:
            if os.path.exists(candidate) and candidate not in files:
                files.insert(0, candidate)

        return jsonify({"files": files})

    except Exception as e:
        return jsonify({
            "error": str(e),
            "trace": traceback.format_exc()
        }), 500


@app.route("/datasets/download", methods=["GET"])
def datasets_download():
    name = request.args.get("name")

    if not name:
        return jsonify({
            "error": "name query parameter required"
        }), 400

    safe_name = secure_filename(name)

    paths_to_try = [
        os.path.join(DATA_DIR, safe_name),
        os.path.join(os.getcwd(), safe_name)
    ]

    for path in paths_to_try:
        if os.path.exists(path) and os.path.isfile(path):
            directory = os.path.dirname(path)

            return send_from_directory(
                directory,
                os.path.basename(path),
                as_attachment=True
            )

    return jsonify({
        "error": f"File '{name}' not found on server."
    }), 404


@app.route("/datasets/preview", methods=["GET"])
def datasets_preview():
    name = request.args.get("name")
    lines = int(request.args.get("lines", 15))

    if not name:
        return jsonify({
            "error": "name query parameter required"
        }), 400

    safe_name = secure_filename(name)
    path = os.path.join(DATA_DIR, safe_name)

    if not os.path.exists(path):
        path = os.path.join(os.getcwd(), safe_name)

    if not os.path.exists(path):
        return jsonify({
            "error": f"File '{name}' not found."
        }), 404

    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as file:
            all_lines = []

            for _ in range(lines):
                line = file.readline()

                if not line:
                    break

                all_lines.append(line.rstrip("\n"))

        return jsonify({
            "name": name,
            "lines": all_lines
        })

    except Exception as e:
        return jsonify({
            "error": str(e),
            "trace": traceback.format_exc()
        }), 500


@app.route("/datasets/<path:filename>", methods=["GET"])
def datasets_serve(filename):
    safe_name = secure_filename(filename)
    path = os.path.join(DATA_DIR, safe_name)

    if not os.path.exists(path):
        path = os.path.join(os.getcwd(), safe_name)

    if not os.path.exists(path):
        return jsonify({
            "error": f"File '{filename}' not found."
        }), 404

    try:
        directory = os.path.dirname(path)

        return send_from_directory(
            directory,
            os.path.basename(path),
            as_attachment=False
        )

    except Exception as e:
        return jsonify({
            "error": str(e),
            "trace": traceback.format_exc()
        }), 500


@app.route("/datasets/upload", methods=["POST"])
def datasets_upload():
    if "file" not in request.files:
        return jsonify({"error": "No file part in request"}), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    if not allowed_file(file.filename):
        return jsonify({
            "error": f"File extension not allowed. Allowed: {ALLOWED_EXTENSIONS}"
        }), 400

    filename = secure_filename(file.filename)
    save_path = os.path.join(DATA_DIR, filename)

    try:
        file.save(save_path)

        return jsonify({
            "name": filename,
            "message": f"Saved to data/{filename}"
        })

    except Exception as e:
        return jsonify({
            "error": str(e),
            "trace": traceback.format_exc()
        }), 500


# ---------- Mock Prices & Items endpoints ----------
@app.route("/api/prices", methods=["GET"])
def get_mock_prices():
    mock_prices = [
        {
            "id": 1,
            "date": "2026-08-19",
            "price": 4500.0,
            "currency": "NGN",
            "quantity": 1,
            "source": "Lagos Central Market",
            "item": {
                "id": 101,
                "name": "Live Broiler Chicken (2.5kg)"
            },
            "vendor": {
                "id": 201,
                "name": "Standard Poultry Farm"
            }
        },
        {
            "id": 2,
            "date": "2026-08-19",
            "price": 2800.0,
            "currency": "NGN",
            "quantity": 1,
            "source": "Kano Cattle Depot",
            "item": {
                "id": 102,
                "name": "Cattle Beef Cut (1kg)"
            },
            "vendor": {
                "id": 202,
                "name": "Northern Meat Suppliers"
            }
        },
        {
            "id": 3,
            "date": "2026-08-18",
            "price": 3200.0,
            "currency": "NGN",
            "quantity": 30,
            "source": "Ibadan Feed Depot",
            "item": {
                "id": 103,
                "name": "Fresh Dairy Milk (1 Liter)"
            },
            "vendor": {
                "id": 203,
                "name": "Oyo Dairy Cooperatives"
            }
        },
        {
            "id": 4,
            "date": "2026-08-17",
            "price": 1400.0,
            "currency": "NGN",
            "quantity": 1,
            "source": "Lagos Poultry Market",
            "item": {
                "id": 104,
                "name": "Crate of Eggs (30)"
            },
            "vendor": {
                "id": 201,
                "name": "Standard Poultry Farm"
            }
        }
    ]

    return jsonify(mock_prices)


@app.route("/api/items", methods=["GET"])
def get_mock_items():
    return jsonify([
        {"id": 101, "name": "Live Broiler Chicken (2.5kg)"},
        {"id": 102, "name": "Cattle Beef Cut (1kg)"},
        {"id": 103, "name": "Fresh Dairy Milk (1 Liter)"},
        {"id": 104, "name": "Crate of Eggs (30)"}
    ])


# ---------- Run ----------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)
