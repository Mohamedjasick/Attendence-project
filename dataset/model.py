import os
import cv2
import numpy as np
import pickle
import faiss  # Import for fast similarity search

# Define paths
MODEL_PATH = "dataset/face_encodings.faiss"
ID_MAP_PATH = "dataset/student_ids.pkl"
EMBEDDING_DIM = 1024

# NEW: Path to the Haar Cascade XML file
# IMPORTANT: You must download 'haarcascade_frontalface_default.xml' and put it in the 'dataset' folder.
HAAR_CASCADE_PATH = os.path.join(os.path.dirname(__file__), "dataset", "haarcascade_frontalface_default.xml")


# ---- Utility: extract face crop -> small grayscale vector (embedding) ----
def crop_face_and_embed(bgr_image, bbox):
    # bbox here is (x, y, w, h) from Haar Cascade, not MediaPipe's normalized box
    x, y, w, h = bbox
    h_img, w_img = bgr_image.shape[:2]

    # Simple boundary checking
    x1 = int(max(0, x))
    y1 = int(max(0, y))
    x2 = int(min(w_img, x + w))
    y2 = int(min(h_img, y + h))

    if x2 <= x1 or y2 <= y1:
        return None

    face = bgr_image[y1:y2, x1:x2]
    face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    face = cv2.resize(face, (32, 32), interpolation=cv2.INTER_AREA)
    emb = face.flatten().astype(np.float32) / 255.0
    return emb


# ---- Face Detection (MODIFIED to use Haar Cascade) ----
def extract_embedding_for_image(stream_or_bytes):
    # Load the Haar Cascade Classifier (optimized for CPU)
    face_cascade = cv2.CascadeClassifier(HAAR_CASCADE_PATH)
    if face_cascade.empty():
        print("Error: Haar Cascade classifier file not loaded.")
        return None

    data = stream_or_bytes.read()
    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Run FAST Haar Cascade Detection
    # detectMultiScale is much faster on CPU than MediaPipe's solution
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(faces) == 0:
        return None

    # faces[0] is the (x, y, w, h) bounding box
    emb = crop_face_and_embed(img, faces[0])
    return emb


# ---- Load model helpers (NO CHANGE - still loading Faiss) ----
def load_model_if_exists():
    """Loads the Faiss index and the student ID map."""
    if not os.path.exists(MODEL_PATH) or not os.path.exists(ID_MAP_PATH):
        return None

    try:
        index = faiss.read_index(MODEL_PATH)
        with open(ID_MAP_PATH, 'rb') as f:
            student_ids_list = pickle.load(f)
        return (index, student_ids_list)
    except Exception as e:
        print(f"Error loading Faiss model: {e}")
        return None


def predict_with_model(clf_tuple, emb):
    """Uses the Faiss index to find the nearest neighbor quickly."""
    index, student_ids = clf_tuple

    query_vector = np.array(emb).astype('float32').reshape(1, -1)
    k = 1
    D, I = index.search(query_vector, k)

    nearest_index = I[0][0]
    distance = D[0][0]

    if nearest_index == -1:
        return ("Unknown", 0.0)

    predicted_student_id = student_ids[nearest_index]

    # --- Distance Thresholding ---
    # Relaxed threshold (1.2) for low-quality embeddings
    DISTANCE_THRESHOLD = 1.2

    if distance < DISTANCE_THRESHOLD:
        confidence = 1.0 - (distance / DISTANCE_THRESHOLD)
        if confidence < 0: confidence = 0.0
        return (str(predicted_student_id), confidence)
    else:
        return ("Unknown", 0.1)


# ---- Training function (MODIFIED to use Haar Cascade) ----
def train_model_background(dataset_dir, progress_callback=None):
    """
    Trains the Faiss index, now using the faster Haar Cascade detection.
    """
    face_cascade = cv2.CascadeClassifier(HAAR_CASCADE_PATH)
    if face_cascade.empty():
        progress_callback(0, "Error: Haar Cascade file not loaded during training.")
        return

    X = []  # List to hold all face embeddings (vectors)
    y_student_id = []  # List to hold the student IDs (labels)

    student_dirs = [d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d)) and d.isdigit()]
    total_students = max(1, len(student_dirs))
    processed = 0

    # 1. Gather all embeddings and labels
    for sid_str in student_dirs:
        folder = os.path.join(dataset_dir, sid_str)
        files = [f for f in os.listdir(folder) if f.lower().endswith((".jpg", ".jpeg", ".png"))]

        for fn in files:
            path = os.path.join(folder, fn)
            img = cv2.imread(path)
            if img is None:
                continue

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # Find faces using Haar Cascade
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            if len(faces) == 0:
                continue

            # Assume the first detected face is the target
            emb = crop_face_and_embed(img, faces[0])

            if emb is None:
                continue

            X.append(emb)
            y_student_id.append(int(sid_str))

        processed += 1
        if progress_callback:
            pct = int((processed / total_students) * 80)
            progress_callback(pct, f"Processed {processed}/{total_students} students")

    if len(X) == 0:
        if progress_callback:
            progress_callback(0, "No training data found")
        return

    X = np.stack(X).astype('float32')
    y_student_id = np.array(y_student_id)
    d = X.shape[1]

    # 2. Build the Faiss Index
    if progress_callback:
        progress_callback(85, "Building Faiss Index...")

    index = faiss.IndexFlatL2(d)
    index.add(X)

    # 3. Save the Index and the ID map
    if progress_callback:
        progress_callback(95, "Saving Index and ID map...")

    faiss.write_index(index, MODEL_PATH)

    with open(ID_MAP_PATH, 'wb') as f:
        pickle.dump(y_student_id.tolist(), f)

    if progress_callback:
        progress_callback(100, "Training complete. Faiss index created.")