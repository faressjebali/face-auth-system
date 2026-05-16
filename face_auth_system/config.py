"""
Configuration constants for the Face Recognition-Based Authentication System.

All hardcoded values, paths, thresholds, and tunable parameters are defined
here so that every module imports from a single source of truth.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT: Path = Path(__file__).parent.resolve()
DATA_DIR: Path = PROJECT_ROOT / "data"
ENROLLED_FACES_DIR: Path = DATA_DIR / "enrolled_faces"
LOGS_DIR: Path = DATA_DIR / "logs"
MODELS_DIR: Path = PROJECT_ROOT / "models"
VAULT_DIR: Path = DATA_DIR / "vault"
EVALUATION_OUTPUT_DIR: Path = PROJECT_ROOT / "evaluation"

# Dlib landmark model – download separately and place at project root
# wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
DLIB_LANDMARK_MODEL: Path = PROJECT_ROOT / "shape_predictor_68_face_landmarks.dat"

# ---------------------------------------------------------------------------
# Face Detection
# ---------------------------------------------------------------------------
FACE_SCALE_FACTOR: float = 1.1
FACE_MIN_NEIGHBORS: int = 5
FACE_MIN_SIZE: tuple = (30, 30)

# ---------------------------------------------------------------------------
# Preprocessing / Image Dimensions
# ---------------------------------------------------------------------------
IMAGE_SIZE: tuple = (128, 128)      # (width, height)

# ---------------------------------------------------------------------------
# Classical Model
# ---------------------------------------------------------------------------
PCA_N_COMPONENTS: int = 150
LBPH_RADIUS: int = 1
LBPH_NEIGHBORS: int = 8
LBPH_GRID_X: int = 8
LBPH_GRID_Y: int = 8
EIGENFACES_MODEL_PATH: Path = MODELS_DIR / "eigenfaces.pkl"
LBPH_MODEL_PATH: Path = MODELS_DIR / "lbph_model.yml"
LBPH_MAX_CONFIDENCE: float = 100.0  # Raw LBPH confidence ceiling for normalization

# ---------------------------------------------------------------------------
# Deep Model
# ---------------------------------------------------------------------------
FACENET_BACKEND: str = "Facenet"
ARCFACE_BACKEND: str = "ArcFace"
# Cosine *distance* threshold (lower = more similar; match if distance < threshold)
DEEP_DISTANCE_THRESHOLD: float = 0.40

# ---------------------------------------------------------------------------
# Score Fusion / Authenticator
# ---------------------------------------------------------------------------
CLASSICAL_WEIGHT: float = 0.4
DEEP_WEIGHT: float = 0.6
AUTH_THRESHOLD: float = 0.6         # Combined score >= threshold → GRANTED

# ---------------------------------------------------------------------------
# Liveness Detection
# ---------------------------------------------------------------------------
EAR_THRESHOLD: float = 0.25         # EAR below this → blink detected
EAR_CONSEC_FRAMES: int = 3          # Min consecutive low-EAR frames for blink
BLINKS_REQUIRED: int = 1            # Blinks needed to pass liveness
LBP_LIVENESS_THRESHOLD: float = 0.55

# ---------------------------------------------------------------------------
# Deepfake Detection
# ---------------------------------------------------------------------------
FFT_ARTIFACT_THRESHOLD: float = 0.55   # Normalized spectral score above this → fake

# ---------------------------------------------------------------------------
# Security Monitor
# ---------------------------------------------------------------------------
MAX_FAILED_ATTEMPTS: int = 5
LOCKOUT_DURATION_SEC: int = 300        # 5-minute lockout
RATE_LIMIT_WINDOW_SEC: int = 30        # Rolling window
RATE_LIMIT_MAX_ATTEMPTS: int = 3

# ---------------------------------------------------------------------------
# Embedding Vault (AES-256-GCM)
# ---------------------------------------------------------------------------
KEY_DERIVATION_ITERATIONS: int = 100_000
VAULT_SALT_SIZE: int = 16              # bytes
VAULT_NONCE_SIZE: int = 12             # bytes for AES-GCM
VAULT_COSINE_THRESHOLD: float = 0.80   # Similarity required for vault verify

# ---------------------------------------------------------------------------
# Cancelable Biometrics
# ---------------------------------------------------------------------------
PROJECTION_INPUT_DIM: int = 512        # FaceNet / ArcFace embedding size
PROJECTION_OUTPUT_DIM: int = 256       # Reduced protected template dimension

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_FILENAME_PREFIX: str = "auth_log"

# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
ROC_PLOT_PATH: Path = EVALUATION_OUTPUT_DIR / "roc_curve.png"
FAR_FRR_PLOT_PATH: Path = EVALUATION_OUTPUT_DIR / "far_frr_curve.png"
METRICS_CSV_PATH: Path = EVALUATION_OUTPUT_DIR / "metrics.csv"
