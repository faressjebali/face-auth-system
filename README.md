# Face Authentication System

A face recognition-based authentication system combining classical computer vision and deep learning, with liveness detection, deepfake detection, cancelable biometrics, and an encrypted embedding vault.

## Features

- **Dual-pipeline recognition** — Eigenfaces (PCA) + LBPH for the classical path; FaceNet / ArcFace for the deep path; scores fused via configurable weighted average
- **Liveness detection** — blink detection (Eye Aspect Ratio) and LBP texture analysis to reject printed/screen spoofs
- **Deepfake detection** — FFT-based spectral artifact analysis
- **Cancelable biometrics** — BioHashing (random orthogonal projection seeded by a user token) produces non-invertible, revocable templates
- **Encrypted vault** — face embeddings stored with AES-256-GCM; key derived via PBKDF2 from a user password, never stored on disk
- **Security monitor** — rate limiting, failed-attempt lockout (5 attempts → 5-minute lockout)
- **Structured logging** — JSON auth event logs and security alert logs
- **Evaluation** — ROC curve, FAR/FRR analysis from historical auth logs

## Project Structure

```
face_auth_system/
├── main.py                  # CLI entry point
├── config.py                # All thresholds, paths, and tunable parameters
├── app.py                   # Desktop GUI (Tkinter)
├── authenticator.py         # Score fusion and GRANTED/DENIED decision
├── enrollment.py            # User enrollment (images or webcam)
├── classical_model.py       # Eigenfaces (PCA) and LBPH models
├── deep_model.py            # FaceNet / ArcFace via DeepFace
├── liveness.py              # EAR blink detection + LBP texture analysis
├── deepfake_detector.py     # FFT spectral artifact detector
├── cancelable_biometrics.py # BioHashing — non-invertible protected templates
├── vault.py                 # AES-256-GCM encrypted embedding storage
├── preprocessing.py         # Face detection, alignment, normalization
├── lighting.py              # Illumination normalization
├── security_monitor.py      # Rate limiting and lockout logic
├── logger.py                # Structured JSON auth and security logging
├── user_store.py            # User registry (name, DOB)
├── evaluation.py            # FAR, FRR, ROC evaluation from logs
├── tests/                   # Unit tests (pytest)
├── data/
│   ├── enrolled_faces/      # Face crops (populated at enrollment)
│   ├── logs/                # Auth and security event logs
│   ├── vault/               # Encrypted face embeddings
│   └── keys/                # Per-user derived key material
└── models/                  # Trained classical model files
```

## Requirements

- Python 3.10+
- dlib requires CMake and a C++ compiler

```bash
pip install -r requirements.txt
```

### Dlib landmark model

Download separately and place in the project root:

```bash
wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
bunzip2 shape_predictor_68_face_landmarks.dat.bz2
```

## Usage

### Enroll a user

```bash
# From image files
python main.py enroll --user alice --images face1.jpg face2.jpg \
                      --password SECRET --token MYTOKEN

# From webcam (captures 10 frames)
python main.py enroll --user alice --camera --password SECRET --token MYTOKEN
```

### Authenticate

```bash
# From an image file (with liveness and deepfake checks)
python main.py auth --user alice --image probe.jpg \
                    --password SECRET --token MYTOKEN --liveness --deepfake

# From webcam
python main.py auth --user alice --camera --password SECRET --token MYTOKEN
```

### Other commands

```bash
python main.py list           # List enrolled users
python main.py delete --user alice   # Remove a user
python main.py train          # Retrain classical models from enrolled faces
python main.py eval           # Run FAR/FRR evaluation from auth logs
```

### Desktop GUI

```bash
python app.py
```

## Configuration

All thresholds and parameters are in `config.py`:

| Parameter | Default | Description |
|---|---|---|
| `AUTH_THRESHOLD` | 0.6 | Fused score required for GRANTED |
| `CLASSICAL_WEIGHT` | 0.4 | Weight of Eigenfaces/LBPH score |
| `DEEP_WEIGHT` | 0.6 | Weight of FaceNet/ArcFace score |
| `DEEP_DISTANCE_THRESHOLD` | 0.40 | Cosine distance cutoff for deep match |
| `BLINKS_REQUIRED` | 1 | Blinks needed to pass liveness |
| `MAX_FAILED_ATTEMPTS` | 5 | Attempts before lockout |
| `LOCKOUT_DURATION_SEC` | 300 | Lockout duration in seconds |

## Running Tests

```bash
pytest tests/
```

## Security Notes

- The `--password` and `--token` flags are per-user secrets. The password encrypts the vault; the token controls the cancelable biometric projection. Neither is stored on disk.
- To revoke a compromised template, re-enroll with a new `--token`. The old template becomes uncorrelated and unusable.
- `data/users.json`, `data/keys/`, and `data/vault/` are excluded from version control and must never be committed.
