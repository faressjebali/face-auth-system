"""
Face Authentication System — CLI entry point.

Sub-commands
------------
enroll  — register a new user from images or webcam
auth    — authenticate a claimed identity
delete  — remove a user's enrollment
list    — show all enrolled users
train   — retrain classical models from enrolled face crops
eval    — run performance evaluation against labelled log data

Quick start
-----------
  # Enroll from image files
  python main.py enroll --user alice --images face1.jpg face2.jpg \\
                        --password SECRET --token MYTOKEN

  # Enroll interactively from webcam
  python main.py enroll --user alice --camera --password SECRET --token MYTOKEN

  # Authenticate from an image file
  python main.py auth --user alice --image probe.jpg \\
                      --password SECRET --token MYTOKEN --liveness --deepfake

  # Authenticate from webcam snapshot
  python main.py auth --user alice --camera --password SECRET --token MYTOKEN

  # List enrolled users
  python main.py list

  # Retrain classical models
  python main.py train

  # Evaluate system performance
  python main.py eval
"""

import argparse
import logging
import sys

import cv2

import config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("main")


# ---------------------------------------------------------------------------
# Lazy module loader (avoids importing heavy deps at parse time)
# ---------------------------------------------------------------------------

def _imports() -> dict:
    from authenticator import Authenticator
    from cancelable_biometrics import CancelableBiometrics
    from classical_model import EigenfacesModel
    from deep_model import DeepFaceModel
    from deepfake_detector import DeepfakeDetector
    from enrollment import EnrollmentManager, _load_gallery
    from liveness import LBPLivenessDetector, check_liveness_single_frame
    from logger import log_auth_event, log_security_alert
    from preprocessing import preprocess_image
    from security_monitor import SecurityMonitor
    from vault import EmbeddingVault
    return locals()


# ---------------------------------------------------------------------------
# enroll
# ---------------------------------------------------------------------------

def cmd_enroll(args) -> int:
    m = _imports()
    manager = m["EnrollmentManager"]()

    if args.camera:
        ok = manager.enroll_from_camera(
            user_id=args.user,
            password=args.password,
            token=args.token,
            n_captures=args.n_captures,
            camera_id=args.camera_id,
        )
    else:
        if not args.images:
            logger.error("Provide --images <paths> or use --camera.")
            return 1
        ok = manager.enroll_from_images(
            user_id=args.user,
            image_paths=args.images,
            password=args.password,
            token=args.token,
        )

    if ok:
        print(f"[OK] '{args.user}' enrolled successfully.")
        return 0
    print(f"[FAIL] Enrollment failed for '{args.user}'.")
    return 1


# ---------------------------------------------------------------------------
# auth
# ---------------------------------------------------------------------------

def cmd_auth(args) -> int:
    m = _imports()

    # ── Security gate ──────────────────────────────────────────────────
    security = m["SecurityMonitor"]()
    allowed, reason = security.check(args.user)
    if not allowed:
        print(f"[BLOCKED] {reason}")
        m["log_security_alert"](args.user, "blocked", {"reason": reason})
        return 1

    # ── Acquire image ──────────────────────────────────────────────────
    img = None
    if args.image:
        img = cv2.imread(args.image)
        if img is None:
            logger.error("Cannot read image: %s", args.image)
            return 1
    elif args.camera:
        cap = cv2.VideoCapture(args.camera_id)
        ret, img = cap.read()
        cap.release()
        if not ret or img is None:
            logger.error("Cannot capture from camera %d.", args.camera_id)
            return 1
    else:
        logger.error("Provide --image <path> or --camera.")
        return 1

    # ── Liveness check ─────────────────────────────────────────────────
    if args.liveness:
        faces_pre = m["preprocess_image"](img)
        if faces_pre:
            is_live, live_score = m["check_liveness_single_frame"](
                faces_pre[0]["gray"], m["LBPLivenessDetector"]()
            )
            if not is_live:
                print(f"[DENIED] Liveness failed (score={live_score:.3f}). Possible spoof.")
                security.record_failure(args.user)
                return 1
            logger.info("Liveness passed (score=%.3f).", live_score)

    # ── Deepfake check ─────────────────────────────────────────────────
    if args.deepfake:
        is_fake, fake_score = m["DeepfakeDetector"]().predict(img)
        if is_fake:
            print(f"[DENIED] Deepfake detected (score={fake_score:.3f}).")
            security.record_failure(args.user)
            m["log_security_alert"](args.user, "deepfake", {"score": fake_score})
            return 1
        logger.info("Deepfake check passed (score=%.3f).", fake_score)

    # ── Face preprocessing ─────────────────────────────────────────────
    faces = m["preprocess_image"](img)
    if not faces:
        print("[DENIED] No face detected.")
        security.record_failure(args.user)
        return 1
    face = faces[0]

    # ── Classical model score ──────────────────────────────────────────
    classical_score = 0.0
    classical_identity = "unknown"
    if config.EIGENFACES_MODEL_PATH.exists():
        try:
            ef = m["EigenfacesModel"]()
            ef.load(config.EIGENFACES_MODEL_PATH)
            classical_identity, classical_score = ef.predict(face["gray"])
        except Exception as exc:
            logger.warning("Eigenfaces predict error: %s", exc)

    # ── Deep model + vault verification ────────────────────────────────
    deep_model = m["DeepFaceModel"]()
    emb = deep_model.extract_embedding(face["color"])

    deep_score = 0.0
    deep_identity = "unknown"

    if emb is not None:
        import numpy as np
        # Pad / truncate to match cancelable projection input dim
        target = config.PROJECTION_INPUT_DIM
        if emb.shape[0] < target:
            emb = np.pad(emb, (0, target - emb.shape[0]))
        elif emb.shape[0] > target:
            emb = emb[:target]

        cancelable = m["CancelableBiometrics"]()
        protected = cancelable.protect(emb, args.token)
        vault = m["EmbeddingVault"]()
        matched, vault_sim = vault.verify(args.user, protected, args.password)
        deep_score = vault_sim
        if matched:
            deep_identity = args.user

        # Gallery identification (1-to-N)
        gallery = m["_load_gallery"]()
        if gallery:
            gid, gsim = deep_model.identify(face["color"], gallery)
            if gid != "unknown":
                deep_identity = gid
                deep_score = max(deep_score, gsim)

    # ── Score fusion ───────────────────────────────────────────────────
    result = m["Authenticator"]().authenticate(
        classical_score=classical_score,
        deep_score=deep_score,
        claimed_identity=args.user,
        deep_identity=deep_identity,
    )

    m["log_auth_event"](
        user_id=args.user,
        decision=result.decision,
        combined_score=result.combined_score,
        classical_score=result.classical_score,
        deep_score=result.deep_score,
    )

    if result.decision == "GRANTED":
        security.record_success(args.user)
        print(f"[GRANTED] Welcome, {args.user}! Score={result.combined_score:.3f}")
        return 0

    security.record_failure(args.user)
    print(
        f"[DENIED] Score={result.combined_score:.3f} "
        f"(threshold={config.AUTH_THRESHOLD}). {result.detail}"
    )
    return 1


# ---------------------------------------------------------------------------
# delete / list / train / eval
# ---------------------------------------------------------------------------

def cmd_delete(args) -> int:
    m = _imports()
    ok = m["EnrollmentManager"]().delete_user(args.user)
    print(f"[OK] '{args.user}' deleted." if ok else f"[WARN] '{args.user}' not found.")
    return 0


def cmd_list(_args) -> int:
    m = _imports()
    users = m["EnrollmentManager"]().list_enrolled()
    if not users:
        print("No users enrolled.")
    else:
        print(f"Enrolled users ({len(users)}):")
        for u in sorted(users):
            print(f"  • {u}")
    return 0


def cmd_train(_args) -> int:
    m = _imports()
    eigen_ok, lbph_ok = m["EnrollmentManager"]().train_classical_models()
    print(f"Eigenfaces: {'OK' if eigen_ok else 'FAIL'}  |  LBPH: {'OK' if lbph_ok else 'FAIL'}")
    return 0 if (eigen_ok or lbph_ok) else 1


def cmd_eval(_args) -> int:
    from evaluation import Evaluator
    ev = Evaluator()
    if not ev.load_from_logs(days=90):
        print("[WARN] Not enough labelled log data for evaluation.")
        return 1
    metrics = ev.run()
    if metrics:
        print("\n=== Evaluation Results ===")
        for k, v in metrics.items():
            print(f"  {k:<35} {v}")
        return 0
    return 1


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="face_auth",
        description="Face Recognition-Based Authentication System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # enroll
    pe = sub.add_parser("enroll", help="Register a new user")
    pe.add_argument("--user", required=True, help="Unique user ID")
    pe.add_argument("--images", nargs="+", default=[], metavar="IMG",
                    help="Image file paths to enroll from")
    pe.add_argument("--password", required=True, help="Vault encryption password")
    pe.add_argument("--token", required=True, help="Cancelable biometrics token")
    pe.add_argument("--camera", action="store_true", help="Use webcam instead of files")
    pe.add_argument("--camera-id", type=int, default=0, dest="camera_id")
    pe.add_argument("--n-captures", type=int, default=10, dest="n_captures",
                    help="Number of webcam frames to capture")

    # auth
    pa = sub.add_parser("auth", help="Authenticate a user")
    pa.add_argument("--user", required=True)
    pa.add_argument("--image", default=None, metavar="IMG")
    pa.add_argument("--password", required=True)
    pa.add_argument("--token", required=True)
    pa.add_argument("--camera", action="store_true")
    pa.add_argument("--camera-id", type=int, default=0, dest="camera_id")
    pa.add_argument("--liveness", action="store_true", help="Run liveness check")
    pa.add_argument("--deepfake", action="store_true", help="Run deepfake check")

    # delete
    pd = sub.add_parser("delete", help="Remove a user's enrollment")
    pd.add_argument("--user", required=True)

    # list
    sub.add_parser("list", help="Show enrolled users")

    # train
    sub.add_parser("train", help="Retrain classical models from enrolled faces")

    # eval
    sub.add_parser("eval", help="Run performance evaluation from auth logs")

    return parser


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    dispatch = {
        "enroll": cmd_enroll,
        "auth":   cmd_auth,
        "delete": cmd_delete,
        "list":   cmd_list,
        "train":  cmd_train,
        "eval":   cmd_eval,
    }
    return dispatch[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
