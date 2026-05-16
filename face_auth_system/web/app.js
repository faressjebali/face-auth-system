'use strict';

// ── Config ─────────────────────────────────────────────────────────────────
const MODEL_URL         = 'https://vladmandic.github.io/face-api/model/';
const MATCH_THRESHOLD   = 0.50;   // euclidean distance (lower = stricter)
const CONFIRM_NEEDED    = 3;      // consecutive matching frames before login

// ── UserStore (localStorage) ───────────────────────────────────────────────
class UserStore {
  constructor() {
    this._users   = JSON.parse(localStorage.getItem('nexus_users')   || '{}');
    this._gallery = this._loadGallery();
  }

  _loadGallery() {
    const raw = JSON.parse(localStorage.getItem('nexus_gallery') || '{}');
    const out = {};
    for (const [uid, descs] of Object.entries(raw)) {
      out[uid] = descs.map(d => new Float32Array(d));
    }
    return out;
  }

  _saveGallery() {
    const raw = {};
    for (const [uid, descs] of Object.entries(this._gallery)) {
      raw[uid] = descs.map(d => Array.from(d));
    }
    localStorage.setItem('nexus_gallery', JSON.stringify(raw));
  }

  create(firstName, lastName, dob) {
    const uid = 'u_' + Date.now() + '_' + Math.random().toString(36).slice(2, 8);
    this._users[uid] = { firstName, lastName, dob, createdAt: Date.now() };
    localStorage.setItem('nexus_users', JSON.stringify(this._users));
    return uid;
  }

  addFace(uid, descriptor) {
    if (!this._gallery[uid]) this._gallery[uid] = [];
    this._gallery[uid].push(new Float32Array(descriptor));
    this._saveGallery();
  }

  getUser(uid) { return this._users[uid] || null; }

  hasUsers() { return Object.keys(this._gallery).length > 0; }

  findBestMatch(descriptor) {
    let bestUid = null, bestDist = Infinity;
    for (const [uid, descs] of Object.entries(this._gallery)) {
      for (const stored of descs) {
        const dist = faceapi.euclideanDistance(descriptor, stored);
        if (dist < bestDist) { bestDist = dist; bestUid = uid; }
      }
    }
    return bestDist <= MATCH_THRESHOLD ? { uid: bestUid, dist: bestDist } : null;
  }
}

const store = new UserStore();
let modelsLoaded = false;

// ── Neural mesh canvas ─────────────────────────────────────────────────────
(function () {
  const canvas = document.getElementById('neural-canvas');
  const ctx    = canvas.getContext('2d');
  const COUNT  = 55, DIST = 118;
  let W, H, pts;

  function init() {
    W = canvas.width  = innerWidth;
    H = canvas.height = innerHeight;
    pts = Array.from({ length: COUNT }, () => ({
      x:  Math.random() * W,
      y:  Math.random() * H,
      vx: (Math.random() - 0.5) * 0.28,
      vy: (Math.random() - 0.5) * 0.28,
      r:  Math.random() * 1.2 + 0.5,
    }));
  }

  function tick() {
    ctx.clearRect(0, 0, W, H);
    for (const p of pts) {
      p.x += p.vx; p.y += p.vy;
      if (p.x < 0 || p.x > W) p.vx *= -1;
      if (p.y < 0 || p.y > H) p.vy *= -1;
    }
    for (let i = 0; i < pts.length; i++) {
      for (let j = i + 1; j < pts.length; j++) {
        const dx = pts[i].x - pts[j].x;
        const dy = pts[i].y - pts[j].y;
        const d  = Math.hypot(dx, dy);
        if (d < DIST) {
          ctx.beginPath();
          ctx.moveTo(pts[i].x, pts[i].y);
          ctx.lineTo(pts[j].x, pts[j].y);
          ctx.strokeStyle = `rgba(0,245,255,${(1 - d / DIST) * 0.15})`;
          ctx.lineWidth = 0.5;
          ctx.stroke();
        }
      }
      ctx.beginPath();
      ctx.arc(pts[i].x, pts[i].y, pts[i].r, 0, 6.283);
      ctx.fillStyle = 'rgba(0,245,255,0.32)';
      ctx.fill();
    }
    requestAnimationFrame(tick);
  }

  init(); tick();
  addEventListener('resize', init);
})();

// ── Status cycling ─────────────────────────────────────────────────────────
const STATUS_SEQ = [
  'INITIALIZING',
  'LOADING NEURAL MODELS',
  'SCANNING ENVIRONMENT',
  'CALIBRATING SENSORS',
  'SYSTEM READY',
];
let statusIdx = 0;
const statusEl = document.getElementById('status-text');
statusEl.style.transition = 'opacity 0.26s ease, transform 0.26s ease';

function cycleStatus() {
  const next = modelsLoaded ? STATUS_SEQ.length - 1 : (statusIdx + 1) % (STATUS_SEQ.length - 1);
  if (next === statusIdx && modelsLoaded) return;
  statusIdx = next;
  statusEl.style.opacity   = '0';
  statusEl.style.transform = 'translateY(-8px)';
  setTimeout(() => {
    statusEl.textContent     = STATUS_SEQ[statusIdx];
    statusEl.style.opacity   = '1';
    statusEl.style.transform = 'translateY(0)';
  }, 260);
}

setTimeout(() => setInterval(cycleStatus, 2500), 1800);

// ── Model loading ──────────────────────────────────────────────────────────
async function loadModels() {
  await Promise.all([
    faceapi.nets.tinyFaceDetector.loadFromUri(MODEL_URL),
    faceapi.nets.faceLandmark68TinyNet.loadFromUri(MODEL_URL),
    faceapi.nets.faceRecognitionNet.loadFromUri(MODEL_URL),
  ]);
  modelsLoaded = true;
}

loadModels().catch(err => console.error('Model load failed:', err));

// ── Wipe transition ────────────────────────────────────────────────────────
function triggerWipe() {
  const el = document.getElementById('wipe');
  el.classList.remove('fire');
  void el.offsetWidth;
  el.classList.add('fire');
}

// ── Toast ──────────────────────────────────────────────────────────────────
function showToast(msg, ms = 2700) {
  const el = document.getElementById('toast');
  el.textContent = msg;
  el.classList.add('visible');
  clearTimeout(el._tid);
  el._tid = setTimeout(() => el.classList.remove('visible'), ms);
}

// ── Screen navigation ──────────────────────────────────────────────────────
function showScreen(id) {
  triggerWipe();
  setTimeout(() => {
    document.querySelectorAll('.screen').forEach(s => s.classList.remove('active'));
    document.getElementById(id).classList.add('active');
  }, 120);
}

// ── Canvas helpers ─────────────────────────────────────────────────────────
function drawBrackets(ctx, x, y, w, h, color) {
  const blen = Math.max(Math.min(w, h) / 4, 12);
  ctx.strokeStyle = color;
  ctx.lineWidth   = 2;
  ctx.lineCap     = 'square';
  ctx.beginPath();
  ctx.moveTo(x,         y + blen); ctx.lineTo(x,         y); ctx.lineTo(x + blen, y);
  ctx.moveTo(x + w - blen, y);     ctx.lineTo(x + w,     y); ctx.lineTo(x + w,    y + blen);
  ctx.moveTo(x,         y + h - blen); ctx.lineTo(x,     y + h); ctx.lineTo(x + blen, y + h);
  ctx.moveTo(x + w - blen, y + h); ctx.lineTo(x + w, y + h); ctx.lineTo(x + w,    y + h - blen);
  ctx.stroke();
}

function syncOverlay(video, canvas) {
  canvas.width  = video.videoWidth  || 640;
  canvas.height = video.videoHeight || 480;
}

const DET_OPTS = new faceapi.TinyFaceDetectorOptions({ inputSize: 320, scoreThreshold: 0.5 });

// ══════════════════════════════════════════════════════════════════
// HOME SCREEN
// ══════════════════════════════════════════════════════════════════
document.getElementById('btn-create').addEventListener('click', function (e) {
  const btn  = this;
  const rect = btn.getBoundingClientRect();
  const size = Math.max(rect.width, rect.height) * 2;
  const r    = document.createElement('span');
  r.className = 'ripple';
  Object.assign(r.style, {
    width:  size + 'px',
    height: size + 'px',
    left:   (e.clientX - rect.left - size / 2) + 'px',
    top:    (e.clientY - rect.top  - size / 2) + 'px',
  });
  btn.appendChild(r);
  r.addEventListener('animationend', () => r.remove());
  goRegister();
});

document.getElementById('btn-create').addEventListener('mouseenter', function () {
  const btn    = this;
  const frames = [
    'hue-rotate(-14deg) brightness(1.09)',
    'hue-rotate(7deg)   brightness(1.05)',
    'hue-rotate(-5deg)  brightness(1.07)',
    'none',
    'hue-rotate(9deg)   brightness(1.04)',
    'hue-rotate(-2deg)',
    'none',
  ];
  let f = 0;
  (function step() {
    btn.style.filter = frames[f];
    if (++f < frames.length) requestAnimationFrame(step);
    else btn.style.filter = '';
  })();
});

document.getElementById('btn-login').addEventListener('click', function () {
  const btn = this;
  if (btn.classList.contains('is-loading')) return;
  btn.classList.add('is-loading');
  const spinner = document.createElement('span');
  spinner.className = 'spinner';
  btn.appendChild(spinner);
  setTimeout(() => {
    spinner.remove();
    btn.classList.remove('is-loading');
    goLogin();
  }, 1200);
});

// ══════════════════════════════════════════════════════════════════
// REGISTER SCREEN
// ══════════════════════════════════════════════════════════════════
let regStream        = null;
let regDetectHandle  = null;
let regCaptures      = [];
let regCaptured      = false;

function goRegister() {
  resetRegisterState();
  showScreen('screen-register');
  updateRegProgress(0);
  setTimeout(startRegCam, 450);
}

function resetRegisterState() {
  regCaptures  = [];
  regCaptured  = false;
  const first  = document.getElementById('reg-first');
  const last   = document.getElementById('reg-last');
  const dob    = document.getElementById('reg-dob');
  if (first) first.value = '';
  if (last)  last.value  = '';
  if (dob)   dob.value   = '';
  document.getElementById('reg-capture').disabled = false;
  document.getElementById('reg-capture').textContent = 'CAPTURE  1/1';
  document.getElementById('reg-submit').disabled = true;
  document.getElementById('reg-pose-instr').textContent = 'Look straight at the camera';
  document.getElementById('reg-step-label').textContent = 'STEP 1 OF 2';
  setRegStatus('Position your face in the camera', 'var(--text-dim)');
  setRegCamStatus('Waiting for camera…', 'var(--text-dim)');
  paintDot('pose-dot-0', '#1a2030');
}

async function startRegCam() {
  stopRegCam();
  try {
    regStream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'user', width: 640, height: 480 } });
    const video = document.getElementById('reg-video');
    video.srcObject = regStream;
    await video.play();
    runRegDetection();
  } catch (e) {
    setRegCamStatus('Camera access denied', 'var(--error)');
  }
}

async function runRegDetection() {
  const video   = document.getElementById('reg-video');
  const overlay = document.getElementById('reg-overlay');
  const ctx     = overlay.getContext('2d');

  async function loop() {
    if (!document.getElementById('screen-register').classList.contains('active')) return;
    syncOverlay(video, overlay);
    ctx.clearRect(0, 0, overlay.width, overlay.height);

    if (modelsLoaded && video.readyState >= 2) {
      const det = await faceapi.detectSingleFace(video, DET_OPTS);
      if (det) {
        drawBrackets(ctx, det.box.x, det.box.y, det.box.width, det.box.height, '#00F5FF');
        if (!regCaptured) setRegCamStatus('Face detected — ready to capture', 'var(--success)');
      } else {
        if (!regCaptured) setRegCamStatus('No face detected', 'var(--text-dim)');
      }
    } else if (!modelsLoaded) {
      setRegCamStatus('Loading AI models…', 'var(--warn)');
    }

    regDetectHandle = requestAnimationFrame(loop);
  }
  loop();
}

async function captureRegFace() {
  if (regCaptured) return;
  if (!modelsLoaded) { setRegStatus('AI models still loading — please wait.', 'var(--warn)'); return; }

  const video = document.getElementById('reg-video');
  if (!video.srcObject) { setRegStatus('Camera not available.', 'var(--error)'); return; }

  setRegStatus('Extracting face embedding…', 'var(--text-dim)');

  const result = await faceapi
    .detectSingleFace(video, DET_OPTS)
    .withFaceLandmarks(true)
    .withFaceDescriptor();

  if (!result) {
    setRegStatus('No face detected — try again.', 'var(--error)');
    return;
  }

  regCaptures.push(result.descriptor);
  regCaptured = true;

  paintDot('pose-dot-0', '#2adf80');
  document.getElementById('reg-capture').disabled = true;
  document.getElementById('reg-submit').disabled  = false;
  document.getElementById('reg-pose-instr').textContent = 'Face captured successfully';
  document.getElementById('reg-step-label').textContent = 'READY TO REGISTER';
  setRegStatus('Face captured — fill in your details and click REGISTER.', 'var(--success)');
  setRegCamStatus('Capture complete', 'var(--success)');
  updateRegProgress(1);
}

async function submitRegistration() {
  const firstName = document.getElementById('reg-first').value.trim();
  const lastName  = document.getElementById('reg-last').value.trim();
  const dob       = document.getElementById('reg-dob').value.trim();

  if (!firstName || !lastName) { setRegStatus('First and last name are required.', 'var(--error)'); return; }
  if (!dob)                     { setRegStatus('Date of birth is required.', 'var(--error)'); return; }
  if (regCaptures.length === 0) { setRegStatus('Capture your face first.', 'var(--error)'); return; }

  const uid = store.create(firstName, lastName, dob);
  for (const desc of regCaptures) store.addFace(uid, desc);

  stopRegCam();
  showToast('ACCOUNT CREATED SUCCESSFULLY');
  setTimeout(() => showScreen('screen-home'), 500);
}

function stopRegCam() {
  if (regDetectHandle) { cancelAnimationFrame(regDetectHandle); regDetectHandle = null; }
  if (regStream)       { regStream.getTracks().forEach(t => t.stop()); regStream = null; }
}

function updateRegProgress(step) {
  const pct = (step / 2) * 100 + '%';
  document.getElementById('reg-progress').style.width = pct;
}

function setRegStatus(msg, color) {
  const el = document.getElementById('reg-status');
  el.textContent   = msg;
  el.style.color   = color;
}

function setRegCamStatus(msg, color) {
  const el = document.getElementById('reg-cam-status');
  el.textContent = msg;
  el.style.color = color;
}

function paintDot(id, fill) {
  const c   = document.getElementById(id);
  if (!c) return;
  const ctx = c.getContext('2d');
  ctx.clearRect(0, 0, 10, 10);
  ctx.beginPath();
  ctx.arc(5, 5, 4, 0, Math.PI * 2);
  ctx.fillStyle = fill;
  ctx.fill();
}

document.getElementById('reg-back').addEventListener('click', () => {
  stopRegCam();
  showScreen('screen-home');
});
document.getElementById('reg-capture').addEventListener('click', captureRegFace);
document.getElementById('reg-submit').addEventListener('click', submitRegistration);

// ══════════════════════════════════════════════════════════════════
// LOGIN SCREEN
// ══════════════════════════════════════════════════════════════════
let loginStream       = null;
let loginDetectHandle = null;
let loginDone         = false;
let confirmUid        = null;
let confirmCount      = 0;

function goLogin() {
  if (!store.hasUsers()) {
    showToast('NO ACCOUNTS FOUND — CREATE AN ACCOUNT FIRST');
    return;
  }
  loginDone    = false;
  confirmUid   = null;
  confirmCount = 0;
  document.getElementById('conf-fill').style.width = '0';
  setLoginStatus('SCANNING…', 'var(--text-dim)');
  showScreen('screen-login');
  setTimeout(startLoginCam, 450);
}

async function startLoginCam() {
  stopLoginCam();
  try {
    loginStream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'user', width: 640, height: 480 } });
    const video = document.getElementById('login-video');
    video.srcObject = loginStream;
    await video.play();
    runLoginDetection();
  } catch (e) {
    setLoginStatus('CAMERA NOT FOUND', 'var(--error)');
  }
}

async function runLoginDetection() {
  const video   = document.getElementById('login-video');
  const overlay = document.getElementById('login-overlay');
  const ctx     = overlay.getContext('2d');

  async function loop() {
    if (loginDone || !document.getElementById('screen-login').classList.contains('active')) return;

    syncOverlay(video, overlay);
    ctx.clearRect(0, 0, overlay.width, overlay.height);

    if (!modelsLoaded) {
      setLoginStatus('LOADING AI MODELS…', 'var(--warn)');
      loginDetectHandle = requestAnimationFrame(loop);
      return;
    }

    if (video.readyState < 2) {
      loginDetectHandle = requestAnimationFrame(loop);
      return;
    }

    const result = await faceapi
      .detectSingleFace(video, DET_OPTS)
      .withFaceLandmarks(true)
      .withFaceDescriptor();

    if (!result) {
      drawBrackets(ctx, overlay.width / 2 - 80, overlay.height / 2 - 80, 160, 160, 'rgba(74,90,114,0.4)');
      setLoginStatus('SCANNING…', 'var(--text-dim)');
      confirmUid   = null;
      confirmCount = 0;
      document.getElementById('conf-fill').style.width = '0';
      loginDetectHandle = requestAnimationFrame(loop);
      return;
    }

    const match = store.findBestMatch(result.descriptor);

    if (match) {
      const user = store.getUser(match.uid);
      const name = user ? `${user.firstName} ${user.lastName}` : 'UNKNOWN';
      const sim  = Math.round((1 - match.dist) * 100);

      drawBrackets(ctx, result.box.x, result.box.y, result.box.width, result.box.height, '#2adf80');
      setLoginStatus(`${name.toUpperCase()}  ${sim}%`, '#00F5FF');

      if (match.uid === confirmUid) {
        confirmCount++;
      } else {
        confirmUid   = match.uid;
        confirmCount = 1;
      }

      const pct = Math.min(confirmCount / CONFIRM_NEEDED, 1);
      document.getElementById('conf-fill').style.width = (pct * 100) + '%';

      if (confirmCount >= CONFIRM_NEEDED && !loginDone) {
        loginDone = true;
        stopLoginCam();
        goWelcome(match.uid);
        return;
      }
    } else {
      drawBrackets(ctx, result.box.x, result.box.y, result.box.width, result.box.height, '#ff2255');
      setLoginStatus('FACE NOT RECOGNISED', 'var(--error)');
      confirmUid   = null;
      confirmCount = 0;
      document.getElementById('conf-fill').style.width = '0';
    }

    loginDetectHandle = requestAnimationFrame(loop);
  }

  loop();
}

function stopLoginCam() {
  if (loginDetectHandle) { cancelAnimationFrame(loginDetectHandle); loginDetectHandle = null; }
  if (loginStream)       { loginStream.getTracks().forEach(t => t.stop()); loginStream = null; }
}

function setLoginStatus(msg, color) {
  const el = document.getElementById('login-status');
  el.textContent = msg;
  el.style.color = color;
}

document.getElementById('login-back').addEventListener('click', () => {
  loginDone = true;
  stopLoginCam();
  showScreen('screen-home');
});

// ══════════════════════════════════════════════════════════════════
// WELCOME SCREEN
// ══════════════════════════════════════════════════════════════════
function goWelcome(uid) {
  const user = store.getUser(uid);
  if (!user) { showScreen('screen-home'); return; }
  document.getElementById('welcome-name').textContent = `${user.firstName} ${user.lastName}`;
  document.getElementById('welcome-dob').textContent  = user.dob;
  showToast('IDENTITY VERIFIED — WELCOME BACK');
  showScreen('screen-welcome');
}

document.getElementById('welcome-signout').addEventListener('click', () => {
  showScreen('screen-home');
});

// ── Draw initial pose dot ──────────────────────────────────────────────────
paintDot('pose-dot-0', '#1a2030');
