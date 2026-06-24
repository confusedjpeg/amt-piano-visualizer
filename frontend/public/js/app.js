import { initScene, setPulse, resize } from './scene.js';

const API_BASE = window.location.origin;

const STEP_NAMES = ['Stem Separation', 'Vocal Transcription', 'Arrangement Generation', 'MIDI Processing', 'Video Rendering'];

const STEP_KEY_TO_INDEX = {
  'stem_separation': 0,
  'vocal_transcription': 1,
  'vocal_skipped': 1,
  'accompaniment_generation': 2,
  'piano_transcription': 2,
  'algorithmic_arrangement': 2,
  'noise_gate': 2,
  'midi_processing': 3,
  'playability_filter': 3,
  'video_rendering': 4,
};

const MAX_POLL_ITERATIONS = 300; // 5 minutes at 1s intervals

const state = {
  file: null,
  includeVocals: true,
  hasPiano: true,
  style: 'pop_ballad',
  hands: 'both',
  running: false,
  completed: false,
  backendAvailable: false,
  runId: null,
};

const $ = s => document.querySelector(s);
const $$ = s => document.querySelectorAll(s);

// DOM refs
const uploadZone = $('#uploadZone');
const fileInput = $('#fileInput');
const fileName = $('#fileName');
const fileSize = $('#fileSize');
const clearFile = $('#clearFile');
const submitBtn = $('#submitBtn');
const progressSection = $('#progressSection');
const progressFill = $('#progressFill');
const progressPct = $('#progressPct');
const progressElapsed = $('#progressElapsed');
const stepsContainer = $('#stepsContainer');
const resultsSection = $('#resultsSection');
const midiPath = $('#midiPath');
const videoPath = $('#videoPath');
const midiDownload = $('#midiDownload');
const videoToggle = $('#videoToggle');
const videoContainer = $('#videoContainer');
const videoPlayer = $('#videoPlayer');
const resultsMeta = $('#resultsMeta');
const toasts = $('#toasts');
const navBadge = $('#navBadge');

function init() {
  initScene(document.getElementById('scene-container'));
  detectBackend();
  bindEvents();
}

function bindEvents() {
  // Toggles
  $$('.toggle-group').forEach(group => {
    group.querySelectorAll('.toggle-btn').forEach(btn => {
      btn.addEventListener('click', () => {
        if (state.running) return;
        group.querySelectorAll('.toggle-btn').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        const key = group.id === 'vocalsToggle' ? 'includeVocals' : 'hasPiano';
        state[key] = btn.dataset.value === 'true';
      });
    });
  });

  $('#styleSelect').addEventListener('change', e => { state.style = e.target.value; });
  $('#handsSelect').addEventListener('change', e => { state.hands = e.target.value; });

  // Upload
  uploadZone.addEventListener('click', () => { if (!state.running) fileInput.click(); });
  fileInput.addEventListener('change', e => { if (e.target.files.length) handleFile(e.target.files[0]); });

  uploadZone.addEventListener('dragover', e => { e.preventDefault(); uploadZone.classList.add('dragover'); });
  uploadZone.addEventListener('dragleave', () => uploadZone.classList.remove('dragover'));
  uploadZone.addEventListener('drop', e => {
    e.preventDefault();
    uploadZone.classList.remove('dragover');
    if (e.dataTransfer.files.length) handleFile(e.dataTransfer.files[0]);
  });

  clearFile.addEventListener('click', e => { e.stopPropagation(); resetFile(); });
  submitBtn.addEventListener('click', runPipeline);
}

function detectBackend() {
  function check() {
    fetch(API_BASE + '/api/runs', { method: 'HEAD', cache: 'no-store' })
      .then(r => {
        state.backendAvailable = r.ok;
        if (r.ok) navBadge.classList.add('visible');
      })
      .catch(() => {});
  }
  check();
  setTimeout(check, 2000);
}

function handleFile(file) {
  if (!file.type.startsWith('audio/') && !file.name.match(/\.(mp3|wav|m4a|flac|ogg|aac)$/i)) {
    return showToast('Please select a valid audio file.', 'error');
  }
  if (file.size > 100 * 1024 * 1024) {
    return showToast('File exceeds 100 MB limit.', 'error');
  }
  state.file = file;
  fileName.textContent = file.name;
  fileSize.textContent = formatSize(file.size);
  uploadZone.classList.add('has-file');
  submitBtn.disabled = false;
}

function resetFile() {
  state.file = null;
  fileInput.value = '';
  uploadZone.classList.remove('has-file');
  submitBtn.disabled = true;
}

function formatSize(bytes) {
  if (bytes < 1024) return bytes + ' B';
  if (bytes < 1048576) return (bytes / 1024).toFixed(1) + ' KB';
  return (bytes / 1048576).toFixed(1) + ' MB';
}

async function runPipeline() {
  if (!state.file || state.running) return;

  state.running = true;
  state.completed = false;
  submitBtn.classList.add('loading');
  submitBtn.disabled = true;
  uploadZone.style.pointerEvents = 'none';
  clearFile.style.display = 'none';

  progressSection.classList.add('visible');
  resetSteps();
  setPulse(0);

  if (state.backendAvailable) {
    await runWithBackend();
  } else {
    await runSimulated();
  }

  state.running = false;
  submitBtn.classList.remove('loading');
  submitBtn.disabled = state.completed;
  uploadZone.style.pointerEvents = '';
  clearFile.style.display = '';
}

async function runWithBackend() {
  try {
    const fd = new FormData();
    fd.append('file', state.file);
    fd.append('include_vocals', String(state.includeVocals));
    fd.append('has_piano', String(state.hasPiano));
    fd.append('pattern', state.style);
    fd.append('hands', state.hands);

    const runRes = await fetch(API_BASE + '/api/run', { method: 'POST', body: fd });
    if (!runRes.ok) throw new Error('Server error ' + runRes.status);
    const { run_id } = await runRes.json();
    state.runId = run_id;
    const startTime = Date.now();

    let done = false;
    let iterations = 0;

    while (!done) {
      if (iterations++ >= MAX_POLL_ITERATIONS) throw new Error('Pipeline timed out after 5 minutes');
      await sleep(1000);
      const res = await fetch(API_BASE + '/api/status/' + run_id);
      if (!res.ok) throw new Error('Status check failed');
      const data = await res.json();

      if (data.status === 'failed') throw new Error(data.error || 'Pipeline failed');
      if (data.status === 'completed') done = true;

      const completed = data.steps_completed || [];
      const pct = data.progress || Math.min(Math.round((completed.length / STEP_NAMES.length) * 100), 99);
      progressFill.style.width = pct + '%';
      progressPct.textContent = pct + '%';
      setPulse(pct / 100);

      const elapsedSecs = Math.floor((Date.now() - startTime) / 1000);
      progressElapsed.textContent = elapsedSecs < 60 ? elapsedSecs + 's' : Math.floor(elapsedSecs/60) + 'm ' + (elapsedSecs%60) + 's';

      for (const key of completed) {
        const idx = STEP_KEY_TO_INDEX[key];
        if (idx !== undefined) {
          const el = stepsContainer.querySelector(`[data-step="${idx}"]`);
          if (el && !el.classList.contains('done') && !el.classList.contains('skipped')) {
            el.classList.remove('active');
            el.classList.add('done');
            el.querySelector('.step-indicator').textContent = '\u2713';
          }
        }
      }

      const activeIdx = (() => {
        let idx = completed.length < STEP_NAMES.length ? completed.length : STEP_NAMES.length - 1;
        while (idx < STEP_NAMES.length) {
          const el = stepsContainer.querySelector(`[data-step="${idx}"]`);
          if (!el || !el.classList.contains('skipped')) break;
          idx++;
        }
        return Math.min(idx, STEP_NAMES.length - 1);
      })();
      stepsContainer.querySelectorAll('.step').forEach(el => {
        const i = parseInt(el.dataset.step);
        if (i === activeIdx && !el.classList.contains('done') && !el.classList.contains('skipped')) {
          el.classList.add('active');
        }
      });

    }

    progressFill.style.width = '100%';
    progressPct.textContent = '100%';
    setPulse(1);

    const final = await (await fetch(API_BASE + '/api/status/' + run_id)).json();

    stepsContainer.querySelectorAll('.step').forEach(el => {
      if (el.classList.contains('skipped')) return;
      el.classList.remove('active');
      el.classList.add('done');
      el.querySelector('.step-indicator').textContent = '\u2713';
    });

    showResults(run_id, final);
    setTimeout(() => setPulse(0), 2000);

  } catch (err) {
    showToast('Pipeline failed: ' + (err.message || 'Unknown error'), 'error');
    setPulse(0);
    state.completed = false;
    resetSteps();
  }
}

async function runSimulated() {
  const stepDurations = [[1800, 3000], [2000, 3500], [2800, 4800], [1000, 2000], [3500, 6000]];
  const n = stepDurations.length;
  let elapsed = 0;

  for (let i = 0; i < n; i++) {
    if (i === 1 && !state.includeVocals) {
      const el = stepsContainer.querySelector(`[data-step="${i}"]`);
      el.classList.add('skipped');
      el.querySelector('.step-indicator').textContent = '\u2014';
      continue;
    }
    const delay = rand(stepDurations[i][0], stepDurations[i][1]);
    const el = stepsContainer.querySelector(`[data-step="${i}"]`);
    el.classList.add('active');

    await sleep(delay);

    el.classList.remove('active');
    el.classList.add('done');
    el.querySelector('.step-indicator').textContent = '✓';
    el.querySelector('.step-time').textContent = (delay / 1000).toFixed(1) + 's';

    const pct = Math.round(((i + 1) / n) * 100);
    progressFill.style.width = pct + '%';
    progressPct.textContent = pct + '%';
    setPct(pct);
    elapsed += delay;
  }

  const runId = 'demo_' + Date.now().toString(36);
  showResults(runId, {
    duration_seconds: elapsed / 1000,
    steps_completed: (state.includeVocals ? STEP_NAMES : STEP_NAMES.filter((_, i) => i !== 1)).map(s => s.toLowerCase().replace(/\s+/g, '_')),
    warnings: [],
    midi_path: runId + '_final.mid',
    video_path: runId + '_synthesia.mp4',
  });

  showToast('Arrangement complete!', 'success');
  setTimeout(() => setPulse(0), 2000);
}

let pulseTimer = null;
function setPct(pct) {
  clearTimeout(pulseTimer);
  setPulse(pct / 100);
  if (pct >= 100) {
    pulseTimer = setTimeout(() => setPulse(0), 2500);
  }
}

function showResults(runId, data) {
  state.completed = true;
  state.runId = runId;
  resultsSection.classList.add('visible');
  videoContainer.classList.remove('visible');
  videoToggle.textContent = 'Show Video';

  const midiName = data.midi_path ? data.midi_path.replace(/\\/g, '/').split('/').pop() : runId + '_final.mid';
  const videoName = data.video_path ? data.video_path.replace(/\\/g, '/').split('/').pop() : runId + '_synthesia.mp4';

  midiPath.textContent = midiName;
  videoPath.textContent = videoName;

  if (state.backendAvailable && data.midi_path) {
    midiDownload.href = API_BASE + '/api/download/' + runId + '/midi';
    midiDownload.download = midiName;
    midiDownload.style.pointerEvents = '';
    midiDownload.style.opacity = '';
  } else {
    midiDownload.removeAttribute('href');
    midiDownload.download = '';
    midiDownload.style.pointerEvents = 'none';
    midiDownload.style.opacity = '0.5';
  }

  videoToggle.onclick = () => {
    if (videoContainer.classList.contains('visible')) {
      videoContainer.classList.remove('visible');
      videoToggle.textContent = 'Show Video';
    } else if (state.backendAvailable && data.video_path) {
      videoContainer.classList.add('visible');
      videoToggle.textContent = 'Hide Video';
      videoPlayer.src = API_BASE + '/api/download/' + runId + '/video';
      videoPlayer.load();
    } else {
      videoContainer.classList.add('visible');
      videoToggle.textContent = 'Hide Video';
    }
  };

  const secs = data.duration_seconds || 0;
  const dur = secs < 60 ? secs.toFixed(1) + 's' : (secs / 60).toFixed(1) + 'm';
  const warnings = data.warnings || [];

  resultsMeta.innerHTML = `
    <span class="meta-item">Run: <strong>${runId}</strong></span>
    <span class="meta-item">Duration: <strong>${dur}</strong></span>
    <span class="meta-item">Warnings: <strong>${warnings.length}</strong></span>
  `;
}

function resetSteps() {
  stepsContainer.querySelectorAll('.step').forEach(el => {
    el.classList.remove('active', 'done', 'skipped');
    el.querySelector('.step-indicator').textContent = parseInt(el.dataset.step) + 1;
    el.querySelector('.step-time').textContent = '';
  });
  const vocalStep = stepsContainer.querySelector('[data-step="1"]');
  if (!state.includeVocals) {
    vocalStep.classList.add('skipped');
    vocalStep.querySelector('.step-indicator').textContent = '\u2014';
  } else {
    vocalStep.classList.remove('skipped');
  }
  progressFill.style.width = '0%';
  progressPct.textContent = '0%';
  if (typeof progressElapsed !== 'undefined') progressElapsed.textContent = '0s';
  resultsSection.classList.remove('visible');
  videoContainer.classList.remove('visible');
}

function sleep(ms) { return new Promise(r => setTimeout(r, ms)); }
function rand(min, max) { return Math.floor(Math.random() * (max - min + 1)) + min; }

function showToast(msg, type) {
  const el = document.createElement('div');
  el.className = 'toast' + (type ? ' ' + type : '');
  el.textContent = msg;
  toasts.appendChild(el);
  setTimeout(() => { el.style.opacity = '0'; el.style.transition = 'opacity 0.3s'; }, 3500);
  setTimeout(() => el.remove(), 4000);
}

// Boot
document.addEventListener('DOMContentLoaded', init);
