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

const MAX_POLL_ITERATIONS = 300;

const state = {
  file: null,
  includeVocals: true,
  hasPiano: true,
  style: 'pop_ballad',
  backendAvailable: false,
  runs: new Map(),
  activeRunId: null,
  pollTimers: new Map(),
};

const $ = s => document.querySelector(s);
const $$ = s => document.querySelectorAll(s);

const uploadZone = $('#uploadZone');
const fileInput = $('#fileInput');
const fileName = $('#fileName');
const fileSize = $('#fileSize');
const clearFile = $('#clearFile');
const submitBtn = $('#submitBtn');
const progressSection = $('#progressSection');
const progressTitle = $('#progressTitle');
const progressFilename = $('#progressFilename');
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
const queuePanel = $('#queuePanel');
const queueList = $('#queueList');
const queueCount = $('#queueCount');
const cancelBtn = $('#cancelBtn');

function init() {
  initScene(document.getElementById('scene-container'));
  detectBackend();
  bindEvents();
}

function bindEvents() {
  $$('.toggle-group').forEach(group => {
    group.querySelectorAll('.toggle-btn').forEach(btn => {
      btn.addEventListener('click', () => {
        const activeRun = state.activeRunId ? state.runs.get(state.activeRunId) : null;
        if (activeRun && (activeRun.status === 'running' || activeRun.status === 'pending')) return;
        group.querySelectorAll('.toggle-btn').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        const key = group.id === 'vocalsToggle' ? 'includeVocals' : 'hasPiano';
        state[key] = btn.dataset.value === 'true';
      });
    });
  });

  $('#styleSelect').addEventListener('change', e => { state.style = e.target.value; });

  uploadZone.addEventListener('click', () => {
    const activeRun = state.activeRunId ? state.runs.get(state.activeRunId) : null;
    if (activeRun && (activeRun.status === 'running' || activeRun.status === 'pending')) return;
    fileInput.click();
  });
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
  cancelBtn.addEventListener('click', cancelActiveRun);
}

function detectBackend() {
  function check() {
    fetch(API_BASE + '/api/runs', { method: 'HEAD', cache: 'no-store' })
      .then(r => {
        state.backendAvailable = r.ok;
        if (r.ok) {
          navBadge.classList.add('visible');
          loadExistingRuns();
        }
      })
      .catch(() => {});
  }
  check();
  setTimeout(check, 2000);
}

async function loadExistingRuns() {
  try {
    const res = await fetch(API_BASE + '/api/runs', { cache: 'no-store' });
    if (!res.ok) return;
    const runs = await res.json();

    for (const run of runs) {
      state.runs.set(run.run_id, run);
      if (run.status === 'running' || run.status === 'pending') {
        startPolling(run.run_id);
      }
    }

    if (runs.length > 0) {
      const activeRun = runs.find(r => r.status === 'running' || r.status === 'pending');
      if (activeRun) {
        selectRun(activeRun.run_id);
      } else {
        selectRun(runs[0].run_id);
      }
      renderQueue();
    }
  } catch (e) {
    console.error('Failed to load runs:', e);
  }
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
  if (!state.file) return;

  const fd = new FormData();
  fd.append('file', state.file);
  fd.append('include_vocals', String(state.includeVocals));
  fd.append('has_piano', String(state.hasPiano));
  fd.append('pattern', state.style);

  resetFile();

  if (state.backendAvailable) {
    await runWithBackend(fd);
  } else {
    await runSimulated();
  }
}

async function runWithBackend(fd) {
  try {
    const runRes = await fetch(API_BASE + '/api/run', { method: 'POST', body: fd });
    if (!runRes.ok) throw new Error('Server error ' + runRes.status);
    const { run_id } = await runRes.json();

    const runData = {
      run_id,
      status: 'pending',
      progress: 0,
      step: '',
      steps_completed: [],
      filename: state.file ? state.file.name : 'unknown',
      created_at: new Date().toISOString(),
    };
    state.runs.set(run_id, runData);
    selectRun(run_id);
    renderQueue();
    startPolling(run_id);

    showToast('Pipeline queued: ' + runData.filename, 'success');
  } catch (err) {
    showToast('Failed to start pipeline: ' + (err.message || 'Unknown error'), 'error');
  }
}

function startPolling(runId) {
  if (state.pollTimers.has(runId)) return;

  let iterations = 0;
  const startTime = Date.now();

  const poll = async () => {
    if (iterations++ >= MAX_POLL_ITERATIONS) {
      stopPolling(runId);
      const run = state.runs.get(runId);
      if (run) {
        run.status = 'failed';
        run.error = 'Pipeline timed out after 5 minutes';
      }
      renderQueue();
      if (state.activeRunId === runId) showRunStatus(runId);
      return;
    }

    try {
      const res = await fetch(API_BASE + '/api/status/' + runId);
      if (!res.ok) throw new Error('Status check failed');
      const data = await res.json();

      state.runs.set(runId, data);
      renderQueue();

      if (state.activeRunId === runId) {
        showRunStatus(runId);
      }

      if (data.status === 'completed') {
        stopPolling(runId);
        if (state.activeRunId === runId) {
          showResults(runId, data);
          setPulse(1);
          setTimeout(() => setPulse(0), 2000);
        }
        showToast('Pipeline complete: ' + (data.filename || runId), 'success');
      } else if (data.status === 'failed' || data.status === 'cancelled') {
        stopPolling(runId);
        if (state.activeRunId === runId) {
          showToast('Pipeline ' + data.status + ': ' + (data.error || 'Unknown error'), 'error');
          setPulse(0);
        }
      }
    } catch (e) {
      console.error('Poll error for', runId, e);
    }
  };

  const timer = setInterval(poll, 1000);
  state.pollTimers.set(runId, timer);
  poll();
}

function stopPolling(runId) {
  const timer = state.pollTimers.get(runId);
  if (timer) {
    clearInterval(timer);
    state.pollTimers.delete(runId);
  }
}

function selectRun(runId) {
  state.activeRunId = runId;
  renderQueue();
  showRunStatus(runId);
}

function showRunStatus(runId) {
  const run = state.runs.get(runId);
  if (!run) return;

  progressSection.classList.add('visible');
  progressTitle.textContent = 'Pipeline: ' + (run.run_id || '');
  progressFilename.textContent = run.filename || '';

  resetSteps();

  if (run.status === 'completed') {
    progressFill.style.width = '100%';
    progressPct.textContent = '100%';
    progressElapsed.textContent = run.duration ? run.duration.toFixed(1) + 's' : 'Done';
    cancelBtn.style.display = 'none';

    stepsContainer.querySelectorAll('.step').forEach(el => {
      if (el.classList.contains('skipped')) return;
      el.classList.remove('active');
      el.classList.add('done');
      el.querySelector('.step-indicator').textContent = '\u2713';
    });

    showResults(runId, run);
  } else if (run.status === 'failed' || run.status === 'cancelled') {
    progressFill.style.width = '100%';
    progressFill.style.background = 'var(--error)';
    progressPct.textContent = run.status === 'cancelled' ? 'Cancelled' : 'Failed';
    cancelBtn.style.display = 'none';
    showToast(run.error || ('Pipeline ' + run.status), 'error');
  } else {
    cancelBtn.style.display = '';
    const completed = run.steps_completed || [];
    const pct = run.progress || Math.min(Math.round((completed.length / STEP_NAMES.length) * 100), 99);
    progressFill.style.width = pct + '%';
    progressFill.style.background = '';
    progressPct.textContent = pct + '%';

    if (run.created_at) {
      const elapsedSecs = Math.floor((Date.now() - new Date(run.created_at).getTime()) / 1000);
      progressElapsed.textContent = elapsedSecs < 60 ? elapsedSecs + 's' : Math.floor(elapsedSecs/60) + 'm ' + (elapsedSecs%60) + 's';
    }

    setPulse(pct / 100);

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
        if (!el || (!el.classList.contains('skipped') && !el.classList.contains('done'))) break;
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
}

async function cancelActiveRun() {
  if (!state.activeRunId) return;
  const run = state.runs.get(state.activeRunId);
  if (!run || (run.status !== 'running' && run.status !== 'pending')) return;

  try {
    const res = await fetch(API_BASE + '/api/cancel/' + state.activeRunId, { method: 'POST' });
    if (!res.ok) throw new Error('Cancel failed');
    run.status = 'cancelled';
    run.error = 'Cancelled by user';
    stopPolling(state.activeRunId);
    renderQueue();
    showRunStatus(state.activeRunId);
    showToast('Pipeline cancelled', 'error');
  } catch (e) {
    showToast('Failed to cancel: ' + e.message, 'error');
  }
}

function renderQueue() {
  const runs = Array.from(state.runs.values());
  runs.sort((a, b) => new Date(b.created_at) - new Date(a.created_at));

  const activeRuns = runs.filter(r => r.status === 'running' || r.status === 'pending');
  const completedRuns = runs.filter(r => r.status === 'completed' || r.status === 'failed' || r.status === 'cancelled');

  if (runs.length === 0) {
    queuePanel.style.display = 'none';
    return;
  }

  queuePanel.style.display = '';
  queueCount.textContent = runs.length + ' run' + (runs.length !== 1 ? 's' : '');

  queueList.innerHTML = '';
  for (const run of runs) {
    const item = document.createElement('div');
    item.className = 'queue-item' + (run.run_id === state.activeRunId ? ' active' : '');
    item.dataset.runId = run.run_id;

    const statusClass = run.status || 'pending';
    const progressText = run.status === 'completed' ? '100%' :
                         run.status === 'failed' ? 'Failed' :
                         run.status === 'cancelled' ? 'Cancelled' :
                         run.status === 'running' ? (run.progress || 0) + '%' :
                         'Queued';

    item.innerHTML = `
      <div class="queue-item-status ${statusClass}"></div>
      <div class="queue-item-info">
        <div class="queue-item-name">${run.filename || run.run_id}</div>
        <div class="queue-item-meta">${run.run_id} &middot; ${formatTimeAgo(run.created_at)}</div>
      </div>
      <div class="queue-item-progress">${progressText}</div>
    `;

    item.addEventListener('click', () => selectRun(run.run_id));
    queueList.appendChild(item);
  }
}

function formatTimeAgo(isoString) {
  if (!isoString) return '';
  const secs = Math.floor((Date.now() - new Date(isoString).getTime()) / 1000);
  if (secs < 60) return secs + 's ago';
  if (secs < 3600) return Math.floor(secs / 60) + 'm ago';
  return Math.floor(secs / 3600) + 'h ago';
}

async function runSimulated() {
  const runId = 'demo_' + Date.now().toString(36);
  const runData = {
    run_id: runId,
    status: 'running',
    progress: 0,
    steps_completed: [],
    filename: state.file ? state.file.name : 'demo.mp3',
    created_at: new Date().toISOString(),
  };
  state.runs.set(runId, runData);
  selectRun(runId);
  renderQueue();

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
    el.querySelector('.step-indicator').textContent = '\u2713';
    el.querySelector('.step-time').textContent = (delay / 1000).toFixed(1) + 's';

    const pct = Math.round(((i + 1) / n) * 100);
    progressFill.style.width = pct + '%';
    progressPct.textContent = pct + '%';
    setPulse(pct / 100);
    elapsed += delay;

    runData.progress = pct;
    runData.steps_completed = STEP_NAMES.slice(0, i + 1).map(s => s.toLowerCase().replace(/\s+/g, '_'));
    renderQueue();

    const elapsedSecs = Math.floor(elapsed / 1000);
    progressElapsed.textContent = elapsedSecs < 60 ? elapsedSecs + 's' : Math.floor(elapsedSecs/60) + 'm ' + (elapsedSecs%60) + 's';
  }

  runData.status = 'completed';
  runData.progress = 100;
  runData.duration = elapsed / 1000;
  renderQueue();

  showResults(runId, {
    duration_seconds: elapsed / 1000,
    steps_completed: STEP_NAMES.map(s => s.toLowerCase().replace(/\s+/g, '_')),
    warnings: [],
    midi_path: runId + '_final.mid',
    video_path: runId + '_synthesia.mp4',
  });

  showToast('Arrangement complete!', 'success');
  setTimeout(() => setPulse(0), 2000);
}

function showResults(runId, data) {
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

  const secs = data.duration_seconds || data.duration || 0;
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
  progressFill.style.width = '0%';
  progressFill.style.background = '';
  progressPct.textContent = '0%';
  progressElapsed.textContent = '0s';
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

document.addEventListener('DOMContentLoaded', init);
