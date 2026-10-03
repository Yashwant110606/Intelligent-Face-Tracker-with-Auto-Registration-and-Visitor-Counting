/**
 * AetherVision - Intelligent Face Tracker Frontend Application
 * Handles live stream playback, real-time telemetry polling, interactive charts,
 * visitor gallery exploration, event timeline modals, and system calibration.
 */

// Application State
const state = {
  currentTab: 'tabStream',
  pipelineRunning: false,
  activeTracks: [],
  uniqueCount: 0,
  entriesCount: 0,
  exitsCount: 0,
  activeInFrame: 0,
  currentFPS: 0,
  selectedSource: 'data/video_sample1.mp4',
  auditFilter: 'all',
  auditOffset: 0,
  auditLimit: 25,
  visitorSearch: '',
  visitorSort: 'newest',
  pollInterval: null,
  logInterval: null,
  isLiveFeeding: true
};

// DOM Element Selectors
const elements = {
  // Navigation Tabs
  navTabs: document.querySelectorAll('.nav-tab'),
  tabContents: document.querySelectorAll('.tab-content'),

  // Hero Stats
  statUniqueVisitors: document.getElementById('statUniqueVisitors'),
  statActiveFaces: document.getElementById('statActiveFaces'),
  statTotalEntries: document.getElementById('statTotalEntries'),
  statTotalExits: document.getElementById('statTotalExits'),

  // Header
  systemOnlinePill: document.getElementById('systemOnlinePill'),
  systemStatusText: document.getElementById('systemStatusText'),
  btnRefreshAll: document.getElementById('btnRefreshAll'),

  // Stream Player & Viewport
  videoContainer: document.getElementById('videoContainer'),
  liveVideoFeed: document.getElementById('liveVideoFeed'),
  streamLiveBadge: document.getElementById('streamLiveBadge'),
  streamLiveText: document.getElementById('streamLiveText'),
  currentSourceLabel: document.getElementById('currentSourceLabel'),
  hudResolution: document.getElementById('hudResolution'),
  hudFPS: document.getElementById('hudFPS'),
  btnTogglePipeline: document.getElementById('btnTogglePipeline'),
  pipelineBtnText: document.getElementById('pipelineBtnText'),
  pipelineBtnIcon: document.getElementById('pipelineBtnIcon'),
  inputSourceSelect: document.getElementById('inputSourceSelect'),
  uploadMp4Btn: document.getElementById('uploadMp4Btn'),
  videoFileInput: document.getElementById('videoFileInput'),
  badgeCurrentSkip: document.getElementById('badgeCurrentSkip'),
  btnSnapshot: document.getElementById('btnSnapshot'),
  btnFullscreen: document.getElementById('btnFullscreen'),

  // Calibration Controls
  sliderSkip: document.getElementById('sliderSkip'),
  valSkip: document.getElementById('valSkip'),
  sliderConf: document.getElementById('sliderConf'),
  valConf: document.getElementById('valConf'),
  sliderSim: document.getElementById('sliderSim'),
  valSim: document.getElementById('valSim'),
  btnSaveConfig: document.getElementById('btnSaveConfig'),

  // Active Tracks Sidebar
  sidebarActiveCount: document.getElementById('sidebarActiveCount'),
  activeTracksContainer: document.getElementById('activeTracksContainer'),

  // Visitors Directory
  searchVisitorsInput: document.getElementById('searchVisitorsInput'),
  sortVisitorsSelect: document.getElementById('sortVisitorsSelect'),
  visitorsCountBadge: document.getElementById('visitorsCountBadge'),
  visitorsGridContainer: document.getElementById('visitorsGridContainer'),

  // Audit Logs
  btnFilterAllEvents: document.getElementById('btnFilterAllEvents'),
  btnFilterEntryEvents: document.getElementById('btnFilterEntryEvents'),
  btnFilterExitEvents: document.getElementById('btnFilterExitEvents'),
  btnExportCSV: document.getElementById('btnExportCSV'),
  auditTableBody: document.getElementById('auditTableBody'),
  auditPaginationText: document.getElementById('auditPaginationText'),
  btnAuditPrev: document.getElementById('btnAuditPrev'),
  btnAuditNext: document.getElementById('btnAuditNext'),

  // System Terminal
  terminalLogBody: document.getElementById('terminalLogBody'),
  checkAutoScroll: document.getElementById('checkAutoScroll'),
  btnClearLogsView: document.getElementById('btnClearLogsView'),
  btnResetDatabase: document.getElementById('btnResetDatabase'),

  // Modals
  visitorModalBackdrop: document.getElementById('visitorModalBackdrop'),
  modalVisitorTitle: document.getElementById('modalVisitorTitle'),
  modalVisitorBody: document.getElementById('modalVisitorBody'),
  btnCloseVisitorModal: document.getElementById('btnCloseVisitorModal'),
  btnDismissVisitorModal: document.getElementById('btnDismissVisitorModal'),

  rtspModalBackdrop: document.getElementById('rtspModalBackdrop'),
  rtspUrlInput: document.getElementById('rtspUrlInput'),
  btnCloseRtspModal: document.getElementById('btnCloseRtspModal'),
  btnCancelRtsp: document.getElementById('btnCancelRtsp'),
  btnConfirmRtsp: document.getElementById('btnConfirmRtsp'),

  resetModalBackdrop: document.getElementById('resetModalBackdrop'),
  btnCloseResetModal: document.getElementById('btnCloseResetModal'),
  btnCancelReset: document.getElementById('btnCancelReset'),
  btnConfirmReset: document.getElementById('btnConfirmReset'),

  // Charts
  footfallCanvas: document.getElementById('footfallChartCanvas'),
  donutCanvas: document.getElementById('donutChartCanvas'),

  // Toast Container
  toastContainer: document.getElementById('toastContainer')
};

// ============================================================================
// TOAST NOTIFICATIONS
// ============================================================================
function showToast(message, type = 'success') {
  const toast = document.createElement('div');
  toast.className = `toast toast-${type}`;
  toast.innerHTML = `
    <span style="font-weight: 700; color: ${type === 'success' ? '#10b981' : '#f43f5e'}">
      ${type === 'success' ? '✓' : '⚠'}
    </span>
    <span>${message}</span>
  `;
  elements.toastContainer.appendChild(toast);
  setTimeout(() => {
    toast.style.opacity = '0';
    toast.style.transform = 'translateY(10px)';
    toast.style.transition = 'all 0.3s ease';
    setTimeout(() => toast.remove(), 300);
  }, 3500);
}

// ============================================================================
// TAB NAVIGATION
// ============================================================================
function initTabs() {
  elements.navTabs.forEach(tab => {
    tab.addEventListener('click', () => {
      const targetId = tab.getAttribute('data-tab');
      switchTab(targetId);
    });
  });
}

function switchTab(targetId) {
  state.currentTab = targetId;

  elements.navTabs.forEach(tab => {
    if (tab.getAttribute('data-tab') === targetId) {
      tab.classList.add('active');
    } else {
      tab.classList.remove('active');
    }
  });

  elements.tabContents.forEach(content => {
    if (content.id === targetId) {
      content.classList.add('active');
    } else {
      content.classList.remove('active');
    }
  });

  // Lazy load tab data
  if (targetId === 'tabVisitors') {
    fetchVisitors();
  } else if (targetId === 'tabAudit') {
    fetchAuditEvents();
  } else if (targetId === 'tabDashboard') {
    renderCharts();
  } else if (targetId === 'tabLogs') {
    fetchLogs();
  }
}

// ============================================================================
// REAL-TIME STATS & TELEMETRY POLLING
// ============================================================================
async function fetchStats() {
  try {
    const res = await fetch('/api/stats');
    if (!res.ok) return;
    const data = await res.json();

    // Update state
    state.uniqueCount = data.unique_visitors || 0;
    state.entriesCount = data.total_entries || 0;
    state.exitsCount = data.total_exits || 0;
    state.activeInFrame = data.active_in_frame || 0;
    state.currentFPS = data.fps || 0;
    state.pipelineRunning = !!data.pipeline_running;
    state.activeTracks = data.active_tracks || [];

    // Update Hero UI
    elements.statUniqueVisitors.textContent = state.uniqueCount.toLocaleString();
    elements.statActiveFaces.textContent = state.activeInFrame;
    elements.statTotalEntries.textContent = state.entriesCount.toLocaleString();
    elements.statTotalExits.textContent = state.exitsCount.toLocaleString();

    // Update HUD
    elements.hudFPS.textContent = `FPS: ${state.currentFPS.toFixed(1)}`;
    elements.sidebarActiveCount.textContent = `${state.activeInFrame} in frame`;

    // Update Status Pill
    if (state.pipelineRunning) {
      elements.systemStatusText.textContent = `Pipeline Tracking (${state.currentFPS.toFixed(0)} FPS)`;
      elements.systemOnlinePill.style.borderColor = 'rgba(16, 185, 129, 0.4)';
      elements.streamLiveBadge.style.display = 'flex';
      elements.streamLiveText.textContent = 'STREAM LIVE';
    } else {
      elements.systemStatusText.textContent = 'Engine Standby';
      elements.streamLiveBadge.style.display = 'flex';
      elements.streamLiveText.textContent = 'STANDBY';
    }

    // Update Toggle Button
    updatePipelineButtonState();

    // Update Active Scene Faces Sidebar
    renderActiveTracks(state.activeTracks);

  } catch (err) {
    console.error('Error fetching stats:', err);
    elements.systemStatusText.textContent = 'Connection Reconnecting...';
  }
}

function updatePipelineButtonState() {
  if (state.pipelineRunning) {
    elements.btnTogglePipeline.className = 'btn-danger';
    elements.pipelineBtnText.textContent = 'Stop Pipeline';
    elements.pipelineBtnIcon.innerHTML = '<path d="M6 6h12v12H6z"/>';
  } else {
    elements.btnTogglePipeline.className = 'btn-primary';
    elements.pipelineBtnText.textContent = 'Start Tracking';
    elements.pipelineBtnIcon.innerHTML = '<path d="M8 5v14l11-7z"/>';
  }
}

function renderActiveTracks(tracks) {
  if (!tracks || tracks.length === 0) {
    elements.activeTracksContainer.innerHTML = `
      <div style="font-size: 12px; color: var(--text-dim); text-align: center; padding: 16px 0;">
        ${state.pipelineRunning ? 'No faces in current frame.' : 'Tracking is stopped. Click Start Tracking.'}
      </div>
    `;
    return;
  }

  let html = '';
  tracks.forEach(tr => {
    html += `
      <div class="track-item-mini">
        <div class="track-item-left">
          <span class="track-id-badge">#${tr.track_id}</span>
          <span style="font-weight: 600; color: #fff;">${tr.visitor_id}</span>
        </div>
        <div class="track-conf-badge">Conf: ${(tr.confidence * 100).toFixed(0)}%</div>
      </div>
    `;
  });
  elements.activeTracksContainer.innerHTML = html;
}

// ============================================================================
// PIPELINE CONTROL (START / STOP / SOURCE / RTSP)
// ============================================================================
async function togglePipeline() {
  elements.btnTogglePipeline.disabled = true;

  if (state.pipelineRunning) {
    try {
      const res = await fetch('/api/pipeline/stop', { method: 'POST' });
      const data = await res.json();
      if (data.success) {
        showToast('Tracking pipeline stopped cleanly.');
      }
    } catch (e) {
      showToast('Error stopping pipeline.', 'error');
    }
  } else {
    const selectedSource = elements.inputSourceSelect.value;
    if (selectedSource === 'custom_rtsp') {
      elements.rtspModalBackdrop.classList.add('active');
      elements.btnTogglePipeline.disabled = false;
      return;
    }

    await startPipelineWithSource(selectedSource);
  }

  elements.btnTogglePipeline.disabled = false;
  await fetchStats();
}

async function startPipelineWithSource(src) {
  try {
    showToast(`Initializing pipeline on: ${src}...`);
    const res = await fetch('/api/pipeline/start', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        source: src,
        skip_frames: parseInt(elements.sliderSkip.value),
        confidence_threshold: parseFloat(elements.sliderConf.value),
        similarity_threshold: parseFloat(elements.sliderSim.value)
      })
    });
    const data = await res.json();
    if (data.success) {
      showToast('Face Tracking Pipeline active!');
      state.pipelineRunning = true;
      updatePipelineButtonState();
      // Reload stream image to reconnect stream endpoint cleanly
      elements.liveVideoFeed.src = `/api/video_feed?t=${Date.now()}`;
    } else {
      showToast(data.error || 'Failed to start pipeline.', 'error');
    }
  } catch (err) {
    showToast('Failed to connect to backend pipeline.', 'error');
  }
}

// RTSP Modal Handlers
elements.inputSourceSelect.addEventListener('change', (e) => {
  if (e.target.value === 'custom_rtsp') {
    elements.rtspModalBackdrop.classList.add('active');
  } else {
    state.selectedSource = e.target.value;
    elements.currentSourceLabel.textContent = `Source: ${e.target.options[e.target.selectedIndex].text}`;
  }
});

elements.btnCancelRtsp.addEventListener('click', () => {
  elements.rtspModalBackdrop.classList.remove('active');
  elements.inputSourceSelect.value = state.selectedSource;
});

elements.btnCloseRtspModal.addEventListener('click', () => {
  elements.rtspModalBackdrop.classList.remove('active');
  elements.inputSourceSelect.value = state.selectedSource;
});

elements.btnConfirmRtsp.addEventListener('click', async () => {
  const rtspUrl = elements.rtspUrlInput.value.trim();
  if (!rtspUrl) {
    showToast('Please enter an RTSP URL', 'error');
    return;
  }
  elements.rtspModalBackdrop.classList.remove('active');
  state.selectedSource = rtspUrl;
  elements.currentSourceLabel.textContent = `RTSP: ${rtspUrl}`;
  await startPipelineWithSource(rtspUrl);
});

elements.btnTogglePipeline.addEventListener('click', togglePipeline);

// ============================================================================
// VIDEO FILE UPLOAD HANDLERS & MP4 INTEGRATION
// ============================================================================
async function fetchUploadedVideos() {
  try {
    const res = await fetch('/api/uploaded_videos');
    const data = await res.json();
    if (data.videos && data.videos.length > 0) {
      data.videos.forEach(v => {
        addUploadedOptionToSelect(v.path, v.filename);
      });
    }
  } catch (err) {
    console.warn('Could not fetch uploaded video list:', err);
  }
}

function addUploadedOptionToSelect(path, filename) {
  if (!elements.inputSourceSelect) return;
  const existing = Array.from(elements.inputSourceSelect.options).find(opt => opt.value === path);
  if (!existing) {
    const opt = document.createElement('option');
    opt.value = path;
    opt.textContent = `Uploaded MP4: ${filename}`;
    elements.inputSourceSelect.appendChild(opt);
  }
}

async function handleFileUpload(file) {
  if (!file) return;

  const validTypes = ['video/mp4', 'video/x-m4v', 'video/quicktime', 'video/x-msvideo', 'video/mkv', 'video/webm'];
  if (!validTypes.includes(file.type) && !file.name.match(/\.(mp4|avi|mkv|mov|m4v)$/i)) {
    showToast('Invalid video format. Please upload an MP4, AVI, or MKV file.', 'error');
    return;
  }

  showToast(`Uploading '${file.name}'...`, 'info');
  if (elements.uploadMp4Btn) {
    elements.uploadMp4Btn.style.opacity = '0.6';
    elements.uploadMp4Btn.style.pointerEvents = 'none';
  }

  try {
    const formData = new FormData();
    formData.append('file', file, file.name);

    const res = await fetch('/api/upload_video', {
      method: 'POST',
      body: formData
    });

    const data = await res.json();

    if (data.success && data.filepath) {
      showToast(`Upload complete: ${data.filename}`, 'success');
      addUploadedOptionToSelect(data.filepath, data.filename);

      elements.inputSourceSelect.value = data.filepath;
      state.selectedSource = data.filepath;
      elements.currentSourceLabel.textContent = `Source: Uploaded (${data.filename})`;

      if (state.pipelineRunning) {
        await fetch('/api/pipeline/stop', { method: 'POST' });
        await new Promise(r => setTimeout(r, 400));
        await startPipelineWithSource(data.filepath);
      } else {
        await startPipelineWithSource(data.filepath);
      }
    } else {
      showToast(data.error || 'Video upload failed.', 'error');
    }
  } catch (err) {
    console.error('Upload error:', err);
    showToast('Network error during video upload.', 'error');
  } finally {
    if (elements.uploadMp4Btn) {
      elements.uploadMp4Btn.style.opacity = '1';
      elements.uploadMp4Btn.style.pointerEvents = 'auto';
    }
  }
}

// Upload Button & File Input Event Listeners
if (elements.uploadMp4Btn && elements.videoFileInput) {
  elements.uploadMp4Btn.addEventListener('click', () => {
    elements.videoFileInput.click();
  });

  elements.videoFileInput.addEventListener('change', (e) => {
    if (e.target.files && e.target.files[0]) {
      handleFileUpload(e.target.files[0]);
    }
  });

  elements.uploadMp4Btn.addEventListener('dragover', (e) => {
    e.preventDefault();
    elements.uploadMp4Btn.classList.add('drag-over');
  });

  elements.uploadMp4Btn.addEventListener('dragleave', () => {
    elements.uploadMp4Btn.classList.remove('drag-over');
  });

  elements.uploadMp4Btn.addEventListener('drop', (e) => {
    e.preventDefault();
    elements.uploadMp4Btn.classList.remove('drag-over');
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFileUpload(e.dataTransfer.files[0]);
    }
  });
}

// Fullscreen
elements.btnFullscreen.addEventListener('click', () => {
  if (!document.fullscreenElement) {
    elements.videoContainer.requestFullscreen().catch(err => {
      showToast(`Fullscreen error: ${err.message}`, 'error');
    });
  } else {
    document.exitFullscreen();
  }
});

// Snapshot
elements.btnSnapshot.addEventListener('click', () => {
  const canvas = document.createElement('canvas');
  const img = elements.liveVideoFeed;
  canvas.width = img.naturalWidth || 1280;
  canvas.height = img.naturalHeight || 720;
  const ctx = canvas.getContext('2d');
  ctx.drawImage(img, 0, 0, canvas.width, canvas.height);

  const link = document.createElement('a');
  link.download = `AetherVision_Snapshot_${Date.now()}.jpg`;
  link.href = canvas.toDataURL('image/jpeg', 0.95);
  link.click();
  showToast('Frame snapshot downloaded!');
});

// Calibration Sliders
elements.sliderSkip.addEventListener('input', (e) => {
  elements.valSkip.textContent = e.target.value;
  elements.badgeCurrentSkip.textContent = `${e.target.value} frames`;
});

elements.sliderConf.addEventListener('input', (e) => {
  elements.valConf.textContent = parseFloat(e.target.value).toFixed(2);
});

elements.sliderSim.addEventListener('input', (e) => {
  elements.valSim.textContent = parseFloat(e.target.value).toFixed(2);
});

async function fetchInitialConfig() {
  try {
    const res = await fetch('/api/config');
    if (!res.ok) return;
    const cfg = await res.json();
    if (cfg) {
      if (cfg.detection_skip_frames !== undefined) {
        elements.sliderSkip.value = cfg.detection_skip_frames;
        elements.valSkip.textContent = cfg.detection_skip_frames;
        elements.badgeCurrentSkip.textContent = `${cfg.detection_skip_frames} frames`;
      }
      if (cfg.confidence_threshold !== undefined) {
        elements.sliderConf.value = cfg.confidence_threshold;
        elements.valConf.textContent = parseFloat(cfg.confidence_threshold).toFixed(2);
      }
      if (cfg.similarity_threshold !== undefined) {
        elements.sliderSim.value = cfg.similarity_threshold;
        elements.valSim.textContent = parseFloat(cfg.similarity_threshold).toFixed(2);
      }
      if (cfg.input_source) {
        state.selectedSource = cfg.input_source;
        if (elements.inputSourceSelect) {
          const optExists = Array.from(elements.inputSourceSelect.options).some(o => o.value === cfg.input_source);
          if (optExists) {
            elements.inputSourceSelect.value = cfg.input_source;
          }
        }
      }
    }
  } catch (err) {
    console.warn('Could not load initial config:', err);
  }
}

elements.btnSaveConfig.addEventListener('click', async () => {
  try {
    const payload = {
      detection_skip_frames: parseInt(elements.sliderSkip.value),
      confidence_threshold: parseFloat(elements.sliderConf.value),
      similarity_threshold: parseFloat(elements.sliderSim.value),
      input_source: state.selectedSource
    };
    const res = await fetch('/api/config', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    if (res.ok) {
      showToast(`Vision parameters applied! Similarity Threshold set to ${parseFloat(elements.sliderSim.value).toFixed(2)}.`);
    }
  } catch (e) {
    showToast('Failed to apply configuration', 'error');
  }
});

// ============================================================================
// TAB 3: VISITORS GALLERY & PROFILES
// ============================================================================
async function fetchVisitors() {
  try {
    const q = encodeURIComponent(state.visitorSearch);
    const sort = encodeURIComponent(state.visitorSort);
    const res = await fetch(`/api/visitors?search=${q}&sort=${sort}`);
    if (!res.ok) return;
    const data = await res.json();

    const visitors = data.visitors || [];
    elements.visitorsCountBadge.textContent = `${visitors.length} Profiles`;

    if (visitors.length === 0) {
      elements.visitorsGridContainer.innerHTML = `
        <div style="grid-column: 1/-1; text-align: center; padding: 40px; color: var(--text-dim);">
          No registered visitors match your query. Run the video stream to automatically register visitors.
        </div>
      `;
      return;
    }

    let html = '';
    visitors.forEach(v => {
      const thumb = v.thumbnail_path ? `/${v.thumbnail_path.replace(/\\/g, '/')}` : '';
      const aliasHtml = v.alias ? `<div class="visitor-alias">${v.alias}</div>` : '';
      const firstSeenTime = v.first_seen ? v.first_seen.split('T')[0] : '';

      html += `
        <div class="visitor-profile-card" onclick="openVisitorModal('${v.visitor_id}')">
          <div class="visitor-avatar-wrap">
            <img class="visitor-avatar" src="${thumb}" alt="${v.visitor_id}" onerror="this.src='https://via.placeholder.com/100?text=Face'">
          </div>
          <div class="visitor-id-tag">${v.visitor_id}</div>
          ${aliasHtml}
          <div class="visitor-meta">
            <span class="badge-visits">${v.total_visits} Visits</span>
            <span>${firstSeenTime}</span>
          </div>
          <button class="btn-delete-visitor" title="Delete this visitor permanently"
            onclick="event.stopPropagation(); deleteVisitor('${v.visitor_id}')">
            &#x1F5D1; Delete
          </button>
        </div>
      `;
    });
    elements.visitorsGridContainer.innerHTML = html;
  } catch (err) {
    console.error('Error fetching visitors:', err);
  }
}

elements.searchVisitorsInput.addEventListener('input', (e) => {
  state.visitorSearch = e.target.value;
  fetchVisitors();
});

elements.sortVisitorsSelect.addEventListener('change', (e) => {
  state.visitorSort = e.target.value;
  fetchVisitors();
});

// ============================================================================
// VISITOR PROFILE & TIMELINE MODAL
// ============================================================================
window.openVisitorModal = async function(visitorId) {
  elements.visitorModalBackdrop.classList.add('active');
  elements.modalVisitorTitle.textContent = `Biometric Identity: ${visitorId}`;
  elements.modalVisitorBody.innerHTML = `
    <div style="text-align: center; padding: 24px; color: var(--text-dim);">
      Loading historical timeline...
    </div>
  `;

  try {
    const res = await fetch(`/api/visitor/${visitorId}`);
    if (!res.ok) {
      elements.modalVisitorBody.innerHTML = `<p style="color: var(--rose-danger);">Visitor not found.</p>`;
      return;
    }
    const data = await res.json();
    const v = data.visitor;
    const events = data.events || [];
    const thumb = v.thumbnail_path ? `/${v.thumbnail_path.replace(/\\/g, '/')}` : '';

    let timelineHtml = '';
    events.forEach(ev => {
      const evThumb = ev.image_path ? `/${ev.image_path.replace(/\\/g, '/')}` : '';
      const isEntry = ev.event_type === 'entry';
      timelineHtml += `
        <div class="timeline-item ${isEntry ? '' : 'item-exit'}">
          <div class="timeline-card">
            <img src="${evThumb}" class="thumb-circle" alt="Face" onerror="this.src='https://via.placeholder.com/42?text=Face'">
            <div style="flex: 1;">
              <div style="display: flex; justify-content: space-between; align-items: center;">
                <span class="${isEntry ? 'badge-entry' : 'badge-exit'}">${ev.event_type.toUpperCase()}</span>
                <span style="font-family: var(--font-mono); font-size: 11px; color: var(--text-dim);">${ev.timestamp}</span>
              </div>
              <div style="font-size: 11px; color: var(--text-muted); margin-top: 4px;">
                Confidence: ${(ev.confidence * 100).toFixed(1)}% &bull; ${ev.details || ''}
              </div>
            </div>
          </div>
        </div>
      `;
    });

    if (events.length === 0) {
      timelineHtml = `<p style="color: var(--text-dim); font-size: 12px;">No logged entry/exit events for this visitor yet.</p>`;
    }

    elements.modalVisitorBody.innerHTML = `
      <div style="display: flex; gap: 20px; align-items: center; padding-bottom: 18px; border-bottom: 1px solid var(--border-subtle);">
        <img src="${thumb}" style="width: 80px; height: 80px; border-radius: 8px; object-fit: cover; border: 2px solid var(--border-glow);" onerror="this.src='https://via.placeholder.com/80?text=Face'">
        <div style="flex: 1;">
          <div style="font-size: 18px; font-weight: 700; color: var(--cyan-primary);">${v.visitor_id}</div>
          <div style="display: flex; gap: 10px; margin-top: 6px; flex-wrap: wrap;">
            <input type="text" class="search-input" id="inputEditAlias" value="${v.alias || ''}" placeholder="Assign Custom Name/Tag..." style="padding: 6px 12px; font-size: 12px; max-width: 220px;">
            <button class="btn-secondary" onclick="saveVisitorAlias('${v.visitor_id}')" style="padding: 6px 12px; font-size: 12px;">Save Alias</button>
            <button class="btn-delete-modal" onclick="deleteVisitor('${v.visitor_id}')" title="Permanently delete this visitor" style="padding: 6px 14px; font-size: 12px; background: linear-gradient(135deg,#7f1d1d,#dc2626); border: 1px solid #ef4444; border-radius: 6px; color: #fff; cursor: pointer; font-weight: 600;">&#x1F5D1; Delete Visitor</button>
          </div>
          <div style="font-size: 11px; color: var(--text-dim); margin-top: 8px;">
            Total Visits: <strong>${v.total_visits}</strong> &bull; First Seen: ${v.first_seen.replace('T', ' ')}
          </div>
        </div>
      </div>

      <div style="margin-top: 18px;">
        <div style="font-size: 13px; font-weight: 600; color: #fff; margin-bottom: 8px;">
          Chronological Event History (${events.length} Events)
        </div>
        <div class="timeline-list">
          ${timelineHtml}
        </div>
      </div>
    `;
  } catch (err) {
    elements.modalVisitorBody.innerHTML = `<p style="color: var(--rose-danger);">Error loading details.</p>`;
  }
};

window.saveVisitorAlias = async function(visitorId) {
  const aliasInput = document.getElementById('inputEditAlias');
  if (!aliasInput) return;
  const newAlias = aliasInput.value.trim();

  try {
    const res = await fetch(`/api/visitor/${visitorId}/alias`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ alias: newAlias })
    });
    if (res.ok) {
      showToast(`Updated alias for ${visitorId}`);
      fetchVisitors();
    }
  } catch (e) {
    showToast('Failed to update alias', 'error');
  }
};

// ============================================================================
// DELETE VISITOR
// ============================================================================
window.deleteVisitor = async function(visitorId) {
  const confirmed = window.confirm(
    `⚠️ Permanently delete visitor "${visitorId}"?\n\nThis will remove:\n• The visitor profile from the database\n• All entry/exit event records\n• All associated face crop images\n\nThis action CANNOT be undone.`
  );
  if (!confirmed) return;

  try {
    const res = await fetch(`/api/visitor/${encodeURIComponent(visitorId)}`, {
      method: 'DELETE'
    });
    const data = await res.json();

    if (data.success) {
      showToast(`🗑️ ${visitorId} deleted — ${data.events_deleted} event(s) removed.`, 'success');
      // Close modal if open for this visitor
      if (elements.visitorModalBackdrop.classList.contains('active')) {
        elements.visitorModalBackdrop.classList.remove('active');
      }
      // Refresh gallery and stats
      fetchVisitors();
      fetchStats();
    } else {
      showToast(data.error || 'Delete failed.', 'error');
    }
  } catch (err) {
    showToast('Network error — could not delete visitor.', 'error');
  }
};


elements.btnCloseVisitorModal.addEventListener('click', () => {
  elements.visitorModalBackdrop.classList.remove('active');
});

elements.btnDismissVisitorModal.addEventListener('click', () => {
  elements.visitorModalBackdrop.classList.remove('active');
});

// ============================================================================
// TAB 4: SECURITY AUDIT TRAIL & EVENT LOGS TABLE
// ============================================================================
async function fetchAuditEvents() {
  try {
    const url = `/api/events?type=${state.auditFilter}&limit=${state.auditLimit}&offset=${state.auditOffset}`;
    const res = await fetch(url);
    if (!res.ok) return;
    const data = await res.json();
    const events = data.events || [];

    if (events.length === 0) {
      elements.auditTableBody.innerHTML = `
        <tr>
          <td colspan="7" style="text-align: center; padding: 24px; color: var(--text-dim);">
            No audit events found for current filter.
          </td>
        </tr>
      `;
      return;
    }

    let rows = '';
    events.forEach(ev => {
      const imgPath = ev.image_path ? `/${ev.image_path.replace(/\\/g, '/')}` : '';
      const isEntry = ev.event_type === 'entry';
      const badgeCls = isEntry ? 'badge-entry' : 'badge-exit';
      const confPercent = ev.confidence ? (ev.confidence * 100).toFixed(1) : 'N/A';

      rows += `
        <tr>
          <td style="font-family: var(--font-mono); color: var(--text-dim);">#${ev.id}</td>
          <td>
            <img src="${imgPath}" class="thumb-circle" alt="Face" onerror="this.src='https://via.placeholder.com/42?text=Face'">
          </td>
          <td>
            <span style="font-weight: 600; color: var(--cyan-primary); cursor: pointer;" onclick="openVisitorModal('${ev.visitor_id}')">
              ${ev.visitor_id}
            </span>
          </td>
          <td>
            <span class="${badgeCls}">${ev.event_type.toUpperCase()}</span>
          </td>
          <td style="font-family: var(--font-mono); font-size: 12px; color: var(--text-bright);">
            ${ev.timestamp}
          </td>
          <td>
            <div style="display: flex; align-items: center; gap: 8px;">
              <span style="font-family: var(--font-mono); font-size: 11px;">${confPercent}%</span>
              <div style="width: 50px; height: 4px; background: rgba(255,255,255,0.1); border-radius: 2px; overflow: hidden;">
                <div style="width: ${confPercent}%; height: 100%; background: ${isEntry ? '#10b981' : '#f59e0b'};"></div>
              </div>
            </div>
          </td>
          <td style="font-size: 12px; color: var(--text-muted);">
            ${ev.details || ''}
          </td>
        </tr>
      `;
    });

    elements.auditTableBody.innerHTML = rows;
    elements.auditPaginationText.textContent = `Showing ${events.length} events (offset ${state.auditOffset})`;
  } catch (err) {
    console.error('Error fetching audit events:', err);
  }
}

// Filter Event Buttons
elements.btnFilterAllEvents.addEventListener('click', () => {
  state.auditFilter = 'all';
  state.auditOffset = 0;
  elements.btnFilterAllEvents.classList.add('active');
  elements.btnFilterEntryEvents.classList.remove('active');
  elements.btnFilterExitEvents.classList.remove('active');
  fetchAuditEvents();
});

elements.btnFilterEntryEvents.addEventListener('click', () => {
  state.auditFilter = 'entry';
  state.auditOffset = 0;
  elements.btnFilterEntryEvents.classList.add('active');
  elements.btnFilterAllEvents.classList.remove('active');
  elements.btnFilterExitEvents.classList.remove('active');
  fetchAuditEvents();
});

elements.btnFilterExitEvents.addEventListener('click', () => {
  state.auditFilter = 'exit';
  state.auditOffset = 0;
  elements.btnFilterExitEvents.classList.add('active');
  elements.btnFilterAllEvents.classList.remove('active');
  elements.btnFilterEntryEvents.classList.remove('active');
  fetchAuditEvents();
});

// Pagination
elements.btnAuditPrev.addEventListener('click', () => {
  if (state.auditOffset >= state.auditLimit) {
    state.auditOffset -= state.auditLimit;
    fetchAuditEvents();
  }
});

elements.btnAuditNext.addEventListener('click', () => {
  state.auditOffset += state.auditLimit;
  fetchAuditEvents();
});

// CSV Export
elements.btnExportCSV.addEventListener('click', () => {
  window.location.href = '/api/export_csv';
  showToast('Downloading Audit CSV report...');
});

// ============================================================================
// TAB 5: SYSTEM LOGS & CONSOLE TERMINAL
// ============================================================================
async function fetchLogs() {
  try {
    const res = await fetch('/api/logs');
    if (!res.ok) return;
    const data = await res.json();
    const lines = data.logs || [];

    let html = '';
    lines.forEach(line => {
      let cls = 'log-info';
      if (line.includes('[REGISTRATION]')) cls = 'log-reg';
      else if (line.includes('[RECOGNITION]')) cls = 'log-match';
      else if (line.includes('[ENTRY]')) cls = 'log-entry';
      else if (line.includes('[EXIT]')) cls = 'log-exit';
      else if (line.includes('[ERROR]')) cls = 'log-err';

      html += `<div class="log-line ${cls}">${escapeHtml(line)}</div>`;
    });

    elements.terminalLogBody.innerHTML = html;

    if (elements.checkAutoScroll.checked) {
      elements.terminalLogBody.scrollTop = elements.terminalLogBody.scrollHeight;
    }
  } catch (err) {
    console.error('Error fetching logs:', err);
  }
}

function escapeHtml(str) {
  return str.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

elements.btnClearLogsView.addEventListener('click', () => {
  elements.terminalLogBody.innerHTML = '';
});

// Database Reset Modal Handlers
elements.btnResetDatabase.addEventListener('click', () => {
  elements.resetModalBackdrop.classList.add('active');
});

elements.btnCancelReset.addEventListener('click', () => {
  elements.resetModalBackdrop.classList.remove('active');
});

elements.btnCloseResetModal.addEventListener('click', () => {
  elements.resetModalBackdrop.classList.remove('active');
});

elements.btnConfirmReset.addEventListener('click', async () => {
  elements.resetModalBackdrop.classList.remove('active');
  try {
    showToast('Resetting database and log directory...');
    const res = await fetch('/api/reset_db', { method: 'POST' });
    const data = await res.json();
    if (data.success) {
      showToast('Database wiped cleanly. System ready for fresh run!');
      await fetchStats();
      if (state.currentTab === 'tabVisitors') fetchVisitors();
      if (state.currentTab === 'tabAudit') fetchAuditEvents();
      if (state.currentTab === 'tabDashboard') renderCharts();
    }
  } catch (err) {
    showToast('Error resetting database.', 'error');
  }
});

// ============================================================================
// TAB 2: INTERACTIVE HTML5 CANVAS CHARTS (FOOTFALL + DONUT)
// ============================================================================
async function renderCharts() {
  await renderFootfallChart();
  renderDonutChart();
}

async function renderFootfallChart() {
  const canvas = elements.footfallCanvas;
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  canvas.width = rect.width * dpr;
  canvas.height = rect.height * dpr;
  ctx.scale(dpr, dpr);

  const w = rect.width;
  const h = rect.height;

  // Clear canvas
  ctx.clearRect(0, 0, w, h);

  // Fetch hourly metrics from API
  let hourlyData = [];
  try {
    const res = await fetch('/api/charts/footfall');
    if (res.ok) {
      const data = await res.json();
      hourlyData = data.footfall || [];
    }
  } catch (e) {
    console.error('Error fetching footfall data:', e);
  }

  // Parse hour buckets
  const buckets = {};
  hourlyData.forEach(item => {
    const key = item.hour_group || 'Now';
    if (!buckets[key]) buckets[key] = { entry: 0, exit: 0 };
    if (item.event_type === 'entry') buckets[key].entry += item.count;
    if (item.event_type === 'exit') buckets[key].exit += item.count;
  });

  const labels = Object.keys(buckets);
  if (labels.length === 0) {
    // Render placeholder grid
    ctx.fillStyle = '#64748b';
    ctx.font = '13px Inter';
    ctx.textAlign = 'center';
    ctx.fillText('No hourly traffic data recorded yet. Start tracking to generate footfall telemetry.', w / 2, h / 2);
    return;
  }

  const paddingLeft = 40;
  const paddingRight = 20;
  const paddingTop = 20;
  const paddingBottom = 30;

  const chartW = w - paddingLeft - paddingRight;
  const chartH = h - paddingTop - paddingBottom;

  let maxVal = 5;
  labels.forEach(k => {
    maxVal = Math.max(maxVal, buckets[k].entry, buckets[k].exit);
  });
  maxVal = Math.ceil(maxVal * 1.25);

  // Draw background horizontal grid lines
  ctx.strokeStyle = 'rgba(255, 255, 255, 0.05)';
  ctx.lineWidth = 1;
  const gridLines = 4;
  for (let i = 0; i <= gridLines; i++) {
    const y = paddingTop + (chartH / gridLines) * i;
    ctx.beginPath();
    ctx.moveTo(paddingLeft, y);
    ctx.lineTo(w - paddingRight, y);
    ctx.stroke();

    const val = Math.round(maxVal - (maxVal / gridLines) * i);
    ctx.fillStyle = '#64748b';
    ctx.font = '10px JetBrains Mono';
    ctx.textAlign = 'right';
    ctx.fillText(val, paddingLeft - 8, y + 3);
  }

  // Draw bars for each bucket
  const groupW = chartW / labels.length;
  const barW = Math.max(8, Math.min(24, groupW * 0.35));

  labels.forEach((label, idx) => {
    const centerX = paddingLeft + groupW * idx + groupW / 2;
    const entryH = (buckets[label].entry / maxVal) * chartH;
    const exitH = (buckets[label].exit / maxVal) * chartH;

    // Entry Bar (Emerald)
    ctx.fillStyle = '#10b981';
    ctx.fillRect(centerX - barW - 2, paddingTop + chartH - entryH, barW, entryH);

    // Exit Bar (Amber)
    ctx.fillStyle = '#f59e0b';
    ctx.fillRect(centerX + 2, paddingTop + chartH - exitH, barW, exitH);

    // Label on x-axis
    ctx.fillStyle = '#94a3b8';
    ctx.font = '10px Inter';
    ctx.textAlign = 'center';
    const displayLabel = label.length > 10 ? label.slice(-5) : label;
    ctx.fillText(displayLabel, centerX, h - 10);
  });

  // Legend
  ctx.fillStyle = '#10b981';
  ctx.fillRect(w - 140, 10, 10, 10);
  ctx.fillStyle = '#f1f5f9';
  ctx.font = '11px Inter';
  ctx.textAlign = 'left';
  ctx.fillText('Entries', w - 125, 19);

  ctx.fillStyle = '#f59e0b';
  ctx.fillRect(w - 70, 10, 10, 10);
  ctx.fillStyle = '#f1f5f9';
  ctx.fillText('Exits', w - 55, 19);
}

function renderDonutChart() {
  const canvas = elements.donutCanvas;
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  canvas.width = rect.width * dpr;
  canvas.height = rect.height * dpr;
  ctx.scale(dpr, dpr);

  const w = rect.width;
  const h = rect.height;
  ctx.clearRect(0, 0, w, h);

  const entries = state.entriesCount || 0;
  const exits = state.exitsCount || 0;
  const total = entries + exits;

  if (total === 0) {
    ctx.fillStyle = '#64748b';
    ctx.font = '12px Inter';
    ctx.textAlign = 'center';
    ctx.fillText('No events recorded', w / 2, h / 2);
    return;
  }

  const cx = w / 2;
  const cy = h / 2 - 10;
  const radius = Math.min(cx, cy) - 25;
  const innerRadius = radius * 0.65;

  const entryAngle = (entries / total) * 2 * Math.PI;

  // Segment 1: Entries (Emerald)
  ctx.beginPath();
  ctx.arc(cx, cy, radius, -Math.PI / 2, -Math.PI / 2 + entryAngle);
  ctx.arc(cx, cy, innerRadius, -Math.PI / 2 + entryAngle, -Math.PI / 2, true);
  ctx.closePath();
  ctx.fillStyle = '#10b981';
  ctx.fill();

  // Segment 2: Exits (Amber)
  ctx.beginPath();
  ctx.arc(cx, cy, radius, -Math.PI / 2 + entryAngle, 1.5 * Math.PI);
  ctx.arc(cx, cy, innerRadius, 1.5 * Math.PI, -Math.PI / 2 + entryAngle, true);
  ctx.closePath();
  ctx.fillStyle = '#f59e0b';
  ctx.fill();

  // Center Text
  ctx.fillStyle = '#ffffff';
  ctx.font = 'bold 22px Outfit';
  ctx.textAlign = 'center';
  ctx.fillText(total.toString(), cx, cy + 4);
  ctx.fillStyle = '#94a3b8';
  ctx.font = '10px Inter';
  ctx.fillText('TOTAL EVENTS', cx, cy + 18);

  // Bottom Legend
  ctx.font = '11px Inter';
  ctx.fillStyle = '#10b981';
  ctx.fillText(`Entries: ${entries} (${((entries/total)*100).toFixed(0)}%)`, cx - 60, h - 14);
  ctx.fillStyle = '#f59e0b';
  ctx.fillText(`Exits: ${exits} (${((exits/total)*100).toFixed(0)}%)`, cx + 60, h - 14);
}

// Global Refresh Action
elements.btnRefreshAll.addEventListener('click', async () => {
  elements.btnRefreshAll.style.transform = 'rotate(180deg)';
  elements.btnRefreshAll.style.transition = 'transform 0.4s ease';
  await fetchStats();
  if (state.currentTab === 'tabVisitors') await fetchVisitors();
  if (state.currentTab === 'tabAudit') await fetchAuditEvents();
  if (state.currentTab === 'tabDashboard') await renderCharts();
  if (state.currentTab === 'tabLogs') await fetchLogs();
  setTimeout(() => {
    elements.btnRefreshAll.style.transform = 'none';
  }, 400);
  showToast('Synchronized with tracker state.');
});

// ============================================================================
// INITIALIZATION
// ============================================================================
document.addEventListener('DOMContentLoaded', () => {
  initTabs();
  fetchInitialConfig();
  fetchStats();
  fetchUploadedVideos();

  // Real-time polling timer (1.5 seconds)
  state.pollInterval = setInterval(fetchStats, 1500);

  // Background log polling if on log tab
  setInterval(() => {
    if (state.currentTab === 'tabLogs') {
      fetchLogs();
    }
  }, 2500);

  window.addEventListener('resize', () => {
    if (state.currentTab === 'tabDashboard') {
      renderCharts();
    }
  });
});
