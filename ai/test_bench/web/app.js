/**
 * CCTV Vision Bench — Interactive Frontend Controller
 * P2 Krithick: YuNet + SFace + RapidOCR
 */

(function () {
  'use strict';

  // State Management
  const state = {
    mode: 'both', // 'both' | 'faces' | 'plates'
    source: 'file', // 'file' | 'webcam' | 'preset' | 'rtsp'
    activeTab: 'playground',
    mediaType: null, // 'image' | 'video' | 'rtsp'
    currentImageBgr: null, // Image object or base64
    currentVideoFile: null,
    webcamStream: null,
    webcamInterval: null,
    rtsp: {
      running: false,
      seq: 0,
      timer: null,
      configTimer: null,
      lastImage: null,
    },
    isProcessing: false,
    lastDetections: { faces: [], plates: [], telemetry: {}, events: [] },
    galleryData: [],
    knownPlates: {},
    tuning: {
      faceDetThresh: 0.60,
      faceRecThresh: 0.80,
      minFaceArea: 750,
      minPlateConf: 0.80,
      minPlateArea: 1000,
      videoStride: 5,
      claheBoost: false,
    },
    toggles: {
      boxes: true,
      landmarks: true,
      labels: true,
    }
  };

  // DOM Elements
  const el = {
    // Navigation
    mainNav: document.getElementById('main-nav'),
    tabs: document.querySelectorAll('.tab-content'),

    // Status Pills
    pillYuNet: document.getElementById('pill-yunet'),
    pillSFace: document.getElementById('pill-sface'),
    pillOcr: document.getElementById('pill-ocr'),
    galleryVal: document.getElementById('gallery-val'),
    galleryDot: document.getElementById('gallery-dot'),

    // Control Column
    modeBtns: document.querySelectorAll('.mode-btn'),
    sourcePills: document.querySelectorAll('.source-pill'),
    sourceContainers: {
      file: document.getElementById('source-file-box'),
      webcam: document.getElementById('source-webcam-box'),
      preset: document.getElementById('source-preset-box'),
      rtsp: document.getElementById('source-rtsp-box'),
    },
    currentSourceBadge: document.getElementById('current-source-badge'),
    fileDropZone: document.getElementById('file-drop-zone'),
    mediaFileInput: document.getElementById('media-file-input'),
    btnBrowse: document.getElementById('btn-browse'),
    btnToggleCam: document.getElementById('btn-toggle-cam'),
    camFpsSelect: document.getElementById('cam-fps-select'),
    presetChips: document.querySelectorAll('.preset-chip'),

    // Live RTSP Source
    rtspUrl: document.getElementById('rtsp-url'),
    rtspFpsSelect: document.getElementById('rtsp-fps-select'),
    rtspPlateSelect: document.getElementById('rtsp-plate-select'),
    btnToggleRtsp: document.getElementById('btn-toggle-rtsp'),
    rtspDot: document.getElementById('rtsp-dot'),
    rtspStatusText: document.getElementById('rtsp-status-text'),

    // Sliders & Tuning
    tuningHeader: document.getElementById('tuning-header'),
    tuningAccordion: document.getElementById('tuning-accordion'),
    sliderFaceDet: document.getElementById('slider-face-det'),
    valFaceDet: document.getElementById('val-face-det'),
    sliderFaceRec: document.getElementById('slider-face-rec'),
    valFaceRec: document.getElementById('val-face-rec'),
    sliderFaceArea: document.getElementById('slider-face-area'),
    valFaceArea: document.getElementById('val-face-area'),
    sliderPlateConf: document.getElementById('slider-plate-conf'),
    valPlateConf: document.getElementById('val-plate-conf'),
    sliderPlateArea: document.getElementById('slider-plate-area'),
    valPlateArea: document.getElementById('val-plate-area'),
    sliderVideoStride: document.getElementById('slider-video-stride'),
    valVideoStride: document.getElementById('val-video-stride'),
    checkClahe: document.getElementById('check-clahe'),
    btnRunPipeline: document.getElementById('btn-run-pipeline'),

    // Viewport Stage
    stagePlaceholder: document.getElementById('stage-placeholder'),
    stageCanvas: document.getElementById('stage-canvas'),
    stageVideo: document.getElementById('stage-video'),
    scannerOverlay: document.getElementById('scanner-overlay'),
    mediaTitle: document.getElementById('media-title'),
    mediaResBadge: document.getElementById('media-res-badge'),
    btnFitScreen: document.getElementById('btn-fit-screen'),
    toggleBoxes: document.getElementById('toggle-boxes'),
    toggleLandmarks: document.getElementById('toggle-landmarks'),
    toggleLabels: document.getElementById('toggle-labels'),
    btnQuickFace: document.getElementById('btn-quick-face'),
    btnQuickPlate: document.getElementById('btn-quick-plate'),

    // Video Toolbar
    videoBar: document.getElementById('video-bar'),
    btnPlayPause: document.getElementById('btn-play-pause'),
    playIcon: document.getElementById('play-icon'),
    timelineTime: document.getElementById('timeline-time'),
    timelineScrubber: document.getElementById('timeline-scrubber'),
    timelineFill: document.getElementById('timeline-fill'),
    timelineMarkers: document.getElementById('timeline-markers'),
    frameCounter: document.getElementById('frame-counter'),

    // Telemetry Panel
    budgetBadge: document.getElementById('budget-badge'),
    metricTotalMs: document.getElementById('metric-total-ms'),
    metricFaceDet: document.getElementById('metric-face-det'),
    metricFaceEmbed: document.getElementById('metric-face-embed'),
    metricOcr: document.getElementById('metric-ocr'),
    metricPlatePost: document.getElementById('metric-plate-post'),
    detectionCountBadge: document.getElementById('detection-count-badge'),
    detectionList: document.getElementById('detection-list'),
    eventsFeed: document.getElementById('events-feed'),
    btnCopyEvents: document.getElementById('btn-copy-events'),

    // Gallery Studio
    btnRefreshGallery: document.getElementById('btn-refresh-gallery'),
    btnRebuildGallery: document.getElementById('btn-rebuild-gallery'),
    newPersonName: document.getElementById('new-person-name'),
    galleryPhotoDrop: document.getElementById('gallery-photo-drop'),
    galleryPhotoInput: document.getElementById('gallery-photo-input'),
    selectedPhotosCount: document.getElementById('selected-photos-count'),
    btnSavePerson: document.getElementById('btn-save-person'),
    peopleGrid: document.getElementById('people-grid'),
    galleryConsoleBody: document.getElementById('gallery-console-body'),
    btnClearConsole: document.getElementById('btn-clear-console'),

    // Plates Registry
    fuzzyTestInput: document.getElementById('fuzzy-test-input'),
    fuzzyVal: document.getElementById('fuzzy-val'),
    platesTableBody: document.getElementById('plates-table-body'),
    btnAddPlateDialog: document.getElementById('btn-add-plate-dialog'),
  };

  const ctx = el.stageCanvas.getContext('2d');
  let selectedGalleryFiles = [];

  // =========================================================================
  // INITIALIZATION
  // =========================================================================
  async function init() {
    setupEventListeners();
    await checkSystemStatus();
    await loadGalleryList();
    await loadKnownPlates();
  }

  async function checkSystemStatus() {
    try {
      const res = await fetch('/api/status');
      const data = await res.json();
      if (data.models) {
        setIndicator(el.pillYuNet, data.models.yunet_loaded);
        setIndicator(el.pillSFace, data.models.sface_loaded);
        setIndicator(el.pillOcr, data.models.rapidocr_loaded);
      }
      if (data.gallery) {
        el.galleryVal.textContent = `${data.gallery.people_count} ${data.gallery.people_count === 1 ? 'person' : 'people'}`;
        if (data.gallery.people_count > 0) {
          el.galleryDot.classList.add('active');
        } else {
          el.galleryDot.classList.remove('active');
        }
      }
    } catch (e) {
      console.warn('Status check failed:', e);
    }
  }

  function setIndicator(pill, isActive) {
    const dot = pill.querySelector('.indicator');
    if (isActive) dot.classList.add('active');
    else dot.classList.remove('active');
  }

  // =========================================================================
  // EVENT LISTENERS & UI WIRING
  // =========================================================================
  function setupEventListeners() {
    // Nav Tabs
    el.mainNav.addEventListener('click', (e) => {
      const btn = e.target.closest('.nav-btn');
      if (!btn) return;
      document.querySelectorAll('.nav-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      const tabName = btn.dataset.tab;
      state.activeTab = tabName;
      el.tabs.forEach(t => t.classList.remove('active'));
      document.getElementById(`tab-${tabName}`).classList.add('active');
      if (tabName === 'gallery') loadGalleryList();
      if (tabName === 'plates') loadKnownPlates();
    });

    // Mode Selector
    el.modeBtns.forEach(btn => {
      btn.addEventListener('click', () => {
        el.modeBtns.forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        state.mode = btn.dataset.mode;
        if (state.rtsp.running) {
          pushLiveConfig();
        } else if (state.currentImageBgr && state.source !== 'webcam') {
          runPipeline();
        }
      });
    });

    // Input Source Selector
    el.sourcePills.forEach(pill => {
      pill.addEventListener('click', () => {
        el.sourcePills.forEach(p => p.classList.remove('active'));
        pill.classList.add('active');
        const src = pill.dataset.source;
        state.source = src;
        Object.keys(el.sourceContainers).forEach(k => {
          el.sourceContainers[k].classList.toggle('active', k === src);
        });
        el.currentSourceBadge.textContent = src.toUpperCase();
        if (src !== 'webcam' && state.webcamStream) {
          stopWebcam();
        }
        if (src !== 'rtsp' && state.rtsp.running) {
          stopRtsp();
        }
      });
    });

    // File Browse & Drag/Drop
    el.btnBrowse.addEventListener('click', () => el.mediaFileInput.click());
    el.fileDropZone.addEventListener('click', (e) => {
      if (e.target !== el.btnBrowse) el.mediaFileInput.click();
    });

    el.fileDropZone.addEventListener('dragover', (e) => {
      e.preventDefault();
      el.fileDropZone.classList.add('dragover');
    });

    el.fileDropZone.addEventListener('dragleave', () => {
      el.fileDropZone.classList.remove('dragover');
    });

    el.fileDropZone.addEventListener('drop', (e) => {
      e.preventDefault();
      el.fileDropZone.classList.remove('dragover');
      if (e.dataTransfer.files.length) {
        handleFileSelection(e.dataTransfer.files[0]);
      }
    });

    el.mediaFileInput.addEventListener('change', (e) => {
      if (e.target.files.length) {
        handleFileSelection(e.target.files[0]);
      }
    });

    // Presets
    el.presetChips.forEach(chip => {
      chip.addEventListener('click', () => loadPreset(chip.dataset.preset));
    });
    el.btnQuickFace.addEventListener('click', () => loadPreset('face_sample.jpg'));
    el.btnQuickPlate.addEventListener('click', () => loadPreset('plate_sample.png'));

    // Tuning Sliders
    wireSlider(el.sliderFaceDet, el.valFaceDet, (v) => { state.tuning.faceDetThresh = parseFloat(v); });
    wireSlider(el.sliderFaceRec, el.valFaceRec, (v) => { state.tuning.faceRecThresh = parseFloat(v); });
    wireSlider(el.sliderFaceArea, el.valFaceArea, (v) => { state.tuning.minFaceArea = parseInt(v); });
    wireSlider(el.sliderPlateConf, el.valPlateConf, (v) => { state.tuning.minPlateConf = parseFloat(v); });
    wireSlider(el.sliderPlateArea, el.valPlateArea, (v) => { state.tuning.minPlateArea = parseInt(v); });
    wireSlider(el.sliderVideoStride, el.valVideoStride, (v) => { state.tuning.videoStride = parseInt(v); });
    el.checkClahe.addEventListener('change', (e) => {
      state.tuning.claheBoost = e.target.checked;
      scheduleLiveConfig();
    });

    // Accordion Toggle
    el.tuningHeader.addEventListener('click', () => {
      el.tuningHeader.classList.toggle('collapsed');
      el.tuningAccordion.style.display = el.tuningHeader.classList.contains('collapsed') ? 'none' : 'flex';
    });

    // Overlay Toggles
    el.toggleBoxes.addEventListener('change', (e) => { state.toggles.boxes = e.target.checked; redrawStage(); });
    el.toggleLandmarks.addEventListener('change', (e) => { state.toggles.landmarks = e.target.checked; redrawStage(); });
    el.toggleLabels.addEventListener('change', (e) => { state.toggles.labels = e.target.checked; redrawStage(); });

    // Primary Run Button & Ctrl+Enter shortcut
    el.btnRunPipeline.addEventListener('click', runPipeline);
    window.addEventListener('keydown', (e) => {
      if (e.ctrlKey && e.key === 'Enter') {
        e.preventDefault();
        runPipeline();
      }
    });

    // Webcam Button
    el.btnToggleCam.addEventListener('click', toggleWebcam);

    // Live RTSP Button
    el.btnToggleRtsp.addEventListener('click', toggleRtsp);
    el.rtspUrl.addEventListener('keydown', (e) => {
      if (e.key === 'Enter') { e.preventDefault(); startRtsp(); }
    });
    el.rtspFpsSelect.addEventListener('change', () => pushLiveConfig());
    el.rtspPlateSelect.addEventListener('change', () => pushLiveConfig());

    // Never leave a camera session running after the tab closes
    window.addEventListener('beforeunload', () => {
      if (state.rtsp.running) navigator.sendBeacon('/api/live/stop');
    });

    // Copy Events JSON
    el.btnCopyEvents.addEventListener('click', () => {
      navigator.clipboard.writeText(JSON.stringify(state.lastDetections.events, null, 2));
      el.btnCopyEvents.textContent = 'Copied!';
      setTimeout(() => el.btnCopyEvents.textContent = 'Copy JSON', 1500);
    });

    // Gallery Actions
    el.btnRefreshGallery.addEventListener('click', loadGalleryList);
    el.btnRebuildGallery.addEventListener('click', triggerGalleryRebuild);
    el.galleryPhotoDrop.addEventListener('click', () => el.galleryPhotoInput.click());
    el.galleryPhotoInput.addEventListener('change', (e) => {
      selectedGalleryFiles = Array.from(e.target.files);
      el.selectedPhotosCount.textContent = `${selectedGalleryFiles.length} selected`;
    });
    el.btnSavePerson.addEventListener('click', uploadPersonPhotos);
    el.btnClearConsole.addEventListener('click', () => { el.galleryConsoleBody.textContent = ''; });

    // Fuzzy Matcher Realtime Input
    el.fuzzyTestInput.addEventListener('input', (e) => {
      simulateFuzzyPlate(e.target.value.trim());
    });

    // Add Plate Dialog
    el.btnAddPlateDialog.addEventListener('click', promptAddPlate);
  }

  function wireSlider(slider, display, onChange) {
    slider.addEventListener('input', (e) => {
      display.textContent = e.target.value;
      onChange(e.target.value);
      scheduleLiveConfig();
    });
  }

  // =========================================================================
  // FILE & MEDIA HANDLING
  // =========================================================================
  function handleFileSelection(file) {
    if (state.rtsp.running) stopRtsp();
    if (file.type.startsWith('image/')) {
      state.mediaType = 'image';
      el.videoBar.style.display = 'none';
      el.stageVideo.pause();
      el.stageVideo.style.display = 'none';
      el.stageCanvas.style.display = 'block';

      const reader = new FileReader();
      reader.onload = (e) => {
        const dataUrl = e.target.result;
        state.currentImageDataUrl = dataUrl;
        const img = new Image();
        img.onload = () => {
          state.currentImageBgr = img;
          el.stagePlaceholder.style.display = 'none';
          el.mediaTitle.textContent = file.name;
          el.mediaResBadge.textContent = `${img.naturalWidth}x${img.naturalHeight}`;
          setupCanvas(img.naturalWidth, img.naturalHeight);
          ctx.drawImage(img, 0, 0);
          runPipeline(dataUrl);
        };
        img.src = dataUrl;
      };
      reader.readAsDataURL(file);
    } else if (file.type.startsWith('video/')) {
      state.mediaType = 'video';
      state.currentVideoFile = file;
      el.stagePlaceholder.style.display = 'none';
      el.stageCanvas.style.display = 'none'; // video renders on its own element
      el.stageVideo.style.display = 'block';
      el.videoBar.style.display = 'flex';
      el.mediaTitle.textContent = file.name;

      const url = URL.createObjectURL(file);
      el.stageVideo.src = url;
      el.stageVideo.onloadedmetadata = () => {
        el.mediaResBadge.textContent = `${el.stageVideo.videoWidth}x${el.stageVideo.videoHeight}`;
        setupCanvas(el.stageVideo.videoWidth, el.stageVideo.videoHeight);
        runVideoPipeline(file);
      };
    }
  }

  async function loadPreset(presetName) {
    if (state.rtsp.running) stopRtsp();
    state.source = 'preset';
    el.sourcePills.forEach(p => p.classList.toggle('active', p.dataset.source === 'preset'));
    Object.keys(el.sourceContainers).forEach(k => {
      el.sourceContainers[k].classList.toggle('active', k === 'preset');
    });
    el.currentSourceBadge.textContent = 'PRESET';

    const url = `/api/sample?name=${encodeURIComponent(presetName)}`;
    try {
      const res = await fetch(url);
      const blob = await res.blob();
      const reader = new FileReader();
      reader.onload = (e) => {
        const dataUrl = e.target.result;
        state.currentImageDataUrl = dataUrl;
        const img = new Image();
        img.onload = () => {
          state.mediaType = 'image';
          state.currentImageBgr = img;
          el.stagePlaceholder.style.display = 'none';
          el.stageVideo.style.display = 'none';
          el.stageCanvas.style.display = 'block';
          el.videoBar.style.display = 'none';
          el.mediaTitle.textContent = presetName;
          el.mediaResBadge.textContent = `${img.naturalWidth}x${img.naturalHeight}`;
          setupCanvas(img.naturalWidth, img.naturalHeight);
          ctx.drawImage(img, 0, 0);
          runPipeline(dataUrl);
        };
        img.src = dataUrl;
      };
      reader.readAsDataURL(blob);
    } catch (err) {
      console.error('Failed to load preset:', err);
    }
  }

  function setupCanvas(w, h) {
    el.stageCanvas.width = w;
    el.stageCanvas.height = h;
  }

  // =========================================================================
  // PIPELINE EXECUTION
  // =========================================================================
  async function runPipeline(overrideDataUrl) {
    if (state.rtsp.running) {
      // In live mode the tuning is pushed to the running session instead.
      pushLiveConfig();
      return;
    }
    if ((!state.currentImageBgr && !overrideDataUrl) || state.isProcessing) return;
    state.isProcessing = true;
    el.scannerOverlay.style.display = 'flex';

    let dataUrl = overrideDataUrl || state.currentImageDataUrl;
    if (!dataUrl && state.currentImageBgr) {
      const tempCanvas = document.createElement('canvas');
      tempCanvas.width = state.currentImageBgr.naturalWidth || state.currentImageBgr.width;
      tempCanvas.height = state.currentImageBgr.naturalHeight || state.currentImageBgr.height;
      const tctx = tempCanvas.getContext('2d');
      tctx.drawImage(state.currentImageBgr, 0, 0);
      dataUrl = tempCanvas.toDataURL('image/jpeg', 0.95);
      state.currentImageDataUrl = dataUrl;
    }

    try {
      const payload = {
        image: dataUrl,
        mode: state.mode,
        contrast_boost: state.tuning.claheBoost,
        min_face_area: state.tuning.minFaceArea,
        face_det_thresh: state.tuning.faceDetThresh,
        face_rec_thresh: state.tuning.faceRecThresh,
        min_plate_area: state.tuning.minPlateArea,
        min_plate_conf: state.tuning.minPlateConf,
      };

      const res = await fetch('/api/process/image', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });

      const result = await res.json();
      state.lastDetections = result;
      renderDetections(result);
    } catch (err) {
      console.error('Inference error:', err);
    } finally {
      state.isProcessing = false;
      el.scannerOverlay.style.display = 'none';
    }
  }

  // Video Pipeline Runner
  async function runVideoPipeline(file) {
    el.scannerOverlay.style.display = 'flex';
    const reader = new FileReader();
    reader.onload = async (e) => {
      const b64 = e.target.result;
      try {
        const payload = {
          video: b64,
          stride: state.tuning.videoStride,
          mode: state.mode,
          max_frames: 120,
          face_rec_thresh: state.tuning.faceRecThresh,
          min_face_area: state.tuning.minFaceArea,
          min_plate_area: state.tuning.minPlateArea,
          min_plate_conf: state.tuning.minPlateConf,
          contrast_boost: state.tuning.claheBoost,
        };
        const res = await fetch('/api/process/video', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload),
        });
        const summary = await res.json();
        if (summary.error) {
          console.error('Video processing failed:', summary.error);
          return;
        }
        renderVideoPlayback(summary);
      } catch (err) {
        console.error('Video processing error:', err);
      } finally {
        el.scannerOverlay.style.display = 'none';
      }
    };
    reader.readAsDataURL(file);
  }

  // =========================================================================
  // WEBCAM LIVE STREAMING
  // =========================================================================
  async function toggleWebcam() {
    if (state.webcamStream) {
      stopWebcam();
    } else {
      await startWebcam();
    }
  }

  async function startWebcam() {
    if (state.rtsp.running) stopRtsp();
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 640 }, height: { ideal: 480 } },
        audio: false,
      });
      state.webcamStream = stream;
      el.stageVideo.srcObject = stream;
      el.stageVideo.play();
      el.stageVideo.style.display = 'none'; // Render purely via canvas
      el.stageCanvas.style.display = 'block';
      el.stagePlaceholder.style.display = 'none';
      el.btnToggleCam.innerHTML = `
        <svg viewBox="0 0 24 24" width="16" height="16" stroke="currentColor" fill="none" stroke-width="2"><rect x="3" y="3" width="18" height="18" rx="2" ry="2"/><line x1="9" y1="9" x2="15" y2="15"/><line x1="15" y1="9" x2="9" y2="15"/></svg>
        Stop Live Camera
      `;
      el.btnToggleCam.classList.add('btn-secondary');
      el.btnToggleCam.classList.remove('btn-primary');

      el.stageVideo.onloadedmetadata = () => {
        setupCanvas(el.stageVideo.videoWidth, el.stageVideo.videoHeight);
        const fps = parseInt(el.camFpsSelect.value) || 5;
        const intervalMs = Math.round(1000 / fps);

        state.webcamInterval = setInterval(processWebcamFrame, intervalMs);
      };
    } catch (e) {
      alert(`Could not access camera: ${e.message}`);
    }
  }

  function stopWebcam() {
    if (state.webcamInterval) clearInterval(state.webcamInterval);
    if (state.webcamStream) {
      state.webcamStream.getTracks().forEach(t => t.stop());
      state.webcamStream = null;
    }
    el.stageVideo.srcObject = null;
    el.btnToggleCam.innerHTML = `
      <svg viewBox="0 0 24 24" width="16" height="16" stroke="currentColor" fill="none" stroke-width="2"><path d="M23 7l-7 5 7 5V7z"/><rect x="1" y="5" width="15" height="14" rx="2" ry="2"/></svg>
      Start Live Camera
    `;
    el.btnToggleCam.classList.remove('btn-secondary');
    el.btnToggleCam.classList.add('btn-primary');
  }

  async function processWebcamFrame() {
    if (!state.webcamStream || state.isProcessing) return;
    const w = el.stageVideo.videoWidth;
    const h = el.stageVideo.videoHeight;
    if (!w || !h) return;

    ctx.drawImage(el.stageVideo, 0, 0, w, h);
    const dataUrl = el.stageCanvas.toDataURL('image/jpeg', 0.85);

    state.isProcessing = true;
    try {
      const res = await fetch('/api/process/frame', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          image: dataUrl,
          mode: state.mode,
          contrast_boost: state.tuning.claheBoost,
          min_face_area: state.tuning.minFaceArea,
          face_det_thresh: state.tuning.faceDetThresh,
          face_rec_thresh: state.tuning.faceRecThresh,
          min_plate_area: state.tuning.minPlateArea,
          min_plate_conf: state.tuning.minPlateConf,
        }),
      });
      const result = await res.json();
      state.lastDetections = result;

      // Redraw video frame & paint overlays
      ctx.drawImage(el.stageVideo, 0, 0, w, h);
      paintOverlays(result);
      updateTelemetry(result);
    } catch (e) {
      // quiet webcam error
    } finally {
      state.isProcessing = false;
    }
  }

  // =========================================================================
  // RTSP LIVE STREAMING
  // =========================================================================
  function liveTuningPayload() {
    return {
      mode: state.mode,
      contrast_boost: state.tuning.claheBoost,
      min_face_area: state.tuning.minFaceArea,
      face_det_thresh: state.tuning.faceDetThresh,
      face_rec_thresh: state.tuning.faceRecThresh,
      min_plate_area: state.tuning.minPlateArea,
      min_plate_conf: state.tuning.minPlateConf,
      fps: parseInt(el.rtspFpsSelect.value) || 5,
      plate_every: parseInt(el.rtspPlateSelect.value) || 5,
    };
  }

  function scheduleLiveConfig() {
    if (!state.rtsp.running) return;
    clearTimeout(state.rtsp.configTimer);
    state.rtsp.configTimer = setTimeout(pushLiveConfig, 250);
  }

  async function pushLiveConfig() {
    if (!state.rtsp.running) return;
    try {
      await fetch('/api/live/config', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(liveTuningPayload()),
      });
    } catch (e) {
      // The next frame poll will surface a server-side error.
    }
  }

  function toggleRtsp() {
    if (state.rtsp.running) stopRtsp();
    else startRtsp();
  }

  async function startRtsp() {
    const url = el.rtspUrl.value.trim();
    if (!url) {
      el.rtspUrl.focus();
      setRtspStatus('error', 'Enter an RTSP URL first');
      return;
    }
    if (state.webcamStream) stopWebcam();

    el.btnToggleRtsp.disabled = true;
    setRtspStatus('warn', 'Connecting to source…');
    try {
      const res = await fetch('/api/live/start', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ url, ...liveTuningPayload() }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || `HTTP ${res.status}`);

      state.source = 'rtsp';
      el.sourcePills.forEach(p => p.classList.toggle('active', p.dataset.source === 'rtsp'));
      Object.keys(el.sourceContainers).forEach(k => {
        el.sourceContainers[k].classList.toggle('active', k === 'rtsp');
      });
      el.currentSourceBadge.textContent = 'RTSP';

      state.rtsp.running = true;
      state.rtsp.seq = 0;
      state.rtsp.lastImage = null;
      state.mediaType = 'rtsp';
      state.currentImageBgr = null;

      el.stagePlaceholder.style.display = 'none';
      el.stageVideo.pause();
      el.stageVideo.style.display = 'none';
      el.stageCanvas.style.display = 'block';
      el.videoBar.style.display = 'none';
      el.mediaTitle.textContent = `LIVE · ${data.source || url}`;
      el.mediaResBadge.textContent = '…';

      setRtspButton(true);
      pollLive();
    } catch (e) {
      setRtspStatus('error', e.message || 'Could not start stream');
      setRtspButton(false);
    } finally {
      el.btnToggleRtsp.disabled = false;
    }
  }

  async function stopRtsp() {
    state.rtsp.running = false;
    clearTimeout(state.rtsp.timer);
    clearTimeout(state.rtsp.configTimer);
    state.rtsp.timer = null;
    setRtspButton(false);
    setRtspStatus('idle', 'Not connected');
    try {
      await fetch('/api/live/stop', { method: 'POST' });
    } catch (e) {
      // The reader thread also stops when the server goes away.
    }
  }

  async function pollLive() {
    if (!state.rtsp.running) return;
    let delay = 25;
    try {
      const res = await fetch(`/api/live/frame?after=${state.rtsp.seq}`, { cache: 'no-store' });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      if (!state.rtsp.running) return;

      if (data.status === 'stopped') {
        state.rtsp.running = false;
        setRtspButton(false);
        setRtspStatus('idle', 'Session ended');
        return;
      }
      updateRtspStatus(data);

      if (!data.unchanged && data.image) {
        state.rtsp.seq = data.seq;
        const img = new Image();
        img.onload = () => {
          if (!state.rtsp.running) return;
          state.rtsp.lastImage = img;
          if (el.stageCanvas.width !== data.width || el.stageCanvas.height !== data.height) {
            setupCanvas(data.width, data.height);
            el.mediaResBadge.textContent = `${data.width}x${data.height}`;
          }
          ctx.drawImage(img, 0, 0, data.width, data.height);
          state.lastDetections = data;
          paintOverlays(data);
          updateTelemetry(data);
          renderDetectionCards(data);
          renderEvents(data.events);
        };
        img.src = data.image;
        delay = 10;
      } else {
        delay = 30;
      }
    } catch (e) {
      delay = 1000;
      if (state.rtsp.running) setRtspStatus('error', `Stream lost: ${e.message}`);
    }
    if (state.rtsp.running) state.rtsp.timer = setTimeout(pollLive, delay);
  }

  function updateRtspStatus(data) {
    const status = data.status || 'stopped';
    const stats = data.stats || {};
    if (data.source) el.mediaTitle.textContent = `LIVE · ${data.source}`;

    if (status === 'live') {
      const fps = (stats.inference_fps || 0).toFixed(1);
      const lag = stats.capture_lag_ms != null ? `${Math.round(stats.capture_lag_ms)} ms` : '—';
      const skipped = stats.frames_skipped || 0;
      setRtspStatus('live', `LIVE · ${fps} inf/s · ${lag} lag · ${skipped} skipped`);
    } else if (status === 'connecting') {
      setRtspStatus('warn', 'Connecting…');
    } else if (status === 'reconnecting') {
      setRtspStatus('warn', data.error ? `Reconnecting — ${data.error}` : 'Reconnecting…');
    } else if (status === 'error') {
      setRtspStatus('error', data.error || 'Stream error');
    } else {
      setRtspStatus('idle', 'Not connected');
    }
  }

  function setRtspStatus(kind, text) {
    el.rtspDot.className = 'indicator' + (kind === 'idle' ? '' : ` ${kind}`);
    el.rtspStatusText.textContent = text;
    el.rtspStatusText.style.color =
      kind === 'error' ? 'var(--danger)' : kind === 'warn' ? 'var(--warning)' : '';
  }

  function setRtspButton(running) {
    el.btnToggleRtsp.innerHTML = running
      ? `<svg viewBox="0 0 24 24" width="16" height="16" stroke="currentColor" fill="none" stroke-width="2"><rect x="3" y="3" width="18" height="18" rx="2" ry="2"/><line x1="9" y1="9" x2="15" y2="15"/><line x1="15" y1="9" x2="9" y2="15"/></svg> Stop Live RTSP`
      : `<svg viewBox="0 0 24 24" width="16" height="16" stroke="currentColor" fill="none" stroke-width="2"><path d="M4.93 4.93a10 10 0 0 0 0 14.14"/><path d="M7.76 7.76a6 6 0 0 0 0 8.48"/><circle cx="12" cy="12" r="2"/><path d="M19.07 19.07a10 10 0 0 0 0-14.14"/><path d="M16.24 16.24a6 6 0 0 0 0-8.48"/></svg> Start Live RTSP`;
    el.btnToggleRtsp.classList.toggle('btn-secondary', running);
    el.btnToggleRtsp.classList.toggle('btn-primary', !running);
  }

  // =========================================================================
  // RENDERING HUD & OVERLAYS ON CANVAS
  // =========================================================================
  function redrawStage() {
    if (state.source === 'rtsp' && state.rtsp.lastImage) {
      ctx.drawImage(
        state.rtsp.lastImage, 0, 0,
        el.stageCanvas.width || state.rtsp.lastImage.naturalWidth,
        el.stageCanvas.height || state.rtsp.lastImage.naturalHeight
      );
      paintOverlays(state.lastDetections);
      return;
    }
    if (state.currentImageBgr && state.mediaType === 'image') {
      ctx.drawImage(state.currentImageBgr, 0, 0);
      paintOverlays(state.lastDetections);
    }
  }

  function renderDetections(result) {
    redrawStage();
    updateTelemetry(result);
    renderDetectionCards(result);
    renderEvents(result.events);
  }

  function paintOverlays(result) {
    if (!result) return;
    const { faces = [], plates = [] } = result;

    // 1. Paint Face Overlays
    faces.forEach(f => {
      const [x, y, w, h] = f.box;
      const color = f.is_known ? '#38bdf8' : '#f59e0b';

      if (state.toggles.boxes) {
        drawReticleBox(ctx, x, y, w, h, color);
      }

      // Landmarks (5 points)
      if (state.toggles.landmarks && f.landmarks && f.landmarks.length === 5) {
        ctx.fillStyle = '#f43f5e';
        f.landmarks.forEach(([lx, ly]) => {
          ctx.beginPath();
          ctx.arc(lx, ly, 3.5, 0, 2 * Math.PI);
          ctx.fill();
          ctx.strokeStyle = '#ffffff';
          ctx.lineWidth = 1;
          ctx.stroke();
        });
      }

      // Label Pill
      if (state.toggles.labels) {
        const title = f.is_known ? `${f.name.toUpperCase()} (${Math.round(f.confidence * 100)}%)` : `UNKNOWN (${Math.round(f.confidence * 100)}%)`;
        drawLabelPill(ctx, x, y - 8, title, color);
      }
    });

    // 2. Paint Plate Overlays
    plates.forEach(p => {
      const [x, y, w, h] = p.box;
      const isOk = p.is_valid_indian;
      const color = isOk ? '#10b981' : '#a855f7';

      if (state.toggles.boxes) {
        drawReticleBox(ctx, x, y, w, h, color);
      }

      if (state.toggles.labels) {
        const text = p.repaired_plate || p.raw_text;
        const owner = p.known_as ? ` [${p.known_as}]` : '';
        const title = `${text} (${Math.round(p.confidence * 100)}%)${owner}`;
        drawLabelPill(ctx, x, y - 8, title, color);
      }
    });
  }

  function drawReticleBox(c, x, y, w, h, color) {
    const len = Math.min(w, h, 20);
    c.strokeStyle = color;
    c.lineWidth = 2.5;

    // Corner brackets
    c.beginPath();
    // Top-Left
    c.moveTo(x, y + len); c.lineTo(x, y); c.lineTo(x + len, y);
    // Top-Right
    c.moveTo(x + w - len, y); c.lineTo(x + w, y); c.lineTo(x + w, y + len);
    // Bottom-Right
    c.moveTo(x + w, y + h - len); c.lineTo(x + w, y + h); c.lineTo(x + w - len, y + h);
    // Bottom-Left
    c.moveTo(x + len, y + h); c.lineTo(x, y + h); c.lineTo(x, y + h - len);
    c.stroke();

    // Subtle transparent fill
    c.fillStyle = color.replace(')', ', 0.12)').replace('rgb', 'rgba').replace('#38bdf8', 'rgba(56, 189, 248, 0.12)').replace('#10b981', 'rgba(16, 185, 129, 0.12)').replace('#f59e0b', 'rgba(245, 158, 11, 0.12)');
    c.fillRect(x, y, w, h);
  }

  function drawLabelPill(c, x, y, text, color) {
    c.font = 'bold 12px "JetBrains Mono", monospace';
    const textWidth = c.measureText(text).width;
    const pad = 6;
    const h = 20;

    c.fillStyle = 'rgba(8, 11, 17, 0.88)';
    c.strokeStyle = color;
    c.lineWidth = 1;
    c.beginPath();
    c.roundRect(x, y - h, textWidth + pad * 2, h, 4);
    c.fill();
    c.stroke();

    c.fillStyle = color;
    c.fillText(text, x + pad, y - 6);
  }

  // =========================================================================
  // TELEMETRY & CARDS RENDERING
  // =========================================================================
  function updateTelemetry(result) {
    const t = result.telemetry || {};
    const totalMs = t.total_ms || 0;
    el.metricTotalMs.textContent = totalMs.toFixed(1);
    el.metricFaceDet.textContent = `${t.face_detection_ms || 0} ms`;
    el.metricFaceEmbed.textContent = `${t.face_embedding_ms || 0} ms`;
    el.metricOcr.textContent = `${t.ocr_inference_ms || 0} ms`;
    el.metricPlatePost.textContent = `${t.plate_postprocessing_ms || 0} ms`;

    // Budget check (<100ms)
    if (totalMs <= 100) {
      el.budgetBadge.className = 'budget-badge pass';
      el.budgetBadge.textContent = '<100ms BUDGET PASS';
    } else {
      el.budgetBadge.className = 'budget-badge fail';
      el.budgetBadge.textContent = '>100ms BUDGET EXCEEDED';
    }

    const count = (result.faces?.length || 0) + (result.plates?.length || 0);
    el.detectionCountBadge.textContent = `${count} ${count === 1 ? 'item' : 'items'}`;
  }

  function renderDetectionCards(result) {
    el.detectionList.innerHTML = '';
    const { faces = [], plates = [] } = result;

    if (!faces.length && !plates.length) {
      el.detectionList.innerHTML = '<div class="empty-state-notice">No faces or plates met detection thresholds.</div>';
      return;
    }

    // Face Cards
    faces.forEach((f, idx) => {
      const card = document.createElement('div');
      card.className = 'det-card face-card';
      const sparkHtml = (f.embedding_preview || []).map(v => {
        const heightPct = Math.min(100, Math.max(10, Math.abs(v) * 200));
        return `<div class="spark-bar" style="height: ${heightPct}%;"></div>`;
      }).join('');

      card.innerHTML = `
        <div class="det-header">
          <span class="det-title">👤 ${f.is_known ? f.name : 'Unknown Identity'}</span>
          <span class="badge ${f.is_known ? 'badge-primary' : 'badge-dim'}">${f.is_known ? 'VERIFIED' : 'STRANGER'}</span>
        </div>
        <div class="det-meta-grid">
          <div class="det-meta-item"><span class="det-meta-label">YuNet Score:</span><span class="det-meta-val">${f.score}</span></div>
          <div class="det-meta-item"><span class="det-meta-label">Cosine Sim:</span><span class="det-meta-val">${f.similarity}</span></div>
          <div class="det-meta-item"><span class="det-meta-label">Confidence:</span><span class="det-meta-val">${Math.round(f.confidence * 100)}%</span></div>
          <div class="det-meta-item"><span class="det-meta-label">Blur Penalty:</span><span class="det-meta-val">-${f.blur_penalty}</span></div>
        </div>
        <div>
          <div class="det-meta-label" style="margin-bottom: 3px; font-size: 0.68rem;">SFace 128-D Vector Sparkline:</div>
          <div class="vector-sparkline">${sparkHtml}</div>
        </div>
      `;
      el.detectionList.appendChild(card);
    });

    // Plate Cards
    plates.forEach(p => {
      const card = document.createElement('div');
      card.className = 'det-card plate-card';
      const isIndian = p.is_valid_indian;
      const typeStr = p.plate_type === 'bharat' ? 'BHARAT SERIES' : 'STANDARD INDIAN';

      card.innerHTML = `
        <div class="det-header">
          <span class="det-title">🚗 ${p.repaired_plate || p.raw_text}</span>
          <span class="badge ${isIndian ? 'badge-success' : 'badge-dim'}">${isIndian ? typeStr : 'UNVERIFIED'}</span>
        </div>
        <div class="det-meta-grid">
          <div class="det-meta-item"><span class="det-meta-label">Raw OCR:</span><span class="det-meta-val">${p.raw_text}</span></div>
          <div class="det-meta-item"><span class="det-meta-label">Avg Conf:</span><span class="det-meta-val">${Math.round(p.confidence * 100)}%</span></div>
          <div class="det-meta-item"><span class="det-meta-label">Two-Line:</span><span class="det-meta-val">${p.is_two_line ? 'Yes (Bike)' : 'No'}</span></div>
          <div class="det-meta-item"><span class="det-meta-label">Known Owner:</span><span class="det-meta-val" style="color: var(--primary);">${p.known_as || 'None'}</span></div>
        </div>
      `;
      el.detectionList.appendChild(card);
    });
  }

  function renderEvents(events) {
    el.eventsFeed.innerHTML = '';
    if (!events || !events.length) {
      el.eventsFeed.innerHTML = '<div class="empty-state-notice">No events emitted.</div>';
      return;
    }
    events.forEach(ev => {
      const chip = document.createElement('div');
      chip.className = 'event-chip';
      chip.innerHTML = `
        <span class="event-type">${ev.type}</span>
        <span class="event-label">${ev.label} (${Math.round(ev.confidence * 100)}%)</span>
      `;
      el.eventsFeed.appendChild(chip);
    });
  }

const PLAY_SVG = '<svg viewBox="0 0 24 24" width="16" height="16" fill="currentColor"><polygon points="5 3 19 12 5 21 5 3"/></svg>';
const PAUSE_SVG = '<svg viewBox="0 0 24 24" width="16" height="16" fill="currentColor"><rect x="5" y="3" width="5" height="18"/><rect x="14" y="3" width="5" height="18"/></svg>';

function renderVideoSummary(summary) {
    // Non-blocking summary card on top of the detection list
    const fv = summary.face_vote_summary || {};
    const pc = summary.plate_cluster_summary || {};
    const card = document.createElement('div');
    card.className = 'det-card video-summary-card';
    card.innerHTML = `
      <div class="det-header">
        <span class="det-title">📊 Video Analysis Summary</span>
        <span class="badge badge-primary">${summary.processed_frames}/${summary.total_video_frames} frames</span>
      </div>
      <div class="det-meta-item"><span class="det-meta-label">Face Decision:</span><span class="det-meta-val">${fv.decision_name || 'None'} (${Math.round((fv.confidence || 0) * 100)}%)</span></div>
      <div class="det-meta-item"><span class="det-meta-label">Plate Winner:</span><span class="det-meta-val">${pc.best_plate || 'None'} (${pc.agreeing_reads || 0} agreeing reads)</span></div>
      <div class="det-meta-item"><span class="det-meta-label">Events:</span><span class="det-meta-val">${(summary.detections || []).length} timeline marker(s) — click one to jump</span></div>
    `;
    el.detectionList.prepend(card);
  }

  function renderDetectionCardsFromDetections(detections, video) {
    el.detectionList.innerHTML = '';
    if (!detections || !detections.length) {
      el.detectionList.innerHTML = '<div class="empty-state-notice">No faces or plates detected in this video.</div>';
      return;
    }

    detections.forEach((det) => {
      const isFace = det.kind === 'face';
      const card = document.createElement('div');
      card.className = `det-card ${isFace ? 'face-card' : 'plate-card'}`;
      card.style.cursor = 'pointer';
      card.title = 'Click to jump to this moment';
      card.addEventListener('click', () => {
        if (video && det.timestamp_sec != null) {
          video.currentTime = det.timestamp_sec;
          if (video.paused) video.play();
        }
      });

      if (isFace) {
        card.innerHTML = `
          <div class="det-header">
            <span class="det-title">👤 ${det.is_known ? det.label : 'Unknown Identity'}</span>
            <span class="badge ${det.is_known ? 'badge-primary' : 'badge-dim'}">${det.is_known ? 'VERIFIED' : 'STRANGER'}</span>
          </div>
          <div class="det-meta-item"><span class="det-meta-label">Snapshot:</span><span class="det-meta-val"><img src="${det.snapshot}" style="max-height: 60px; max-width: 100%;"></span></div>
          <div class="det-meta-item"><span class="det-meta-label">Confidence:</span><span class="det-meta-val">${Math.round(det.confidence * 100)}%</span></div>
          <div class="det-meta-item"><span class="det-meta-label">Timestamp:</span><span class="det-meta-val">${det.timestamp_sec}s (frame ${det.frame_index})</span></div>
        `;
      } else {
        const type = (det.plate_type || 'PLATE').toUpperCase();
        card.innerHTML = `
          <div class="det-header">
            <span class="det-title">🚗 ${det.label}</span>
            <span class="badge ${det.is_known ? 'badge-success' : 'badge-dim'}">${det.is_known ? type : 'UNVERIFIED'}</span>
          </div>
          <div class="det-meta-item"><span class="det-meta-label">Snapshot:</span><span class="det-meta-val"><img src="${det.snapshot}" style="max-height: 60px; max-width: 100%;"></span></div>
          <div class="det-meta-item"><span class="det-meta-label">Confidence:</span><span class="det-meta-val">${Math.round(det.confidence * 100)}%</span></div>
          <div class="det-meta-item"><span class="det-meta-label">Timestamp:</span><span class="det-meta-val">${det.timestamp_sec}s (frame ${det.frame_index})</span></div>
          <div class="det-meta-item"><span class="det-meta-label">Known Owner:</span><span class="det-meta-val" style="color: var(--primary);">${det.known_as || 'None'}</span></div>
        `;
      }
      el.detectionList.appendChild(card);
    });
  }

  function renderVideoPlayback(summary) {
    const detections = summary.detections || [];
    const video = el.stageVideo;
    if (!video || !video.src) return;
    // Prefer the real media duration for timeline math — the runner's
    // timestamps (frame_index / fps) map onto the same clock.
    const fps = summary.fps || 25;
    const framesFallback = (summary.total_video_frames || 0) / fps;
    const durationSec = (isFinite(video.duration) && video.duration > 0)
      ? video.duration : framesFallback;
    if (!durationSec) return;

    el.videoBar.style.display = 'flex';
    el.timelineTime.textContent = `00:00 / ${formatTime(durationSec)}`;
    el.timelineFill.style.width = '0%';

    // Timeline markers: percentage-positioned, click -> seek + play
    el.timelineMarkers.innerHTML = '';
    detections.forEach((det) => {
      const pct = durationSec ? Math.min(100, Math.max(0, (det.timestamp_sec / durationSec) * 100)) : 0;
      const marker = document.createElement('div');
      marker.className = `timeline-marker ${det.kind === 'face' ? 'marker-face' : 'marker-plate'}`;
      marker.style.left = `${pct}%`;
      marker.title = `${det.kind === 'face' ? '👤' : '🚗'} ${det.label} @ ${det.timestamp_sec}s (${Math.round(det.confidence * 100)}%)`;
      marker.addEventListener('click', (e) => {
        e.stopPropagation();
        video.currentTime = det.timestamp_sec || 0;
        if (video.paused) video.play();
      });
      el.timelineMarkers.appendChild(marker);
    });

    // Detection cards (clickable -> seek)
    renderDetectionCardsFromDetections(detections, video);
    renderVideoSummary(summary);

    // Live position: fill + time + frame counter
    video.ontimeupdate = () => {
      const pct = durationSec ? (video.currentTime / durationSec) * 100 : 0;
      el.timelineFill.style.width = `${Math.min(100, pct)}%`;
      el.timelineTime.textContent = `${formatTime(video.currentTime)} / ${formatTime(durationSec)}`;
      el.frameCounter.textContent = `Frame ${Math.round(video.currentTime * fps)}`;
    };
    video.onplay = () => { el.btnPlayPause.innerHTML = PAUSE_SVG; };
    video.onpause = () => { el.btnPlayPause.innerHTML = PLAY_SVG; };

    // Play/pause toggle (onclick assignment: never accumulates listeners)
    el.btnPlayPause.onclick = () => {
      if (video.paused) { video.play().catch(() => {}); } else { video.pause(); }
    };
    el.btnPlayPause.innerHTML = video.paused ? PLAY_SVG : PAUSE_SVG;

    // Seek: click anywhere on the scrubber
    el.timelineScrubber.onclick = (e) => {
      const rect = el.timelineScrubber.getBoundingClientRect();
      const pct = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
      video.currentTime = durationSec * pct;
    };

    // Auto-start playback (element is muted, so autoplay is allowed)
    video.play().catch(() => {});
  }

  function formatTime(seconds) {
    const m = Math.floor(seconds / 60);
    const s = Math.floor(seconds % 60);
    return `${m}:${s < 10 ? '0' : ''}${s}`;
  }

  // =========================================================================
  // GALLERY STUDIO CONTROLLERS
  // =========================================================================
  async function loadGalleryList() {
    try {
      const res = await fetch('/api/gallery');
      const data = await res.json();
      state.galleryData = data.people || [];
      el.peopleGrid.innerHTML = '';

      if (!state.galleryData.length) {
        el.peopleGrid.innerHTML = '<div class="empty-state-notice">No face folders found in ai/plugins/face_recognition/faces/</div>';
        return;
      }

      state.galleryData.forEach(p => {
        const card = document.createElement('div');
        card.className = 'person-card';
        const isEnrolled = data.enrolled_in_model.includes(p.name);
        card.innerHTML = `
          <div class="person-header">
            <div class="person-avatar">${p.name.charAt(0).toUpperCase()}</div>
            <div>
              <span class="person-name">${p.name}</span>
              <span class="person-details">${p.photo_count} photos stored</span>
            </div>
          </div>
          <div>
            <span class="badge ${isEnrolled ? 'badge-primary' : 'badge-dim'}">${isEnrolled ? 'EMBEDDED IN MODEL' : 'AWAITING REBUILD'}</span>
          </div>
        `;
        el.peopleGrid.appendChild(card);
      });
    } catch (e) {
      console.warn('Failed to load gallery:', e);
    }
  }

  async function uploadPersonPhotos() {
    const name = el.newPersonName.value.trim().toLowerCase();
    if (!name) return alert('Please enter person name (ID).');
    if (!selectedGalleryFiles.length) return alert('Please select at least 1 photo.');

    const b64Promises = selectedGalleryFiles.map(file => {
      return new Promise((resolve) => {
        const r = new FileReader();
        r.onload = () => resolve(r.result);
        r.readAsDataURL(file);
      });
    });

    const photosB64 = await Promise.all(b64Promises);
    try {
      const res = await fetch('/api/gallery/upload', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name, photos: photosB64 }),
      });
      const data = await res.json();
      alert(data.message || 'Photos saved successfully!');
      el.newPersonName.value = '';
      selectedGalleryFiles = [];
      el.selectedPhotosCount.textContent = '0 selected';
      loadGalleryList();
    } catch (e) {
      alert(`Error saving photos: ${e.message}`);
    }
  }

  async function triggerGalleryRebuild() {
    el.galleryConsoleBody.textContent = 'Rebuilding gallery (running YuNet + SFace + outlier drop + trimmed mean)...\n';
    try {
      const res = await fetch('/api/gallery/rebuild', { method: 'POST' });
      const data = await res.json();
      if (res.ok) {
        el.galleryConsoleBody.textContent += `SUCCESS: ${data.message}\nIdentities: ${data.identities.join(', ')}\n`;
        checkSystemStatus();
        loadGalleryList();
      } else {
        el.galleryConsoleBody.textContent += `FAILED: ${data.error}\n`;
      }
    } catch (e) {
      el.galleryConsoleBody.textContent += `NETWORK ERROR: ${e.message}\n`;
    }
  }

  // =========================================================================
  // KNOWN PLATES REGISTRY CONTROLLERS
  // =========================================================================
  async function loadKnownPlates() {
    try {
      const res = await fetch('/api/plates');
      const data = await res.json();
      state.knownPlates = data.plates || {};
      el.platesTableBody.innerHTML = '';

      const entries = Object.entries(state.knownPlates);
      if (!entries.length) {
        el.platesTableBody.innerHTML = '<tr><td colspan="3" style="text-align:center; color: var(--text-dim);">No known plates in data/known_plates.yaml yet.</td></tr>';
        return;
      }

      entries.forEach(([owner, pList]) => {
        const row = document.createElement('tr');
        const platePills = pList.map(p => `<span class="plate-pill">${p}</span>`).join('');
        row.innerHTML = `
          <td><strong>${owner}</strong></td>
          <td>${platePills}</td>
          <td>
            <button class="icon-btn btn-delete-plate" data-owner="${owner}" title="Remove owner">
              <svg viewBox="0 0 24 24" width="16" height="16" stroke="currentColor" fill="none" stroke-width="2"><polyline points="3 6 5 6 21 6"/><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/></svg>
            </button>
          </td>
        `;
        el.platesTableBody.appendChild(row);
      });

      // Hook up delete buttons
      document.querySelectorAll('.btn-delete-plate').forEach(btn => {
        btn.addEventListener('click', () => {
          const owner = btn.dataset.owner;
          if (confirm(`Remove all plates for ${owner}?`)) {
            delete state.knownPlates[owner];
            saveKnownPlates();
          }
        });
      });
    } catch (e) {
      console.warn('Failed to load known plates:', e);
    }
  }

  function simulateFuzzyPlate(candidate) {
    if (!candidate) {
      el.fuzzyVal.className = 'fuzzy-val';
      el.fuzzyVal.textContent = 'Awaiting input...';
      return;
    }
    const cleanCand = candidate.toUpperCase().replace(/[^A-Z0-9]/g, '');
    let matchedOwner = null;
    let matchedPlate = null;
    let minD = 999;

    Object.entries(state.knownPlates).forEach(([owner, pList]) => {
      pList.forEach(p => {
        const d = levenshtein(cleanCand, p);
        if (d <= 1 && d < minD) {
          minD = d;
          matchedOwner = owner;
          matchedPlate = p;
        }
      });
    });

    if (matchedOwner) {
      el.fuzzyVal.className = 'fuzzy-val match';
      el.fuzzyVal.textContent = `MATCH FOUND: "${matchedPlate}" (Owner: ${matchedOwner}, Distance: ${minD})`;
    } else {
      el.fuzzyVal.className = 'fuzzy-val nomatch';
      el.fuzzyVal.textContent = `NO MATCH (Distance > 1 from any registered vehicle)`;
    }
  }

  function levenshtein(a, b) {
    const matrix = [];
    for (let i = 0; i <= b.length; i++) matrix[i] = [i];
    for (let j = 0; j <= a.length; j++) matrix[0][j] = j;
    for (let i = 1; i <= b.length; i++) {
      for (let j = 1; j <= a.length; j++) {
        if (b.charAt(i - 1) === a.charAt(j - 1)) {
          matrix[i][j] = matrix[i - 1][j - 1];
        } else {
          matrix[i][j] = Math.min(
            matrix[i - 1][j - 1] + 1,
            matrix[i][j - 1] + 1,
            matrix[i - 1][j] + 1
          );
        }
      }
    }
    return matrix[b.length][a.length];
  }

  function promptAddPlate() {
    const owner = prompt('Enter Owner / Role Name (e.g. Principal, Security, Visitor):');
    if (!owner) return;
    const plate = prompt('Enter Plate Number (e.g. TN09AB1234):');
    if (!plate) return;
    const cleanP = plate.toUpperCase().replace(/[^A-Z0-9]/g, '');

    if (!state.knownPlates[owner]) state.knownPlates[owner] = [];
    if (!state.knownPlates[owner].includes(cleanP)) {
      state.knownPlates[owner].push(cleanP);
    }
    saveKnownPlates();
  }

  async function saveKnownPlates() {
    try {
      await fetch('/api/plates', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ plates: state.knownPlates }),
      });
      loadKnownPlates();
    } catch (e) {
      alert(`Failed to save known plates: ${e.message}`);
    }
  }

  // Boot
  window.addEventListener('DOMContentLoaded', init);

})();
