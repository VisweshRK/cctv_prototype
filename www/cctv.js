// DOM code for the CCTV dashboard (www/cctv.html). Pure logic lives in
// cctv-logic.js. Text that comes from the server only reaches the DOM through
// textContent (see el()), never through innerHTML.
import './video-stream.js';
import {USE_CASES} from './cctv-logic.js';

const STATUS_EVERY = 5000; // ms between GET api/cctv/status
const MAX_PILLS = 3; // use-case chips shown on a tile before "+N"

const $ = (id) => document.getElementById(id);
const params = new URLSearchParams(location.search);

/** A new element with an optional class and text (set as textContent). */
function el(tag, cls, text) {
    const e = document.createElement(tag);
    if (cls) e.className = cls;
    if (text !== undefined) e.textContent = String(text);
    return e;
}

const useCaseTitle = (uc) => (Object.hasOwn(USE_CASES, uc) && USE_CASES[uc].title) || String(uc);

/** "for 42 s", "for 3 min", "for 2 h 5 min", "for 4 d"; '' if since is unknown. */
function forHowLong(since, now) {
    const t = Date.parse(since);
    if (Number.isNaN(t) || t <= 0) return ''; // Go's zero time parses as year 1
    const s = Math.max(0, Math.round((now - t) / 1000));
    if (s < 60) return `for ${s} s`;
    const m = Math.floor(s / 60);
    if (m < 60) return `for ${m} min`;
    const h = Math.floor(m / 60);
    if (h < 24) return m % 60 ? `for ${h} h ${m % 60} min` : `for ${h} h`;
    return `for ${Math.floor(h / 24)} d`;
}

// ---------------------------------------------------------------------------
// Clock (top bar, every 1 s)
// ---------------------------------------------------------------------------

function tick() {
    const d = new Date();
    const date = d.toLocaleDateString('en-GB', {day: '2-digit', month: 'short', year: 'numeric'});
    const time = d.toLocaleTimeString('en-GB', {hour12: false});
    $('clock').textContent = `${date}  ${time}`;
}

// ---------------------------------------------------------------------------
// Camera tiles and status (GET api/cctv/status every 5 s)
// ---------------------------------------------------------------------------

/** name -> {el, hud parts, key}. Map order = tile order in #cams. */
const tiles = new Map();
let noCameras = null; // the "No cameras configured" message, while shown

function makeTile(name) {
    const tile = el('div', 'tile');
    tile.dataset.camera = name;

    // Created once per camera and never replaced: replacing it restarts the video.
    /** @type {VideoStream} */
    const video = document.createElement('video-stream');
    video.src = new URL('api/ws?src=' + encodeURIComponent(name), location.href);
    tile.appendChild(video); // its <video> is made when the tile joins the document

    const off = el('div', 'off');
    const offText = el('div', 't', 'Camera offline');
    const offWhy = el('div', 'why');
    off.append(offText, offWhy);

    const hud = el('div', 'hud');
    const rec = el('span', 'rec');
    rec.append(el('i'), 'LIVE');
    const sub = el('span', 'sub');
    hud.append(rec, el('span', 'name', name), sub);

    const hudb = el('div', 'hudb');
    const pills = el('div', 'pills');
    hudb.append(el('span', 'k', 'configured'), pills);

    tile.append(off, hud, hudb);
    return {el: tile, video, offText, offWhy, sub, pills, hudb, key: null};
}

/** Place and use-case chips; redrawn only when they change. */
function describeTile(t, place, useCases) {
    const key = JSON.stringify([place, useCases]);
    if (key === t.key) return;
    t.key = key;

    t.sub.textContent = place ? `· ${place}` : '';

    const titles = useCases.map(useCaseTitle);
    const shown = titles.slice(0, MAX_PILLS);
    const rest = titles.slice(MAX_PILLS);
    t.pills.replaceChildren(...shown.map((s) => el('span', 'pill', s)));
    if (rest.length) {
        const more = el('span', 'pill', `+${rest.length}`);
        more.title = rest.join('\n');
        t.pills.append(more);
    }
    t.hudb.hidden = titles.length === 0;
}

function updateTile(t, c, now) {
    const place = typeof c.place === 'string' ? c.place : '';
    const useCases = Array.isArray(c.use_cases) ? c.use_cases.map(String) : [];
    describeTile(t, place, useCases);

    const offline = c.online === false;
    t.el.classList.toggle('offline', offline);
    if (offline) {
        const since = forHowLong(c.since, now);
        t.offText.textContent = since ? `Camera offline · ${since}` : 'Camera offline';
        t.offWhy.textContent = typeof c.error === 'string' ? c.error : '';
    }
}

function setOnlineChip(online, total) {
    const chip = $('online');
    chip.classList.remove('bad');
    chip.classList.toggle('warn', online < total);
    chip.replaceChildren(el('i', 'pulse'), el('b', '', `${online}/${total}`), ' cameras online');

    $('kpi-online').textContent = `${online}/${total}`;
    const offline = total - online;
    $('kpi-online-note').textContent = total === 0 ? 'no cameras configured'
        : offline ? `${offline} offline` : 'all cameras streaming';
}

function showStatusUnavailable() {
    const chip = $('online');
    chip.classList.remove('warn');
    chip.classList.add('bad');
    chip.textContent = 'status unavailable';
    $('kpi-online-note').textContent = 'status unavailable';
}

function renderStatus(list) {
    const now = Date.now();
    const cams = $('cams');
    const seen = new Set();
    const useCases = new Set();
    let online = 0;

    for (const c of list) {
        if (!c || typeof c.name !== 'string' || c.name === '' || seen.has(c.name)) continue;
        seen.add(c.name);

        let t = tiles.get(c.name);
        if (!t) {
            t = makeTile(c.name);
            tiles.set(c.name, t);
            cams.appendChild(t.el);
            const v = t.video.video; // made by video-stream's connectedCallback
            if (v) {
                v.muted = true; // so autoplay is allowed
                v.controls = false; // the HUD covers the tile
            }
        }
        updateTile(t, c, now);

        if (c.online === true) online++;
        if (Array.isArray(c.use_cases)) c.use_cases.forEach((uc) => useCases.add(String(uc)));
    }

    for (const [name, t] of tiles) {
        if (seen.has(name)) continue;
        t.el.remove();
        tiles.delete(name);
    }

    if (seen.size === 0 && !noCameras) {
        noCameras = el('div', 'empty', 'No cameras configured in go2rtc.yaml (cctv.cameras)');
        cams.appendChild(noCameras);
    } else if (seen.size > 0 && noCameras) {
        noCameras.remove();
        noCameras = null;
    }

    setOnlineChip(online, seen.size);
    $('armed').replaceChildren(el('b', '', useCases.size), ' use cases configured');
}

async function refreshStatus() {
    try {
        const r = await fetch('api/cctv/status', {cache: 'no-cache'});
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        const list = await r.json();
        if (!Array.isArray(list)) throw new Error('status is not a list');
        renderStatus(list);
    } catch {
        showStatusUnavailable(); // tiles stay as they are
    }
}

async function pollStatus() {
    await refreshStatus();
    setTimeout(pollStatus, STATUS_EVERY);
}

// ---------------------------------------------------------------------------
// Alerts: feed, live push, popup, acknowledge (Task 9)
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Numbers, timeline, simulate (Task 10)
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Startup
// ---------------------------------------------------------------------------

$('simulate').hidden = params.get('demo') !== '1';

tick();
setInterval(tick, 1000);
pollStatus();
