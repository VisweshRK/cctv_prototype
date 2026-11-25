// DOM code for the CCTV dashboard (www/cctv.html). Pure logic lives in
// cctv-logic.js. Text that comes from the server only reaches the DOM through
// textContent (see el()), never through innerHTML.
import './video-stream.js';
import {
    USE_CASES, describe, areaOf, detailOf, boxOf, snapshotState, snapshotSrc, sirenText, Incidents, queueList,
    summarize, timelineHours, barFractions, parseTime, occupancyNow, energyWaste, TARIFF,
} from './cctv-logic.js';

const STATUS_EVERY = 5000; // ms between GET api/cctv/status
const MAX_PILLS = 3; // use-case chips shown on a tile before "+N"
const FETCH_TIMEOUT = 5000; // ms before any fetch gives up

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
    t.place = place;
    t.useCases = useCases;

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
    $('kpi-online').textContent = '–';
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
    syncAlarm(); // a tile made just now may belong to the open incident
    if (analyticsLoaded) renderAnalytics(); // the boxes name cameras by their place
}

async function refreshStatus() {
    try {
        const r = await fetch('api/cctv/status', {cache: 'no-cache', signal: AbortSignal.timeout(FETCH_TIMEOUT)});
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        const list = await r.json();
        if (!Array.isArray(list)) throw new Error('status is not a list');
        renderStatus(list);
    } catch (err) {
        console.warn('[cctv] camera status unavailable:', err);
        showStatusUnavailable(); // tiles stay as they are
    }
}

async function pollStatus() {
    await refreshStatus();
    setTimeout(pollStatus, STATUS_EVERY);
}

async function pollWorkerHealth() {
    const chip = $('worker-health');
    try {
        const r = await fetch('api/cctv/health', {cache: 'no-cache', signal: AbortSignal.timeout(FETCH_TIMEOUT)});
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        const health = await r.json();
        const cases = Array.isArray(health.use_cases) ? health.use_cases : [];
        const ready = cases.filter((item) => item.state === 'ready').length;
        const degraded = cases.filter((item) => item.state === 'degraded').length;
        chip.classList.toggle('bad', health.worker !== 'online');
        chip.classList.toggle('warn', health.worker === 'online' && degraded > 0);
        chip.replaceChildren(el('i', 'pulse'), `AI worker: ${health.worker === 'online' ? `${ready}/${cases.length} ready` : 'offline'}`);
        const caseLines = cases.map((item) => `${item.id}: ${item.state}${item.reason ? ` — ${item.reason}` : ''}`);
        const audio = health.audio && typeof health.audio === 'object' ? health.audio : {};
        const audioLines = Object.entries(audio).map(([camera, state]) => `Audio ${camera}: ${state.state}${state.reason ? ` — ${state.reason}` : ''}`);
        const metrics = health.metrics && typeof health.metrics === 'object' ? health.metrics : {};
        const captureLines = Object.entries(metrics.capture || {}).map(([camera, stat]) => `${camera} capture: ${stat.fps ?? '–'} FPS`);
        const laneLines = Object.entries(metrics.lanes || {}).map(([lane, stat]) => `${lane}: ${stat.last_ms ?? '–'} ms · ${stat.dropped ?? 0} stale frames dropped`);
        chip.title = [...caseLines, ...audioLines, ...captureLines, ...laneLines].join('\n') || 'No use-case health reported';
    } catch (err) {
        chip.classList.remove('warn');
        chip.classList.add('bad');
        chip.textContent = 'AI worker: unavailable';
        chip.title = 'Worker health endpoint unavailable';
        console.warn('[cctv] worker health unavailable:', err);
    }
    setTimeout(pollWorkerHealth, STATUS_EVERY);
}

// ---------------------------------------------------------------------------
// Alerts: feed, live push, popup, acknowledge (Task 9)
// ---------------------------------------------------------------------------

const FEED_MAX = 50; // rows kept in #feed, as many as GET api/cctv/events?limit=50 gives
const SNAP_RECHECK = 1000; // ms between re-checks while the popup waits for a snapshot
const REACHED_MAX = 200; // arrival times kept

const feed = $('feed');
const modal = $('modal');
const incidents = new Incidents();
/** id -> latest version of each event in the feed. */
const known = new Map();
/** id -> feed row element. */
const rows = new Map();
/** id -> ms from the incident to this page, for events that arrived by push. */
const reachedIn = new Map();
const feedEmpty = el('div', 'empty', 'No alerts yet');

/**
 * What the open popup follows: {} = incidents.current(), {id} = the event
 * clicked in the feed (or its incident). null while the popup is closed.
 */
let view = null;
let shown = null; // what renderPopup drew: {events, latest, ackIds, others}
let snapTimer = 0; // the one pending snapshot re-check
let ackError = null; // {ids, msg} after a failed Acknowledge
let bigTile = null; // the tile enlarged by "Open live view"

const isEvent = (e) => e !== null && typeof e === 'object' && Number.isFinite(e.id)
    && typeof e.camera === 'string' && typeof e.use_case === 'string';
const isAcked = (e) => Boolean(e.acknowledged_at);
const newestFirst = (a, b) => b.id - a.id;

const pad = (n) => String(n).padStart(2, '0');

function hms(t) {
    const d = new Date(parseTime(t));
    return Number.isNaN(d.getTime()) ? '–' : `${pad(d.getHours())}:${pad(d.getMinutes())}:${pad(d.getSeconds())}`;
}

function dayOf(t) {
    const d = new Date(parseTime(t));
    return Number.isNaN(d.getTime()) ? '' : d.toLocaleDateString('en-GB', {day: '2-digit', month: 'short', year: 'numeric'});
}

/** Confidence as a whole percent 0..100, or null if it isn't a number. */
function percent(c) {
    return typeof c === 'number' && Number.isFinite(c) ? Math.round(Math.min(1, Math.max(0, c)) * 100) : null;
}

const ICONS = {
    fall: ['M12 3a2 2 0 100 4 2 2 0 000-4z', 'M5 20l4-6 4 2 4-5', 'M3 21h18'],
    tampering: ['M3 7h12v10H3z', 'M15 10l6-3v10l-6-3', 'M2 3l20 18'],
    baby_crying: ['M4 10v4h4l5 4V6L8 10z', 'M16 9a4 4 0 010 6', 'M19 6a8 8 0 010 12'],
    face_recognition: ['M4 8V5a1 1 0 011-1h3', 'M16 4h3a1 1 0 011 1v3', 'M20 16v3a1 1 0 01-1 1h-3',
        'M8 20H5a1 1 0 01-1-1v-3', 'M12 8a3 3 0 100 6 3 3 0 000-6z'],
    power_when_absent: ['M13 2L4 14h7l-1 8 9-12h-7z'],
    occupancy: ['M9 7a3 3 0 100 6 3 3 0 000-6z', 'M3 20c1-4 3.5-5 6-5s5 1 6 5', 'M16 8a2.5 2.5 0 110 5',
        'M18 15c2 .5 3 2 3.5 5'],
    vehicle_lpr: ['M3 13l2-5h14l2 5v5H3z', 'M7 18v2', 'M17 18v2', 'M3 13h18'],
};
const BELL = ['M6 8a6 6 0 1112 0c0 7 3 8 3 8H3s3-1 3-8', 'M10 20a2 2 0 004 0'];

/** The use case's icon as an <svg>, built node by node. */
function icon(useCase) {
    const ns = 'http://www.w3.org/2000/svg';
    const svg = document.createElementNS(ns, 'svg');
    svg.setAttribute('viewBox', '0 0 24 24');
    svg.setAttribute('fill', 'none');
    svg.setAttribute('stroke', 'currentColor');
    svg.setAttribute('stroke-width', '2');
    svg.setAttribute('stroke-linecap', 'round');
    svg.setAttribute('stroke-linejoin', 'round');
    for (const d of Object.hasOwn(ICONS, useCase) ? ICONS[useCase] : BELL) {
        const p = document.createElementNS(ns, 'path');
        p.setAttribute('d', d);
        svg.appendChild(p);
    }
    return svg;
}

const placeOf = (camera) => (tiles.has(camera) && tiles.get(camera).place) || '';

/** The tile of the current incident's camera glows. */
function syncAlarm() {
    const cur = incidents.current();
    for (const [name, t] of tiles) t.el.classList.toggle('alarm', cur !== null && name === cur.camera);
}

// --- feed ------------------------------------------------------------------

function feedRow(e, isNew) {
    const d = describe(e);
    const row = el('div', `ev sev-${d.severity}`);
    row.classList.toggle('new', isNew);
    row.classList.toggle('acked', isAcked(e));
    row.classList.toggle('quiet', !d.popup);
    row.tabIndex = 0;
    row.setAttribute('role', 'button');
    row.addEventListener('click', () => openEvent(e.id));
    row.addEventListener('keydown', (ev) => {
        if (ev.key !== 'Enter' && ev.key !== ' ') return;
        ev.preventDefault();
        openEvent(e.id);
    });

    const th = el('div', 'th');
    const src = snapshotSrc(e);
    if (src) {
        const img = el('img');
        img.alt = '';
        img.loading = 'lazy';
        img.addEventListener('error', () => img.replaceWith(icon(e.use_case)), {once: true});
        img.src = src;
        th.appendChild(img);
    } else {
        th.appendChild(icon(e.use_case));
    }

    const t1 = el('div', 't1');
    t1.appendChild(el('span', 'tt', d.title));
    if (d.popup && !isAcked(e)) t1.appendChild(el('span', 'ua', 'Unacknowledged'));

    const t3 = el('div', 't3');
    t3.appendChild(el('span', 'mono', hms(e.time)));
    const p = percent(e.confidence);
    if (p !== null) {
        const bar = el('span', 'b');
        const fill = el('i');
        fill.style.width = `${p}%`;
        bar.appendChild(fill);
        const conf = el('span', 'conf');
        conf.append(`${p}%`, bar);
        t3.appendChild(conf);
    }

    const body = el('div', 'bd');
    body.append(t1, el('div', 't2', `${e.camera} · ${areaOf(e)}`), t3);
    row.append(th, body);
    return row;
}

/**
 * Empty state and the unacknowledged count in the panel header, then the
 * dashboard numbers. Every change to known (list, alert, update, ack) ends here.
 */
function feedChanged() {
    if (rows.size === 0 && !feedEmpty.isConnected) feed.replaceChildren(feedEmpty);
    if (rows.size > 0) feedEmpty.remove();

    let unacked = 0;
    for (const e of known.values()) if (describe(e).popup && !isAcked(e)) unacked++;
    const count = $('feed-count');
    count.hidden = unacked === 0;
    count.textContent = `${unacked} unacknowledged`;
    renderNumbers();
}

function renderFeed() {
    rows.clear();
    const list = [...known.values()].sort(newestFirst);
    feed.replaceChildren(...list.map((e) => {
        const row = feedRow(e, false);
        rows.set(e.id, row);
        return row;
    }));
    feedChanged();
}

function feedAdd(e) {
    if (rows.has(e.id)) {
        feedReplace(e);
        return;
    }
    known.set(e.id, e);
    const row = feedRow(e, true);
    rows.set(e.id, row);
    feed.prepend(row);
    while (rows.size > FEED_MAX) {
        const oldest = Math.min(...rows.keys());
        rows.get(oldest).remove();
        rows.delete(oldest);
        known.delete(oldest);
    }
    feedChanged();
}

function feedReplace(e) {
    const old = rows.get(e.id);
    if (!old) return; // not in the feed (older than its last 50)
    known.set(e.id, e);
    const row = feedRow(e, false);
    old.replaceWith(row);
    rows.set(e.id, row);
    feedChanged();
}

// --- list + push -------------------------------------------------------------

let held = null; // push messages held back while the list loads, replayed after it
let listLoading = false;
let listAgain = false;
let listLoaded = false; // a list has loaded at least once; the numbers wait for it

/**
 * GET api/cctv/events?limit=50 into the feed. Pushes that arrive meanwhile
 * are held and replayed afterwards, so a list fetched a moment too early
 * can't undo them. Old events don't pop up, except a siren still ringing.
 * A reload asked for while one fails still runs; after a failure with none
 * asked for, the stream's next open retries.
 */
async function loadFeed() {
    if (listLoading) {
        listAgain = true; // e.g. the stream opened during the first load
        return;
    }
    listLoading = true;
    held = [];
    try {
        do {
            listAgain = false;
            try {
                const r = await fetch('api/cctv/events?limit=50', {cache: 'no-cache', signal: AbortSignal.timeout(FETCH_TIMEOUT)});
                if (!r.ok) throw new Error(`HTTP ${r.status}`);
                const list = await r.json();
                if (!Array.isArray(list)) throw new Error('events is not a list');
                applyList(list.filter(isEvent));
            } catch (err) {
                console.warn('[cctv] alert list unavailable:', err);
                if (rows.size === 0) feedEmpty.textContent = 'Alert list unavailable';
            }
        } while (listAgain); // a failed load doesn't drop a reload queued meanwhile
    } finally {
        const replay = held;
        held = null;
        listLoading = false;
        for (const [type, e] of replay) onPush(type, e);
    }
}

function applyList(list) {
    listLoaded = true;
    known.clear();
    for (const e of list) known.set(e.id, e);
    feedEmpty.textContent = 'No alerts yet';
    renderFeed();
    // every unacknowledged popup event is queued (so a group's Acknowledge
    // acks all of it); only a siren still ringing pops up
    alertsChanged(queueList(incidents, [...known.values()]));
}

function onPush(type, e) {
    if (held) {
        held.push([type, e]);
        return;
    }
    if (type === 'alert') {
        analyticsAdd(e);
        feedAdd(e);
        const before = incidents.groupOf(e.id);
        incidents.add(e);
        const cur = incidents.current();
        const pop = cur !== null && cur.events.some((x) => x.id === e.id)
            && (before === null || before.key !== cur.key);
        alertsChanged(pop);
    } else {
        applyUpdate(e);
        alertsChanged(false);
    }
}

/** An event changed: SSE "update", or the answer to Acknowledge. */
function applyUpdate(e) {
    feedReplace(e);
    incidents.update(e);
}

/** After any change to incidents: the glow, then pop up or refresh the popup. */
function alertsChanged(pop) {
    syncAlarm();
    if (pop) {
        view = {};
        renderPopup();
    } else if (view) {
        renderPopup();
    }
}

function noteArrival(e) {
    if (reachedIn.has(e.id)) return;
    const t = parseTime(e.time);
    if (Number.isNaN(t)) return;
    reachedIn.set(e.id, Math.max(0, Date.now() - t));
    if (reachedIn.size > REACHED_MAX) reachedIn.delete(reachedIn.keys().next().value);
}

function connect() {
    const es = new EventSource('api/cctv/events/stream');
    es.addEventListener('open', () => loadFeed()); // also after a reconnect, for what was missed
    for (const type of ['alert', 'update']) {
        es.addEventListener(type, (m) => {
            let e;
            try {
                e = JSON.parse(m.data);
            } catch (err) {
                console.warn('[cctv] bad push message:', err);
                return;
            }
            if (!isEvent(e)) return;
            if (type === 'alert') noteArrival(e); // now, not when a held message is replayed
            onPush(type, e);
        });
    }
    es.addEventListener('error', () => {
        if (es.readyState !== EventSource.CLOSED) return; // the browser retries by itself
        console.warn('[cctv] alert stream closed; retrying in 5 s');
        setTimeout(connect, 5000);
    });
}

// --- popup ---------------------------------------------------------------------

/** {events, latest, ackIds, others} for the current view, or null if nothing is left to show. */
function popupData() {
    const groups = incidents.current() ? incidents.others() + 1 : 0;
    const g = view.id === undefined ? incidents.current() : incidents.groupOf(view.id);
    if (g) return {events: g.events, latest: g.latest, ackIds: incidents.ids(g.key), others: groups - 1};

    const e = known.get(view.id);
    if (!e) return null;
    const ackIds = describe(e).popup && !isAcked(e) ? [e.id] : [];
    return {events: [e], latest: e, ackIds, others: groups};
}

function renderPopup() {
    clearTimeout(snapTimer);
    snapTimer = 0;
    const p = view && popupData();
    if (!p) {
        closePopup();
        return;
    }
    shown = p;
    const e = p.latest;
    const d = describe(e);

    modal.className = `sev-${d.severity}`;
    $('m-icon').replaceChildren(icon(e.use_case));
    $('m-title').textContent = d.title;
    $('m-detail').textContent = detailOf(e);
    $('m-sev').textContent = d.severity;
    const more = [];
    if (p.events.length > 1) more.push(`×${p.events.length} detections`);
    if (p.others > 0) more.push(`+${p.others} more`);
    $('m-more').textContent = more.join(' · ');

    const opening = modal.hidden;
    modal.hidden = false; // before renderSnap, which measures the snapshot area
    renderSnap(e);

    $('m-time').textContent = `${dayOf(e.time)} ${hms(e.time)}`.trim();
    const place = placeOf(e.camera);
    $('m-cam').textContent = place ? `${e.camera} · ${place}` : e.camera;
    $('m-area').textContent = areaOf(e);
    const pc = percent(e.confidence);
    $('m-conf').textContent = pc === null ? '–' : `${pc}%`;
    $('m-bar').style.width = `${pc === null ? 0 : pc}%`;

    const siren = sirenText(e);
    const sirenEl = $('m-siren');
    sirenEl.textContent = siren;
    sirenEl.classList.toggle('ringing', e.siren === 'ringing');
    sirenEl.parentElement.hidden = siren === '';

    const ids = p.ackIds.join(',');
    // an old "Could not acknowledge" is for that one view: drop it on opening or on another incident
    if (ackError && (opening || ackError.ids !== ids)) ackError = null;
    const ms = reachedIn.get(e.id);
    $('m-hint').textContent = ackError ? ackError.msg
        : ms !== undefined ? `Reached screen ${(ms / 1000).toFixed(1)} s after the incident`
            : isAcked(e) ? `Acknowledged at ${hms(e.acknowledged_at)}` : '';
    $('m-ack').hidden = p.ackIds.length === 0;
    $('m-live').hidden = !tiles.has(e.camera);

    if (opening) modal.querySelector('.modal').focus({preventScroll: true});
}

function renderSnap(e) {
    const snap = $('m-snap');
    const img = $('m-img');
    const msg = $('m-snap-msg');
    const src = snapshotSrc(e);
    // a snapshot string that isn't ours is never shown, and not waited for either
    const state = src ? 'ready' : e.snapshot ? 'none' : snapshotState(e, Date.now());

    $('m-stamp').textContent = `${dayOf(e.time)} ${hms(e.time)} · ${e.camera}`;
    if (state === 'ready') {
        if (img.getAttribute('src') !== src) {
            delete img.dataset.failed;
            img.src = src; // the load handler shows it
        }
        const loaded = img.complete && img.naturalWidth > 0;
        const failed = Boolean(img.dataset.failed);
        img.hidden = !loaded;
        $('m-rec').hidden = !loaded; // the SNAPSHOT badge only next to a picture
        snap.classList.toggle('loading', !loaded && !failed);
        msg.textContent = failed ? 'No snapshot available' : '';
        if (loaded) placeBox(e);
        else $('m-box').hidden = true;
        return;
    }

    img.hidden = true;
    $('m-rec').hidden = true;
    img.removeAttribute('src');
    delete img.dataset.failed;
    $('m-box').hidden = true;
    snap.classList.toggle('loading', state === 'loading');
    msg.textContent = state === 'loading' ? 'Getting snapshot…' : 'No snapshot available';
    if (state === 'loading') snapTimer = setTimeout(renderPopup, SNAP_RECHECK);
}

/** data.box over the picture itself, not over the letterbox around it. */
function placeBox(e) {
    const box = $('m-box');
    const img = $('m-img');
    const snap = $('m-snap');
    const b = boxOf(e);
    const cw = snap.clientWidth;
    const ch = snap.clientHeight;
    if (!b || img.hidden || !img.naturalWidth || !cw || !ch) {
        box.hidden = true;
        return;
    }
    const k = Math.min(cw / img.naturalWidth, ch / img.naturalHeight); // object-fit: contain
    const w = img.naturalWidth * k;
    const h = img.naturalHeight * k;
    const x0 = (cw - w) / 2;
    const y0 = (ch - h) / 2;
    const top = y0 + b[1] * h;
    box.style.left = `${(x0 + b[0] * w) / cw * 100}%`;
    box.style.top = `${top / ch * 100}%`;
    box.style.width = `${(b[2] - b[0]) * w / cw * 100}%`;
    box.style.height = `${(b[3] - b[1]) * h / ch * 100}%`;

    const pc = percent(e.confidence);
    const label = [e.label, pc === null ? '' : `${pc}%`].filter(Boolean).join(' ');
    box.replaceChildren(...(label ? [el('span', 'lb', label)] : []));
    box.classList.toggle('in', top < 22); // no room for the label above the box
    box.hidden = false;
}

$('m-img').addEventListener('load', () => {
    if (!shown) return;
    const img = $('m-img');
    img.hidden = false;
    $('m-rec').hidden = false;
    $('m-snap').classList.remove('loading');
    $('m-snap-msg').textContent = '';
    placeBox(shown.latest);
});

$('m-img').addEventListener('error', () => {
    const img = $('m-img');
    if (!img.getAttribute('src')) return; // src removed on purpose
    img.dataset.failed = '1';
    img.hidden = true;
    $('m-rec').hidden = true;
    $('m-box').hidden = true;
    $('m-snap').classList.remove('loading');
    $('m-snap-msg').textContent = 'No snapshot available';
});

function closePopup() {
    clearTimeout(snapTimer);
    snapTimer = 0;
    view = null;
    shown = null;
    modal.hidden = true;
}

function openEvent(id) {
    view = {id};
    renderPopup();
}

async function acknowledge() {
    const p = shown;
    if (!p || p.ackIds.length === 0) return;
    const ids = p.ackIds.join(',');
    const btn = $('m-ack');
    btn.disabled = true;
    try {
        const r = await fetch('api/cctv/events/ack', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({ids: p.ackIds}),
            signal: AbortSignal.timeout(FETCH_TIMEOUT),
        });
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        const list = await r.json();
        if (!Array.isArray(list)) throw new Error('ack answer is not a list');
        ackError = null;
        if (view) view = {}; // then the next incident shows, or the popup closes
        for (const e of list) if (isEvent(e)) applyUpdate(e);
        alertsChanged(false);
    } catch (err) {
        console.warn('[cctv] acknowledge failed:', err);
        ackError = {ids, msg: 'Could not acknowledge. Try again.'};
        if (view) renderPopup();
    } finally {
        btn.disabled = false;
    }
}

function setBig(tile) {
    if (bigTile) {
        bigTile.classList.remove('big');
        bigTile.title = '';
    }
    bigTile = tile;
    if (tile) {
        tile.classList.add('big');
        tile.title = 'Click or press Escape to restore';
    }
}

function openLiveView() {
    const camera = shown && shown.latest.camera;
    closePopup(); // without acknowledging
    if (tiles.has(camera)) setBig(tiles.get(camera).el);
}

$('m-ack').addEventListener('click', acknowledge);
$('m-live').addEventListener('click', openLiveView);
$('m-close').addEventListener('click', closePopup);
modal.addEventListener('click', (ev) => {
    if (ev.target === modal) closePopup(); // a click on the dimmed backdrop
});
$('cams').addEventListener('click', (ev) => {
    if (bigTile && bigTile.contains(ev.target)) setBig(null); // the second click restores it
});
document.addEventListener('keydown', (ev) => {
    if (ev.key !== 'Escape') return;
    if (!modal.hidden) closePopup();
    else if (bigTile) setBig(null);
});

// ---------------------------------------------------------------------------
// Numbers and timeline
// ---------------------------------------------------------------------------

const NUMBERS_EVERY = 60000; // ms; "today" and the hour buckets move on without new events

/**
 * The KPIs, the 12-hour timeline and today's per-use-case counts, from the
 * feed's events (known). They stay '–' until a list has loaded: numbers from
 * pushes alone would be incomplete. #kpi-online comes from the status poll.
 */
function renderNumbers() {
    if (!listLoaded) return;
    const now = new Date();
    const s = summarize([...known.values()], now);
    $('kpi-alerts').textContent = String(s.alertsToday);
    $('kpi-unacked').textContent = String(s.unacked);
    renderTimeline(s.timeline, timelineHours(now));
    renderUseCases(s.perUseCase);
}

/** One bar per clock hour; the tallest is full height, an empty hour has none. */
function renderTimeline(counts, hours) {
    const fractions = barFractions(counts);
    const bars = counts.map((n, i) => {
        const bar = el('i', fractions[i] > 0 ? '' : 'zero');
        bar.style.left = `calc(${(i + 0.5) * 100 / counts.length}% - 3px)`; // centred in its twelfth
        bar.style.height = `calc((100% - 16px) * ${fractions[i]})`; // 6px bottom gap, 10px headroom
        bar.title = `${pad(hours[i])}:00 · ${n} popup alert${n === 1 ? '' : 's'}`;
        return bar;
    });
    if (counts.every((n) => n === 0)) bars.push(el('div', 'none', 'No popup alerts in the last 12 hours'));
    $('timeline').replaceChildren(...bars);
    $('timeline-ticks').replaceChildren(...hours.map((h) => el('span', '', `${pad(h)}:00`)));
}

/** "Today:" and a chip per use case, most alerts first. */
function renderUseCases(perUseCase) {
    const list = Object.entries(perUseCase).sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0]));
    const box = $('timeline-uc');
    if (list.length === 0) {
        box.replaceChildren(el('span', '', 'No popup alerts today'));
        return;
    }
    box.replaceChildren(el('span', '', 'Today:'), ...list.map(([uc, n]) => {
        const severity = Object.hasOwn(USE_CASES, uc) ? USE_CASES[uc].severity : 'info';
        const chip = el('span', `uc sev-${severity}`);
        chip.append(el('i'), useCaseTitle(uc), el('b', '', n));
        return chip;
    }));
}

// ---------------------------------------------------------------------------
// Zone occupancy and energy wasted (boxes under the cameras)
// ---------------------------------------------------------------------------

const ANALYTICS = ['occupancy', 'power_when_absent'];
const ANALYTICS_EVERY = 30000; // ms between reloads; pushes update the boxes in between
const tariff = Number(params.get('tariff')) > 0 ? Number(params.get('tariff')) : TARIFF; // ₹/kWh, ?tariff=9.5
/** use case -> (id -> event). Loaded on their own: the feed's last 50 events can push them out. */
const analytics = new Map(ANALYTICS.map((uc) => [uc, new Map()]));
let analyticsLoaded = false;

const rupees = (n) => '₹' + n.toLocaleString('en-IN', {minimumFractionDigits: 2, maximumFractionDigits: 2});
const kwhText = (n) => `${n.toLocaleString('en-IN', {maximumFractionDigits: 2})} kWh`;
const cameraLabel = (camera) => {
    const place = placeOf(camera);
    return place ? `${camera} · ${place}` : camera;
};

async function loadAnalytics() {
    const lists = await Promise.all(ANALYTICS.map(async (uc) => {
        try {
            const r = await fetch(`api/cctv/events?use_case=${uc}&limit=100`, {cache: 'no-cache', signal: AbortSignal.timeout(FETCH_TIMEOUT)});
            if (!r.ok) throw new Error(`HTTP ${r.status}`);
            const list = await r.json();
            if (!Array.isArray(list)) throw new Error('events is not a list');
            return list.filter(isEvent);
        } catch (err) {
            console.warn(`[cctv] ${uc} events unavailable:`, err);
            return null; // keep what we have
        }
    }));
    ANALYTICS.forEach((uc, i) => {
        if (lists[i] === null) return;
        const m = analytics.get(uc);
        for (const e of lists[i]) m.set(e.id, e);
    });
    analyticsLoaded = true;
    renderAnalytics();
}

function analyticsAdd(e) {
    if (!analytics.has(e.use_case)) return;
    analytics.get(e.use_case).set(e.id, e);
    if (analyticsLoaded) renderAnalytics();
}

/** One line of a box: name, a bar (share of the largest), the number. */
function anaRow(name, sub, fraction, value, title) {
    const row = el('div', 'row');
    const n = el('span', 'n');
    n.append(el('b', '', name));
    if (sub) n.append(` · ${sub}`);
    const bar = el('span', 'bar');
    const fill = el('i');
    fill.style.width = `${Math.round(Math.min(1, Math.max(0, fraction)) * 100)}%`;
    bar.appendChild(fill);
    row.append(n, bar, el('span', 'x mono', value));
    if (title) row.title = title;
    return row;
}

function renderAnalytics() {
    const occ = occupancyNow([...analytics.get('occupancy').values()]);
    const occTotal = $('occ-total');
    occTotal.classList.toggle('waiting', occ === null);
    if (occ === null) {
        occTotal.textContent = 'waiting for occupancy detector';
        $('occ-rows').replaceChildren(el('div', 'none', 'Counts show here once the occupancy use case sends events'));
    } else {
        occTotal.textContent = `${occ.total} ${occ.total === 1 ? 'person' : 'people'}`;
        const max = Math.max(1, ...occ.zones.map((z) => z.count));
        $('occ-rows').replaceChildren(...occ.zones.map((z) => anaRow(
            z.zone || cameraLabel(z.camera), z.zone ? z.camera : '', z.count / max, String(z.count),
            `Since ${hms(z.time)}`)));
    }

    const w = energyWaste([...analytics.get('power_when_absent').values()], new Date(), tariff);
    const total = $('energy-total');
    total.classList.toggle('waiting', w === null);
    if (w === null) {
        total.textContent = 'waiting for power detector';
        $('energy-note').textContent = '';
        $('energy-rows').replaceChildren(el('div', 'none', 'Costs show here once the power-when-absent use case sends events'));
        return;
    }
    total.textContent = rupees(w.rupees);
    $('energy-note').textContent = `${kwhText(w.kwh)} · ${w.times} time${w.times === 1 ? '' : 's'} · at ${rupees(tariff)}/kWh`;
    if (w.cameras.length === 0) {
        $('energy-rows').replaceChildren(el('div', 'none', 'Nothing left on in an empty room today'));
        return;
    }
    const max = Math.max(...w.cameras.map((c) => c.rupees));
    $('energy-rows').replaceChildren(...w.cameras.map((c) => anaRow(
        cameraLabel(c.camera), '', max > 0 ? c.rupees / max : 0, rupees(c.rupees),
        `${kwhText(c.kwh)} · ${c.times} time${c.times === 1 ? '' : 's'}`)));
}

// Main-site face enrollment. The server validates image bytes and the worker
// notices a newly rebuilt gallery without reconnecting live camera streams.
const facesPanel = $('faces-panel');
const facesStatus = $('faces-status');

async function loadFacePeople() {
    const list = $('faces-list');
    try {
        const r = await fetch('api/cctv/faces', {cache: 'no-cache', signal: AbortSignal.timeout(FETCH_TIMEOUT)});
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        const people = await r.json();
        if (!Array.isArray(people)) throw new Error('gallery response is not a list');
        list.replaceChildren(...people.map((person) => {
            const row = el('div', 'face-row');
            const info = el('span');
            info.append(el('b', '', person.name), el('br'), el('small', '', `${person.photos} photo${person.photos === 1 ? '' : 's'}`));
            const remove = el('button', 'btn', 'Remove');
            remove.type = 'button';
            remove.addEventListener('click', async () => {
                if (!confirm(`Remove all locally stored photos for ${person.name}?`)) return;
                const url = `api/cctv/faces?name=${encodeURIComponent(person.name)}`;
                const response = await fetch(url, {method: 'DELETE', signal: AbortSignal.timeout(FETCH_TIMEOUT)});
                if (!response.ok) { facesStatus.textContent = 'Could not remove this person.'; return; }
                facesStatus.textContent = `${person.name} photos removed. Rebuild the gallery to apply the change.`;
                await loadFacePeople();
            });
            row.append(info, remove);
            return row;
        }));
        if (people.length === 0) list.append(el('div', 'empty', 'No enrolled people yet'));
        facesStatus.textContent = `${people.length} enrolled ${people.length === 1 ? 'person' : 'people'}`;
    } catch (err) {
        list.replaceChildren();
        facesStatus.textContent = 'Face gallery API unavailable. Start the main CCTV server first.';
        console.warn('[cctv] face gallery unavailable:', err);
    }
}

$('faces-open').addEventListener('click', () => {
    facesPanel.hidden = false;
    loadFacePeople();
});
$('faces-close').addEventListener('click', () => { facesPanel.hidden = true; });
facesPanel.addEventListener('click', (event) => { if (event.target === facesPanel) facesPanel.hidden = true; });
$('faces-form').addEventListener('submit', async (event) => {
    event.preventDefault();
    const name = $('faces-name').value.trim();
    const files = [...$('faces-files').files];
    if (!name || files.length === 0) { facesStatus.textContent = 'Enter a name and choose at least one photo.'; return; }
    $('faces-upload').disabled = true;
    let done = 0;
    for (const file of files) {
        const body = new FormData();
        body.set('name', name);
        body.set('photo', file);
        try {
            const r = await fetch('api/cctv/faces', {method: 'POST', body, signal: AbortSignal.timeout(FETCH_TIMEOUT)});
            const data = await r.json();
            if (!r.ok) throw new Error(data.error || `HTTP ${r.status}`);
            done++;
            facesStatus.textContent = `Uploaded ${done}/${files.length} photos…`;
        } catch (err) {
            facesStatus.textContent = `Uploaded ${done}/${files.length}. ${err.message}`;
            break;
        }
    }
    $('faces-upload').disabled = false;
    $('faces-files').value = '';
    if (done === files.length) facesStatus.textContent = `Added ${done} photo${done === 1 ? '' : 's'} for ${name}. Build the gallery to apply them.`;
    await loadFacePeople();
});
$('faces-rebuild').addEventListener('click', async () => {
    const button = $('faces-rebuild');
    button.disabled = true;
    facesStatus.textContent = 'Building face embeddings…';
    try {
        const r = await fetch('api/cctv/faces/rebuild', {method: 'POST', signal: AbortSignal.timeout(130000)});
        const data = await r.json();
        if (!r.ok) throw new Error(data.error + (data.detail ? ` · ${data.detail}` : ''));
        facesStatus.textContent = data.message || 'Gallery rebuilt. Recognition will update on the next frames.';
    } catch (err) {
        facesStatus.textContent = err.message || 'Gallery rebuild failed.';
    } finally {
        button.disabled = false;
        await loadFacePeople();
    }
});

// ---------------------------------------------------------------------------
// Startup
// ---------------------------------------------------------------------------

tick();
setInterval(tick, 1000);
setInterval(renderNumbers, NUMBERS_EVERY); // the only numbers timer; event changes redraw at once
pollStatus();
pollWorkerHealth();
connect(); // its open event loads the alert list
loadFeed();
loadAnalytics();
setInterval(loadAnalytics, ANALYTICS_EVERY); // also moves "today" on past midnight
