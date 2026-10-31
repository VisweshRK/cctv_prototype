// Pure logic for the CCTV dashboard (www/cctv.html): how each event is shown,
// the popup queue and the summary numbers. No DOM, no dependencies, so it can
// be tested with `node --test`.
//
// Events are the JSON of internal/cctv Event: id, camera, use_case, label,
// confidence (0..1), time (RFC3339), snapshot, data, acknowledged_at, siren.

// The spec table, in one place. Label overrides live in describe().
export const USE_CASES = {
    intrusion: {severity: 'critical', title: 'Burglary alarm · intrusion', popup: true, siren: true},
    restricted_area: {severity: 'critical', title: 'Breach · restricted area', popup: true, siren: true},
    fire: {severity: 'critical', title: 'Fire / smoke detected', popup: true, siren: true},
    fall: {severity: 'high', title: 'Person fell', popup: true, siren: false},
    tampering: {severity: 'high', title: 'Camera tampering / offline', popup: true, siren: false},
    baby_crying: {severity: 'medium', title: 'Baby crying', popup: true, siren: false},
    face_recognition: {severity: 'medium', title: 'Unknown person', popup: true, siren: false},
    power_when_absent: {severity: 'medium', title: 'Power on, nobody here', popup: true, siren: false},
    occupancy: {severity: 'info', title: 'Occupancy changed', popup: false, siren: false},
    vehicle_lpr: {severity: 'info', title: 'Vehicle', popup: true, siren: false},
};

const SNAPSHOT_WAIT = 12000; // ms to wait for a snapshot before showing "none"
const HOUR = 3600000;

const dataOf = (e) => (e && e.data) || {};
const nonEmpty = (v) => typeof v === 'string' && v !== '';
const mod = (a, n) => ((a % n) + n) % n; // never negative, unlike %

/**
 * An event time in ms, or NaN. Go writes nanoseconds; only milliseconds are
 * kept, so every browser parses it.
 */
export function parseTime(t) {
    return Date.parse(typeof t === 'string' ? t.replace(/(\.\d{3})\d+/, '$1') : '');
}

/**
 * How an event is shown. Titles are raw text: escape them when rendering.
 * @returns {{severity: string, title: string, popup: boolean, siren: boolean}}
 */
export function describe(e) {
    const d = dataOf(e);
    const base = Object.hasOwn(USE_CASES, e.use_case)
        ? USE_CASES[e.use_case]
        : {severity: 'info', title: String(e.use_case), popup: false, siren: false};
    let over = {};

    switch (e.use_case) {
        case 'tampering':
            if (e.label === 'camera_online') {
                over = {severity: 'info', title: 'Camera back online', popup: false, siren: false};
            } else if (e.label === 'restored') {
                over = {severity: 'info', title: 'Camera view restored', popup: false, siren: false};
            }
            break;
        case 'face_recognition':
            if (e.label === 'known_face') {
                over = {severity: 'info', title: `Known: ${d.name || 'person'}`, popup: false, siren: false};
            }
            break;
        case 'vehicle_lpr':
            over = nonEmpty(d.plate)
                ? {title: `${base.title} · ${d.plate}`, popup: true}
                : {popup: false};
            break;
    }

    const r = {...base, ...over};
    return {severity: r.severity, title: r.title, popup: r.popup, siren: r.siren};
}

export function areaOf(e) {
    const d = dataOf(e);
    return d.area || d.zone || 'Whole view';
}

/**
 * The popup subtitle: data.detail, else label, plate or name, and the area
 * (only when the event names one) joined with " · ". Raw text.
 */
export function detailOf(e) {
    const d = dataOf(e);
    if (nonEmpty(d.detail)) return d.detail;
    const parts = [e.label, d.plate || d.name, d.area || d.zone ? areaOf(e) : ''];
    return parts.filter(Boolean).map(String).join(' · ');
}

/** data.box as [x1, y1, x2, y2] fractions of the frame, or null if it isn't a valid box. */
export function boxOf(e) {
    const b = dataOf(e).box;
    if (!Array.isArray(b) || b.length !== 4) return null;
    if (!b.every((v) => typeof v === 'number' && v >= 0 && v <= 1)) return null; // NaN fails the range
    if (!(b[0] < b[2] && b[1] < b[3])) return null;
    return [...b];
}

const ESC = {'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'};

export function esc(s) {
    return String(s).replace(/[&<>"']/g, (c) => ESC[c]);
}

// Labels the server never grabs a frame for (internal/cctv/snapshots.go handle()).
const NO_GRAB = new Set(['camera_offline', 'camera_online']);

/**
 * "ready" if the event has a snapshot, "loading" for the first 12 s after
 * e.time (the server may still be grabbing a frame), else "none". A camera
 * that went offline or came back gets no grabbed frame, so it is "none" at once.
 * @param {number} now ms, e.g. Date.now()
 */
export function snapshotState(e, now) {
    if (e.snapshot) return 'ready';
    if (NO_GRAB.has(e.label)) return 'none';
    const t = parseTime(e.time);
    if (Number.isNaN(t)) return 'none';
    return now - t < SNAPSHOT_WAIT ? 'loading' : 'none';
}

const SNAPSHOT_PREFIX = 'api/cctv/snapshot?id=';

/**
 * The image URL to show for an event, or '' if none. Only the server's own
 * snapshot URLs are used: clients can POST any string as snapshot.
 */
export function snapshotSrc(e) {
    const s = e && e.snapshot;
    return typeof s === 'string' && s.startsWith(SNAPSHOT_PREFIX) ? s : '';
}

/**
 * The popup's siren line (shown after a "Siren" label), or '' when the use
 * case has no siren. Raw text.
 */
export function sirenText(e) {
    if (!describe(e).siren) return '';
    if (e.siren === 'ringing') return `ringing on ${e.camera}`;
    if (e.siren === 'unavailable') return 'camera speaker not available';
    return 'off'; // stopped, or not (yet) rung
}

const isAcked = (e) => Boolean(e.acknowledged_at);
const keyOf = (e) => `${e.camera}|${e.use_case}`;

/**
 * The popup queue. Unacknowledged popup events are grouped by camera and use
 * case; current() is the group that most recently got a new event.
 */
export class Incidents {
    // key -> {key, camera, useCase, events}; Map order = oldest group first
    #groups = new Map();

    #find(id) {
        for (const g of this.#groups.values()) {
            const i = g.events.findIndex((x) => x.id === id);
            if (i >= 0) return {g, i};
        }
        return null;
    }

    /** Adds a new event; an id already queued is handled like update(e). */
    add(e) {
        if (this.#find(e.id)) {
            this.update(e);
            return;
        }
        if (isAcked(e) || !describe(e).popup) return;

        const key = keyOf(e);
        const g = this.#groups.get(key) || {key, camera: e.camera, useCase: e.use_case, events: []};
        g.events.push(e);
        this.#groups.delete(key);
        this.#groups.set(key, g); // now the newest group
    }

    /**
     * Replaces a queued event by id, without reordering groups. An unknown id
     * is ignored. An event that is now acknowledged leaves its group (an empty
     * group is removed); one whose camera|use_case or popup changed leaves its
     * group and goes through add().
     */
    update(e) {
        const found = this.#find(e.id);
        if (!found) return;
        const {g, i} = found;
        if (!isAcked(e) && keyOf(e) === g.key && describe(e).popup) {
            g.events[i] = e;
            return;
        }
        g.events.splice(i, 1);
        if (!g.events.length) this.#groups.delete(g.key);
        this.add(e); // ignores it if acknowledged or not a popup
    }

    static #view(g) {
        const events = [...g.events];
        return {key: g.key, camera: g.camera, useCase: g.useCase, events, latest: events[events.length - 1]};
    }

    /** @returns {{key, camera, useCase, events, latest} | null} */
    current() {
        const g = [...this.#groups.values()].pop();
        return g ? Incidents.#view(g) : null;
    }

    /** The queued group holding event id, shaped like current(), or null. */
    groupOf(id) {
        const found = this.#find(id);
        return found ? Incidents.#view(found.g) : null;
    }

    others() {
        return Math.max(0, this.#groups.size - 1);
    }

    ids(key) {
        const g = this.#groups.get(key);
        return g ? g.events.map((x) => x.id) : [];
    }
}

/**
 * Files a freshly loaded event list (page load, or a reconnect) into the
 * queue. Every unacknowledged popup event joins it, so nothing that still
 * needs an ack drops out after a reload, and acks or changes missed while
 * disconnected are applied. Only a siren still ringing should pop up: it
 * returns true if one wasn't queued before, and ringing groups are filed
 * last, so current() is then a ringing group.
 * @param {Incidents} incidents
 * @returns {boolean} whether to open the popup
 */
export function queueList(incidents, events) {
    const oldestFirst = [...events].sort((a, b) => a.id - b.id); // the newest group ends current
    const ringing = (e) => e.siren === 'ringing' && !isAcked(e) && describe(e).popup;
    for (const e of oldestFirst) {
        if (!ringing(e)) incidents.add(e); // add() updates a queued id, ignores acked and feed-only ones
    }
    let pop = false;
    for (const e of oldestFirst) {
        if (!ringing(e)) continue;
        if (!incidents.groupOf(e.id)) pop = true;
        incidents.add(e);
    }
    return pop;
}

// The start of n's local clock hour, in ms. Not setMinutes(0): in a DST
// fall-back hour that picks the earlier of the two instants.
function hourStartOf(n) {
    return n.getTime() - (n.getMinutes() * 60000 + n.getSeconds() * 1000 + n.getMilliseconds());
}

/**
 * Dashboard numbers over popup events only. "Today" is now's local calendar
 * date; timeline holds 12 local clock-hour counts, oldest first, the last one
 * being the current hour up to now.
 * @param {Date|number} now
 * @returns {{alertsToday: number, unacked: number, perUseCase: Object<string, number>, timeline: number[]}}
 */
export function summarize(events, now) {
    const n = new Date(now);
    const nowMs = n.getTime();
    const hourStart = hourStartOf(n);
    const sameDay = (t) => {
        const d = new Date(t);
        return d.getFullYear() === n.getFullYear() && d.getMonth() === n.getMonth() && d.getDate() === n.getDate();
    };

    const s = {alertsToday: 0, unacked: 0, perUseCase: {}, timeline: Array(12).fill(0)};
    for (const e of events) {
        if (!describe(e).popup) continue;
        if (!isAcked(e)) s.unacked++;

        const t = parseTime(e.time);
        if (Number.isNaN(t)) continue;
        if (sameDay(t)) {
            s.alertsToday++;
            s.perUseCase[e.use_case] = (s.perUseCase[e.use_case] || 0) + 1;
        }
        const i = 11 + Math.floor((t - hourStart) / HOUR);
        if (i >= 0 && i < 12 && t <= nowMs) s.timeline[i]++;
    }
    return s;
}

/**
 * The local clock hour (0..23) at which each of summarize()'s 12 timeline
 * buckets starts, oldest first; the last one is the current hour.
 * @param {Date|number} now
 */
export function timelineHours(now) {
    const start = hourStartOf(new Date(now));
    return Array.from({length: 12}, (_, i) => new Date(start - (11 - i) * HOUR).getHours());
}

/**
 * Bar heights as fractions 0..1: the largest count is 1, an empty bucket is 0
 * (no minimum height), and all are 0 when every count is 0.
 * @param {number[]} counts
 */
export function barFractions(counts) {
    const max = Math.max(0, ...counts);
    return counts.map((c) => (max > 0 && c > 0 ? c / max : 0));
}

// What "Simulate alert" cycles through: one popup event per use case.
const CYCLE = [
    {use_case: 'intrusion', label: 'person', confidence: 0.91,
        data: {area: 'Main gate', detail: '1 person crossed the perimeter line', box: [0.42, 0.28, 0.58, 0.86]}},
    {use_case: 'restricted_area', label: 'person', confidence: 0.88,
        data: {area: 'Server room door', detail: '1 person in zone for 6 s', box: [0.18, 0.22, 0.36, 0.9]}},
    {use_case: 'fire', label: 'smoke', confidence: 0.84,
        data: {area: 'Kitchen', detail: 'Smoke over 15% of the view', box: [0.55, 0.05, 0.95, 0.5]}},
    {use_case: 'tampering', label: 'lens_covered', confidence: 0.95,
        data: {area: 'Whole view', detail: 'View blocked for 10 s', box: [0.05, 0.05, 0.95, 0.95]}},
    {use_case: 'face_recognition', label: 'unknown_face', confidence: 0.86,
        data: {area: 'Reception', detail: 'Face not in the known list', box: [0.45, 0.2, 0.57, 0.42]}},
    {use_case: 'vehicle_lpr', label: 'plate', confidence: 0.9,
        data: {area: 'Main gate', plate: 'TN09BX4821', detail: 'Car entering, plate read', box: [0.38, 0.62, 0.56, 0.74]}},
];

/**
 * A POST /api/cctv/events body like an AI worker would send, deterministic
 * per i. The server fills id and time.
 *
 * cameras are {name, use_cases} as in /api/cctv/status (a plain name also
 * works). The event goes to a camera configured for its use case, taking
 * turns if there are several; if none is, round-robin over all of them.
 */
export function simulated(i, cameras) {
    const cams = (Array.isArray(cameras) ? cameras : [])
        .map((x) => (typeof x === 'string' ? {name: x} : x))
        .filter((x) => x && nonEmpty(x.name));
    const c = CYCLE[mod(i, CYCLE.length)];
    const fit = cams.filter((x) => Array.isArray(x.use_cases) && x.use_cases.includes(c.use_case));
    const camera = fit.length ? fit[mod(Math.floor(i / CYCLE.length), fit.length)].name
        : cams.length ? cams[mod(i, cams.length)].name : 'cam1';
    const wobble = mod(i * 7, 5) / 100; // 0..0.04, so repeated alerts differ a bit
    return {
        camera,
        use_case: c.use_case,
        label: c.label,
        confidence: Math.round((c.confidence + wobble - 0.02) * 100) / 100,
        data: {...c.data, box: [...c.data.box]},
    };
}
