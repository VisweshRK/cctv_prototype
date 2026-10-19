// Tests for www/cctv-logic.js. Run with: node --test "www/tests/*.test.mjs"
import {test} from 'node:test';
import assert from 'node:assert/strict';
import {spawnSync} from 'node:child_process';

import {
    USE_CASES, describe, areaOf, detailOf, boxOf, esc, snapshotState, snapshotSrc, sirenText,
    Incidents, summarize, simulated, timelineHours, barFractions, parseTime,
} from '../cctv-logic.js';

const ev = (fields) => ({
    id: 1, camera: 'cam1', use_case: 'intrusion', label: 'person', confidence: 0.9,
    time: '2026-10-19T10:00:00Z', ...fields,
});

test('USE_CASES holds every use case of the spec table', () => {
    for (const uc of ['intrusion', 'restricted_area', 'fire', 'fall', 'tampering', 'baby_crying',
        'face_recognition', 'power_when_absent', 'occupancy', 'vehicle_lpr']) {
        assert.ok(USE_CASES[uc], uc);
    }
    assert.equal(USE_CASES.fire.severity, 'critical');
    assert.equal(USE_CASES.fire.siren, true);
    assert.equal(USE_CASES.fall.siren, false);
});

test('describe: intrusion is critical with siren and popup', () => {
    assert.deepEqual(describe(ev({})),
        {severity: 'critical', title: 'Burglary alarm · intrusion', popup: true, siren: true});
});

test('describe: rows of the spec table', () => {
    const d = (use_case, label = 'x') => describe(ev({use_case, label}));
    assert.deepEqual(d('restricted_area'),
        {severity: 'critical', title: 'Breach · restricted area', popup: true, siren: true});
    assert.deepEqual(d('fire', 'smoke'),
        {severity: 'critical', title: 'Fire / smoke detected', popup: true, siren: true});
    assert.deepEqual(d('fall', 'person'),
        {severity: 'high', title: 'Person fell', popup: true, siren: false});
    assert.deepEqual(d('baby_crying', 'crying'),
        {severity: 'medium', title: 'Baby crying', popup: true, siren: false});
    assert.deepEqual(d('power_when_absent', 'power_on'),
        {severity: 'medium', title: 'Power on, nobody here', popup: true, siren: false});
});

test('describe: tampering labels', () => {
    assert.deepEqual(describe(ev({use_case: 'tampering', label: 'camera_offline'})),
        {severity: 'high', title: 'Camera tampering / offline', popup: true, siren: false});
    assert.deepEqual(describe(ev({use_case: 'tampering', label: 'lens_covered'})),
        {severity: 'high', title: 'Camera tampering / offline', popup: true, siren: false});
    assert.deepEqual(describe(ev({use_case: 'tampering', label: 'camera_online'})),
        {severity: 'info', title: 'Camera back online', popup: false, siren: false});
});

test('describe: known and unknown faces', () => {
    const known = describe(ev({use_case: 'face_recognition', label: 'known_face', data: {name: 'Asha'}}));
    assert.equal(known.popup, false);
    assert.equal(known.severity, 'info');
    assert.equal(known.title, 'Known: Asha');
    assert.equal(describe(ev({use_case: 'face_recognition', label: 'known_face'})).title, 'Known: person');

    for (const label of ['unknown_face', 'face']) {
        assert.deepEqual(describe(ev({use_case: 'face_recognition', label})),
            {severity: 'medium', title: 'Unknown person', popup: true, siren: false});
    }
});

test('describe: vehicles pop up only with a plate', () => {
    const plate = describe(ev({use_case: 'vehicle_lpr', label: 'plate', data: {plate: 'TN09AB1234'}}));
    assert.equal(plate.popup, true);
    assert.equal(plate.severity, 'info');
    assert.equal(plate.title, 'Vehicle · TN09AB1234');

    for (const e of [
        ev({use_case: 'vehicle_lpr', label: 'vehicle'}),
        ev({use_case: 'vehicle_lpr', label: 'plate'}),
        ev({use_case: 'vehicle_lpr', label: 'plate', data: {plate: ''}}),
    ]) {
        assert.deepEqual(describe(e), {severity: 'info', title: 'Vehicle', popup: false, siren: false});
    }
});

test('describe: occupancy is feed only, an unknown use case shows raw', () => {
    assert.equal(describe(ev({use_case: 'occupancy', label: 'count'})).popup, false);
    assert.equal(describe(ev({use_case: 'occupancy', label: 'count'})).title, 'Occupancy changed');
    assert.deepEqual(describe(ev({use_case: 'weird'})),
        {severity: 'info', title: 'weird', popup: false, siren: false});
    // inherited object keys are not use cases
    assert.equal(describe(ev({use_case: 'constructor'})).title, 'constructor');
});

test('describe: titles are raw, not escaped', () => {
    const e = ev({use_case: 'face_recognition', label: 'known_face', data: {name: '<b>Asha & co</b>'}});
    assert.equal(describe(e).title, 'Known: <b>Asha & co</b>');
});

test('areaOf', () => {
    assert.equal(areaOf(ev({data: {area: 'Server room door', zone: 'z1'}})), 'Server room door');
    assert.equal(areaOf(ev({data: {zone: 'z1'}})), 'z1');
    assert.equal(areaOf(ev({data: {}})), 'Whole view');
    assert.equal(areaOf(ev({})), 'Whole view');
});

test('detailOf', () => {
    assert.equal(detailOf(ev({data: {detail: '1 person in zone for 6 s', area: 'Gate'}})),
        '1 person in zone for 6 s');
    assert.equal(detailOf(ev({data: {area: 'Server room door'}})), 'person · Server room door');
    assert.equal(detailOf(ev({data: {zone: 'z1', detail: ''}})), 'person · z1');
    assert.equal(detailOf(ev({})), 'person');
    assert.equal(detailOf(ev({use_case: 'vehicle_lpr', label: 'plate', data: {plate: 'TN09AB1234'}})),
        'plate · TN09AB1234');
    assert.equal(detailOf(ev({use_case: 'face_recognition', label: 'known_face', data: {name: 'Asha', area: 'Lobby'}})),
        'known_face · Asha · Lobby');
    assert.equal(detailOf(ev({data: {area: '<i>'}})), 'person · <i>');
});

test('boxOf', () => {
    assert.deepEqual(boxOf(ev({data: {box: [0.1, 0.2, 0.4, 0.9]}})), [0.1, 0.2, 0.4, 0.9]);
    assert.deepEqual(boxOf(ev({data: {box: [0, 0, 1, 1]}})), [0, 0, 1, 1]);
    for (const box of [
        [1, 2, 3, 4], [0.5, 0.2, 0.4, 0.9], [0.1, 0.9, 0.4, 0.2], [0.1, 0.2, 0.1, 0.9],
        [0.1, 0.2, 0.4], [0.1, 0.2, 0.4, 0.9, 1], [NaN, 0.2, 0.4, 0.9], [0.1, 0.2, Infinity, 0.9],
        ['0.1', 0.2, 0.4, 0.9], [-0.1, 0.2, 0.4, 0.9], '0.1,0.2,0.4,0.9', {0: 0.1}, null,
    ]) {
        assert.equal(boxOf(ev({data: {box}})), null, JSON.stringify(box));
    }
    assert.equal(boxOf(ev({data: {}})), null);
    assert.equal(boxOf(ev({})), null);
});

test('esc', () => {
    const out = esc('<img src=x onerror=alert(1)> & "q"');
    assert.ok(!out.includes('<'));
    assert.ok(!out.includes('"'));
    assert.equal(out, '&lt;img src=x onerror=alert(1)&gt; &amp; &quot;q&quot;');
    assert.equal(esc("it's"), 'it&#39;s');
    assert.equal(esc('&amp;'), '&amp;amp;');
    assert.equal(esc(42), '42');
});

test('snapshotState', () => {
    const t = Date.parse('2026-10-19T10:00:00Z');
    assert.equal(snapshotState(ev({snapshot: '/api/cctv/snapshot?id=1'}), t + 60000), 'ready');
    assert.equal(snapshotState(ev({}), t + 5000), 'loading');
    assert.equal(snapshotState(ev({}), t + 13000), 'none');
    assert.equal(snapshotState(ev({time: 'not a time'}), t), 'none');
});

test('parseTime: Go nanoseconds are cut to milliseconds', () => {
    assert.equal(parseTime('2026-10-19T10:00:00.123456789+05:30'), Date.parse('2026-10-19T10:00:00.123+05:30'));
    assert.equal(parseTime('2026-10-19T10:00:00Z'), Date.parse('2026-10-19T10:00:00Z'));
    assert.ok(Number.isNaN(parseTime('not a time')));
    assert.ok(Number.isNaN(parseTime(undefined)));
    assert.ok(Number.isNaN(parseTime(5)));
});

test('snapshotState: no waiting for cameras that went offline or came back', () => {
    const t = Date.parse('2026-10-19T10:00:00Z');
    for (const label of ['camera_offline', 'camera_online']) {
        assert.equal(snapshotState(ev({use_case: 'tampering', label}), t + 1000), 'none', label);
        assert.equal(snapshotState(ev({use_case: 'intrusion', label}), t + 1000), 'none', label); // any use case
        // a snapshot that arrives anyway (an image POSTed with the event) still shows
        assert.equal(snapshotState(ev({label, snapshot: 'api/cctv/snapshot?id=1'}), t + 1000), 'ready', label);
    }
    assert.equal(snapshotState(ev({use_case: 'tampering', label: 'lens_covered'}), t + 1000), 'loading');
});

test('snapshotSrc: only the server snapshot URLs', () => {
    assert.equal(snapshotSrc(ev({snapshot: 'api/cctv/snapshot?id=7'})), 'api/cctv/snapshot?id=7');
    assert.equal(snapshotSrc(ev({})), '');
    assert.equal(snapshotSrc(ev({snapshot: ''})), '');
    assert.equal(snapshotSrc(ev({snapshot: 'http://evil.example/x.jpg'})), '');
    assert.equal(snapshotSrc(ev({snapshot: 'javascript:alert(1)'})), '');
    assert.equal(snapshotSrc(ev({snapshot: '/api/cctv/snapshot?id=7'})), '');
    assert.equal(snapshotSrc(ev({snapshot: 7})), '');
    assert.equal(snapshotSrc(null), '');
});

test('sirenText', () => {
    assert.equal(sirenText(ev({siren: 'ringing'})), 'ringing on cam1');
    assert.equal(sirenText(ev({siren: 'unavailable'})), 'camera speaker not available');
    assert.equal(sirenText(ev({siren: 'stopped'})), 'off');
    assert.equal(sirenText(ev({})), 'off');
    assert.equal(sirenText(ev({use_case: 'fire', camera: 'cam2', siren: 'ringing'})), 'ringing on cam2');
    assert.equal(sirenText(ev({use_case: 'fall', siren: 'ringing'})), ''); // no siren for this use case
    assert.equal(sirenText(ev({use_case: 'nope'})), '');
});

test('Incidents: groupOf finds the group of an event id', () => {
    const q = new Incidents();
    q.add(ev({id: 1}));
    q.add(ev({id: 2, use_case: 'fire'}));
    q.add(ev({id: 3}));
    const g = q.groupOf(1);
    assert.equal(g.key, 'cam1|intrusion');
    assert.deepEqual(g.events.map((x) => x.id), [1, 3]);
    assert.equal(g.latest.id, 3);
    assert.equal(q.groupOf(2).key, 'cam1|fire');
    assert.equal(q.groupOf(99), null);
    g.events.push(ev({id: 50})); // a copy
    assert.deepEqual(q.ids('cam1|intrusion'), [1, 3]);
});

test('Incidents: groups by camera and use case, newest group first', () => {
    const q = new Incidents();
    assert.equal(q.current(), null);
    assert.equal(q.others(), 0);

    q.add(ev({id: 1}));
    q.add(ev({id: 2}));
    q.add(ev({id: 4, camera: 'cam2', use_case: 'fire', label: 'smoke'}));
    assert.equal(q.current().useCase, 'fire');
    assert.equal(q.current().camera, 'cam2');
    assert.equal(q.current().key, 'cam2|fire');
    assert.equal(q.others(), 1);

    q.add(ev({id: 3}));
    const cur = q.current();
    assert.equal(cur.key, 'cam1|intrusion');
    assert.equal(cur.useCase, 'intrusion');
    assert.equal(cur.events.length, 3);
    assert.deepEqual(cur.events.map((e) => e.id), [1, 2, 3]);
    assert.equal(cur.latest.id, 3);
    assert.deepEqual(q.ids('cam1|intrusion'), [1, 2, 3]);
    assert.deepEqual(q.ids('nope'), []);

    const acked = {acknowledged_at: '2026-10-19T10:01:00Z'};
    q.update(ev({id: 1, ...acked}));
    assert.deepEqual(q.ids('cam1|intrusion'), [2, 3]);
    q.update(ev({id: 2, ...acked}));
    q.update(ev({id: 3, ...acked}));
    assert.equal(q.current().useCase, 'fire');
    assert.equal(q.others(), 0);
    assert.deepEqual(q.ids('cam1|intrusion'), []);

    q.update(ev({id: 4, camera: 'cam2', use_case: 'fire', label: 'smoke', ...acked}));
    assert.equal(q.current(), null);
});

test('Incidents: ignores feed-only and acknowledged events', () => {
    const q = new Incidents();
    q.add(ev({id: 1, use_case: 'face_recognition', label: 'known_face', data: {name: 'Asha'}}));
    q.add(ev({id: 2, use_case: 'occupancy', label: 'count'}));
    q.add(ev({id: 3, acknowledged_at: '2026-10-19T10:01:00Z'}));
    assert.equal(q.current(), null);
});

test('Incidents: update replaces in place and never adds or reorders', () => {
    const q = new Incidents();
    q.add(ev({id: 1}));
    q.add(ev({id: 2, camera: 'cam2', use_case: 'fire', label: 'smoke'}));

    // a late snapshot for the older group must not steal the popup
    q.update(ev({id: 1, snapshot: '/snap/1.jpg'}));
    assert.equal(q.current().useCase, 'fire');
    assert.equal(q.others(), 1);
    q.update(ev({id: 2, camera: 'cam2', use_case: 'fire', label: 'smoke', snapshot: '/snap/2.jpg'}));
    assert.equal(q.current().latest.snapshot, '/snap/2.jpg');

    // an unknown id is ignored
    q.update(ev({id: 99, camera: 'cam3'}));
    assert.equal(q.others(), 1);
    assert.deepEqual(q.ids('cam3|intrusion'), []);
});

test('Incidents: update that changes group or popup re-files the event', () => {
    const q = new Incidents();
    q.add(ev({id: 1}));
    q.add(ev({id: 2}));
    q.add(ev({id: 3, camera: 'cam2', use_case: 'fire', label: 'smoke'}));

    // id 1 now reports cam3: it leaves cam1|intrusion and is added as cam3|intrusion
    q.update(ev({id: 1, camera: 'cam3'}));
    assert.deepEqual(q.ids('cam1|intrusion'), [2]);
    assert.equal(q.current().key, 'cam3|intrusion');
    assert.deepEqual(q.ids('cam3|intrusion'), [1]);
    assert.equal(q.others(), 2);

    // same group, but no longer a popup: it leaves the queue
    q.add(ev({id: 4, use_case: 'vehicle_lpr', label: 'plate', data: {plate: 'TN09AB1234'}}));
    assert.deepEqual(q.ids('cam1|vehicle_lpr'), [4]);
    q.update(ev({id: 4, use_case: 'vehicle_lpr', label: 'plate', data: {}}));
    assert.deepEqual(q.ids('cam1|vehicle_lpr'), []);
    assert.equal(q.current().key, 'cam3|intrusion');
});

test('Incidents: re-adding an id replaces that event in place', () => {
    const q = new Incidents();
    q.add(ev({id: 1}));
    q.add(ev({id: 2}));
    q.add(ev({id: 1, confidence: 0.5}));
    const cur = q.current();
    assert.deepEqual(cur.events.map((e) => e.id), [1, 2]);
    assert.equal(cur.events[0].confidence, 0.5);
    assert.equal(cur.latest.id, 2);

    // re-added as acknowledged: it leaves the queue
    q.add(ev({id: 2, acknowledged_at: '2026-10-19T10:01:00Z'}));
    assert.deepEqual(q.ids('cam1|intrusion'), [1]);
});

test('Incidents: current() does not expose the internal list', () => {
    const q = new Incidents();
    q.add(ev({id: 1}));
    q.current().events.push(ev({id: 7}));
    assert.deepEqual(q.ids('cam1|intrusion'), [1]);
});

test('summarize', () => {
    // fixed local time with minutes >= 30, so now-30 min stays in the current clock hour
    const now = new Date(2026, 9, 19, 14, 45, 0);
    const at = (ms) => new Date(ms).toISOString();
    const yesterday23 = new Date(2026, 9, 18, 23, 0, 0).getTime();
    const events = [
        ev({id: 1, time: at(now - 30 * 60000)}),
        ev({id: 2, camera: 'cam2', use_case: 'fire', label: 'smoke', time: at(now - 3 * 3600000),
            acknowledged_at: at(now - 2 * 3600000)}),
        ev({id: 3, time: at(yesterday23)}),
        ev({id: 4, use_case: 'occupancy', label: 'count', time: at(now - 10 * 60000)}),
    ];

    for (const n of [now, now.getTime()]) {
        const s = summarize(events, n);
        assert.equal(s.alertsToday, 2);
        assert.equal(s.unacked, 2); // ids 1 and 3; 2 is acknowledged, 4 is not a popup
        assert.deepEqual(s.perUseCase, {intrusion: 1, fire: 1});
        assert.equal(s.timeline.length, 12);
        assert.equal(s.timeline[11], 1);
        assert.equal(s.timeline[8], 1);
        assert.equal(s.timeline.reduce((a, b) => a + b, 0), 2);
    }

    const empty = summarize([], now);
    assert.deepEqual(empty, {alertsToday: 0, unacked: 0, perUseCase: {}, timeline: Array(12).fill(0)});
});

test('summarize: timeline covers the current clock hour from its start', () => {
    const now = new Date(2026, 9, 19, 14, 45, 0);
    const at = (h, m) => new Date(2026, 9, 19, h, m, 0).toISOString();
    const s = summarize([
        ev({id: 1, time: at(14, 0)}),   // start of current hour
        ev({id: 2, time: at(13, 59)}),  // previous hour
        ev({id: 3, time: at(3, 0)}),    // oldest bucket
        ev({id: 4, time: at(2, 59)}),   // before the timeline, still today
        ev({id: 5, time: at(14, 50)}),  // after now
    ], now);
    assert.deepEqual(s.timeline, [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]);
    assert.equal(s.alertsToday, 5);
});

test('summarize: DST fall-back hour (America/New_York)', (t) => {
    // TZ must be set before the process starts, so run the check in a child.
    // 2026-11-01 01:30 EST: 01:00-02:00 happens twice, now is in the second one.
    const code = `
        const {summarize, timelineHours} = await import(${JSON.stringify(new URL('../cctv-logic.js', import.meta.url).href)});
        const now = Date.parse('2026-11-01T06:30:00Z');
        const ev = (time) => ({id: 1, camera: 'cam1', use_case: 'intrusion', label: 'person', confidence: 0.9, time});
        const s = summarize([
            ev('2026-11-01T06:10:00Z'), // 01:10 EST, current hour
            ev('2026-11-01T05:40:00Z'), // 01:40 EDT, previous hour
            ev('2026-11-01T04:59:00Z'), // 00:59 EDT
        ], now);
        console.log(JSON.stringify({offset: new Date(now).getTimezoneOffset(), timeline: s.timeline,
            hours: timelineHours(now)}));
    `;
    const r = spawnSync(process.execPath, ['--no-warnings', '--input-type=module', '-e', code],
        {env: {...process.env, TZ: 'America/New_York'}, encoding: 'utf8'});
    assert.equal(r.status, 0, r.stderr);
    const out = JSON.parse(r.stdout);
    if (out.offset !== 300) {
        t.skip(`TZ not applied in child (offset ${out.offset})`);
        return;
    }
    assert.equal(out.timeline.length, 12);
    assert.ok(out.timeline.every(Number.isFinite), JSON.stringify(out.timeline));
    assert.deepEqual(out.timeline, [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1]);
    assert.deepEqual(out.hours, [15, 16, 17, 18, 19, 20, 21, 22, 23, 0, 1, 1]); // 1 am twice
});

test('timelineHours: the local clock hour of each summarize() bucket', () => {
    const now = new Date(2026, 9, 19, 14, 45, 0);
    for (const n of [now, now.getTime()]) {
        assert.deepEqual(timelineHours(n), [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]);
    }
    // across midnight
    assert.deepEqual(timelineHours(new Date(2026, 9, 19, 4, 0, 0)), [17, 18, 19, 20, 21, 22, 23, 0, 1, 2, 3, 4]);

    // the bucket an event lands in starts at that bucket's label
    const s = summarize([ev({time: new Date(2026, 9, 19, 9, 20, 0).toISOString()})], now);
    assert.equal(timelineHours(now)[s.timeline.indexOf(1)], 9);
});

test('barFractions: the tallest bar is 1, an empty hour is 0', () => {
    assert.deepEqual(barFractions([0, 2, 4, 1]), [0, 0.5, 1, 0.25]);
    assert.deepEqual(barFractions([0, 0, 0]), [0, 0, 0]);
    assert.deepEqual(barFractions([3]), [1]);
    assert.deepEqual(barFractions([0, 0, 1]), [0, 0, 1]);
    assert.deepEqual(barFractions([]), []);
});

test('simulated', () => {
    const cams = ['cam1', 'cam2'];
    const a = simulated(0, cams);
    assert.equal(a.use_case, 'intrusion');
    assert.equal(a.camera, 'cam1');
    assert.ok(boxOf(a));
    assert.ok(a.label);
    assert.ok(a.data.area);
    assert.ok(a.data.detail);
    assert.ok(a.confidence > 0.5 && a.confidence <= 1);
    assert.equal(simulated(1, cams).camera, 'cam2');

    const cycle = [0, 1, 2, 3, 4, 5].map((i) => simulated(i, cams).use_case);
    assert.deepEqual(cycle,
        ['intrusion', 'restricted_area', 'fire', 'tampering', 'face_recognition', 'vehicle_lpr']);
    assert.equal(simulated(6, cams).use_case, 'intrusion');

    const face = simulated(4, cams);
    assert.equal(face.label, 'unknown_face');
    assert.equal(describe(face).popup, true);
    const car = simulated(5, cams);
    assert.equal(car.label, 'plate');
    assert.match(car.data.plate, /^[A-Z]{2}\d{2}[A-Z]{1,2}\d{4}$/);
    assert.equal(describe(car).popup, true);

    for (let i = 0; i < 12; i++) {
        const s = simulated(i, cams);
        assert.deepEqual(simulated(i, cams), s, 'deterministic');
        assert.equal(describe(s).popup, true);
        assert.ok(s.confidence >= 0 && s.confidence <= 1);
        for (const k of ['id', 'time', 'snapshot', 'acknowledged_at', 'siren']) {
            assert.ok(!(k in s), k);
        }
    }

    assert.equal(simulated(3, []).camera, 'cam1');
    assert.equal(simulated(3).camera, 'cam1');
    assert.equal(simulated(3, 'cam9').camera, 'cam1');
    assert.equal(simulated(3, null).camera, 'cam1');

    // a negative i still cycles: -1 is the last use case and the last camera
    const neg = simulated(-1, ['cam1', 'cam2', 'cam3']);
    assert.equal(neg.use_case, 'vehicle_lpr');
    assert.equal(neg.camera, 'cam3');
    assert.equal(simulated(-6, cams).use_case, 'intrusion');

    // each call returns fresh objects
    simulated(0, cams).data.box[0] = 0.99;
    assert.deepEqual(simulated(0, cams), a);
});
