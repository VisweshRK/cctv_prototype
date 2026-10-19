package cctv

import (
	"bufio"
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
)

func TestParseDotEnvLine(t *testing.T) {
	tests := []struct {
		line, key, value string
		ok               bool
	}{
		{"CAM1_URL=rtsp://u:p@1.2.3.4/s", "CAM1_URL", "rtsp://u:p@1.2.3.4/s", true},
		{`  CAM2_URL = "rtsp://a b"  `, "CAM2_URL", "rtsp://a b", true},
		{"export X='y=z'", "X", "y=z", true},
		{"# comment", "", "", false},
		{"", "", "", false},
		{"NOEQUALS", "", "", false},
		{"=value", "", "", false},
	}
	for _, tt := range tests {
		key, value, ok := parseDotEnvLine(tt.line)
		require.Equal(t, tt.ok, ok, tt.line)
		require.Equal(t, tt.key, key, tt.line)
		require.Equal(t, tt.value, value, tt.line)
	}
}

func TestLoadDotEnvKeepsExisting(t *testing.T) {
	path := filepath.Join(t.TempDir(), ".env")
	require.NoError(t, os.WriteFile(path, []byte("CCTV_TEST_A=file\nCCTV_TEST_B=file\n"), 0644))
	t.Setenv("CCTV_TEST_A", "shell")
	t.Setenv("CCTV_TEST_B", "") // registers cleanup
	require.NoError(t, os.Unsetenv("CCTV_TEST_B"))

	LoadDotEnv(path)

	require.Equal(t, "shell", os.Getenv("CCTV_TEST_A"))
	require.Equal(t, "file", os.Getenv("CCTV_TEST_B"))
}

func TestEventsAPI(t *testing.T) {
	path := filepath.Join(t.TempDir(), "events.jsonl")
	s := newEventStore(path, 10)

	post := func(body string) *httptest.ResponseRecorder {
		r := httptest.NewRequest("POST", "/api/cctv/events", strings.NewReader(body))
		w := httptest.NewRecorder()
		s.apiEvents(w, r)
		return w
	}

	w := post(`{"camera":"cam1","use_case":"intrusion","label":"person","confidence":0.9}`)
	require.Equal(t, http.StatusCreated, w.Code, w.Body.String())
	w = post(`{"camera":"cam2","use_case":"fire","label":"smoke","confidence":0.5}`)
	require.Equal(t, http.StatusCreated, w.Code)

	require.Equal(t, http.StatusBadRequest, post(`{"camera":"cam1"}`).Code)
	require.Equal(t, http.StatusBadRequest, post(`{"camera":"c","use_case":"u","label":"l","confidence":2}`).Code)
	require.Equal(t, http.StatusBadRequest, post(`not json`).Code)

	// newest first, filters work
	r := httptest.NewRequest("GET", "/api/cctv/events?camera=cam1", nil)
	rec := httptest.NewRecorder()
	s.apiEvents(rec, r)
	var got []Event
	require.NoError(t, json.Unmarshal(rec.Body.Bytes(), &got))
	require.Len(t, got, 1)
	require.Equal(t, "intrusion", got[0].UseCase)
	require.Equal(t, int64(1), got[0].ID)
	require.False(t, got[0].Time.IsZero())

	// events survive a restart and IDs keep counting
	s2 := newEventStore(path, 10)
	require.Len(t, s2.list("", "", 100), 2)
	e, err := s2.add(Event{Camera: "cam1", UseCase: "fall", Label: "fall", Confidence: 1})
	require.NoError(t, err)
	require.Equal(t, int64(3), e.ID)
}

func TestEventStoreKeepsMax(t *testing.T) {
	s := newEventStore("", 3)
	for i := 0; i < 5; i++ {
		_, err := s.add(Event{Camera: "c", UseCase: "u", Label: "l"})
		require.NoError(t, err)
	}
	got := s.list("", "", 100)
	require.Len(t, got, 3)
	require.Equal(t, int64(5), got[0].ID)
}

func TestUpdatePublishesAndPersists(t *testing.T) {
	path := filepath.Join(t.TempDir(), "events.jsonl")
	s := newEventStore(path, 10)
	ch, cancel := s.subscribe()
	defer cancel()
	e, _ := s.add(Event{Camera: "cam1", UseCase: "intrusion", Label: "person", Confidence: 0.9})
	require.Equal(t, "alert", (<-ch).Type)
	got, ok := s.update(e.ID, func(e *Event) { e.Snapshot = "api/cctv/snapshot?id=1" })
	require.True(t, ok)
	m := <-ch
	require.Equal(t, "update", m.Type)
	require.Equal(t, got, m.Event)
	_, ok = s.update(999, func(*Event) {})
	require.False(t, ok)
	s2 := newEventStore(path, 10)
	require.Len(t, s2.list("", "", 10), 1) // replaced, not duplicated
	require.Equal(t, "api/cctv/snapshot?id=1", s2.list("", "", 10)[0].Snapshot)
}

func TestLoadResetsRingingAndAppliesUpdates(t *testing.T) {
	path := filepath.Join(t.TempDir(), "events.jsonl")
	s := newEventStore(path, 10)
	e, _ := s.add(Event{Camera: "cam1", UseCase: "fire", Label: "smoke", Confidence: 0.8})
	s.update(e.ID, func(e *Event) { e.Siren = sirenRinging })
	got, _ := newEventStore(path, 10).get(e.ID)
	require.Equal(t, sirenStopped, got.Siren)
}

func TestSlowSubscriberIsDropped(t *testing.T) {
	s := newEventStore("", 1000)
	ch, cancel := s.subscribe()
	defer cancel()
	start := time.Now()
	for i := 0; i < 200; i++ {
		_, _ = s.add(Event{Camera: "c", UseCase: "u", Label: "l"})
	}
	require.Less(t, time.Since(start), time.Second)
	n := 0
	for range ch { // must be closed, or this loop never ends
		n++
	}
	require.LessOrEqual(t, n, 64)
}

func TestAddCallsOnAddHook(t *testing.T) {
	s := newEventStore("", 10)
	var gotImage []byte
	s.onAdd = func(e Event, image []byte) { gotImage = image }
	_, err := s.addWithImage(Event{Camera: "c", UseCase: "u", Label: "l"}, []byte{0xFF, 0xD8})
	require.NoError(t, err)
	require.Equal(t, []byte{0xFF, 0xD8}, gotImage)
}

// Concurrent updates must reach subscribers in the same order as the file
// lines, otherwise a stale whole-event message can win on the dashboard.
func TestSubscribersSeeFileOrder(t *testing.T) {
	path := filepath.Join(t.TempDir(), "events.jsonl")
	s := newEventStore(path, 10)
	ch, cancel := s.subscribe()
	e, _ := s.add(Event{Camera: "c", UseCase: "u", Label: "l"})

	var wg sync.WaitGroup
	for i := 0; i < 50; i++ { // 1 alert + 50 updates fits the 64 buffer
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			s.update(e.ID, func(e *Event) { e.Label = strconv.Itoa(i) })
		}(i)
	}
	wg.Wait()
	cancel()

	var got []string
	for m := range ch {
		got = append(got, m.Type+":"+m.Event.Label)
	}

	b, err := os.ReadFile(path)
	require.NoError(t, err)
	var want []string
	for i, line := range strings.Split(strings.TrimSpace(string(b)), "\n") {
		var fe Event
		require.NoError(t, json.Unmarshal([]byte(line), &fe))
		typ := "update"
		if i == 0 {
			typ = "alert"
		}
		want = append(want, typ+":"+fe.Label)
	}
	require.Equal(t, want, got)
}

func TestCameraUpdate(t *testing.T) {
	const offlineAfter = 20 * time.Second
	t0 := time.Now()
	c := &camera{}

	// first decision is never reported as a change
	require.False(t, c.update(0, errors.New("dial failed"), t0, offlineAfter))
	require.False(t, c.Online)
	require.Equal(t, "dial failed", c.Error)

	// data starts flowing -> online
	require.True(t, c.update(1000, nil, t0.Add(5*time.Second), offlineAfter))
	require.True(t, c.Online)
	require.Empty(t, c.Error)

	// still growing -> no change
	require.False(t, c.update(2000, nil, t0.Add(10*time.Second), offlineAfter))

	// bytes stop growing, but not long enough yet
	require.False(t, c.update(2000, nil, t0.Add(25*time.Second), offlineAfter))
	require.True(t, c.Online)

	// stalled for offlineAfter -> offline
	require.True(t, c.update(2000, nil, t0.Add(31*time.Second), offlineAfter))
	require.False(t, c.Online)

	// reconnect resets the counter to a smaller number -> online again
	require.True(t, c.update(300, nil, t0.Add(35*time.Second), offlineAfter))
	require.True(t, c.Online)
}

func TestWatcherEmitsTamperingEvents(t *testing.T) {
	events := newEventStore("", 10)
	// no go2rtc streams are configured in tests, so the camera stays offline
	w := newWatcher([]string{"cam1"}, time.Second, time.Second, events)
	w.check()

	st := w.statuses()
	require.Len(t, st, 1)
	require.False(t, st[0].Online)
	require.Equal(t, errNotConfigured.Error(), st[0].Error)
	require.Empty(t, events.list("", "", 10)) // startup state is not an event
}

func TestStreamSendsAlertsAndPings(t *testing.T) {
	defer func(d time.Duration) { pingEvery = d }(pingEvery)
	pingEvery = 50 * time.Millisecond

	s := newEventStore("", 10)
	srv := httptest.NewServer(http.HandlerFunc(s.apiStream))
	defer srv.Close()

	res, err := http.Post(srv.URL, "text/plain", nil)
	require.NoError(t, err)
	res.Body.Close()
	require.Equal(t, http.StatusMethodNotAllowed, res.StatusCode)

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	req, err := http.NewRequestWithContext(ctx, "GET", srv.URL, nil)
	require.NoError(t, err)
	res, err = http.DefaultClient.Do(req)
	require.NoError(t, err)
	defer res.Body.Close()
	require.Equal(t, "text/event-stream", res.Header.Get("Content-Type"))
	require.Equal(t, "no-cache", res.Header.Get("Cache-Control"))

	lines := make(chan string, 100)
	go func() {
		r := bufio.NewReader(res.Body)
		for {
			line, err := r.ReadString('\n')
			if err != nil {
				close(lines)
				return
			}
			lines <- strings.TrimRight(line, "\n")
		}
	}()
	// next returns the next line that isn't blank or a ping
	next := func() string {
		for {
			select {
			case line, ok := <-lines:
				require.True(t, ok, "stream ended")
				if line != "" && line != ": ping" {
					return line
				}
			case <-time.After(time.Second):
				require.FailNow(t, "no line within 1s")
			}
		}
	}

	require.Equal(t, ": hello", next())

	_, err = s.add(Event{Camera: "cam1", UseCase: "intrusion", Label: "person", Confidence: 0.9})
	require.NoError(t, err)
	require.Equal(t, "event: alert", next())
	data := next()
	require.True(t, strings.HasPrefix(data, `data: {"id":1,`), data)
	var e Event
	require.NoError(t, json.Unmarshal([]byte(strings.TrimPrefix(data, "data: ")), &e))
	require.Equal(t, "intrusion", e.UseCase)

	s.update(1, func(e *Event) { e.Siren = sirenUnavailable })
	require.Equal(t, "event: update", next())
	data = next()
	require.True(t, strings.HasPrefix(data, "data: "), data)
	require.Contains(t, data, `"siren":"unavailable"`)

	deadline := time.After(500 * time.Millisecond)
	for gotPing := false; !gotPing; {
		select {
		case line, ok := <-lines:
			require.True(t, ok, "stream ended")
			gotPing = line == ": ping"
		case <-deadline:
			require.FailNow(t, "no ping within 500ms")
		}
	}

	cancel()
	require.Eventually(t, func() bool {
		s.mu.Lock()
		defer s.mu.Unlock()
		return len(s.subs) == 0
	}, time.Second, 10*time.Millisecond)
}

func TestAckEndpoint(t *testing.T) {
	path := filepath.Join(t.TempDir(), "events.jsonl")
	s := newEventStore(path, 10)
	e1, _ := s.add(Event{Camera: "cam1", UseCase: "intrusion", Label: "person", Confidence: 0.9})
	_, _ = s.add(Event{Camera: "cam1", UseCase: "intrusion", Label: "person", Confidence: 0.8})
	s.update(e1.ID, func(e *Event) { e.Siren = sirenRinging })
	var acked []int64
	s.onAck = func(e Event) { acked = append(acked, e.ID) }

	ack := func(method, body string) *httptest.ResponseRecorder {
		r := httptest.NewRequest(method, "/api/cctv/events/ack", strings.NewReader(body))
		w := httptest.NewRecorder()
		s.apiAck(w, r)
		return w
	}
	decode := func(w *httptest.ResponseRecorder) []Event {
		var got []Event
		require.NoError(t, json.Unmarshal(w.Body.Bytes(), &got), w.Body.String())
		return got
	}
	fileLines := func() int {
		b, err := os.ReadFile(path)
		require.NoError(t, err)
		return strings.Count(string(b), "\n")
	}

	w := ack("POST", `{"ids":[1,2]}`)
	require.Equal(t, http.StatusOK, w.Code, w.Body.String())
	got := decode(w)
	require.Len(t, got, 2)
	require.Equal(t, int64(1), got[0].ID)
	require.Equal(t, int64(2), got[1].ID)
	require.NotNil(t, got[0].AcknowledgedAt)
	require.NotNil(t, got[1].AcknowledgedAt)
	require.Equal(t, sirenStopped, got[0].Siren) // was ringing
	require.Equal(t, "", got[1].Siren)           // wasn't ringing, left alone
	require.Equal(t, []int64{1, 2}, acked)
	first := *got[0].AcknowledgedAt

	// already acknowledged: first time kept, no hook, no redundant file line
	lines := fileLines()
	w = ack("POST", `{"id":1}`)
	require.Equal(t, http.StatusOK, w.Code, w.Body.String())
	got = decode(w)
	require.Len(t, got, 1)
	require.True(t, first.Equal(*got[0].AcknowledgedAt))
	require.Equal(t, []int64{1, 2}, acked)
	require.Equal(t, lines, fileLines())

	// some ids unknown: the known ones are returned
	w = ack("POST", `{"ids":[2,99]}`)
	require.Equal(t, http.StatusOK, w.Code, w.Body.String())
	got = decode(w)
	require.Len(t, got, 1)
	require.Equal(t, int64(2), got[0].ID)

	require.Equal(t, http.StatusNotFound, ack("POST", `{"id":99}`).Code)
	require.Equal(t, http.StatusBadRequest, ack("POST", `{}`).Code)
	require.Equal(t, http.StatusBadRequest, ack("POST", `{"ids":[]}`).Code)
	require.Equal(t, http.StatusBadRequest, ack("POST", `not json`).Code)
	require.Equal(t, http.StatusMethodNotAllowed, ack("GET", "").Code)
}

// postEvent POSTs body to s.apiEvents and returns the recorder.
func postEvent(s *eventStore, body string) *httptest.ResponseRecorder {
	r := httptest.NewRequest("POST", "/api/cctv/events", strings.NewReader(body))
	w := httptest.NewRecorder()
	s.apiEvents(w, r)
	return w
}

func getSnapshot(p *snapshots, query string) *httptest.ResponseRecorder {
	r := httptest.NewRequest("GET", "/api/cctv/snapshot?"+query, nil)
	w := httptest.NewRecorder()
	p.apiSnapshot(w, r)
	return w
}

func TestSnapshotFromPostedImage(t *testing.T) {
	dir := filepath.Join(t.TempDir(), "snapshots")
	s := newEventStore("", 10)
	p := newSnapshots(dir, func(camera string) ([]byte, error) {
		t.Error("grab must not be called when an image is posted")
		return nil, errors.New("unexpected grab")
	}, s)
	s.onAdd = p.handle

	jpeg := []byte{0xFF, 0xD8, 0xFF, 0xD9}
	body := `{"camera":"cam1","use_case":"intrusion","label":"person","confidence":0.9,"image":"` +
		base64.StdEncoding.EncodeToString(jpeg) + `"}`
	w := postEvent(s, body)
	require.Equal(t, http.StatusCreated, w.Code, w.Body.String())
	require.NotContains(t, w.Body.String(), "image")

	require.Eventually(t, func() bool {
		e, ok := s.get(1)
		return ok && e.Snapshot == "api/cctv/snapshot?id=1"
	}, 2*time.Second, 10*time.Millisecond)

	w = getSnapshot(p, "id=1")
	require.Equal(t, http.StatusOK, w.Code)
	require.Equal(t, "image/jpeg", w.Header().Get("Content-Type"))
	require.Equal(t, "max-age=86400", w.Header().Get("Cache-Control"))
	require.Equal(t, jpeg, w.Body.Bytes())
}

func TestSnapshotGrabbedWhenNoImage(t *testing.T) {
	dir := t.TempDir()
	s := newEventStore("", 10)
	frame := []byte{0xFF, 0xD8, 0x01, 0xFF, 0xD9}
	var mu sync.Mutex
	var grabbed []string
	p := newSnapshots(dir, func(camera string) ([]byte, error) {
		mu.Lock()
		grabbed = append(grabbed, camera)
		mu.Unlock()
		return frame, nil
	}, s)
	s.onAdd = p.handle

	w := postEvent(s, `{"camera":"cam2","use_case":"intrusion","label":"person","confidence":0.9}`)
	require.Equal(t, http.StatusCreated, w.Code, w.Body.String())

	require.Eventually(t, func() bool {
		e, ok := s.get(1)
		return ok && e.Snapshot == "api/cctv/snapshot?id=1"
	}, 2*time.Second, 10*time.Millisecond)

	mu.Lock()
	require.Equal(t, []string{"cam2"}, grabbed)
	mu.Unlock()
	b, err := os.ReadFile(filepath.Join(dir, "1.jpg"))
	require.NoError(t, err)
	require.Equal(t, frame, b)
}

func TestSnapshotGrabFailureLeavesEventWithoutSnapshot(t *testing.T) {
	dir := t.TempDir()
	s := newEventStore("", 10)
	var calls atomic.Int32
	p := newSnapshots(dir, func(camera string) ([]byte, error) {
		calls.Add(1)
		return nil, errors.New("no frame")
	}, s)
	s.onAdd = p.handle

	w := postEvent(s, `{"camera":"cam1","use_case":"intrusion","label":"person","confidence":0.9}`)
	require.Equal(t, http.StatusCreated, w.Code, w.Body.String())

	time.Sleep(200 * time.Millisecond)
	e, ok := s.get(1)
	require.True(t, ok)
	require.Empty(t, e.Snapshot)
	require.Equal(t, int32(1), calls.Load())
	_, err := os.Stat(filepath.Join(dir, "1.jpg"))
	require.True(t, os.IsNotExist(err), err)
}

func TestSnapshotSkippedForCameraStatusEvents(t *testing.T) {
	s := newEventStore("", 10)
	var calls atomic.Int32
	p := newSnapshots(t.TempDir(), func(camera string) ([]byte, error) {
		calls.Add(1)
		return []byte{0xFF, 0xD8, 0xFF, 0xD9}, nil
	}, s)
	s.onAdd = p.handle

	_, err := s.add(Event{Camera: "cam1", UseCase: "tampering", Label: "camera_offline", Confidence: 1})
	require.NoError(t, err)

	time.Sleep(200 * time.Millisecond)
	require.Equal(t, int32(0), calls.Load())
	e, _ := s.get(1)
	require.Empty(t, e.Snapshot)
}

func TestSnapshotEndpointRejectsBadIDs(t *testing.T) {
	p := newSnapshots(t.TempDir(), nil, newEventStore("", 10))
	for _, q := range []string{"id=../events", "id=", "id=1a"} {
		require.Equal(t, http.StatusBadRequest, getSnapshot(p, q).Code, q)
	}
	require.Equal(t, http.StatusNotFound, getSnapshot(p, "id=42").Code)
}

func TestPostImageRejectsBadData(t *testing.T) {
	s := newEventStore("", 10)
	for _, image := range []string{"%%%", base64.StdEncoding.EncodeToString([]byte("hello"))} {
		body := `{"camera":"cam1","use_case":"intrusion","label":"person","confidence":0.9,"image":"` + image + `"}`
		w := postEvent(s, body)
		require.Equal(t, http.StatusBadRequest, w.Code, image)
		require.Contains(t, w.Body.String(), "image must be a base64 JPEG", image)
	}
	require.Empty(t, s.list("", "", 10))
}

func TestPruneDeletesOldSnapshots(t *testing.T) {
	dir := t.TempDir()
	for _, name := range []string{"1.jpg", "2.jpg", "3.jpg", "notes.txt"} {
		require.NoError(t, os.WriteFile(filepath.Join(dir, name), []byte("x"), 0644))
	}
	p := newSnapshots(dir, nil, newEventStore("", 10))
	p.prune(3)

	entries, err := os.ReadDir(dir)
	require.NoError(t, err)
	var left []string
	for _, e := range entries {
		left = append(left, e.Name())
	}
	require.Equal(t, []string{"3.jpg", "notes.txt"}, left)
}

// fakePlayer records every (camera, source) call; fail makes it error, and
// gate (if set) runs inside the call, so a test can hold a play in flight.
type fakePlayer struct {
	mu    sync.Mutex
	calls [][2]string
	fail  bool
	gate  func(camera, source string)
}

func (p *fakePlayer) play(camera, source string) error {
	p.mu.Lock()
	p.calls = append(p.calls, [2]string{camera, source})
	fail, gate := p.fail, p.gate
	p.mu.Unlock()
	if gate != nil {
		gate(camera, source)
	}
	if fail {
		return errors.New("no speaker")
	}
	return nil
}

func (p *fakePlayer) got() [][2]string {
	p.mu.Lock()
	defer p.mu.Unlock()
	return append([][2]string(nil), p.calls...)
}

const testSirenSrc = "ffmpeg:siren.wav#audio=pcmu#input=file"

// newTestSiren wires a siren to a fresh store the way Init will: through onAdd.
func newTestSiren(p *fakePlayer) (*eventStore, *siren) {
	s := newEventStore("", 100)
	sir := newSiren(p.play, testSirenSrc, []string{"intrusion", "restricted_area", "fire"}, 100*time.Millisecond, s)
	s.onAdd = func(e Event, _ []byte) { sir.onEvent(e) }
	return s, sir
}

func addAlert(t *testing.T, s *eventStore, camera, useCase string) Event {
	t.Helper()
	e, err := s.add(Event{Camera: camera, UseCase: useCase, Label: "person", Confidence: 0.9})
	require.NoError(t, err)
	return e
}

func sirenOf(t *testing.T, s *eventStore, id int64) string {
	t.Helper()
	e, ok := s.get(id)
	require.True(t, ok)
	return e.Siren
}

func ackAlert(t *testing.T, s *eventStore, sir *siren, id int64) {
	t.Helper()
	e, ok := s.get(id)
	require.True(t, ok)
	sir.onAck(e)
}

func TestSirenRingsUntilAck(t *testing.T) {
	p := &fakePlayer{}
	s, sir := newTestSiren(p)

	e1 := addAlert(t, s, "cam1", "intrusion")
	require.Equal(t, [][2]string{{"cam1", testSirenSrc}}, p.got())
	require.Equal(t, sirenRinging, sirenOf(t, s, e1.ID))

	e2 := addAlert(t, s, "cam1", "intrusion")
	require.Len(t, p.got(), 1) // already ringing: no new call
	require.Equal(t, sirenRinging, sirenOf(t, s, e2.ID))

	ackAlert(t, s, sir, e1.ID)
	require.Len(t, p.got(), 1) // e2 still unacknowledged: keeps ringing

	ackAlert(t, s, sir, e2.ID)
	require.Equal(t, [][2]string{{"cam1", testSirenSrc}, {"cam1", ""}}, p.got())
	require.Equal(t, sirenStopped, sirenOf(t, s, e1.ID))
	require.Equal(t, sirenStopped, sirenOf(t, s, e2.ID))
}

func TestSirenOnlyForSirenUseCases(t *testing.T) {
	p := &fakePlayer{}
	s, sir := newTestSiren(p)

	e := addAlert(t, s, "cam1", "fall")
	require.Empty(t, p.got())
	require.Empty(t, sirenOf(t, s, e.ID))

	// an ack of a non-siren event must not disturb a ringing camera
	e2 := addAlert(t, s, "cam1", "intrusion")
	ackAlert(t, s, sir, e.ID)
	require.Len(t, p.got(), 1)
	require.Equal(t, sirenRinging, sirenOf(t, s, e2.ID))
}

func TestSirenOnlyOnCameraThatSawIt(t *testing.T) {
	p := &fakePlayer{}
	s, _ := newTestSiren(p)

	addAlert(t, s, "cam2", "fire")
	require.Equal(t, [][2]string{{"cam2", testSirenSrc}}, p.got())
}

func TestSirenCutOffAndStaysQuiet(t *testing.T) {
	p := &fakePlayer{}
	s, sir := newTestSiren(p)

	e1 := addAlert(t, s, "cam1", "intrusion")
	time.Sleep(200 * time.Millisecond) // past max
	calls := p.got()
	require.Equal(t, [2]string{"cam1", ""}, calls[len(calls)-1])
	require.Equal(t, sirenStopped, sirenOf(t, s, e1.ID))

	e2 := addAlert(t, s, "cam1", "intrusion") // e1 still unacknowledged: stay quiet
	require.Len(t, p.got(), len(calls))
	require.Equal(t, sirenStopped, sirenOf(t, s, e2.ID))

	ackAlert(t, s, sir, e1.ID)
	ackAlert(t, s, sir, e2.ID)
	require.Len(t, p.got(), len(calls)) // nothing was playing: no stop call

	addAlert(t, s, "cam1", "intrusion")
	calls = p.got()
	require.Equal(t, [2]string{"cam1", testSirenSrc}, calls[len(calls)-1])
}

func TestSirenUnavailable(t *testing.T) {
	p := &fakePlayer{fail: true}
	s, _ := newTestSiren(p)

	e1 := addAlert(t, s, "cam1", "intrusion")
	require.Equal(t, sirenUnavailable, sirenOf(t, s, e1.ID))
	e2 := addAlert(t, s, "cam1", "restricted_area")
	require.Equal(t, sirenUnavailable, sirenOf(t, s, e2.ID))
}

// holdFirstPlay makes the first play(camera, src) block until release is
// closed; entered is closed once it is in flight.
func holdFirstPlay(p *fakePlayer) (entered, release chan struct{}) {
	entered, release = make(chan struct{}), make(chan struct{})
	var held atomic.Bool
	p.gate = func(_, source string) {
		if source != "" && held.CompareAndSwap(false, true) {
			close(entered)
			<-release
		}
	}
	return
}

// An event that arrives while the start is in flight joins it: one play call,
// and it gets the start's outcome.
func TestSirenEventDuringStartGetsItsOutcome(t *testing.T) {
	for _, fail := range []bool{false, true} {
		p := &fakePlayer{fail: fail}
		entered, release := holdFirstPlay(p)
		s, _ := newTestSiren(p)

		done := make(chan Event)
		go func() {
			e, _ := s.add(Event{Camera: "cam1", UseCase: "intrusion", Label: "person", Confidence: 0.9})
			done <- e
		}()
		<-entered
		e2 := addAlert(t, s, "cam1", "intrusion") // returns at once: a start is in flight
		close(release)
		e1 := <-done

		want := sirenRinging
		if fail {
			want = sirenUnavailable
		}
		require.Len(t, p.got(), 1, "fail=%v", fail)
		require.Equal(t, want, sirenOf(t, s, e1.ID), "fail=%v", fail)
		require.Equal(t, want, sirenOf(t, s, e2.ID), "fail=%v", fail)
	}
}

// Everything acknowledged while play(cam, src) is still dialling: once it
// returns, the siren is stopped at once instead of ringing until max.
func TestSirenAckWhilePlayInFlight(t *testing.T) {
	p := &fakePlayer{}
	entered, release := holdFirstPlay(p)
	s, sir := newTestSiren(p)

	done := make(chan Event)
	go func() {
		e, _ := s.add(Event{Camera: "cam1", UseCase: "intrusion", Label: "person", Confidence: 0.9})
		done <- e
	}()
	<-entered
	e2 := addAlert(t, s, "cam1", "fire")
	ackAlert(t, s, sir, 1)
	ackAlert(t, s, sir, e2.ID) // both return at once: the start in flight will stop it
	close(release)
	e1 := <-done

	require.Equal(t, [][2]string{{"cam1", testSirenSrc}, {"cam1", ""}}, p.got())
	require.NotEqual(t, sirenRinging, sirenOf(t, s, e1.ID))
	require.NotEqual(t, sirenRinging, sirenOf(t, s, e2.ID))

	time.Sleep(150 * time.Millisecond) // no max timer may be left behind
	require.Len(t, p.got(), 2)
}

func TestWriteSirenWAV(t *testing.T) {
	path := filepath.Join(t.TempDir(), "sub", "siren.wav")
	require.NoError(t, writeSirenWAV(path))

	b, err := os.ReadFile(path)
	require.NoError(t, err)
	require.Equal(t, "RIFF", string(b[:4]))
	require.Equal(t, "WAVE", string(b[8:12]))
	require.Len(t, b, 44+2*8000*2)

	info, err := os.Stat(path)
	require.NoError(t, err)
	old := info.ModTime().Add(-time.Hour)
	require.NoError(t, os.Chtimes(path, old, old))
	require.NoError(t, writeSirenWAV(path))
	info, err = os.Stat(path)
	require.NoError(t, err)
	require.True(t, info.ModTime().Equal(old), "rewritten: %v", info.ModTime())
}
