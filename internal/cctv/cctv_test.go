package cctv

import (
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
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
