package cctv

import (
	"bufio"
	"encoding/json"
	"errors"
	"net/http"
	"os"
	"path/filepath"
	"strconv"
	"sync"
	"time"
)

// Event is the one shared format every use case uses to report a detection:
// "camera X saw Y at time T". Go code calls AddEvent, Python plugins POST the
// same JSON to /api/cctv/events.
type Event struct {
	ID         int64          `json:"id"`
	Camera     string         `json:"camera"`
	UseCase    string         `json:"use_case"`   // e.g. intrusion, fall, fire
	Label      string         `json:"label"`      // e.g. person, camera_offline
	Confidence float64        `json:"confidence"` // 0..1
	Time       time.Time      `json:"time"`
	Snapshot   string         `json:"snapshot,omitempty"` // optional path or URL of an image
	Data       map[string]any `json:"data,omitempty"`     // optional use-case specific details

	AcknowledgedAt *time.Time `json:"acknowledged_at,omitempty"` // set once a guard acknowledges the alert
	Siren          string     `json:"siren,omitempty"`           // sirenRinging, sirenStopped or sirenUnavailable
}

const (
	sirenRinging     = "ringing"
	sirenStopped     = "stopped"
	sirenUnavailable = "unavailable"
)

// message is what subscribers receive: Type is "alert" for a new event and
// "update" for a changed one; Event is always the whole event.
type message struct {
	Type  string
	Event Event
}

func (e *Event) validate() error {
	switch {
	case e.Camera == "":
		return errors.New("camera is required")
	case e.UseCase == "":
		return errors.New("use_case is required")
	case e.Label == "":
		return errors.New("label is required")
	case e.Confidence < 0 || e.Confidence > 1:
		return errors.New("confidence must be between 0 and 1")
	}
	return nil
}

// eventStore keeps the latest events in memory and appends every event to a
// JSON Lines file, so events survive a restart. A real database comes later.
//
// An update is written as a new line holding the whole event; on load a line
// with an id that is already loaded replaces that event.
type eventStore struct {
	mu     sync.Mutex
	events []Event
	max    int
	nextID int64
	path   string
	subs   map[chan message]struct{}

	// hooks, set once in Init before serving; they run outside the lock
	onAdd func(e Event, image []byte)
	onAck func(e Event)
}

func newEventStore(path string, max int) *eventStore {
	s := &eventStore{max: max, nextID: 1, path: path, subs: map[chan message]struct{}{}}
	s.load()
	return s
}

func (s *eventStore) load() {
	if s.path == "" {
		return
	}
	f, err := os.Open(s.path)
	if err != nil {
		return // no file yet is OK
	}
	defer f.Close()

	scanner := bufio.NewScanner(f)
	scanner.Buffer(make([]byte, 0, 64*1024), 1024*1024)
	for scanner.Scan() {
		var e Event
		if json.Unmarshal(scanner.Bytes(), &e) != nil {
			continue
		}
		if i := s.index(e.ID); i >= 0 {
			s.events[i] = e // a later line is an update of that event
			continue
		}
		if e.ID < s.nextID {
			continue // update of an event already trimmed by max
		}
		s.push(e)
		s.nextID = e.ID + 1
	}

	// a siren can't still be ringing after a restart
	for i := range s.events {
		if s.events[i].Siren == sirenRinging {
			s.events[i].Siren = sirenStopped
		}
	}
}

// index returns the position of the event with this id in s.events, or -1.
func (s *eventStore) index(id int64) int {
	for i := len(s.events) - 1; i >= 0; i-- {
		if s.events[i].ID == id {
			return i
		}
	}
	return -1
}

func (s *eventStore) push(e Event) {
	s.events = append(s.events, e)
	if len(s.events) > s.max {
		s.events = s.events[len(s.events)-s.max:]
	}
}

func (s *eventStore) add(e Event) (Event, error) {
	return s.addWithImage(e, nil)
}

// addWithImage stores the event, tells subscribers about it and then hands
// the event and its (optional) JPEG image to the onAdd hook.
func (s *eventStore) addWithImage(e Event, image []byte) (Event, error) {
	if err := e.validate(); err != nil {
		return e, err
	}
	if e.Time.IsZero() {
		e.Time = time.Now()
	}

	s.mu.Lock()
	e.ID = s.nextID
	s.nextID++
	s.push(e)
	s.persist(e)
	s.mu.Unlock()

	s.publish(message{Type: "alert", Event: e})
	if s.onAdd != nil {
		s.onAdd(e, image)
	}
	return e, nil
}

// update applies change to a kept event, writes the whole updated event as a
// new line and tells subscribers. It returns false for an unknown id
// (including one already trimmed by max).
func (s *eventStore) update(id int64, change func(e *Event)) (Event, bool) {
	s.mu.Lock()
	i := s.index(id)
	if i < 0 {
		s.mu.Unlock()
		return Event{}, false
	}
	change(&s.events[i])
	e := s.events[i]
	s.persist(e)
	s.mu.Unlock()

	s.publish(message{Type: "update", Event: e})
	return e, true
}

func (s *eventStore) get(id int64) (Event, bool) {
	s.mu.Lock()
	defer s.mu.Unlock()

	if i := s.index(id); i >= 0 {
		return s.events[i], true
	}
	return Event{}, false
}

// oldestID returns the id of the oldest event still kept, or 0 when empty.
func (s *eventStore) oldestID() int64 {
	s.mu.Lock()
	defer s.mu.Unlock()

	if len(s.events) == 0 {
		return 0
	}
	return s.events[0].ID
}

// persist appends the event to the events file. Call with s.mu held.
func (s *eventStore) persist(e Event) {
	if s.path == "" {
		return
	}
	if err := appendJSONLine(s.path, e); err != nil {
		log.Warn().Err(err).Msg("[cctv] can't write events file")
	}
}

// subscribe returns a channel of new and changed events and a cancel func.
// A subscriber that falls 64 messages behind is dropped: its channel is closed.
func (s *eventStore) subscribe() (<-chan message, func()) {
	ch := make(chan message, 64)

	s.mu.Lock()
	s.subs[ch] = struct{}{}
	s.mu.Unlock()

	cancel := func() {
		s.mu.Lock()
		s.unsubscribe(ch)
		s.mu.Unlock()
	}
	return ch, cancel
}

// unsubscribe removes and closes ch once; safe to call again. Call with s.mu held.
func (s *eventStore) unsubscribe(ch chan message) {
	if _, ok := s.subs[ch]; ok {
		delete(s.subs, ch)
		close(ch)
	}
}

// publish never blocks. It takes the lock on its own, after the change is
// done, so a channel is never closed (by cancel or a drop) while being sent to.
func (s *eventStore) publish(m message) {
	s.mu.Lock()
	defer s.mu.Unlock()

	for ch := range s.subs {
		select {
		case ch <- m:
		default:
			s.unsubscribe(ch)
		}
	}
}

// list returns the newest events first, filtered by camera and use case.
func (s *eventStore) list(camera, useCase string, limit int) []Event {
	s.mu.Lock()
	defer s.mu.Unlock()

	out := []Event{}
	for i := len(s.events) - 1; i >= 0 && len(out) < limit; i-- {
		e := s.events[i]
		if camera != "" && e.Camera != camera {
			continue
		}
		if useCase != "" && e.UseCase != useCase {
			continue
		}
		out = append(out, e)
	}
	return out
}

func appendJSONLine(path string, v any) error {
	if err := os.MkdirAll(filepath.Dir(path), 0755); err != nil {
		return err
	}
	f, err := os.OpenFile(path, os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0644)
	if err != nil {
		return err
	}
	defer f.Close()

	b, err := json.Marshal(v)
	if err != nil {
		return err
	}
	_, err = f.Write(append(b, '\n'))
	return err
}

// apiEvents handles:
//
//	GET  /api/cctv/events?camera=cam1&use_case=intrusion&limit=50
//	POST /api/cctv/events  {"camera":"cam1","use_case":"intrusion","label":"person","confidence":0.9}
func (s *eventStore) apiEvents(w http.ResponseWriter, r *http.Request) {
	switch r.Method {
	case "GET":
		query := r.URL.Query()
		limit, _ := strconv.Atoi(query.Get("limit"))
		if limit <= 0 || limit > s.max {
			limit = 100
		}
		writeJSON(w, http.StatusOK, s.list(query.Get("camera"), query.Get("use_case"), limit))

	case "POST":
		var e Event
		if err := json.NewDecoder(http.MaxBytesReader(w, r.Body, 1024*1024)).Decode(&e); err != nil {
			writeJSON(w, http.StatusBadRequest, map[string]string{"error": "invalid JSON: " + err.Error()})
			return
		}
		e, err := s.add(e)
		if err != nil {
			writeJSON(w, http.StatusBadRequest, map[string]string{"error": err.Error()})
			return
		}
		writeJSON(w, http.StatusCreated, e)

	default:
		http.Error(w, "", http.StatusMethodNotAllowed)
	}
}

func writeJSON(w http.ResponseWriter, code int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(code)
	_ = json.NewEncoder(w).Encode(v)
}
