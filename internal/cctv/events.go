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
type eventStore struct {
	mu     sync.Mutex
	events []Event
	max    int
	nextID int64
	path   string
}

func newEventStore(path string, max int) *eventStore {
	s := &eventStore{max: max, nextID: 1, path: path}
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
		s.push(e)
		if e.ID >= s.nextID {
			s.nextID = e.ID + 1
		}
	}
}

func (s *eventStore) push(e Event) {
	s.events = append(s.events, e)
	if len(s.events) > s.max {
		s.events = s.events[len(s.events)-s.max:]
	}
}

func (s *eventStore) add(e Event) (Event, error) {
	if err := e.validate(); err != nil {
		return e, err
	}
	if e.Time.IsZero() {
		e.Time = time.Now()
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	e.ID = s.nextID
	s.nextID++
	s.push(e)

	if s.path != "" {
		if err := appendJSONLine(s.path, e); err != nil {
			log.Warn().Err(err).Msg("[cctv] can't write events file")
		}
	}
	return e, nil
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
