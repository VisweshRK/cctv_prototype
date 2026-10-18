package cctv

import (
	"encoding/json"
	"net/http"
	"sync"
	"time"

	"github.com/AlexxIT/go2rtc/internal/streams"
)

// CameraStatus is what /api/cctv/status returns for each camera.
type CameraStatus struct {
	Name      string    `json:"name"`
	Online    bool      `json:"online"`
	BytesRecv int       `json:"bytes_recv"`
	LastSeen  time.Time `json:"last_seen,omitempty"` // last time new data arrived
	Since     time.Time `json:"since,omitempty"`     // when Online last changed
	Error     string    `json:"error,omitempty"`
}

// camera tracks one camera. A camera is online while the bytes received from
// it keep growing; if they stop growing for offlineAfter, it goes offline.
type camera struct {
	CameraStatus
	seen bool // Online has been decided at least once
}

// update applies one measurement and reports whether Online flipped.
// It has no go2rtc dependencies so it can be unit tested.
func (c *camera) update(bytes int, err error, now time.Time, offlineAfter time.Duration) (changed bool) {
	if err != nil {
		c.Error = err.Error()
	} else {
		c.Error = ""
	}

	if bytes > c.BytesRecv || (bytes > 0 && bytes < c.BytesRecv) { // growing, or counter reset by reconnect
		c.LastSeen = now
	}
	c.BytesRecv = bytes

	online := !c.LastSeen.IsZero() && now.Sub(c.LastSeen) < offlineAfter
	if c.seen && online == c.Online {
		return false
	}
	changed = c.seen // the first decision is not a change
	c.seen = true
	c.Online = online
	c.Since = now
	return changed
}

type watcher struct {
	mu           sync.Mutex
	cameras      []*camera
	interval     time.Duration
	offlineAfter time.Duration
	events       *eventStore
}

func newWatcher(names []string, interval, offlineAfter time.Duration, events *eventStore) *watcher {
	w := &watcher{interval: interval, offlineAfter: offlineAfter, events: events}
	for _, name := range names {
		w.cameras = append(w.cameras, &camera{CameraStatus: CameraStatus{Name: name}})
	}
	return w
}

func (w *watcher) run() {
	for {
		w.check()
		time.Sleep(w.interval)
	}
}

func (w *watcher) check() {
	now := time.Now()
	for _, c := range w.cameras {
		bytes, err := measure(c.Name)

		w.mu.Lock()
		changed := c.update(bytes, err, now, w.offlineAfter)
		online := c.Online
		w.mu.Unlock()

		if !changed {
			continue
		}
		// Tampering / disconnect use case: report every online <-> offline flip.
		label := "camera_online"
		if !online {
			label = "camera_offline"
			log.Warn().Str("camera", c.Name).Msg("[cctv] camera went offline")
		} else {
			log.Info().Str("camera", c.Name).Msg("[cctv] camera is back online")
		}
		_, _ = w.events.add(Event{Camera: c.Name, UseCase: "tampering", Label: label, Confidence: 1, Time: now})
	}
}

// measure keeps the camera stream connected (go2rtc preload) and returns the
// total bytes received from its sources.
func measure(name string) (int, error) {
	stream := streams.Get(name)
	if stream == nil {
		return 0, errNotConfigured
	}

	// Without a consumer go2rtc doesn't connect to the camera at all. A preload
	// keeps it connected; if the camera is unreachable, AddPreload fails and we
	// simply try again on the next check.
	if _, ok := streams.GetPreloads()[name]; !ok {
		if err := streams.AddPreload(name, ""); err != nil {
			return 0, err
		}
	}

	b, err := json.Marshal(stream)
	if err != nil {
		return 0, err
	}
	var info struct {
		Producers []struct {
			BytesRecv int `json:"bytes_recv"`
		} `json:"producers"`
	}
	if err = json.Unmarshal(b, &info); err != nil {
		return 0, err
	}
	total := 0
	for _, p := range info.Producers {
		total += p.BytesRecv
	}
	return total, nil
}

func (w *watcher) statuses() []CameraStatus {
	w.mu.Lock()
	defer w.mu.Unlock()
	out := make([]CameraStatus, 0, len(w.cameras))
	for _, c := range w.cameras {
		out = append(out, c.CameraStatus)
	}
	return out
}

// apiStatus handles GET /api/cctv/status
func (w *watcher) apiStatus(rw http.ResponseWriter, r *http.Request) {
	writeJSON(rw, http.StatusOK, w.statuses())
}
