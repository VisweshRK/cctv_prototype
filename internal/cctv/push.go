package cctv

import (
	"encoding/json"
	"fmt"
	"net/http"
	"time"
)

// pingEvery is how often apiStream sends a comment line, so proxies and the
// browser keep an idle connection open. A variable so tests can shorten it.
var pingEvery = 15 * time.Second

// apiStream handles GET /api/cctv/events/stream: a Server-Sent Events stream
// of new events ("event: alert") and changed ones ("event: update"), each
// with the whole event as JSON in its data line.
func (s *eventStore) apiStream(w http.ResponseWriter, r *http.Request) {
	if r.Method != "GET" {
		http.Error(w, "", http.StatusMethodNotAllowed)
		return
	}
	flusher, ok := w.(http.Flusher)
	if !ok {
		http.Error(w, "streaming not supported", http.StatusInternalServerError)
		return
	}

	// subscribe before the hello, so a client that saw the hello misses nothing
	ch, cancel := s.subscribe()
	defer cancel()

	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	if _, err := fmt.Fprint(w, ": hello\n\n"); err != nil {
		return
	}
	flusher.Flush()

	ping := time.NewTicker(pingEvery)
	defer ping.Stop()

	for {
		var err error
		select {
		case <-r.Context().Done():
			return
		case m, ok := <-ch:
			if !ok {
				return // dropped for falling behind; the browser reconnects
			}
			b, _ := json.Marshal(m.Event)
			_, err = fmt.Fprintf(w, "event: %s\ndata: %s\n\n", m.Type, b)
		case <-ping.C:
			_, err = fmt.Fprint(w, ": ping\n\n")
		}
		if err != nil {
			return
		}
		flusher.Flush()
	}
}
