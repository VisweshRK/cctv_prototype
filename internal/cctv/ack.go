package cctv

import (
	"encoding/json"
	"net/http"
	"time"
)

// apiAck handles POST /api/cctv/events/ack {"id":17} or {"ids":[17,18]}: a
// guard acknowledges one event or a whole incident. Each event that wasn't
// acknowledged yet gets AcknowledgedAt, a ringing siren is marked stopped and
// onAck is called once for it. It answers with the events after the change,
// in request order; unknown ids are skipped (404 only when none exist).
func (s *eventStore) apiAck(w http.ResponseWriter, r *http.Request) {
	if r.Method != "POST" {
		http.Error(w, "", http.StatusMethodNotAllowed)
		return
	}

	var req struct {
		ID  int64   `json:"id"`
		IDs []int64 `json:"ids"`
	}
	if err := json.NewDecoder(http.MaxBytesReader(w, r.Body, 64*1024)).Decode(&req); err != nil {
		writeJSON(w, http.StatusBadRequest, map[string]string{"error": "invalid JSON: " + err.Error()})
		return
	}
	ids := req.IDs
	if req.ID != 0 {
		ids = append([]int64{req.ID}, ids...)
	}
	if len(ids) == 0 {
		writeJSON(w, http.StatusBadRequest, map[string]string{"error": "id or ids is required"})
		return
	}

	now := time.Now()
	out := []Event{}
	seen := map[int64]bool{}
	for _, id := range ids {
		if seen[id] {
			continue
		}
		seen[id] = true

		e, ok := s.get(id)
		if !ok {
			continue
		}
		if e.AcknowledgedAt == nil { // skip update, so no redundant file line
			first := false // decided under the lock: two concurrent acks can't both win
			e, ok = s.update(id, func(e *Event) {
				if e.AcknowledgedAt != nil {
					return
				}
				first = true
				at := now
				e.AcknowledgedAt = &at
				if e.Siren == sirenRinging {
					e.Siren = sirenStopped
				}
			})
			if !ok {
				continue // trimmed by max since get
			}
			if first && s.onAck != nil {
				s.onAck(e)
			}
		}
		out = append(out, e)
	}

	if len(out) == 0 {
		writeJSON(w, http.StatusNotFound, map[string]string{"error": "no such events"})
		return
	}
	writeJSON(w, http.StatusOK, out)
}
