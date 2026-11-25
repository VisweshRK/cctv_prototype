package cctv

import (
	"encoding/json"
	"net/http"
	"os"
	"time"
)

type healthCase struct {
	ID     string `json:"id"`
	State  string `json:"state"`
	Reason string `json:"reason,omitempty"`
}

type workerHealth struct {
	Worker    string                 `json:"worker"`
	UpdatedAt string                 `json:"updated_at"`
	UseCases  []healthCase           `json:"use_cases"`
	Audio     map[string]audioHealth `json:"audio,omitempty"`
	Metrics   json.RawMessage        `json:"metrics,omitempty"`
}

type audioHealth struct {
	State  string `json:"state"`
	Reason string `json:"reason,omitempty"`
}

var workerHealthPath = "data/ai-health.json"

// apiHealth exposes only the local AI worker's safe readiness summary.
func apiHealth(rw http.ResponseWriter, _ *http.Request) {
	body, err := os.ReadFile(workerHealthPath)
	if err != nil {
		writeJSON(rw, http.StatusOK, workerHealth{Worker: "offline", UseCases: []healthCase{}})
		return
	}
	var health workerHealth
	if json.Unmarshal(body, &health) != nil {
		writeJSON(rw, http.StatusOK, workerHealth{Worker: "offline"})
		return
	}
	updated, err := time.Parse(time.RFC3339Nano, health.UpdatedAt)
	if err != nil || time.Since(updated) > 20*time.Second || time.Until(updated) > time.Minute {
		health.Worker = "offline"
	}
	writeJSON(rw, http.StatusOK, health)
}
