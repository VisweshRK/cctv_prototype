package cctv

import (
	"encoding/json"
	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
)

func TestHealthReturnsWorkerAndAudioReadiness(t *testing.T) {
	t.Chdir(t.TempDir())
	require.NoError(t, os.MkdirAll("data", 0700))
	now := time.Now().UTC().Format(time.RFC3339Nano)
	body := `{"worker":"online","updated_at":"` + now + `","use_cases":[{"id":"baby_crying","state":"disabled"}],"audio":{"cam1":{"state":"unavailable","reason":"no audio frames received"}}}`
	require.NoError(t, os.WriteFile(filepath.Join("data", "ai-health.json"), []byte(body), 0600))

	rw := httptest.NewRecorder()
	apiHealth(rw, httptest.NewRequest("GET", "/api/cctv/health", nil))
	require.Equal(t, 200, rw.Code)
	var got workerHealth
	require.NoError(t, json.Unmarshal(rw.Body.Bytes(), &got))
	require.Equal(t, "online", got.Worker)
	require.Equal(t, "unavailable", got.Audio["cam1"].State)
	require.Equal(t, "no audio frames received", got.Audio["cam1"].Reason)
}

func TestHealthMarksStaleWorkerOffline(t *testing.T) {
	t.Chdir(t.TempDir())
	require.NoError(t, os.MkdirAll("data", 0700))
	old := time.Now().Add(-time.Minute).UTC().Format(time.RFC3339Nano)
	body := `{"worker":"online","updated_at":"` + old + `","use_cases":[]}`
	require.NoError(t, os.WriteFile(filepath.Join("data", "ai-health.json"), []byte(body), 0600))
	rw := httptest.NewRecorder()
	apiHealth(rw, httptest.NewRequest("GET", "/api/cctv/health", nil))
	var got workerHealth
	require.NoError(t, json.Unmarshal(rw.Body.Bytes(), &got))
	require.Equal(t, "offline", got.Worker)
}
