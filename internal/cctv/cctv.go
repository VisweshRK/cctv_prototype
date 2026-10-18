// Package cctv is the CCTV prototype's own go2rtc module. It adds:
//
//   - camera online/offline status:  GET /api/cctv/status
//   - the shared events endpoint:     GET/POST /api/cctv/events
//   - the tampering/disconnect use case (camera_offline / camera_online events)
//
// The AI use cases live in the Python worker under ai/ and report through the
// events endpoint.
package cctv

import (
	"errors"
	"time"

	"github.com/AlexxIT/go2rtc/internal/api"
	"github.com/AlexxIT/go2rtc/internal/app"
	"github.com/rs/zerolog"
)

func Init() {
	var cfg struct {
		Mod struct {
			Cameras      []string      `yaml:"cameras"`
			Interval     time.Duration `yaml:"interval"`
			OfflineAfter time.Duration `yaml:"offline_after"`
			EventsFile   string        `yaml:"events_file"`
			MaxEvents    int           `yaml:"max_events"`
		} `yaml:"cctv"`
	}

	// defaults
	cfg.Mod.Interval = 5 * time.Second
	cfg.Mod.OfflineAfter = 20 * time.Second
	cfg.Mod.EventsFile = "data/events.jsonl"
	cfg.Mod.MaxEvents = 1000

	app.LoadConfig(&cfg)

	log = app.GetLogger("cctv")

	events := newEventStore(cfg.Mod.EventsFile, cfg.Mod.MaxEvents)
	api.HandleFunc("api/cctv/events", events.apiEvents)

	w := newWatcher(cfg.Mod.Cameras, cfg.Mod.Interval, cfg.Mod.OfflineAfter, events)
	api.HandleFunc("api/cctv/status", w.apiStatus)

	if len(cfg.Mod.Cameras) == 0 {
		log.Warn().Msg("[cctv] no cameras listed under cctv.cameras")
		return
	}

	// give go2rtc a moment to finish loading streams
	time.AfterFunc(2*time.Second, w.run)
}

var log zerolog.Logger

var errNotConfigured = errors.New("stream is not configured in go2rtc.yaml")
