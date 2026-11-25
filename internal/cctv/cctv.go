// Package cctv is the CCTV prototype's own go2rtc module. It adds:
//
//   - camera online/offline status:  GET /api/cctv/status (with place names and use cases)
//   - the shared events endpoint:     GET/POST /api/cctv/events
//   - live event push (SSE):          GET /api/cctv/events/stream
//   - acknowledging an alert:         POST /api/cctv/events/ack
//   - alert snapshots:                GET /api/cctv/snapshot?id=1
//   - the tampering/disconnect use case (camera_offline / camera_online events)
//   - a siren on the speaker of the camera that saw a burglary-type alert,
//     until the alert is acknowledged
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

			Names        map[string]string   `yaml:"names"`     // camera -> place name
			UseCases     map[string][]string `yaml:"use_cases"` // camera -> configured use cases
			SnapshotsDir string              `yaml:"snapshots_dir"`
			Siren        struct {
				UseCases []string      `yaml:"use_cases"`
				Max      time.Duration `yaml:"max"`
				File     string        `yaml:"file"`
				Audio    string        `yaml:"audio"`
			} `yaml:"siren"`
		} `yaml:"cctv"`
	}

	// defaults
	cfg.Mod.Interval = 5 * time.Second
	cfg.Mod.OfflineAfter = 20 * time.Second
	cfg.Mod.EventsFile = "data/events.jsonl"
	cfg.Mod.MaxEvents = 1000

	app.LoadConfig(&cfg)

	log = app.GetLogger("cctv")

	// defaults for fields left empty (or a missing siren block)
	if cfg.Mod.SnapshotsDir == "" {
		cfg.Mod.SnapshotsDir = "data/snapshots"
	}
	sc := &cfg.Mod.Siren
	if sc.Max <= 0 {
		sc.Max = 3 * time.Minute
	}
	if sc.File == "" {
		sc.File = "data/siren.wav"
	}
	switch sc.Audio {
	case "pcma", "pcmu", "aac":
	case "":
		sc.Audio = "pcma"
	default:
		log.Warn().Str("audio", sc.Audio).Msg("[cctv] siren.audio must be pcma, pcmu or aac; using pcma")
		sc.Audio = "pcma"
	}

	if err := writeSirenWAV(sc.File); err != nil {
		log.Warn().Err(err).Str("file", sc.File).Msg("[cctv] can't write the siren sound")
	}

	events := newEventStore(cfg.Mod.EventsFile, cfg.Mod.MaxEvents)
	snaps := newSnapshots(cfg.Mod.SnapshotsDir, grabFrame, events)
	source := "ffmpeg:" + sc.File + "#input=loop#audio=" + sc.Audio
	sir := newSiren(playOnCamera, source, sc.UseCases, sc.Max, events)
	sir.speaker = func(camera string) string { return speakerOf(camera, streamSources) }
	events.onAdd = func(e Event, image []byte) {
		snaps.handle(e, image) // returns at once
		sir.onEvent(e)
	}
	events.onAck = sir.onAck

	api.HandleFunc("api/cctv/events", events.apiEvents)
	api.HandleFunc("api/cctv/events/stream", events.apiStream)
	api.HandleFunc("api/cctv/events/ack", events.apiAck)
	api.HandleFunc("api/cctv/snapshot", snaps.apiSnapshot)
	api.HandleFunc("api/cctv/health", apiHealth)
	api.HandleFunc("api/cctv/faces", faceGallery)
	api.HandleFunc("api/cctv/faces/rebuild", rebuildFaceGallery)

	w := newWatcher(cfg.Mod.Cameras, cfg.Mod.Interval, cfg.Mod.OfflineAfter, events)
	w.describe(cfg.Mod.Names, cfg.Mod.UseCases)
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
