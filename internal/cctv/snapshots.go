package cctv

import (
	"errors"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"regexp"
	"strconv"
	"time"

	"github.com/AlexxIT/go2rtc/internal/ffmpeg"
	"github.com/AlexxIT/go2rtc/internal/streams"
	"github.com/AlexxIT/go2rtc/pkg/core"
	"github.com/AlexxIT/go2rtc/pkg/magic"
	"github.com/AlexxIT/go2rtc/pkg/mjpeg"
)

// frameGrabber returns one JPEG frame from a camera.
type frameGrabber func(camera string) ([]byte, error)

// snapshots gives every event a JPEG, saved as <dir>/<id>.jpg: the image the
// AI worker POSTed, or else a frame grabbed from the camera.
type snapshots struct {
	dir   string
	grab  frameGrabber
	store *eventStore
}

func newSnapshots(dir string, grab frameGrabber, store *eventStore) *snapshots {
	return &snapshots{dir: dir, grab: grab, store: store}
}

var snapshotName = regexp.MustCompile(`^([0-9]+)\.jpg$`)

// handle is the store's onAdd hook. It returns at once; the grab and the
// write happen in a goroutine.
func (p *snapshots) handle(e Event, image []byte) {
	if image == nil && (e.Label == "camera_offline" || e.Label == "camera_online") {
		return // no picture to take from a camera that changed state
	}
	go p.save(e, image)
}

func (p *snapshots) save(e Event, image []byte) {
	if image == nil {
		var err error
		if image, err = p.grab(e.Camera); err == nil && len(image) == 0 {
			err = errors.New("empty frame")
		}
		if err != nil {
			log.Debug().Err(err).Str("camera", e.Camera).Int64("id", e.ID).Msg("[cctv] no snapshot")
			return
		}
	}

	id := strconv.FormatInt(e.ID, 10)
	if err := p.write(id, image); err != nil {
		log.Warn().Err(err).Msg("[cctv] can't write snapshot")
		return
	}
	// ids restart at 1 when data/ is wiped and snapshots are cached for a day,
	// so t (ignored by apiSnapshot) makes the URL unique per store lifetime
	src := "api/cctv/snapshot?id=" + id + "&t=" + strconv.FormatInt(e.Time.UnixNano(), 10)
	p.store.update(e.ID, func(e *Event) { e.Snapshot = src })
	p.prune(p.store.oldestID())
}

// write saves the file under a temp name first, so a GET never sees half a file.
func (p *snapshots) write(id string, b []byte) error {
	if err := os.MkdirAll(p.dir, 0755); err != nil {
		return err
	}
	path := filepath.Join(p.dir, id+".jpg")
	if err := os.WriteFile(path+".tmp", b, 0644); err != nil {
		return err
	}
	return os.Rename(path+".tmp", path)
}

// prune deletes the snapshots of events older than oldest; other files are left alone.
func (p *snapshots) prune(oldest int64) {
	entries, err := os.ReadDir(p.dir)
	if err != nil {
		return
	}
	for _, entry := range entries {
		m := snapshotName.FindStringSubmatch(entry.Name())
		if m == nil {
			continue
		}
		if n, err := strconv.ParseInt(m[1], 10, 64); err == nil && n < oldest {
			_ = os.Remove(filepath.Join(p.dir, entry.Name()))
		}
	}
}

var snapshotID = regexp.MustCompile(`^[0-9]+$`)

// apiSnapshot handles GET /api/cctv/snapshot?id=1 (any other parameter, like t, is ignored)
func (p *snapshots) apiSnapshot(w http.ResponseWriter, r *http.Request) {
	id := r.URL.Query().Get("id")
	if !snapshotID.MatchString(id) {
		http.Error(w, "bad id", http.StatusBadRequest)
		return
	}
	b, err := os.ReadFile(filepath.Join(p.dir, id+".jpg"))
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			http.Error(w, "no snapshot", http.StatusNotFound)
		} else {
			http.Error(w, err.Error(), http.StatusInternalServerError)
		}
		return
	}
	h := w.Header()
	h.Set("Content-Type", "image/jpeg")
	h.Set("Cache-Control", "max-age=86400") // a snapshot URL never changes its image (see t in save)
	h.Set("Content-Length", strconv.Itoa(len(b)))
	_, _ = w.Write(b)
}

const grabTimeout = 8 * time.Second

// grabFrame takes one keyframe from the camera's go2rtc stream as a JPEG,
// the same way internal/mjpeg handlerKeyframe does.
func grabFrame(camera string) ([]byte, error) {
	stream := streams.Get(camera)
	if stream == nil {
		return nil, errNotConfigured
	}
	timeout := time.NewTimer(grabTimeout)
	defer timeout.Stop()

	cons := magic.NewKeyframe()
	if err := stream.AddConsumer(cons); err != nil {
		return nil, err
	}

	once := &core.OnceBuffer{} // init and first frame
	done := make(chan struct{})
	go func() {
		_, _ = cons.WriteTo(once)
		close(done)
	}()

	select {
	case <-done:
		stream.RemoveConsumer(cons)
	case <-timeout.C:
		stream.RemoveConsumer(cons) // stops the consumer, so WriteTo returns
		return nil, errors.New("no keyframe within " + grabTimeout.String())
	}

	b := once.Buffer()
	switch cons.CodecName() {
	case core.CodecH264, core.CodecH265:
		var err error
		if b, err = ffmpeg.JPEGWithQuery(b, url.Values{}); err != nil {
			return nil, err
		}
	case core.CodecJPEG:
		b = mjpeg.FixJPEG(b)
	}
	return b, nil
}
