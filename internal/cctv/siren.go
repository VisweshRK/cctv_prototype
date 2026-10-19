package cctv

import (
	"encoding/binary"
	"math"
	"os"
	"path/filepath"
	"sync"
	"time"

	"github.com/AlexxIT/go2rtc/internal/streams"
)

// sirenPlayer plays source on the camera's own speaker (go2rtc two-way
// audio); source == "" stops it.
type sirenPlayer func(camera, source string) error

// playOnCamera is the real sirenPlayer.
func playOnCamera(camera, source string) error {
	stream := streams.Get(camera)
	if stream == nil {
		return errNotConfigured
	}
	return stream.Play(source)
}

// siren rings the speaker of the camera that saw a burglary-type alert until
// a guard acknowledges every such alert on that camera, or for at most max.
//
// play can block while dialling the camera, so it is never called with mu
// held. Instead each camera has one "driver" at a time: whoever changes what
// the speaker should do while nobody is driving calls play until the speaker
// matches (see drive). Callers that arrive meanwhile only record their change
// and return; the driver picks it up.
type siren struct {
	play     sirenPlayer
	source   string
	useCases map[string]bool
	max      time.Duration
	store    *eventStore

	mu      sync.Mutex
	cameras map[string]*sirenCamera
}

type sirenCamera struct {
	pending map[int64]bool // unacknowledged siren events
	joined  []int64        // events waiting for the start in flight
	rung    []int64        // events set ringing in this ring cycle

	want   bool // a ring cycle is active: the speaker should play
	on     bool // the speaker was last told to play, successfully
	busy   bool // a caller is running play for this camera
	cutOff bool // max reached: quiet until every pending event is acknowledged
	warned bool // "camera speaker not available" has been logged

	timer *time.Timer
	gen   int // bumped whenever the timer is replaced or cancelled
}

func newSiren(play sirenPlayer, source string, useCases []string, max time.Duration, store *eventStore) *siren {
	s := &siren{play: play, source: source, useCases: map[string]bool{}, max: max, store: store, cameras: map[string]*sirenCamera{}}
	for _, uc := range useCases {
		s.useCases[uc] = true
	}
	return s
}

func (s *siren) camera(name string) *sirenCamera {
	c := s.cameras[name]
	if c == nil {
		c = &sirenCamera{pending: map[int64]bool{}}
		s.cameras[name] = c
	}
	return c
}

// onEvent is called from the store's onAdd chain, synchronously.
func (s *siren) onEvent(e Event) {
	if !s.useCases[e.UseCase] || e.AcknowledgedAt != nil {
		return
	}

	s.mu.Lock()
	// re-read under mu: an ack that already happened is seen here, and one
	// that hasn't yet will find the event pending (onAck takes mu after us)
	if cur, ok := s.store.get(e.ID); !ok || cur.AcknowledgedAt != nil {
		s.mu.Unlock()
		return
	}
	c := s.camera(e.Camera)
	c.pending[e.ID] = true
	switch {
	case c.cutOff:
		s.setSiren(e.ID, sirenStopped)
	case c.want && c.on: // already ringing: the timer is not restarted
		c.rung = append(c.rung, e.ID)
		s.setSiren(e.ID, sirenRinging)
	default: // start a ring cycle, or join the start in flight
		c.want = true
		c.joined = append(c.joined, e.ID)
	}
	s.drive(e.Camera, c)
}

// onAck is the store's onAck hook, called once per newly acknowledged event.
func (s *siren) onAck(e Event) {
	s.mu.Lock()
	c := s.cameras[e.Camera]
	if c == nil || !c.pending[e.ID] {
		s.mu.Unlock()
		return // not a siren event: leave the camera alone
	}
	delete(c.pending, e.ID)
	s.stopRinging(e.ID)

	if len(c.pending) == 0 {
		if c.timer != nil {
			c.timer.Stop()
			c.timer = nil
		}
		c.gen++
		for _, id := range c.rung {
			s.stopRinging(id)
		}
		c.rung, c.joined = nil, nil
		c.want, c.cutOff = false, false
	}
	s.drive(e.Camera, c)
}

// timeout ends the ring cycle that started the timer with generation gen.
func (s *siren) timeout(camera string, gen int) {
	s.mu.Lock()
	c := s.cameras[camera]
	if c.gen != gen || !c.want {
		s.mu.Unlock()
		return // stopped by an ack, or a newer cycle, since the timer fired
	}
	c.timer = nil
	c.want, c.cutOff = false, true
	for id := range c.pending {
		s.stopRinging(id)
	}
	c.rung = nil
	s.drive(camera, c)
}

// drive calls play until the speaker matches c.want, unless another caller
// is already doing so. Call with s.mu held; it returns with s.mu released.
func (s *siren) drive(camera string, c *sirenCamera) {
	if c.busy {
		s.mu.Unlock()
		return
	}
	c.busy = true
	for c.want != c.on {
		start := c.want
		source := ""
		if start {
			source = s.source
		}

		s.mu.Unlock()
		err := s.play(camera, source)
		s.mu.Lock()

		if !start {
			if err != nil {
				log.Debug().Err(err).Str("camera", camera).Msg("[cctv] can't stop siren")
			}
			c.on = false // nothing more we can do
			continue
		}
		s.started(camera, c, err)
	}
	c.busy = false
	s.mu.Unlock()
}

// started records the outcome of play(camera, source) for every event that
// was waiting for it. Call with s.mu held.
func (s *siren) started(camera string, c *sirenCamera, err error) {
	joined := c.joined
	c.joined = nil

	if err != nil {
		if !c.warned {
			c.warned = true
			log.Warn().Err(err).Str("camera", camera).Msg("[cctv] camera speaker not available")
		}
		c.want = false
		for _, id := range joined {
			if c.pending[id] {
				s.setSiren(id, sirenUnavailable)
			}
		}
		return
	}

	c.on = true
	if !c.want {
		return // all acknowledged meanwhile: drive stops it next
	}
	c.gen++
	gen := c.gen
	c.timer = time.AfterFunc(s.max, func() { s.timeout(camera, gen) })
	for _, id := range joined {
		if c.pending[id] {
			c.rung = append(c.rung, id)
			s.setSiren(id, sirenRinging)
		}
	}
}

func (s *siren) setSiren(id int64, state string) {
	s.store.update(id, func(e *Event) { e.Siren = state })
}

// stopRinging marks the event stopped if it is still ringing.
func (s *siren) stopRinging(id int64) {
	if e, ok := s.store.get(id); !ok || e.Siren != sirenRinging {
		return // nothing to change: skip the file line
	}
	s.store.update(id, func(e *Event) {
		if e.Siren == sirenRinging {
			e.Siren = sirenStopped
		}
	})
}

// writeSirenWAV writes a 2 s loopable two-tone siren (960 Hz and 770 Hz,
// 0.5 s each, twice) as 16-bit mono PCM at 8000 Hz, unless path exists.
func writeSirenWAV(path string) error {
	if _, err := os.Stat(path); err == nil {
		return nil
	}

	const rate = 8000
	const amplitude = 0.6 * math.MaxInt16

	var samples []int16
	for i := 0; i < 2; i++ {
		for _, freq := range []float64{960, 770} {
			// both tones fit a whole number of cycles in 0.5 s, so every
			// segment starts and ends at zero: no clicks, and it loops
			for n := 0; n < rate/2; n++ {
				v := amplitude * math.Sin(2*math.Pi*freq*float64(n)/rate)
				samples = append(samples, int16(math.Round(v)))
			}
		}
	}

	dataSize := uint32(len(samples) * 2)
	b := make([]byte, 0, 44+dataSize)
	b = append(b, "RIFF"...)
	b = binary.LittleEndian.AppendUint32(b, 36+dataSize)
	b = append(b, "WAVE"...)
	b = append(b, "fmt "...)
	b = binary.LittleEndian.AppendUint32(b, 16)     // fmt chunk size
	b = binary.LittleEndian.AppendUint16(b, 1)      // PCM
	b = binary.LittleEndian.AppendUint16(b, 1)      // mono
	b = binary.LittleEndian.AppendUint32(b, rate)   // sample rate
	b = binary.LittleEndian.AppendUint32(b, rate*2) // byte rate
	b = binary.LittleEndian.AppendUint16(b, 2)      // block align
	b = binary.LittleEndian.AppendUint16(b, 16)     // bits per sample
	b = append(b, "data"...)
	b = binary.LittleEndian.AppendUint32(b, dataSize)
	for _, v := range samples {
		b = binary.LittleEndian.AppendUint16(b, uint16(v))
	}

	if err := os.MkdirAll(filepath.Dir(path), 0755); err != nil {
		return err
	}
	tmp := path + ".tmp" // so a half-written file never counts as existing
	if err := os.WriteFile(tmp, b, 0644); err != nil {
		return err
	}
	return os.Rename(tmp, path)
}
