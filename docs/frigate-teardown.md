# Frigate Teardown for CCTV Prototype

> Copy of the Claude Doc "Frigate Teardown for CCTV Prototype" (rev 16, 2026-10-18).
> The live version, with the architecture diagram, is at
> https://claude.ai/code/artifact/41239294-7313-40f1-96fc-b19a557dff41

Frigate already solves 6 of our 10 use cases on top of go2rtc; the other 4 (tampering, fall, electricity, visual fire) need custom models or integrations.

## Use-case map

Native = Frigate ships it; Partial = a piece exists; Custom = we build it.

| # | Use case | Frigate status | How Frigate does it / what we add |
| --- | --- | --- | --- |
| 1 | Intrusion / trespassing alarm | Native | Person detection + zone polygon; hit = bottom-centre of the box inside the zone (`cv2.pointPolygonTest`); `alerts.required_zones` raises an alert |
| 2 | Camera tampering / removed / disconnected | Partial | Disconnect: `CameraWatchdog` marks a stream stale/offline per role. Tamper (covered, blurred, turned): not built; nearest is motion recalibration when >80% of the frame changes (`lightning_threshold`). We add a blur/blackout/scene-shift check |
| 3 | Facial recognition | Native | `FaceDetectorYN` finds faces on person crops, FaceNet (small) or ArcFace (large) embeds them, match sets the event `sub_label` |
| 4 | Restricted area breach | Native | Same zones as #1, plus `loitering_time` and per-zone object filters |
| 5 | Fall / person down | Custom | Not in Frigate. Copy its custom object-classification processor on person crops ("standing" / "fallen"), or add a pose model |
| 6 | Room occupancy analytics | Native | Per-zone live counts (`zone_all_object_counts`, `zone_active_object_counts`) published on MQTT |
| 7 | Electricity use when room is empty | Custom | Not in Frigate. Combine #6 occupancy with a smart-plug or energy-meter feed and alert on "power on, 0 people for N min" |
| 8 | Fire detection | Partial | Audio only: YAMNet labels `fire_alarm`, `smoke_detector`, `fire`. No visual fire/smoke class in the object labelmap. We add a fire/smoke vision model |
| 9 | Vehicle detection + number-plate log | Native | Car detection, then YOLOv9 plate detector + PaddleOCR (ONNX); plate stored as `sub_label` on the Event row |
| 10 | Baby crying detection | Native | YAMNet audio model, `crying` / `baby_cry_infant_cry` labels, gated by `min_volume`. Needs the camera's audio track |

## Architecture

_(Architecture diagram: open the Claude Doc linked above.)_

Each camera connects once, to go2rtc; only motion-gated crops reach the detector, which keeps CPU use low on cheap hardware.

## Frigate parts worth copying

Read from Frigate commit add310d (Oct 2026); paths are relative to the repo root.

| Part | How Frigate does it | Where |
| --- | --- | --- |
| go2rtc integration | go2rtc runs as its own service; Frigate writes its YAML config, then pulls every stream back from `rtsp://127.0.0.1:8554/<camera>` | `docker/main/rootfs/usr/local/go2rtc/create_config.py` |
| Frame capture | ffmpeg decodes to raw YUV420 into a ring of shared-memory slots; only the slot name goes on the queue; a full queue drops and counts the frame | `frigate/video/ffmpeg.py` |
| Motion gate | Running-average background, absdiff, threshold, dilate, contours; recalibrates on sudden global change | `frigate/motion/improved_motion.py` |
| Region-only detection | Motion boxes are grown into model-sized regions; the detector only sees those crops; stationary objects rechecked every N frames | `frigate/video/detect.py` |
| Detector plugins | One class per backend (ONNX, OpenVINO, TensorRT, EdgeTPU, Hailo, RKNN, CPU), picked by `type`; cameras share one detector process via shared memory + ZeroMQ | `frigate/detectors/plugins/`, `frigate/object_detection/base.py` |
| Tracking + zones | Norfair tracker gives stable IDs; zones test the box's bottom-centre; loitering and speed per zone | `frigate/track/` |
| Alerts vs detections | Review items have two severities, set by `alerts.labels` and `required_zones` | `frigate/review/maintainer.py` |
| Add-on processors | `RealTimeProcessorApi`: `process_frame(obj, frame)`, `handle_request`, `expire_object`; face, plate, custom classification all plug in here | `frigate/data_processing/real_time/` |
| Audio events | YAMNet on the audio stream, gated by volume | `frigate/events/audio.py` |
| Recording | ffmpeg `-f segment -segment_time 10 -c copy` into MP4 chunks; keep by mode (all / motion / active objects) | `frigate/record/maintainer.py`, `frigate/ffmpeg_presets.py` |
| Storage + messaging | SQLite (peewee) tables Event, Timeline, Recordings; ZeroMQ between processes; MQTT `<prefix>/<camera>/<feature>/state` outside | `frigate/models.py`, `frigate/comms/` |
| Process model | Each job is its own OS process with a watchdog that restarts it | `frigate/app.py` |

## Where we can beat Frigate on efficiency

Frigate's core is already lean (motion gate, crop-only detection, shared memory). Our gains come from building inside go2rtc, which already holds every camera packet, so we can drop the extra ffmpeg processes Frigate runs around it.

| Where Frigate spends | What we do instead | Gain / cost |
| --- | --- | --- |
| One ffmpeg per role per camera (detect, record, audio), each pulling its own RTSP copy back from go2rtc (`frigate/events/audio.py`, `frigate/config/camera/ffmpeg.py`) | Record inside go2rtc: write the packets it already has into 10 s MP4 segments with its own MP4 muxer | One process and one loopback stream fewer per camera. Cost: segment rotation and retention written in Go |
| Audio: a separate `ffmpeg -vn` per camera decodes sound for YAMNet | Tap go2rtc's audio track directly; if the Q-Tech cameras send G.711 (to verify), decoding it is a simple lookup table in Go | No audio ffmpeg at all. Cost: resampling 8 kHz to the model's 16 kHz in our code |
| Detector called once per region, one after another (`frigate/video/detect.py`, loop over `regions`) | Batch all regions from both cameras into one model call (ONNX Runtime with a dynamic batch size) | Fewer calls; worthwhile on a GPU or NPU, small gain on CPU. Cost: model must be exported with a batch dimension |
| Detect stream fully decoded all the time, even with nothing moving | Idle mode: decode keyframes only, or wake on the camera's own ONVIF motion event, then switch to full rate | Large CPU saving at night and in empty rooms. Cost: up to one keyframe interval of extra delay on the first detection |
| Many Python processes plus ZeroMQ, sized for dozens of cameras | One Go process (goroutines) plus one Python inference worker | Less memory and simpler deploy for a 2-camera demo. Cost: we lose Frigate's per-process restart isolation, so add our own watchdog |

New features can piggyback the same way: tamper checks (blackout, blur, scene shift) run on the small grayscale frame the motion gate already produces, at almost no extra cost.

## Demo hardware

We have two Q-Tech Wi-Fi cameras with two-way audio. The one unknown that decides everything is which protocol they speak locally.

| If the camera offers | go2rtc source | Two-way audio | Notes |
| --- | --- | --- | --- |
| RTSP / ONVIF | `rtsp://`, `onvif://` | Yes, if ONVIF backchannel is exposed | Best case; same path Frigate uses |
| XMEye / DVR-IP (port 34567) | `dvrip://` | Yes | Common on budget Wi-Fi cams |

- [ ] Find the camera's companion app name and model number
- [ ] Port-scan each camera for 554 (RTSP), 80/8899 (ONVIF), 34567 (DVR-IP)
- [ ] Confirm the stream carries audio; baby-crying (#10) and fire-alarm sound (#8) need it
- [ ] Check for a low-res substream to run detection on, keeping main stream for recording

## Next decisions

Recommendation: keep go2rtc in Go for streams, add a Python AI sidecar that pulls go2rtc's RTSP restream, as Frigate does.

| Option | For | Against |
| --- | --- | --- |
| Go2rtc + Python sidecar (recommended) | Face, plate, YAMNet and YOLO models all have mature Python/ONNX tooling; Frigate's code can be followed closely | Two runtimes to ship |
