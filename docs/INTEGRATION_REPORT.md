# CCTV all-use-cases integration report

Date: 2026-11-18
Branch: `integration/all-usecases`

## Current outcome

The local website is running at [http://localhost:1984/cctv.html](http://localhost:1984/cctv.html). The dashboard loaded in a browser and showed both configured cameras online, seven configured use cases, and an online AI worker. This is a local link for this computer and network; it is not a public deployment URL.

This branch integrates the seven use cases specified by the pasted requirements: `fall`, `baby_crying`, `face_recognition`, `vehicle_lpr`, `tampering`, `occupancy`, and `power_when_absent`. Fire/smoke is out of scope per that specification. Each case has its own latest-frame inference lane so a slow case cannot hold the other cases' lane queues or the Go camera viewer. Each lane keeps one running frame and one replaceable pending frame; stale pending frames are dropped.

## Live readiness check

At the live health snapshot taken on 2026-11-18, both cameras reported online and the worker reported online. Readiness was:

| Use case | State | Live requirement / remaining setup |
| --- | --- | --- |
| Fall | Ready | MoveNet pose model loaded |
| Occupancy | Ready | Running on both camera streams |
| Tampering | Ready | Running on both camera streams |
| Vehicle LPR | Ready | OCR models loaded; OCR is gated on detected vehicle crops |
| Face recognition | Degraded | Face models loaded; enroll people in Face gallery and build the gallery |
| Power when absent | Degraded | Configure appliance regions/ROIs for the camera views |
| Baby crying | Disabled | Neither camera stream delivered audio frames during the check |

The worker metrics snapshot reported capture at 9.8 FPS for cam1 and 4.5 FPS for cam2, with no queued or dropped work in the displayed case lanes. These are point-in-time capture numbers, not a sustained latency benchmark or a claim that every case was detection-validated. The dashboard showed live occupancy and tampering/fall events; an alert can also be a false positive and should be reviewed against the camera scene. No known-positive face, plate, crying-audio, or appliance-power scenario was available for a full accuracy test.

To use face recognition, open **Face gallery** in the dashboard, add a person and several clear face photos, then build/refresh the gallery. To use power-when-absent, draw/configure appliance ROIs in `ai/zones.yaml` and restart the worker. Baby-crying requires a camera stream that provides audio; this current feed did not. The gallery API is restricted to loopback requests.

## Verification run

Commands/results from this session:

```text
cd ai && .venv\Scripts\python.exe -m unittest discover -s tests
147 tests passed

node --test "www/tests/*.test.mjs"
34 tests passed

go test ./internal/cctv/...
passed (run with local-loopback test access)

PowerShell AST parse of scripts/run.ps1
syntax OK

git diff --check
passed
```

The full Python suite took 316.953 seconds. The Node runner emitted the existing package module-type warning. The initial Go test attempt was blocked by the sandbox's loopback socket restriction; rerunning with the required local-test access passed.

The browser smoke check verified the rendered dashboard, camera-online indicators, configured-case count, worker readiness display, and live alert feed. A controlled sustained throughput/latency test and known-positive detection tests for all seven use cases remain unperformed because those require labeled test scenes/audio and ROI/gallery configuration.

## Runtime and safety notes

- Camera URLs and credentials remain in `.env` and were not copied into this report or source output.
- Model assets and the local face gallery are not committed.
- `scripts/run.ps1` provides local startup and stop commands; it rejects port 1984 if another listener is already present.
- The current website is served from this machine. `localhost` works only on this computer; another device would require a reachable host address and network/firewall configuration.
- Do not present the seven-case integration as fully validated: three cases are presently blocked by missing user-specific setup/audio, and four only have readiness plus live pipeline evidence.

## Camera reconnection retest — 2026-11-22

ONVIF discovery found the camera on the local network and enumerated four profiles using the saved camera login. The ignored local `.env` was updated to the discovered address, then the Go2RTC dashboard and AI worker were restarted. The dashboard returned HTTP 200; cam1 and cam2 both reported online and the worker reported online. The post-reconnect capture snapshot was approximately 12.5 FPS for cam1 and 12.6 FPS for cam2.

Live readiness remained four ready cases (fall, occupancy, tampering, and vehicle LPR). Face recognition still needs gallery enrollment; power-when-absent still needs appliance ROIs; baby-cry detection is disabled because neither stream produced audio frames. These are setup/input limitations, not camera connectivity failures.
