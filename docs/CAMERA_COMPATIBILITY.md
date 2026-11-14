# CAMERA HARDWARE COMPATIBILITY & DEPLOYMENT MODES
## Section 2 Diagnostics & Hardware Verification

---

## 1. Non-Destructive Camera Compatibility Probe

The Presence B2C platform includes a built-in diagnostic tool (`apps/api/app/cameras/compatibility.py`) reachable via `POST /api/cameras/probe`.

### Verified Checks:
1. **IP Reachability**: TCP handshake on port 554 (RTSP) and port 80 / 8899 (ONVIF).
2. **RTSP Stream Decoding**: OpenCV / FFmpeg probe verifying resolution (1080p/720p), FPS, and H.264 video codec.
3. **Audio Stream & Microphone**: Verification of PCM 16kHz audio track for audio ML modules (Baby crying).
4. **ONVIF Support**: Discovery of PTZ capabilities.
5. **Vendor NPU Toolchain**: Cryptographic check for verified edge AI runtimes (e.g., Rockchip RKNN, Ambarella, Horizon BPU).

---

## 2. Deployment Modes

### Mode A — Existing Camera with External Host AI (Default Production Mode)
- The user's Wi-Fi CCTV camera streams H.264 video and 16 kHz audio via standard RTSP to the local host machine (Windows or Linux).
- The Presence B2C server performs single-stream decoding and runs all 10 AI plugins locally on the host's CPU / GPU / NPU.
- **Safety guarantee**: No vendor firmware modification is required.

### Mode B — Compatible Camera with Onboard AI
- Applicable only when the camera hardware is independently validated with an official vendor SDK (e.g., RK3588, RV1126, Ingenic).
- Models are quantized (INT8) and deployed directly onto the camera's internal NPU.
- **Status**: Currently unverified for generic consumer cameras; requires authorized hardware manufacturer partnership.

### Mode C — Unsupported Camera
- Reported when the camera does not support standard RTSP or when video decoding fails.
- The platform honestly reports exact limitations and allows testing using local media files or the built-in synthetic test stream.
