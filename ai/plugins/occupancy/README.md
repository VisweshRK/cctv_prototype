# Slot: Occupancy counting

**Status:** Completed (`enabled = True`).
**Owner:** Kishore

## Goal
Count how many confirmed people are in each camera's view (or zone) right now, including stationary people. Report when the count changes, when capacity is breached, and emit periodic summaries.

## Events Emitted
1. **`occupancy`** (count changed):
   - `label: "occupancy"`
   - `confidence: 1.0`
   - `data: {"zone": "room", "count": 3, "previous_count": 2, "change": "count_changed"}`
2. **`over_capacity`**:
   - `label: "over_capacity"`
   - `data: {"zone": "room", "count": 5, "capacity_limit": 4}`
   - Fires once on capacity breach; re-arms only when count drops to or below the limit.
3. **`summary`**:
   - Emitted every 5 minutes (`300.0` seconds) per zone for timeseries graphing and dashboards.

## Architecture & Frigate Influences
- **Shared Tracker Integration**: Consumes confirmed `TrackedObject` streams from the shared pipeline (`on_objects()`). Only counts `true_positive == True` persons.
- **Stationary People Support**: Sitting individuals are tracked as stationary objects and remain counted.
- **Zone Filtering**: Evaluates zone membership via `ZoneTracker` (ground-contact `feet()` point-in-polygon with 3-observation inertia).
- **Anti-Flicker Debounce**: Independent 3-second stability debounce window. Transient dropouts (e.g. 1-2 seconds behind a pillar) do not cause count fluctuations.
- **Power When Absent Integration**: Exposes `get_count(camera, zone)` directly so `power_when_absent` queries state without relying on asynchronous message queues.

## Testing
```bash
cd ai
python -m unittest tests/test_occupancy.py -v
```

## Running
```bash
cd ai
python -m cctv_ai.worker --plugins occupancy -v
```
Events show up at http://localhost:1984/api/cctv/events?use_case=occupancy
