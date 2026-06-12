# cyvl

**Build on Cyvl's physical-world data.** Street-level imagery, dense LiDAR, and
infrastructure layers for Somerville, MA — with full 3D↔2D projection powered by the
same math as Cyvl's production pipeline.

```bash
pip install "cyvl[viz] @ git+https://github.com/cyvl/cyvl-sdk"
```

```python
import cyvl

scene = cyvl.load_scene("somerville")

# Find a photo near Davis Square
frame_id = scene.frames_near(-71.1218, 42.3967).index[0]
frame = scene.frame(frame_id)

# Project the surrounding LiDAR into the photo — exact pixel coordinates
pts = frame.points_in_view(radius_m=30)
frame.show(overlay=pts)

# Click two pixels, get a real-world measurement
m = cyvl.measure(frame, (1700, 1200), (2100, 1180))
print(m)  # Measurement(3.42 m / 11.22 ft)
```

## What you get

- **`Scene`** — 311k posed camera frames, 84k pavement distresses, 3.7k signs,
  8k surface assets, 7k road markings, pavement condition scores, and a 514-tile
  LiDAR index, all as (Geo)DataFrames.
- **`Frame.project / unproject`** — pixels ↔ world coordinates, both directions,
  using production camera calibration and SLAM poses.
- **`Frame.points_in_view()`** — colored LiDAR points with exact pixel locations.
- **`cyvl.measure(...)`** — real distances between any two pixels of a photo.

Docs, examples, and schema reference: see [`examples/`](examples/).
