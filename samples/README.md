# Sample videos (not committed)

Put test clips here to use as fake cameras. Everything in this folder except this README is git-ignored,
so footage never ends up in the repo.

Make a generic test clip (moving pattern + tone):

```bash
ffmpeg -f lavfi -i testsrc=size=640x360:rate=15 -f lavfi -i sine=frequency=440:sample_rate=16000 -t 20 -c:v libx264 -pix_fmt yuv420p -g 30 -c:a aac samples/test.mp4
```

For your use case, record a short clip that shows it (a person walking in, a car at the gate, a baby
crying...). Then in `.env`:

```
CAM1_URL=ffmpeg:samples/my_clip.mp4#input=loop#video=copy#audio=copy
```

Phone videos are often H.265; if live view stays black, use `#video=h264#audio=aac` instead of `copy`.
