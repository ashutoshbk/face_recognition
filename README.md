# Office Face Recognition (InsightFace)

## Office GPU and Multi-Camera Update - 20 October 2026

For the new single/bulk registration and configurable multi-camera evaluation,
start with **[face_id_poc/office_demo/README.md](face_id_poc/office_demo/README.md)**.
It covers the two five-photo collections, head tracking, progressive voting,
Ubuntu GPU setup, RTSP configuration and the first-round validation results.
The supplied model files are unchanged; no DigiFace download is required for
this office entry point. Personal photos, videos and runtime databases are not
part of this delivery.

The sections below describe the original standalone script. The earlier
[research demo](face_id_poc/research_demo/README.md) and its instructions are
retained separately; do not mix the three installation paths.

A simple Python script that recognizes known people in a video by comparing detected faces
against a folder of reference (staff) photos, using InsightFace face embeddings.

## What it does

1. Loads the InsightFace `buffalo_l` model (GPU if available, CPU as fallback).
2. Reads every staff photo from a reference folder and builds a "face database"
   (one embedding per person, taken from their photo's filename).
3. Runs face detection + recognition frame-by-frame on an input video.
4. For each detected face, compares it against the face database using cosine similarity.
   - If the best match is above the similarity threshold → labeled with the person's name.
   - Otherwise → labeled `Unknown`.
5. Draws a bounding box + label (green = known, red = unknown) on every frame.
6. Saves the annotated result as a new output video and prints live progress (%, FPS, ETA).

## Folder structure

```
.
├── main.py
├── Office_staff_photos/        # Reference photos, one clear frontal face per person
│   ├── John.jpg
│   ├── Sara.png
│   └── ...
├── videos/
│   └── cam1001_test.mp4        # Input test video (see Test Videos below)
└── README.md
```

- **Reference photos**: filename (without extension) is used as the person's name in the
  output, so `John.jpg` will be labeled `John`. Use one clean, frontal, well-lit photo per
  person — if no face is detected in a photo, it's skipped with a warning.
- **Videos folder**: not committed to GitHub (kept out via `.gitignore` due to size).
  Test videos are hosted on Mega — download from the link below and place them in `videos/`.

## Test videos

Sample test videos are available here (Mega): **https://mega.nz/folder/T7RRWIbI#5zz8AuPvobj4Q4Ue-qZ0mA**

Download the video(s) you want to test and place them inside the `videos/` folder before
running the script.

## Requirements

```
opencv-python
numpy
insightface
onnxruntime-gpu   # for GPU (or onnxruntime for CPU-only)
```

Install with:
```bash
pip install opencv-python numpy insightface onnxruntime-gpu
```

## How to run

1. Put your reference photos in `Office_staff_photos/`.
2. Put the video you want to test in `videos/` and update `video_path` at the bottom of
   `main.py` if needed.
3. Run:
   ```bash
   python main.py
   ```
4. The script auto-detects GPU availability. Output video is saved as
   `<input_filename>_output_gpu.mp4` in the project folder.

## Key settings (in `main.py`)

| Setting | Where | Description |
|---|---|---|
| `threshold` | `face_recognition_from_video(...)` call | Minimum cosine similarity to accept a match (default `0.3`). Raise it (e.g. `0.5`) for stricter matching, fewer false positives. |
| `use_gpu` | same call | Set automatically based on `test_gpu()`; can be forced to `True`/`False`. |
| `det_size` | `initialize_model()` | Face detector input size (default `640x640`). |

## Notes

- GPU mode requires `onnxruntime-gpu` + a working CUDA setup; the script automatically
  falls back to CPU if GPU init fails.
- Recognition accuracy depends heavily on reference photo quality — use frontal,
  well-lit, unobstructed faces.

## Standalone Face ID and Head Tracking PoC

The independent [face_id_poc](face_id_poc/) folder provides enrollment, head
tracking, multi-frame face identification, an optional demo renderer, and a
frame-in/results-out integration API. Start with its README for setup and testing.
It does not replace the original `main.py` application.
