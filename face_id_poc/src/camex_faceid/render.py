"""Stateful OpenCV frame compositor; no windows, video I/O, or inference.

One renderer follows one clip with monotonically increasing result sequences.
Repeated or older sequences do not contribute processing samples or trail points.
Changing camera_id clears clip state; use a new Renderer to restart the same clip.
All captured/event times are clip-relative, including retained evidence frames.
"""

from collections import OrderedDict, deque
from datetime import datetime
import math
import unicodedata
import zlib

import cv2
import numpy as np


WIDTH, HEIGHT = 1920, 1080
CAMERA_RECT = (24, 104, 1408, 792)
PANEL_RECT = (1456, 104, 440, 792)
BACKGROUND = (244, 246, 246)
WHITE = (255, 255, 255)
INK = (43, 48, 48)
MUTED = (112, 119, 119)
RULE = (220, 226, 226)
TEAL = (135, 116, 32)
GREEN = (91, 145, 55)
AMBER = (49, 151, 192)
FONT = cv2.FONT_HERSHEY_SIMPLEX
TRACK_COLORS = (
    (177, 178, 55),
    (116, 189, 115),
    (87, 180, 224),
    (208, 158, 99),
    (172, 136, 200),
    (182, 183, 148),
)


def _number(value):
    try:
        value = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return value if math.isfinite(value) else None


def _ascii(value):
    # Hershey fonts cannot render Unicode. Normalize accents, then replace glyphs.
    value = unicodedata.normalize("NFKD", str(value))
    value = "".join(char for char in value if not unicodedata.combining(char))
    return " ".join(value.encode("ascii", "replace").decode("ascii").split())


def _fit(value, width, scale, thickness=1):
    value = _ascii(value)
    measure = lambda text, size: cv2.getTextSize(text, FONT, size, thickness)[0][0]
    if measure(value, scale) <= width:
        return value, scale
    scale = max(min(scale, 0.38), scale * width / max(1, measure(value, scale)))
    if measure(value, scale) <= width:
        return value, scale
    lo, hi = 0, len(value)
    while lo < hi:
        middle = (lo + hi + 1) // 2
        if measure(value[:middle] + "...", scale) <= width:
            lo = middle
        else:
            hi = middle - 1
    return (value[:lo] + "..." if measure("...", scale) <= width else ""), scale


def _text(image, value, x, y, width, scale=0.5, color=INK, thickness=1):
    value, scale = _fit(value, min(width, image.shape[1] - x - 1), scale, thickness)
    if value:
        cv2.putText(image, value, (x, y), FONT, scale, color, thickness, cv2.LINE_AA)


def _name(image, value, x, y, width):
    words = _ascii(value).split()
    if not words:
        return
    first = words.pop(0)
    while (
        words and cv2.getTextSize(first + " " + words[0], FONT, 0.5, 1)[0][0] <= width
    ):
        first += " " + words.pop(0)
    _text(image, first, x, y, width, 0.5)
    if words:
        _text(image, " ".join(words), x, y + 20, width, 0.5)


def _format_ms(value):
    value = _number(value)
    if value is None:
        return "--:--:--.---"
    seconds, milliseconds = divmod(max(0, int(value)), 1000)
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{milliseconds:03d}"


def _similarity(value):
    value = _number(value)
    return "--" if value is None else f"{value:.3f}"


def _track_color(track_id):
    return TRACK_COLORS[zlib.crc32(str(track_id).encode("utf-8")) % len(TRACK_COLORS)]


def _valid_image(image):
    return (
        isinstance(image, np.ndarray)
        and image.dtype == np.uint8
        and image.ndim == 3
        and image.shape[2] == 3
        and image.shape[0] > 0
        and image.shape[1] > 0
    )


def _place_image(destination, source, rect):
    x, y, width, height = rect
    ratio = min(width / source.shape[1], height / source.shape[0])
    size = (
        max(1, round(source.shape[1] * ratio)),
        max(1, round(source.shape[0] * ratio)),
    )
    x += (width - size[0]) // 2
    y += (height - size[1]) // 2
    interpolation = cv2.INTER_AREA if ratio < 1 else cv2.INTER_LINEAR
    destination[y : y + size[1], x : x + size[0]] = cv2.resize(
        source, size, interpolation=interpolation
    )
    return x, y, size[0], size[1]


def _thumbnail(source):
    image = np.full((92, 92, 3), BACKGROUND, dtype=np.uint8)
    if _valid_image(source):
        _place_image(image, source, (0, 0, 92, 92))
    else:
        _text(image, "Unavailable", 5, 49, 81, 0.38, MUTED)
    return image


def _clip_box(box, width, height):
    try:
        values = np.asarray(box, dtype=float)
    except (TypeError, ValueError):
        return None
    if values.shape != (4,) or not np.isfinite(values).all():
        return None
    x1, y1, x2, y2 = values
    x1, x2 = max(0, x1), min(width - 1, x2)
    y1, y2 = max(0, y1), min(height - 1, y2)
    return (x1, y1, x2, y2) if x2 > x1 and y2 > y1 else None


class Renderer:
    """Compose 1920x1080 uint8 BGR frames for a separate MP4 writer.

    Processing fps is 1000 / mean(processing_ms) over the last 120 valid,
    unique processing results. Playback is labeled separately as 10 FPS.
    Evidence retains up to five last-good pairs, newest per enrolled person_id;
    empty input does not erase it. Rows without person_id remain track-specific.
    Updating an existing pair preserves its row; new pairs enter at the top.
    Revoked tracked identities remain marked for review, including cached images.
    """

    MAX_EVIDENCE = 5
    MAX_TRAILS = 64
    TRAIL_LENGTH = 16
    TRAIL_TTL_MS = 3000
    MAX_EVENTS = 32

    def __init__(self, camera_name="Reception", clock_start="13:51:31"):
        clock = datetime.strptime(clock_start, "%H:%M:%S")
        self.camera_name = str(camera_name)
        self._clock_seconds = clock.hour * 3600 + clock.minute * 60 + clock.second
        self._camera_id = None
        self._reset()

    def _reset(self):
        self._last_sequence = None
        self._processed_frames = 0
        self._processing_times = deque(maxlen=120)
        self._trails = OrderedDict()
        self._track_seen = {}
        self._evidence = OrderedDict()
        self._timeline = deque(maxlen=self.MAX_EVENTS)
        self._event_keys = OrderedDict()
        self._revocations = OrderedDict()

    def _advance(self, result, source_shape):
        sequence = result.get("sequence", 0)
        if self._last_sequence is not None and sequence <= self._last_sequence:
            return
        self._last_sequence = sequence
        self._processed_frames += 1
        processing_ms = _number(result.get("processing_ms"))
        if processing_ms is not None and processing_ms > 0:
            self._processing_times.append(processing_ms)
        now = _number(result.get("timestamp_ms")) or 0
        for track_id in list(self._trails):
            if now - self._track_seen[track_id] > self.TRAIL_TTL_MS:
                del self._trails[track_id]
                del self._track_seen[track_id]
        height, width = source_shape[:2]
        for track in result.get("tracks", ()):
            track_id = track["track_id"]
            box = _clip_box(track.get("head_bbox"), width, height)
            if box is None:
                continue
            points = self._trails.setdefault(track_id, deque(maxlen=self.TRAIL_LENGTH))
            points.append(
                ((box[0] + box[2]) / (2 * width), (box[1] + box[3]) / (2 * height))
            )
            self._track_seen[track_id] = now
            self._trails.move_to_end(track_id)
            if len(self._trails) > self.MAX_TRAILS:
                evicted, _ = self._trails.popitem(last=False)
                del self._track_seen[evicted]
        for event in sorted(
            result.get("events", ()),
            key=lambda event: _number(event.get("timestamp_ms")) or 0,
        ):
            if event.get("event") not in ("identity_confirmed", "identity_revoked"):
                continue
            key = (
                event.get("event"),
                event.get("track_id"),
                event.get("person_id"),
                event.get("timestamp_ms"),
            )
            if key in self._event_keys:
                continue
            self._timeline.append(dict(event))
            self._event_keys[key] = None
            if len(self._event_keys) > self.MAX_EVENTS * 4:
                self._event_keys.popitem(last=False)

    def _remember_evidence(self, evidence):
        candidates = dict(self._evidence)
        for track_id, item in (evidence or {}).items():
            timestamp = _number(item.get("timestamp_ms"))
            if timestamp is None or not _valid_image(item.get("crop")):
                continue
            previous = candidates.get(track_id)
            if previous is None or timestamp > previous["timestamp_ms"]:
                candidates[track_id] = dict(item, timestamp_ms=timestamp)
        newest = []
        seen_people = set()
        # Compact before the five-row limit so repeat tracks cannot crowd out people.
        for track_id in sorted(
            candidates,
            key=lambda key: (
                -candidates[key]["timestamp_ms"],
                key not in self._evidence,
                str(key),
            ),
        ):
            person_id = candidates[track_id].get("person_id")
            if person_id is not None:
                if person_id in seen_people:
                    continue
                seen_people.add(person_id)
            newest.append(track_id)
            if len(newest) == self.MAX_EVIDENCE:
                break
        retained = OrderedDict(
            (key, row) for key, row in self._evidence.items() if key in newest
        )
        # Rank admission by capture time, but do not shuffle already-visible rows.
        for track_id in reversed(newest):
            item = candidates[track_id]
            if item is self._evidence.get(track_id):
                continue
            row = {
                "crop": _thumbnail(item.get("crop")),
                "reference": _thumbnail(item.get("reference")),
                "timestamp_ms": item["timestamp_ms"],
                "person_id": item.get("person_id"),
                "display_name": item.get("display_name"),
                "similarity": item.get("similarity"),
                "revoked": bool(item.get("revoked"))
                or bool(
                    self._evidence.get(track_id, {}).get("revoked")
                    and self._evidence[track_id].get("person_id")
                    == item.get("person_id")
                ),
            }
            is_new = track_id not in retained
            retained[track_id] = row
            if is_new:
                retained.move_to_end(track_id, last=False)
        self._evidence = retained

    def _needs_review(self, track_id, person_id):
        return (track_id, person_id) in self._revocations or (
            track_id,
            None,
        ) in self._revocations

    def _review_evidence(self, result, evidence):
        def revoke(track_id, person_id):
            key = (track_id, person_id)
            self._revocations[key] = None
            self._revocations.move_to_end(key)
            if len(self._revocations) > self.MAX_EVENTS * 4:
                self._revocations.popitem(last=False)

        # Review signals are independent of capture freshness and sequence sampling.
        for track_id, item in (evidence or {}).items():
            if item.get("revoked"):
                revoke(track_id, item.get("person_id"))
        for event in result.get("events", ()):
            if event.get("event") == "identity_revoked":
                revoke(event.get("track_id"), event.get("person_id"))
        for track in result.get("tracks", ()):
            row = self._evidence.get(track["track_id"])
            if row is not None and (
                track.get("recognition_state") in ("pending", "unknown")
                or track.get("person_id") != row["person_id"]
            ):
                revoke(track["track_id"], row["person_id"])
        for track_id, row in self._evidence.items():
            row["revoked"] = row.get("revoked", False) or self._needs_review(
                track_id, row["person_id"]
            )

    def _draw_tracks(self, camera, result, source_shape, image_rect):
        offset_x, offset_y, image_width, image_height = image_rect
        source_height, source_width = source_shape[:2]

        def project(box):
            box = _clip_box(box, source_width, source_height)
            if box is None:
                return None
            x1, y1, x2, y2 = box
            return (
                min(
                    offset_x + image_width - 1,
                    round(offset_x + x1 * image_width / source_width),
                ),
                min(
                    offset_y + image_height - 1,
                    round(offset_y + y1 * image_height / source_height),
                ),
                min(
                    offset_x + image_width - 1,
                    round(offset_x + x2 * image_width / source_width),
                ),
                min(
                    offset_y + image_height - 1,
                    round(offset_y + y2 * image_height / source_height),
                ),
            )

        labels = []
        for track in result.get("tracks", ()):
            track_id = track["track_id"]
            box = project(track.get("head_bbox"))
            if box is None:
                continue
            color = _track_color(track_id)
            points = self._trails.get(track_id, ())
            if len(points) > 1:
                points = np.array(
                    [
                        (
                            round(offset_x + x * image_width),
                            round(offset_y + y * image_height),
                        )
                        for x, y in points
                    ],
                    dtype=np.int32,
                )
                cv2.polylines(camera, [points], False, color, 1, cv2.LINE_AA)
            x1, y1, x2, y2 = box
            cv2.rectangle(camera, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
            face = project(track.get("face_bbox"))
            if face is not None:
                cv2.rectangle(camera, face[:2], face[2:], WHITE, 1, cv2.LINE_AA)
            state = track.get("recognition_state", "pending")
            if state == "identified" and self._needs_review(
                track_id, track.get("person_id")
            ):
                state = "review"
            name = track.get("display_name")
            status = {
                "no_usable_face": "No usable face",
                "pending": "Pending",
                "identified": str(name) if name else "Identified",
                "unknown": "Unknown",
                "stale": f"Stale: {name}" if name else "Stale",
                "review": "Needs review",
            }.get(state, "Pending")
            label, scale = _fit(f"Track {track_id} | {status}", 406, 0.5)
            size, baseline = cv2.getTextSize(label, FONT, scale, 1)
            label_width, label_height = size[0] + 14, size[1] + baseline + 12
            left = max(0, min(x1, camera.shape[1] - label_width - 1))
            top = y1 - label_height - 3 if y1 >= label_height + 3 else y2 + 3
            top = max(0, min(top, camera.shape[0] - label_height - 1))
            label_color = AMBER if state in ("unknown", "stale", "review") else WHITE
            if state == "identified":
                label_color = (177, 224, 181)
            labels.append(
                (
                    left,
                    top,
                    label_width,
                    label_height,
                    label,
                    scale,
                    color,
                    label_color,
                    size[1],
                )
            )
        for (
            left,
            top,
            width,
            height,
            label,
            scale,
            color,
            label_color,
            text_height,
        ) in labels:
            cv2.rectangle(camera, (left, top), (left + width, top + height), INK, -1)
            cv2.rectangle(camera, (left, top), (left + 3, top + height), color, -1)
            _text(
                camera,
                label,
                left + 8,
                top + text_height + 5,
                width - 12,
                scale,
                label_color,
            )

    def _draw_header(self, canvas, result):
        cv2.rectangle(canvas, (0, 0), (WIDTH - 1, 87), WHITE, -1)
        cv2.rectangle(canvas, (24, 20), (28, 69), TEAL, -1)
        _text(canvas, "CAMEX | Face ID", 42, 48, 414, 0.9, INK, 2)
        _text(canvas, "Single-camera preview", 42, 71, 414, 0.43, MUTED)
        _text(canvas, self.camera_name, 480, 45, 690, 0.67)
        _text(
            canvas, f"Camera {result.get('camera_id', '--')}", 480, 69, 690, 0.43, MUTED
        )
        timestamp = _number(result.get("timestamp_ms")) or 0
        seconds = (self._clock_seconds + max(0, int(timestamp)) // 1000) % 86400
        clock = f"{seconds // 3600:02d}:{seconds // 60 % 60:02d}:{seconds % 60:02d}"
        _text(canvas, clock, 1210, 46, 222, 0.82)
        _text(canvas, "Camera clock", 1210, 69, 222, 0.4, MUTED)
        _text(
            canvas,
            f"Clip {_format_ms(result.get('timestamp_ms'))}",
            1456,
            46,
            440,
            0.61,
        )
        _text(canvas, "Clip-relative time", 1456, 69, 440, 0.4, MUTED)

    def _draw_evidence(self, canvas):
        x, y, width, height = PANEL_RECT
        cv2.rectangle(canvas, (x, y), (x + width - 1, y + height - 1), WHITE, -1)
        _text(canvas, "RECENT CONFIRMATIONS", 1472, 132, 408, 0.59)
        _text(canvas, "Last good captures | clip time", 1472, 153, 408, 0.42, MUTED)
        _text(canvas, "Observed", 1472, 176, 92, 0.4, MUTED)
        _text(canvas, "Reference", 1572, 176, 92, 0.4, MUTED)
        if not self._evidence:
            _text(canvas, "No captured evidence yet", 1472, 245, 408, 0.55, MUTED)
        for index, (track_id, row) in enumerate(self._evidence.items()):
            top = 184 + index * 140
            review = row.get("revoked", False)
            cv2.rectangle(
                canvas,
                (1461, top + 5),
                (1464, top + 111),
                AMBER if review else _track_color(track_id),
                -1,
            )
            canvas[top + 5 : top + 97, 1472:1564] = row["crop"]
            canvas[top + 5 : top + 97, 1572:1664] = row["reference"]
            _name(
                canvas,
                row["display_name"]
                or ("Historical identity" if review else "Identified"),
                1672,
                top + 18,
                208,
            )
            _text(
                canvas,
                f"Track {track_id}",
                1672,
                top + 60,
                208,
                0.46,
                AMBER if review else TEAL,
            )
            _text(
                canvas,
                f"Captured {_format_ms(row['timestamp_ms'])}",
                1672,
                top + 81,
                208,
                0.4,
                MUTED,
            )
            _text(
                canvas,
                f"Similarity {_similarity(row['similarity'])}",
                1672,
                top + 104,
                208,
                0.45,
                AMBER if review else GREEN,
            )
            if review:
                _text(canvas, "Needs review", 1472, top + 118, 192, 0.44, AMBER)
            cv2.line(canvas, (1472, top + 129), (1880, top + 129), RULE, 1)

    def _draw_timeline(self, canvas):
        _text(canvas, "CONFIRMATION TIMELINE", 24, 925, 1872, 0.43, MUTED)
        if not self._timeline:
            _text(canvas, "No confirmation events received", 24, 976, 1380, 0.51, MUTED)
        for index, event in enumerate(reversed(list(self._timeline)[-4:])):
            left = 24 + index * 474
            revoked = event.get("event") == "identity_revoked"
            cv2.line(
                canvas,
                (left, 941),
                (left, 1007),
                AMBER if revoked else _track_color(event.get("track_id")),
                2,
            )
            _text(
                canvas,
                f"{_format_ms(event.get('timestamp_ms'))} | Track {event.get('track_id', '--')}",
                left + 12,
                953,
                438,
                0.43,
                MUTED,
            )
            _text(
                canvas,
                event.get("display_name")
                or ("Needs review" if revoked else "Identity confirmed"),
                left + 12,
                978,
                438,
                0.56,
            )
            status = (
                "Identity revoked | Needs review"
                if revoked
                else f"Identity confirmed | Similarity {_similarity(event.get('similarity'))}"
            )
            _text(
                canvas, status, left + 12, 1002, 438, 0.41, AMBER if revoked else TEAL
            )

    def _draw_footer(self, canvas, result):
        cv2.rectangle(canvas, (0, 1032), (WIDTH - 1, HEIGHT - 1), INK, -1)
        fps = (
            1000 * len(self._processing_times) / sum(self._processing_times)
            if self._processing_times
            else None
        )
        fps_text = "--" if fps is None else f"{fps:.1f}"
        processing = _number(result.get("processing_ms"))
        ms_text = "--" if processing is None or processing <= 0 else f"{processing:.1f}"
        _text(
            canvas,
            "Pipeline | Head tracking > Face quality > Identity",
            24,
            1062,
            650,
            0.48,
            WHITE,
        )
        for text, x, width in (
            (f"Processing fps {fps_text}", 710, 226),
            (f"Processing {ms_text} ms", 958, 225),
            (f"Processed frames {self._processed_frames}", 1205, 235),
            ("Playback 10 FPS", 1462, 204),
            (f"Sequence {result.get('sequence', '--')}", 1688, 208),
        ):
            _text(canvas, text, x, 1062, width, 0.46, WHITE)

    def render(self, frame_bgr, result, evidence):
        """Return an owned BGR frame without modifying any supplied arrays/dicts."""
        if not _valid_image(frame_bgr):
            raise ValueError("frame_bgr must be a non-empty HxWx3 uint8 BGR array")
        camera_id = result.get("camera_id")
        if camera_id != self._camera_id:
            self._reset()
            self._camera_id = camera_id
        self._advance(result, frame_bgr.shape)
        self._remember_evidence(evidence)
        self._review_evidence(result, evidence)
        canvas = np.full((HEIGHT, WIDTH, 3), BACKGROUND, dtype=np.uint8)
        x, y, width, height = CAMERA_RECT
        camera = canvas[y : y + height, x : x + width]
        camera[:] = INK
        image_rect = _place_image(camera, frame_bgr, (0, 0, width, height))
        self._draw_tracks(camera, result, frame_bgr.shape, image_rect)
        self._draw_header(canvas, result)
        self._draw_evidence(canvas)
        self._draw_timeline(canvas)
        self._draw_footer(canvas, result)
        return canvas
