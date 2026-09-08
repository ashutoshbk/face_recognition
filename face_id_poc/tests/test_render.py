"""Synthetic-only compositor tests, runnable with unittest or pytest."""

import copy
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from camex_faceid.render import AMBER, CAMERA_RECT, GREEN, INK, Renderer, _track_color


def result(sequence=0, timestamp_ms=0, processing_ms=20, tracks=None, events=None):
    return {
        "camera_id": "camera-1",
        "timestamp_ms": timestamp_ms,
        "sequence": sequence,
        "processing_ms": processing_ms,
        "tracks": tracks or [],
        "events": events or [],
    }


def track(track_id=7, state="pending", name=None):
    return {
        "track_id": track_id,
        "head_bbox": [100, 80, 190, 180],
        "head_score": 0.91,
        "face_bbox": [120, 100, 170, 150],
        "recognition_state": state,
        "person_id": "person-7" if name else None,
        "display_name": name,
        "similarity": 0.734 if name else None,
        "last_face_confirmed_ms": 500 if name else None,
    }


def confirmation(track_id=7, timestamp_ms=500, name="Synthetic Name"):
    return {
        "event": "identity_confirmed",
        "track_id": track_id,
        "person_id": f"person-{track_id}",
        "display_name": name,
        "timestamp_ms": timestamp_ms,
        "similarity": 0.734,
    }


def evidence_item(timestamp_ms=500, name="Synthetic Name"):
    return {
        "crop": np.full((80, 80, 3), (25, 80, 160), np.uint8),
        "reference": np.full((120, 120, 3), (130, 45, 30), np.uint8),
        "timestamp_ms": timestamp_ms,
        "person_id": "person-7",
        "display_name": name,
        "similarity": 0.734,
    }


class RendererTests(unittest.TestCase):
    def setUp(self):
        self.frame = np.full((360, 640, 3), (20, 45, 80), np.uint8)

    def render_text(self, renderer, value, evidence=None):
        with patch("camex_faceid.render.cv2.putText", wraps=cv2.putText) as put_text:
            output = renderer.render(self.frame, value, evidence or {})
        return output, [call.args[1] for call in put_text.call_args_list]

    def test_empty_frame_shape_bgr_and_no_letterbox_or_ui(self):
        self.frame[0, 0] = (250, 10, 30)
        original = self.frame.copy()
        value = result()
        with patch("cv2.imshow", side_effect=AssertionError("No UI")), patch(
            "cv2.VideoWriter", side_effect=AssertionError("No video I/O")
        ):
            output, texts = self.render_text(Renderer(), value)
        self.assertEqual(output.shape, (1080, 1920, 3))
        self.assertEqual(output.dtype, np.uint8)
        self.assertTrue(output.flags.c_contiguous)
        self.assertFalse(np.shares_memory(output, self.frame))
        np.testing.assert_array_equal(original, self.frame)
        x, y, width, height = CAMERA_RECT
        np.testing.assert_array_equal(
            output[y : y + height, x : x + width],
            cv2.resize(self.frame, (width, height)),
        )
        self.assertIn("No captured evidence yet", texts)
        self.assertIn("No confirmation events received", texts)
        self.assertIn("Playback 10 FPS", texts)
        self.assertFalse(any("people" in text.lower() for text in texts))

    def test_evidence_and_event_empty_states_are_independent(self):
        _, texts = self.render_text(Renderer(), result(events=[confirmation()]))
        self.assertIn("No captured evidence yet", texts)
        self.assertNotIn("No confirmation events received", texts)
        self.assertIn("Synthetic Name", texts)
        _, texts = self.render_text(Renderer(), result(), {7: evidence_item()})
        self.assertNotIn("No captured evidence yet", texts)
        self.assertIn("No confirmation events received", texts)
        self.assertIn("Captured 00:00:00.500", texts)

    def test_all_states_have_truthful_head_labels(self):
        renderer = Renderer()
        for sequence, (state, expected) in enumerate(
            (
                ("pending", "Pending"),
                ("no_usable_face", "No usable face"),
                ("unknown", "Unknown"),
                ("identified", "Synthetic Name"),
                ("stale", "Stale: Synthetic Name"),
            )
        ):
            with self.subTest(state=state):
                value = result(
                    sequence, tracks=[track(state=state, name="Synthetic Name")]
                )
                _, texts = self.render_text(renderer, value)
                self.assertIn(f"Track 7 | {expected}", texts)
                if state in ("pending", "no_usable_face", "unknown"):
                    self.assertFalse(any("Synthetic Name" in text for text in texts))
        _, texts = self.render_text(
            renderer, result(6, tracks=[track(state="identified")])
        )
        self.assertIn("Track 7 | Identified", texts)
        self.assertFalse(any("person-" in text for text in texts))

    def test_long_names_and_edge_labels_fit_their_regions(self):
        name = "Alexandria Verylongfamilyname " * 40
        tracks = []
        for index, box in enumerate(
            ([0, 0, 40, 40], [590, 300, 639, 359], [250, 20, 350, 110])
        ):
            item = track(index, "identified", name if index < 2 else "W" * 400)
            item["head_bbox"] = box
            item["face_bbox"] = None
            tracks.append(item)
        evidence = {
            index: dict(evidence_item(index * 100, name), person_id=f"person-{index}")
            for index in range(5)
        }
        value = result(
            tracks=tracks,
            events=[confirmation(index, index * 100, name) for index in range(4)],
        )
        original_put_text = cv2.putText
        texts = []

        def checked_text(image, text, origin, font, scale, color, thickness, line_type):
            (width, height), baseline = cv2.getTextSize(text, font, scale, thickness)
            x, y = origin
            self.assertGreaterEqual(x, 0)
            self.assertGreaterEqual(y - height, 0)
            self.assertLess(x + width, image.shape[1])
            self.assertLess(y + baseline, image.shape[0])
            if image.shape[:2] == (1080, 1920) and x == 1672:
                self.assertLessEqual(width, 208)
            texts.append(text)
            return original_put_text(
                image, text, origin, font, scale, color, thickness, line_type
            )

        with patch("camex_faceid.render.cv2.putText", side_effect=checked_text):
            output = Renderer(camera_name="Reception " * 100).render(
                self.frame, value, evidence
            )
        self.assertEqual(output.shape, (1080, 1920, 3))
        self.assertTrue(any(text.endswith("...") for text in texts))
        self.assertTrue(any(text.startswith("Similarity ") for text in texts))
        self.assertFalse(
            any("%" in text or "probability" in text.lower() for text in texts)
        )

    def test_processing_fps_uses_unique_results_not_playback_time(self):
        renderer = Renderer()
        _, texts = self.render_text(renderer, result(10, 0, 10, [track()]))
        self.assertIn("Processing fps 100.0", texts)
        _, texts = self.render_text(renderer, result(10, 9000, 1000, [track()]))
        self.assertIn("Processing fps 100.0", texts)
        self.assertEqual(renderer._processed_frames, 1)
        self.assertEqual(len(renderer._trails[7]), 1)
        _, texts = self.render_text(renderer, result(20, 9100, 30, [track()]))
        self.assertIn("Processing fps 50.0", texts)
        self.assertIn("Playback 10 FPS", texts)
        self.assertEqual(renderer._processed_frames, 2)
        self.assertEqual(list(renderer._processing_times), [10, 30])
        self.render_text(renderer, result(10, 0, 200))
        self.assertEqual(renderer._processed_frames, 2)

    def test_invalid_processing_times_do_not_create_infinite_fps(self):
        renderer = Renderer()
        for sequence, processing_ms in enumerate(
            (0, -1, None, float("nan"), float("inf"))
        ):
            _, texts = self.render_text(
                renderer, result(sequence, processing_ms=processing_ms)
            )
            self.assertIn("Processing fps --", texts)
        self.assertFalse(renderer._processing_times)

    def test_clock_and_captured_timestamp_are_independent(self):
        renderer = Renderer(clock_start="23:59:59")
        _, texts = self.render_text(renderer, result(1, 2500), {7: evidence_item(500)})
        self.assertIn("00:00:01", texts)
        self.assertIn("Clip 00:00:02.500", texts)
        self.assertIn("Captured 00:00:00.500", texts)
        self.assertNotIn("Captured 00:00:02.500", texts)
        self.assertIn("Similarity 0.734", texts)

    def test_evidence_is_owned_last_good_and_survives_absent_tracks(self):
        renderer = Renderer()
        evidence = {7: evidence_item()}
        value = result(tracks=[track(state="identified", name="Synthetic Name")])
        original_result = copy.deepcopy(value)
        renderer.render(self.frame, value, evidence)
        self.assertEqual(value, original_result)
        np.testing.assert_array_equal(
            evidence[7]["crop"], np.full((80, 80, 3), (25, 80, 160), np.uint8)
        )
        saved_crop = renderer._evidence[7]["crop"].copy()
        saved_reference = renderer._evidence[7]["reference"].copy()
        evidence[7]["crop"][:] = 0
        evidence[7]["reference"][:] = 0
        self.frame[:] = 255
        renderer.render(self.frame, result(1, 5000), {})
        np.testing.assert_array_equal(renderer._evidence[7]["crop"], saved_crop)
        np.testing.assert_array_equal(
            renderer._evidence[7]["reference"], saved_reference
        )
        renderer.render(self.frame, result(2, 5100), evidence)
        np.testing.assert_array_equal(renderer._evidence[7]["crop"], saved_crop)
        renderer.render(self.frame, result(3, 5200), {7: evidence_item(100)})
        self.assertEqual(renderer._evidence[7]["timestamp_ms"], 500)
        newer = evidence_item(5100, "Updated Name")
        newer["crop"][:] = (1, 2, 3)
        renderer.render(self.frame, result(4, 5300), {7: newer})
        self.assertEqual(renderer._evidence[7]["display_name"], "Updated Name")
        np.testing.assert_array_equal(renderer._evidence[7]["crop"][45, 45], (1, 2, 3))

    def test_evidence_rows_are_stable_bounded_and_recent(self):
        renderer = Renderer()
        evidence = {
            index: dict(evidence_item(index * 100), person_id=f"person-{index}")
            for index in range(7)
        }
        renderer.render(self.frame, result(), evidence)
        self.assertEqual(list(renderer._evidence), [6, 5, 4, 3, 2])
        renderer.render(self.frame, result(1), dict(reversed(list(evidence.items()))))
        self.assertEqual(list(renderer._evidence), [6, 5, 4, 3, 2])
        evidence[3] = dict(evidence_item(800), person_id="person-3")
        renderer.render(self.frame, result(2), evidence)
        self.assertEqual(list(renderer._evidence), [6, 5, 4, 3, 2])
        evidence[7] = evidence_item(900)
        renderer.render(self.frame, result(3), evidence)
        self.assertEqual(list(renderer._evidence), [7, 6, 5, 4, 3])
        renderer.render(self.frame, result(4), evidence)
        self.assertEqual(list(renderer._evidence), [7, 6, 5, 4, 3])

    def test_reconfirmation_on_new_track_replaces_person_row_not_events(self):
        renderer = Renderer()
        old = evidence_item(500)
        renderer.render(self.frame, result(events=[confirmation(7, 500)]), {7: old})
        new = evidence_item(1500)
        new["crop"][:] = (1, 2, 3)
        new["reference"][:] = (4, 5, 6)
        event = dict(confirmation(42, 1500), person_id="person-7")
        _, texts = self.render_text(
            renderer, result(1, 1500, events=[event]), {42: new, 7: old}
        )
        self.assertEqual(list(renderer._evidence), [42])
        self.assertEqual(renderer._evidence[42]["timestamp_ms"], 1500)
        np.testing.assert_array_equal(renderer._evidence[42]["crop"][45, 45], (1, 2, 3))
        np.testing.assert_array_equal(
            renderer._evidence[42]["reference"][45, 45], (4, 5, 6)
        )
        self.assertIn("Track 42", texts)
        self.assertNotIn("Track 7", texts)
        self.assertIn("Captured 00:00:01.500", texts)
        self.assertEqual([event["track_id"] for event in renderer._timeline], [7, 42])
        for sequence, incoming in enumerate(({}, {7: old}, {7: old, 42: new}), 2):
            renderer.render(self.frame, result(sequence, 2000), incoming)
            self.assertEqual(list(renderer._evidence), [42])
            self.assertEqual(len(renderer._timeline), 2)

    def test_person_compaction_precedes_panel_limit(self):
        renderer = Renderer()
        evidence = {index: evidence_item(1000 + index) for index in range(7)}
        evidence.update(
            {
                index: dict(evidence_item(index), person_id=f"person-{index}")
                for index in range(10, 14)
            }
        )
        renderer.render(self.frame, result(), evidence)
        self.assertEqual(list(renderer._evidence), [6, 13, 12, 11, 10])
        self.assertEqual(
            len({row["person_id"] for row in renderer._evidence.values()}), 5
        )

    def test_equal_names_and_missing_person_ids_do_not_merge(self):
        renderer = Renderer()
        evidence = {
            1: dict(evidence_item(100), person_id="first-person"),
            2: dict(evidence_item(200), person_id="second-person"),
            3: dict(evidence_item(300), person_id=None),
            4: dict(evidence_item(400), person_id=None),
        }
        renderer.render(self.frame, result(), evidence)
        self.assertEqual(list(renderer._evidence), [4, 3, 2, 1])

    def test_compaction_keeps_newest_review_row_not_older_confirmed_pair(self):
        renderer = Renderer()
        old = evidence_item(500)
        new = dict(evidence_item(1000), revoked=True)
        _, texts = self.render_text(renderer, result(), {7: old, 42: new})
        self.assertEqual(list(renderer._evidence), [42])
        self.assertTrue(renderer._evidence[42]["revoked"])
        self.assertIn("Needs review", texts)
        renderer.render(self.frame, result(1), {7: old})
        self.assertEqual(list(renderer._evidence), [42])
        self.assertTrue(renderer._evidence[42]["revoked"])

    def test_revoked_flag_preserves_audit_and_removes_green_styling(self):
        renderer = Renderer()
        item = evidence_item()
        renderer.render(self.frame, result(), {7: item})
        saved = copy.deepcopy(renderer._evidence[7])
        item["revoked"] = True
        with patch("camex_faceid.render.cv2.putText", wraps=cv2.putText) as put_text:
            renderer.render(self.frame, result(1, 1000), {7: item})
        panel_text = [
            call.args
            for call in put_text.call_args_list
            if call.args[2][0] >= 1456 and 184 <= call.args[2][1] < 896
        ]
        self.assertTrue(
            any(args[1] == "Needs review" and args[5] == AMBER for args in panel_text)
        )
        self.assertFalse(any(args[5] == GREEN for args in panel_text))
        self.assertTrue(renderer._evidence[7]["revoked"])
        self.assertEqual(renderer._evidence[7]["timestamp_ms"], 500)
        self.assertEqual(renderer._evidence[7]["display_name"], saved["display_name"])
        np.testing.assert_array_equal(renderer._evidence[7]["crop"], saved["crop"])
        np.testing.assert_array_equal(
            renderer._evidence[7]["reference"], saved["reference"]
        )
        for sequence, incoming in enumerate(
            ({}, {7: evidence_item()}, {7: evidence_item(1500)}), 2
        ):
            _, texts = self.render_text(renderer, result(sequence, 2000), incoming)
            self.assertIn("Needs review", texts)
            self.assertTrue(renderer._evidence[7]["revoked"])

    def test_revocation_event_and_confirmation_do_not_deduplicate_each_other(self):
        renderer = Renderer()
        confirmed = confirmation()
        revoked = dict(confirmed, event="identity_revoked")
        _, texts = self.render_text(
            renderer, result(events=[confirmed, revoked]), {7: evidence_item()}
        )
        self.assertEqual(len(renderer._timeline), 2)
        self.assertIn("Identity revoked | Needs review", texts)
        self.assertIn("Needs review", texts)
        _, texts = self.render_text(renderer, result(1, 1000))
        self.assertTrue(renderer._evidence[7]["revoked"])
        self.assertIn("Captured 00:00:00.500", texts)
        self.assertIn("Needs review", texts)

    def test_review_signal_before_evidence_arrival_is_retained(self):
        for signal in ("event", "flag"):
            with self.subTest(signal=signal):
                renderer = Renderer()
                events = (
                    [dict(confirmation(), event="identity_revoked")]
                    if signal == "event"
                    else []
                )
                evidence = (
                    {7: {"revoked": True, "person_id": "person-7"}}
                    if signal == "flag"
                    else {}
                )
                renderer.render(self.frame, result(events=events), evidence)
                _, texts = self.render_text(
                    renderer, result(1, 1000), {7: evidence_item()}
                )
                self.assertIn("Needs review", texts)
                self.assertTrue(renderer._evidence[7]["revoked"])

    def test_suspended_or_changed_identity_marks_cached_row_for_review(self):
        for state, person_id in (
            ("pending", None),
            ("unknown", None),
            ("pending", "other-person"),
            ("identified", "other-person"),
        ):
            with self.subTest(state=state, person_id=person_id):
                renderer = Renderer()
                renderer.render(self.frame, result(), {7: evidence_item()})
                current = track(state=state)
                current["person_id"] = person_id
                _, texts = self.render_text(renderer, result(1, 1000, tracks=[current]))
                self.assertIn("Needs review", texts)
                _, texts = self.render_text(renderer, result(2, 2000))
                self.assertIn("Needs review", texts)
                self.assertEqual(renderer._evidence[7]["person_id"], "person-7")

    def test_revocation_does_not_taint_another_identity_or_camera(self):
        renderer = Renderer()
        revoked = dict(confirmation(), event="identity_revoked")
        renderer.render(self.frame, result(events=[revoked]), {7: evidence_item()})
        new_item = dict(evidence_item(1000), person_id="other-person")
        current = dict(
            track(state="identified", name="New identity"), person_id="other-person"
        )
        renderer.render(self.frame, result(1, 1000, tracks=[current]), {7: new_item})
        self.assertFalse(renderer._evidence[7]["revoked"])
        renderer.render(
            self.frame, dict(result(), camera_id="camera-2"), {7: evidence_item()}
        )
        self.assertFalse(renderer._evidence[7]["revoked"])
        self.assertFalse(renderer._revocations)

    def test_event_timeline_persists_deduplicates_and_is_bounded(self):
        renderer = Renderer()
        event = confirmation()
        renderer.render(self.frame, result(events=[event, event]), {})
        renderer.render(self.frame, result(1, events=[event]), {})
        self.assertEqual(len(renderer._timeline), 1)
        event["display_name"] = "Changed outside renderer"
        _, texts = self.render_text(renderer, result(2, 5000))
        self.assertIn("Synthetic Name", texts)
        self.assertEqual(len(renderer._timeline), 1)
        events = [confirmation(index, 1000 + index) for index in range(200)]
        renderer.render(self.frame, result(3, events=events), {})
        self.assertEqual(len(renderer._timeline), renderer.MAX_EVENTS)
        self.assertLessEqual(len(renderer._event_keys), renderer.MAX_EVENTS * 4)
        self.assertEqual(renderer._timeline[-1]["track_id"], 199)

    def test_trails_and_processing_samples_are_bounded(self):
        renderer = Renderer()
        for sequence in range(140):
            renderer._advance(
                result(sequence, sequence * 10, tracks=[track()]), self.frame.shape
            )
        self.assertEqual(len(renderer._trails[7]), renderer.TRAIL_LENGTH)
        self.assertEqual(len(renderer._processing_times), 120)
        renderer.render(
            self.frame,
            result(141, 1500, tracks=[track(index) for index in range(80)]),
            {},
        )
        self.assertEqual(len(renderer._trails), renderer.MAX_TRAILS)
        self.assertEqual(len(renderer._track_seen), renderer.MAX_TRAILS)
        renderer.render(self.frame, result(142, 10000), {})
        self.assertFalse(renderer._trails)
        self.assertFalse(renderer._track_seen)

    def test_track_colors_are_stable_and_face_boxes_are_thin(self):
        renderer = Renderer()
        with patch(
            "camex_faceid.render.cv2.rectangle", wraps=cv2.rectangle
        ) as rectangles:
            renderer.render(self.frame, result(tracks=[track()]), {})
        boxes = [
            call.args
            for call in rectangles.call_args_list
            if call.args[0].shape[:2] == (792, 1408)
        ]
        self.assertEqual(boxes[0][3], _track_color(7))
        self.assertEqual(boxes[0][4], 2)
        self.assertEqual(boxes[1][4], 1)
        with patch(
            "camex_faceid.render.cv2.rectangle", wraps=cv2.rectangle
        ) as rectangles:
            renderer.render(
                self.frame,
                result(1, tracks=[track(state="identified", name="Synthetic Name")]),
                {},
            )
        boxes = [
            call.args
            for call in rectangles.call_args_list
            if call.args[0].shape[:2] == (792, 1408)
        ]
        self.assertEqual(boxes[0][3], _track_color(7))

    def test_camera_change_clears_identity_state(self):
        renderer = Renderer()
        renderer.render(
            self.frame,
            result(10, tracks=[track()], events=[confirmation()]),
            {7: evidence_item()},
        )
        value = result()
        value["camera_id"] = "camera-2"
        renderer.render(self.frame, value, {})
        self.assertFalse(renderer._evidence)
        self.assertFalse(renderer._timeline)
        self.assertFalse(renderer._trails)
        self.assertEqual(renderer._processed_frames, 1)

    def test_non_widescreen_input_is_not_distorted(self):
        frame = np.full((100, 100, 3), (10, 20, 30), np.uint8)
        output = Renderer().render(frame, result(), {})
        x, y, width, height = CAMERA_RECT
        np.testing.assert_array_equal(output[y + 100, x + 10], INK)
        np.testing.assert_array_equal(output[y + 100, x + width // 2], (10, 20, 30))
        np.testing.assert_array_equal(
            output[y + height - 1, x + width // 2], (10, 20, 30)
        )

    def test_bad_or_clipped_boxes_and_missing_reference_do_not_crash(self):
        tracks = []
        for index, box in enumerate(
            (
                None,
                [],
                [0, 0, float("nan"), 2],
                [-10, -10, 50, 50],
                [500, 250, 900, 500],
                [900, 900, 950, 950],
                [30, 30, 10, 10],
            )
        ):
            item = track(index)
            item["head_bbox"] = box
            tracks.append(item)
        evidence = evidence_item()
        evidence["reference"] = None
        output = Renderer().render(self.frame, result(tracks=tracks), {7: evidence})
        self.assertEqual(output.shape, (1080, 1920, 3))

    def test_invalid_source_is_rejected(self):
        for frame in (
            None,
            np.zeros((0, 20, 3), np.uint8),
            np.zeros((20, 20), np.uint8),
            np.zeros((20, 20, 3), np.float32),
        ):
            with self.subTest(frame_type=type(frame)), self.assertRaises(ValueError):
                Renderer().render(frame, result(), {})


if __name__ == "__main__":
    unittest.main()
