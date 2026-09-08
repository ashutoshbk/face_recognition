"""Model-free tests; runnable with unittest or pytest (no model downloads)."""

import unittest
from types import SimpleNamespace

import numpy as np

from camex_faceid.tracking import HeadTracker
from camex_faceid.vendor.bytetrack import UPSTREAM_COMMIT, UPSTREAM_REPOSITORY
from camex_faceid.vendor.bytetrack.byte_tracker import BYTETracker, STrack
from camex_faceid.vendor.bytetrack import matching


FRAME = (480, 640, 3)
HEAD = [100, 80, 150, 140, 0.9]


def detections(*rows):
    return np.asarray(rows, dtype=float).reshape(-1, 5)


def centered_head(cx=200, cy=200, width=40, height=50, score=0.6):
    return [cx - width / 2, cy - height / 2, cx + width / 2, cy + height / 2, score]


class HeadTrackerTests(unittest.TestCase):
    def test_continuity_and_output_contract(self):
        tracker = HeadTracker()
        for frame in range(8):
            row = [100 + frame * 2, 80, 150 + frame * 2, 140, 0.91]
            output = tracker.update(detections(row), FRAME, frame * 100)
            self.assertEqual(len(output), 1)
            track = output[0]
            self.assertEqual(set(track), {"track_id", "bbox", "score"})
            self.assertIs(type(track["track_id"]), int)
            self.assertEqual(track["track_id"], 1)
            self.assertIs(type(track["score"]), float)
            self.assertEqual(track["score"], row[4])
            self.assertIs(type(track["bbox"]), list)
            self.assertEqual(len(track["bbox"]), 4)
            self.assertTrue(all(type(value) is float for value in track["bbox"]))
            np.testing.assert_allclose(track["bbox"], row[:4], atol=2)

    def test_dropout_recovery_without_predicted_ghosts(self):
        tracker = HeadTracker()
        original = tracker.update(detections(HEAD), FRAME, 0)[0]["track_id"]
        for timestamp in (100, 200, 300):
            self.assertEqual(tracker.update(detections(), FRAME, timestamp), [])
        recovered = tracker.update(detections(HEAD), FRAME, 400)
        self.assertEqual([track["track_id"] for track in recovered], [original])
        self.assertEqual(tracker.update(None, FRAME, 500), [])

    def test_low_score_second_association_does_not_birth_tracks(self):
        tracker = HeadTracker()
        tracker.update(detections(HEAD), FRAME, 0)
        weak = HEAD[:4] + [0.2]
        unrelated = [300, 100, 350, 160, 0.3]
        output = tracker.update(detections(weak, unrelated), FRAME, 100)
        self.assertEqual(len(output), 1)
        self.assertEqual(output[0]["track_id"], 1)
        self.assertEqual(output[0]["score"], 0.2)
        self.assertEqual(HeadTracker().update(detections(weak), FRAME, 0), [])

    def test_threshold_is_inclusive_and_controls_births(self):
        row = HEAD[:4] + [0.4]
        tracker = HeadTracker()
        self.assertEqual(tracker.update(detections(row), FRAME, 0)[0]["track_id"], 1)
        self.assertEqual(tracker.update(detections(row), FRAME, 100)[0]["track_id"], 1)

    def test_two_heads_reordered_detections_keep_exclusive_ids(self):
        tracker = HeadTracker()
        left = [40, 60, 100, 120, 0.9]
        right = [320, 60, 380, 120, 0.85]
        first = tracker.update(detections(left, right), FRAME, 0)
        self.assertEqual([track["track_id"] for track in first], [1, 2])
        for frame in range(1, 8):
            left[0] += 2
            left[2] += 2
            right[0] -= 2
            right[2] -= 2
            output = tracker.update(detections(right, left), FRAME, frame * 100)
            self.assertEqual([track["track_id"] for track in output], [1, 2])
            self.assertLess(output[0]["bbox"][0], output[1]["bbox"][0])

    def test_one_observation_cannot_update_two_tracks(self):
        tracker = HeadTracker()
        tracker.update(
            detections([10, 10, 40, 50, 0.95], [50, 10, 80, 50, 0.95]), FRAME, 0
        )
        output = tracker.update(detections([25, 10, 65, 50, 0.95]), FRAME, 100)
        self.assertEqual(output, [])
        self.assertTrue(tracker.association_ambiguous)
        self.assertEqual(tracker.update([], FRAME, 200), [])

    def test_two_instances_have_independent_counters_and_lifecycles(self):
        first, second = HeadTracker(), HeadTracker()
        self.assertEqual(first.update(detections(HEAD), FRAME, 0)[0]["track_id"], 1)
        first.update(detections(HEAD, [300, 60, 350, 120, 0.9]), FRAME, 100)
        self.assertEqual(second.update(detections(HEAD), FRAME, 10)[0]["track_id"], 1)
        self.assertEqual(
            [
                t["track_id"]
                for t in first.update(
                    detections(HEAD, [300, 60, 350, 120, 0.9]), FRAME, 200
                )
            ],
            [1, 2],
        )
        self.assertEqual(first.update(detections(HEAD), FRAME, 2300)[0]["track_id"], 3)
        self.assertEqual(second.update(detections(HEAD), FRAME, 110)[0]["track_id"], 1)
        self.assertEqual(
            HeadTracker().update(detections(HEAD), FRAME, 0)[0]["track_id"], 1
        )

    def test_empty_forms_and_confirmation_after_initial_empty_frame(self):
        for empty in (None, [], (), np.empty((0, 5)), np.array([])):
            with self.subTest(empty=repr(empty)):
                tracker = HeadTracker()
                self.assertEqual(tracker.update(empty, FRAME, 0), [])
                self.assertEqual(tracker.update(detections(HEAD), FRAME, 100), [])
                self.assertEqual(
                    tracker.update(detections(HEAD), FRAME, 200)[0]["track_id"], 1
                )

    def test_new_arrival_needs_confirmation_after_first_frame(self):
        tracker = HeadTracker()
        tracker.update(detections(HEAD), FRAME, 0)
        new_head = [300, 100, 360, 160, 0.9]
        self.assertEqual(tracker.update(detections(new_head), FRAME, 100), [])
        self.assertEqual(
            tracker.update(detections(new_head), FRAME, 200)[0]["track_id"], 2
        )

    def test_dropout_buffer_expiry_does_not_reuse_id(self):
        tracker = HeadTracker(buffer_seconds=0.2)
        tracker.update(detections(HEAD), FRAME, 0)
        tracker.update([], FRAME, 100)
        tracker.update([], FRAME, 200)
        self.assertEqual(tracker.update(detections(HEAD), FRAME, 300), [])
        self.assertEqual(tracker.update(detections(HEAD), FRAME, 400)[0]["track_id"], 2)

    def test_source_time_expires_tracks_even_when_frames_are_skipped(self):
        tracker = HeadTracker(buffer_seconds=0.5)
        tracker.update(detections(HEAD), FRAME, 0)
        self.assertEqual(tracker.update(detections(HEAD), FRAME, 600), [])
        self.assertEqual(tracker.update(detections(HEAD), FRAME, 700)[0]["track_id"], 2)

    def test_source_buffer_boundary_can_recover(self):
        tracker = HeadTracker(buffer_seconds=0.5)
        tracker.update(detections(HEAD), FRAME, 0)
        tracker.update([], FRAME, 100)
        self.assertEqual(tracker.update(detections(HEAD), FRAME, 500)[0]["track_id"], 1)

    def test_large_source_gap_resets_but_ids_never_restart(self):
        tracker = HeadTracker(buffer_seconds=10)
        self.assertEqual(tracker.update(detections(HEAD), FRAME, 0)[0]["track_id"], 1)
        self.assertEqual(
            tracker.update(detections(HEAD), FRAME, 2000)[0]["track_id"], 1
        )
        self.assertEqual(
            tracker.update(detections(HEAD), FRAME, 4001)[0]["track_id"], 2
        )
        self.assertEqual(
            tracker.update(detections(HEAD), FRAME, 6102)[0]["track_id"], 3
        )
        self.assertEqual(tracker.update([], FRAME, 8303), [])
        self.assertEqual(tracker.update(detections(HEAD), FRAME, 8403), [])
        self.assertEqual(
            tracker.update(detections(HEAD), FRAME, 8503)[0]["track_id"], 4
        )

    def test_bad_timestamps_are_rejected_without_state_mutation(self):
        tracker = HeadTracker()
        tracker.update(detections(HEAD), FRAME, 100)
        for bad in (100, 99, -1, 100.5, 200.0, True, np.bool_(False), "200", None):
            with self.subTest(timestamp=bad):
                with self.assertRaisesRegex(ValueError, "timestamp_ms"):
                    tracker.update(detections(HEAD), FRAME, bad)
                self.assertEqual(tracker._tracker.frame_id, 1)
                self.assertEqual(tracker._last_timestamp_ms, 100)
        self.assertEqual(
            tracker.update(detections(HEAD), FRAME, np.int64(200))[0]["track_id"], 1
        )
        self.assertEqual(tracker.update([], FRAME, 300), [])
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            tracker.update([], FRAME, 300)

    def test_source_timestamps_can_have_any_integer_origin(self):
        tracker = HeadTracker()
        self.assertEqual(
            tracker.update(detections(HEAD), FRAME, -100)[0]["track_id"], 1
        )
        self.assertEqual(tracker.update(detections(HEAD), FRAME, 0)[0]["track_id"], 1)

    def test_source_pixels_no_rescaling_no_person_aspect_cutoff(self):
        tracker = HeadTracker()
        wide_head = [800, 400, 1100, 460, 0.9]
        output = tracker.update(detections(wide_head), (1080, 1920, 3), 0)
        self.assertEqual(len(output), 1)
        np.testing.assert_allclose(output[0]["bbox"], wide_head[:4])
        self.assertEqual(
            len(tracker.update(detections(wide_head), (1080, 1920), 100)), 1
        )

    def test_input_copy_clipping_and_degenerate_filtering(self):
        tracker = HeadTracker()
        array = np.array(
            [
                [-5, -10, 700, 500, 1],
                [20, 30, 20, 50, 1],
                [30, 40, 10, 50, 1],
                [650, 0, 700, 50, 1],
            ],
            dtype=int,
        )
        original = array.copy()
        array.setflags(write=False)
        output = tracker.update(array, FRAME, 0)
        self.assertEqual(len(output), 1)
        self.assertEqual(output[0]["bbox"], [0.0, 0.0, 640.0, 480.0])
        np.testing.assert_array_equal(array, original)
        output[0]["bbox"][0] = 300
        self.assertEqual(tracker.update(array[:1], FRAME, 100)[0]["bbox"][0], 0.0)

    def test_invalid_detections_do_not_advance_or_reset_state(self):
        bad_inputs = (
            np.empty((0, 4)),
            np.zeros((1, 6)),
            HEAD,
            [[1, 2, 3, 4, float("nan")]],
            [[1, 2, float("inf"), 4, 0.9]],
            [[1, 2, 3, 4, -0.1]],
            [[1, 2, 3, 4, 1.1]],
            [["x", 2, 3, 4, 0.9]],
        )
        tracker = HeadTracker()
        tracker.update(detections(HEAD), FRAME, 0)
        for bad in bad_inputs:
            with self.subTest(detections=repr(bad)):
                with self.assertRaises(ValueError):
                    tracker.update(bad, FRAME, 5000)
                self.assertEqual(tracker._last_timestamp_ms, 0)
                self.assertEqual(tracker._tracker.frame_id, 1)
        self.assertEqual(tracker.update(detections(HEAD), FRAME, 100)[0]["track_id"], 1)

    def test_invalid_frame_shapes(self):
        for shape in (None, (), (480,), (0, 640), (480, -1), (480.0, 640), (True, 640)):
            with self.subTest(shape=shape), self.assertRaises(ValueError):
                HeadTracker().update(detections(HEAD), shape, 0)

    def test_invalid_configuration(self):
        for kwargs in (
            {"fps": 0},
            {"fps": -1},
            {"fps": float("inf")},
            {"fps": True},
            {"threshold": 0},
            {"threshold": 1.1},
            {"threshold": float("nan")},
            {"buffer_seconds": -1},
            {"buffer_seconds": "1"},
            {"match_threshold": -0.1},
            {"match_threshold": 1.1},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                HeadTracker(**kwargs)

    def test_buffer_uses_update_fps_not_upstream_30fps_default(self):
        self.assertEqual(HeadTracker()._tracker.max_time_lost, 10)
        self.assertEqual(
            HeadTracker(fps=15, buffer_seconds=0.5)._tracker.max_time_lost, 7
        )


class HeadGeometryTests(unittest.TestCase):
    def test_five_minutes_of_centered_scale_switches_keep_one_id(self):
        tracker = HeadTracker()
        for frame in range(3000):
            width, height = (40, 50) if frame // 10 % 2 == 0 else (96, 110)
            output = tracker.update(
                detections(centered_head(width=width, height=height)),
                FRAME,
                frame * 100,
            )
            self.assertEqual([t["track_id"] for t in output], [1], f"frame {frame}")
            np.testing.assert_allclose(
                (np.array(output[0]["bbox"][:2]) + output[0]["bbox"][2:]) / 2,
                [200, 200],
                atol=1e-6,
            )

    def test_scale_switch_after_dropout_recovers_without_ghosts(self):
        for first_size, next_size in (((40, 50), (96, 110)), ((96, 110), (40, 50))):
            with self.subTest(first_size=first_size):
                tracker = HeadTracker()
                tracker.update(
                    detections(
                        centered_head(width=first_size[0], height=first_size[1])
                    ),
                    FRAME,
                    0,
                )
                for timestamp in (100, 200, 300):
                    self.assertEqual(tracker.update([], FRAME, timestamp), [])
                output = tracker.update(
                    detections(centered_head(width=next_size[0], height=next_size[1])),
                    FRAME,
                    400,
                )
                self.assertEqual([t["track_id"] for t in output], [1])

    def test_low_score_and_unconfirmed_scale_switches(self):
        tracker = HeadTracker()
        tracker.update(detections(centered_head()), FRAME, 0)
        output = tracker.update(
            detections(centered_head(width=96, height=110, score=0.2)), FRAME, 100
        )
        self.assertEqual([t["track_id"] for t in output], [1])
        self.assertEqual(output[0]["score"], 0.2)

        tracker = HeadTracker()
        tracker.update([], FRAME, 0)
        self.assertEqual(tracker.update(detections(centered_head()), FRAME, 100), [])
        output = tracker.update(
            detections(centered_head(width=96, height=110)), FRAME, 200
        )
        self.assertEqual([t["track_id"] for t in output], [1])

    def test_adjacent_heads_scale_switch_reordering_jitter_and_dropout(self):
        for separation in (24, 42):
            with self.subTest(separation=separation):
                tracker = HeadTracker()
                for frame in range(600):
                    rows, expected = [], []
                    for person in range(2):
                        if person == 0 and 10 <= frame % 40 < 13:
                            continue
                        wide = (frame // 7 + person) % 2
                        width, height = (96, 110) if wide else (40, 50)
                        cx = 200 + person * separation + 0.5 * np.sin(frame / 3)
                        rows.append(centered_head(cx=cx, width=width, height=height))
                        expected.append(person + 1)
                    if frame % 2:
                        rows.reverse()
                    output = tracker.update(detections(*rows), FRAME, frame * 100)
                    self.assertEqual(
                        [t["track_id"] for t in output],
                        expected,
                        f"separation {separation}, frame {frame}",
                    )
                    for track in output:
                        center = (track["bbox"][0] + track["bbox"][2]) / 2
                        self.assertAlmostEqual(
                            center, 200 + (track["track_id"] - 1) * separation, delta=2
                        )

    def test_center_gate_does_not_join_shifted_scale_change(self):
        tracker = HeadTracker()
        tracker.update(detections(centered_head()), FRAME, 0)
        output = tracker.update(
            detections(centered_head(cx=220, width=96, height=110)), FRAME, 100
        )
        self.assertEqual(output, [])

    def test_per_axis_size_gate_cannot_be_bypassed_by_iou(self):
        tracker = HeadTracker()
        tracker.update(detections(centered_head(score=0.9)), FRAME, 0)
        huge = detections(centered_head(width=104, height=50, score=0.9))
        self.assertEqual(tracker.update(huge, FRAME, 100), [])
        self.assertEqual([t["track_id"] for t in tracker.update(huge, FRAME, 200)], [2])

    def test_collision_is_flagged_and_deferred_without_new_identity(self):
        tracker = HeadTracker()
        separate = detections(
            centered_head(cx=180, width=60, height=80),
            centered_head(cx=200, width=60, height=80),
        )
        self.assertEqual(
            [t["track_id"] for t in tracker.update(separate, FRAME, 0)], [1, 2]
        )
        self.assertFalse(tracker.association_ambiguous)
        self.assertEqual(tracker.ambiguous_track_ids, set())
        merged = detections(centered_head(cx=190, width=60, height=80))
        self.assertEqual(tracker.update(merged, FRAME, 100), [])
        self.assertTrue(tracker.association_ambiguous)
        self.assertEqual(tracker.ambiguous_track_ids, {1, 2})
        tracker.ambiguous_track_ids.clear()
        self.assertEqual(tracker.ambiguous_track_ids, {1, 2})
        self.assertEqual(
            [t["track_id"] for t in tracker.update(separate, FRAME, 200)], [1, 2]
        )
        self.assertFalse(tracker.association_ambiguous)
        self.assertEqual(tracker.ambiguous_track_ids, set())
        tracker.update(merged, FRAME, 250)
        self.assertEqual(tracker.ambiguous_track_ids, {1, 2})
        self.assertEqual(tracker.update([], FRAME, 300), [])
        self.assertFalse(tracker.association_ambiguous)
        self.assertEqual(tracker.ambiguous_track_ids, set())

    def test_split_observation_tie_does_not_birth_duplicate_tracks(self):
        tracker = HeadTracker()
        tracker.update(detections(centered_head()), FRAME, 0)
        split = detections(centered_head(cx=195), centered_head(cx=205))
        self.assertEqual(tracker.update(split, FRAME, 100), [])
        self.assertTrue(tracker.association_ambiguous)
        self.assertEqual(tracker.ambiguous_track_ids, {1})
        output = tracker.update(detections(centered_head()), FRAME, 200)
        self.assertEqual([t["track_id"] for t in output], [1])

    def test_ambiguity_ids_exclude_unaffected_observed_heads(self):
        tracker = HeadTracker()
        left = centered_head(cx=180, width=60, height=80)
        right = centered_head(cx=200, width=60, height=80)
        distant = centered_head(cx=400)
        tracker.update(detections(left, right, distant), FRAME, 0)
        output = tracker.update(
            detections(centered_head(cx=190, width=60, height=80), distant), FRAME, 100
        )
        self.assertEqual([t["track_id"] for t in output], [3])
        self.assertEqual(tracker.ambiguous_track_ids, {1, 2})
        output = tracker.update(detections(left, right, distant), FRAME, 200)
        self.assertEqual([t["track_id"] for t in output], [1, 2, 3])
        self.assertEqual(tracker.ambiguous_track_ids, set())

    def test_iou_only_opt_out_retains_original_behavior(self):
        tracker = HeadTracker(geometry_fallback=False)
        tracker.update(detections(centered_head()), FRAME, 0)
        self.assertEqual(
            tracker.update(detections(centered_head(width=96, height=110)), FRAME, 100),
            [],
        )
        self.assertFalse(tracker.association_ambiguous)

    def test_geometry_is_resolution_independent(self):
        for scale in (0.5, 1, 4):
            with self.subTest(scale=scale):
                tracker = HeadTracker()
                for frame in range(40):
                    width, height = (40, 50) if frame % 2 else (96, 110)
                    row = centered_head(
                        cx=202 if frame % 2 else 200, width=width, height=height
                    )
                    row[:4] = [value * scale for value in row[:4]]
                    output = tracker.update(
                        detections(row),
                        (int(480 * scale), int(640 * scale)),
                        frame * 100,
                    )
                    self.assertEqual([t["track_id"] for t in output], [1])

    def test_ambiguity_is_session_local_and_invalid_update_does_not_clear_it(self):
        tracker, other = HeadTracker(), HeadTracker()
        tracker.update(
            detections(
                centered_head(cx=180, width=60, height=80),
                centered_head(cx=200, width=60, height=80),
            ),
            FRAME,
            0,
        )
        tracker.update(
            detections(centered_head(cx=190, width=60, height=80)), FRAME, 100
        )
        self.assertTrue(tracker.association_ambiguous)
        self.assertFalse(other.association_ambiguous)
        self.assertEqual(tracker.ambiguous_track_ids, {1, 2})
        self.assertEqual(other.ambiguous_track_ids, set())
        with self.assertRaises(ValueError):
            tracker.update([], FRAME, 100)
        self.assertTrue(tracker.association_ambiguous)
        self.assertEqual(tracker.ambiguous_track_ids, {1, 2})
        output = tracker.update(detections(centered_head()), FRAME, 2201)
        self.assertEqual([t["track_id"] for t in output], [3])
        self.assertFalse(tracker.association_ambiguous)
        self.assertEqual(tracker.ambiguous_track_ids, set())

    def test_rejected_ratio_stays_rejected_at_match_threshold_one(self):
        tracker = HeadTracker(match_threshold=1)
        tracker.update(detections(centered_head(score=1)), FRAME, 0)
        self.assertEqual(
            tracker.update(
                detections(centered_head(width=100.01, height=50, score=1)), FRAME, 100
            ),
            [],
        )

    def test_geometry_flag_requires_bool(self):
        for invalid in (None, 1, "true"):
            with self.subTest(value=invalid), self.assertRaises(ValueError):
                HeadTracker(geometry_fallback=invalid)

    def test_geometry_cost_uses_last_measurement_size_not_predicted_extent(self):
        track = STrack(np.array([180, 175, 40, 50], dtype=float), 0.9)
        from camex_faceid.vendor.bytetrack.kalman_filter import KalmanFilter

        track.activate(KalmanFilter(), 1, 1)
        track.mean[2] *= 4
        track.mean[3] *= 2
        allowed = STrack(np.array([150, 137.5, 100, 125]), 0.9)
        rejected = STrack(np.array([145, 137.5, 110, 125]), 0.9)
        costs, _, _ = matching.head_geometry_distance([track], [allowed], 0.8)
        self.assertLess(costs[0, 0], 0.8)
        costs, _, _ = matching.head_geometry_distance([track], [rejected], 1)
        self.assertGreater(costs[0, 0], 1)


class VendoredCoreTests(unittest.TestCase):
    def test_pinned_provenance_constants(self):
        self.assertEqual(
            UPSTREAM_REPOSITORY, "https://github.com/FoundationVision/ByteTrack"
        )
        self.assertEqual(UPSTREAM_COMMIT, "d1bf0191adff59bc8fcfeaa0b33d3d1642552a99")

    def test_assignment_is_cost_limited_and_exclusive(self):
        costs = np.array([[0.1, 0.9], [0.2, 0.85], [0.95, 0.95]])
        matches, unmatched_tracks, unmatched_dets = matching.linear_assignment(
            costs, 0.8
        )
        np.testing.assert_array_equal(matches, [[0, 0]])
        self.assertEqual(set(unmatched_tracks), {1, 2})
        self.assertEqual(set(unmatched_dets), {1})

    def test_empty_and_fully_rejected_assignments_have_stable_shapes(self):
        for shape in ((0, 0), (0, 3), (2, 0), (2, 3)):
            with self.subTest(shape=shape):
                matches, unmatched_tracks, unmatched_dets = matching.linear_assignment(
                    np.ones(shape), 0.8
                )
                self.assertEqual(matches.shape, (0, 2))
                self.assertEqual(set(unmatched_tracks), set(range(shape[0])))
                self.assertEqual(set(unmatched_dets), set(range(shape[1])))

    def test_upstream_inclusive_pixel_iou_is_preserved(self):
        iou = matching.ious([[0, 0, 1, 1]], [[1, 1, 2, 2]])
        np.testing.assert_allclose(iou, [[1 / 7]])
        self.assertEqual(matching.ious([], [[1, 1, 2, 2]]).shape, (0, 1))

    def test_expired_raw_track_cannot_be_resurrected(self):
        args = SimpleNamespace(
            track_thresh=0.4, track_buffer=6, match_thresh=0.8, mot20=False
        )
        tracker = BYTETracker(args, frame_rate=10)
        self.assertEqual(tracker.update(detections(HEAD), FRAME, FRAME)[0].track_id, 1)
        tracker.update(detections(), FRAME, FRAME)
        tracker.update(detections(), FRAME, FRAME)
        self.assertEqual(tracker.update(detections(HEAD), FRAME, FRAME), [])
        self.assertNotIn(1, [track.track_id for track in tracker.lost_stracks])
        self.assertEqual(tracker.update(detections(HEAD), FRAME, FRAME)[0].track_id, 2)

    def test_raw_core_instances_do_not_share_counter(self):
        args = SimpleNamespace(
            track_thresh=0.4, track_buffer=30, match_thresh=0.8, mot20=False
        )
        first, second = BYTETracker(args), BYTETracker(args)
        self.assertEqual(first.update(detections(HEAD), FRAME, FRAME)[0].track_id, 1)
        self.assertEqual(second.update(detections(HEAD), FRAME, FRAME)[0].track_id, 1)

    def test_core_does_not_rescale_input_in_place(self):
        args = SimpleNamespace(
            track_thresh=0.4, track_buffer=30, match_thresh=0.8, mot20=False
        )
        array = detections(HEAD)
        original = array.copy()
        array.setflags(write=False)
        BYTETracker(args).update(array, FRAME, (240, 320))
        np.testing.assert_array_equal(array, original)


if __name__ == "__main__":
    unittest.main()
