"""Causal, bounded floor fusion. No file names, camera roles, or identity exceptions."""

from collections import deque
import copy
import json
from pathlib import Path
import threading
import time
import uuid

import numpy as np

from .geometry import Projector, digest, load_site, validate_site


DEFAULTS = {
    "synchronization_ms": 250,
    "active_ms": 1000,
    "retention_ms": 1800000,
    "track_retention_ms": 30000,
    "evidence_retention_ms": 10000,
    "pair_distance_m": 0.9,
    "merge_distance_m": 0.65,
    "appearance_minimum": 0.58,
    "occluded_appearance_minimum": 0.44,
    "ambiguity_margin": 0.04,
    "merge_support_ms": 700,
    "occluded_support_ms": 5000,
    "trail_seconds": 20,
    "torso_ratio": 3.0,
    "minimum_box_height_fraction": 100 / 720,
    "torso_override_height_fraction": 110 / 720,
    "probation_ms": 5000,
    "high_detector_confidence": 0.75,
    "pose_joint_confidence": 0.45,
    "human_pose_minimum_votes": 5,
    "acceleration_m_s2": 0.8,
    "measurement_sigma_m": 0.08,
    "measurement_uncertainty_m": 0.20,
    "maximum_speed_m_s": 3.0,
    "reject_trajectory_jumps": 0,
}


def unit_vector(vector):
    if vector is None:
        return None
    v = np.asarray(vector, dtype=float)
    if v.ndim != 1 or not np.isfinite(v).all() or np.linalg.norm(v) < 1e-9:
        return None
    return v / np.linalg.norm(v)


def footpoint(track, size, cfg, projector=None):
    box = np.asarray(track["bbox_xyxy"], dtype=float)
    x1, y1, x2, y2 = box
    height = y2 - y1
    if not np.isfinite(box).all() or min(x2 - x1, height) <= 0:
        return None, 0.0, "invalid_box"
    points = track.get("keypoints_xy")
    conf = track.get("keypoints_conf")
    if points is not None and conf is not None:
        points, conf = np.asarray(points, dtype=float), np.asarray(conf, dtype=float)
        if (
            points.shape != (17, 2)
            or conf.shape != (17,)
            or not np.isfinite(points).all()
            or not np.isfinite(conf).all()
        ):
            return None, 0.0, "invalid_pose"
        ankles = [
            i
            for i in (15, 16)
            if conf[i] >= 0.42 and points[i, 1] >= y1 + height * 0.48
        ]
        torso_ok = min(conf[[5, 6, 11, 12]]) >= 0.55
        if ankles and height / size[1] >= cfg["torso_override_height_fraction"]:
            p = np.array(
                [points[ankles, 0].mean(), points[ankles, 1].max() + 0.012 * height]
            )
            return p, float(np.clip(conf[ankles].mean(), 0.62, 1)), "ankles"
        if torso_ok:
            shoulder, hip = points[[5, 6]].mean(0), points[[11, 12]].mean(0)
            if np.linalg.norm(hip - shoulder) >= size[1] * 0.007:
                gravity_point = (
                    projector.gravity_foot(shoulder, hip, cfg["torso_ratio"], size)
                    if projector is not None
                    else None
                )
                p = (
                    gravity_point
                    if gravity_point is not None
                    else shoulder + cfg["torso_ratio"] * (hip - shoulder)
                )
                p[0] = np.clip(
                    p[0],
                    max(0, x1 - 0.35 * (x2 - x1)),
                    min(size[0] - 1, x2 + 0.35 * (x2 - x1)),
                )
                p[1] = np.clip(
                    max(p[1], hip[1] if gravity_point is not None else y2),
                    0,
                    min(size[1] - 1, y2 + 1.10 * height),
                )
                return (
                    p,
                    float(0.68 * min(conf[[5, 6, 11, 12]])),
                    "gravity_torso_estimate"
                    if gravity_point is not None
                    else "torso_estimate",
                )
        if ankles:
            return (
                np.array([points[ankles, 0].mean(), points[ankles, 1].max()]),
                float(conf[ankles].mean()),
                "ankles",
            )
    if height / size[1] < cfg["minimum_box_height_fraction"]:
        return None, 0.0, "insufficient_ground_contact"
    if track.get("pose_available"):
        return None, 0.0, "ground_contact_unobserved"
    return np.array([(x1 + x2) / 2, y2]), 0.34, "bbox_estimate"


class FloorFilter:
    def __init__(self, cfg):
        self.cfg = cfg
        self.state = None
        self.cov = None
        self.timestamp = None

    def update(self, timestamp, measurement, reliability):
        z = np.asarray(measurement, dtype=float)
        self.discontinuity = False
        if (
            self.state is not None
            and self.cfg.get("reject_trajectory_jumps")
            and timestamp > self.timestamp
        ):
            dt = (timestamp - self.timestamp) / 1000
            if (
                dt > 2.5
                or np.linalg.norm(z - self.state[:2])
                > 0.75 + self.cfg["maximum_speed_m_s"] * dt
            ):
                self.state = None
                self.discontinuity = True
        if self.state is None or timestamp - self.timestamp > 2500:
            self.state = np.r_[z, 0.0, 0.0]
            self.cov = np.diag([0.01, 0.01, 1.0, 1.0])
        elif timestamp > self.timestamp:
            dt = (timestamp - self.timestamp) / 1000
            a = np.array(
                [[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]], float
            )
            g = np.array([[dt * dt / 2, 0], [0, dt * dt / 2], [dt, 0], [0, dt]])
            state = a @ self.state
            covariance = (
                a @ self.cov @ a.T + g @ g.T * self.cfg["acceleration_m_s2"] ** 2
            )
            innovation = z - state[:2]
            limit = 0.4 + self.cfg["maximum_speed_m_s"] * dt
            if np.linalg.norm(innovation) > limit:
                innovation *= limit / np.linalg.norm(innovation)
            sigma = self.cfg["measurement_sigma_m"] + self.cfg[
                "measurement_uncertainty_m"
            ] * (1 - reliability)
            h = np.eye(2, 4)
            noise = np.eye(2) * sigma**2
            gain = np.linalg.solve(h @ covariance @ h.T + noise, h @ covariance).T
            self.state = state + gain @ innovation
            residual = np.eye(4) - gain @ h
            self.cov = residual @ covariance @ residual.T + gain @ noise @ gain.T
        self.timestamp = timestamp
        return self.state[:2].copy()


class FloorTrackingService:
    """Process timestamped tracking packets; caller owns stream decoding and SCT."""

    def __init__(self, site, allow_draft=False, session_id=None):
        self.site = copy.deepcopy(site)
        self.validation = validate_site(self.site, not allow_draft)
        self.preview = allow_draft
        self.cfg = {**DEFAULTS, **site.get("runtime", {})}
        for key, value in self.cfg.items():
            if (
                not isinstance(value, (float, int))
                or not np.isfinite(value)
                or value < 0
            ):
                raise ValueError(f"Invalid runtime setting {key}")
        self.ppm = site["floor"].get("pixels_per_metre")
        if not self.ppm:
            if not allow_draft:
                raise ValueError("Metric scale required")
            self.ppm = float(np.linalg.norm(site["floor"]["size_wh"])) / 20
        self.session_id = session_id or uuid.uuid4().hex
        self.projectors = {
            cid: Projector(c, site["floor"]) for cid, c in site["cameras"].items()
        }
        self.local, self.identities, self.aliases, self.latest = {}, {}, {}, {}
        self.pairs, self.conflicts, self.filters, self.trails = {}, {}, {}, {}
        self.last_camera_time = {}
        self.clock = -1
        self.sequence = 0
        self.pending_updates = []
        self.next_id = 1
        self.metrics = {
            "packets": 0,
            "stale_packets": 0,
            "invalid_observations": 0,
            "aliases": 0,
            "rejected_identity_updates": 0,
        }
        self.lock = threading.RLock()

    @classmethod
    def from_json(cls, path, **kwargs):
        return cls(load_site(path), **kwargs)

    def resolve_global_id(self, gid):
        value = str(gid)
        seen = set()
        while value in self.aliases:
            if value in seen:
                raise ValueError("Cyclic identity aliases")
            seen.add(value)
            value = self.aliases[value]
        return value

    def _record(self, gid, timestamp):
        gid = self.resolve_global_id(gid)
        self.identities.setdefault(
            gid, {"first": timestamp, "last": timestamp, "embedding": None}
        )
        self.identities[gid]["last"] = max(timestamp, self.identities[gid]["last"])
        return gid

    def _human(self, track, key, timestamp, size):
        center = np.array(track["bbox_xyxy"], float).reshape(2, 2).mean(0) / np.array(
            size
        )
        history = self.local.setdefault(
            key,
            {
                "first": timestamp,
                "last": timestamp,
                "center": center,
                "confirmed": False,
                "gid": None,
                "human_samples": [],
            },
        )
        history["last"] = timestamp
        conf = track.get("keypoints_conf")
        pose = (
            conf is not None
            and len(conf) == 17
            and min(np.asarray(conf)[[5, 6, 11, 12]])
            >= self.cfg["pose_joint_confidence"]
        )
        supported = pose or float(track.get("pose_box_iou", 0)) >= 0.25
        if supported:
            history["last_pose_support"] = timestamp
        samples = history.setdefault("human_samples", [])
        samples.append(
            [
                timestamp,
                bool(supported),
                float(track.get("detection_confidence", 0)),
                center.tolist(),
            ]
        )
        history["human_samples"] = samples = [
            s for s in samples[-120:] if timestamp - s[0] <= 5000
        ]
        pose_votes = sum(s[1] for s in samples)
        confirmed = (
            track.get("detection_confidence", 0) >= 0.85
            or (
                pose_votes >= self.cfg["human_pose_minimum_votes"]
                and pose_votes / len(samples) >= 0.15
            )
            or (
                len(samples) >= 3
                and np.median([s[2] for s in samples])
                >= self.cfg["high_detector_confidence"]
            )
            or np.linalg.norm(center - history["center"]) > 0.085
        )
        history["confirmed"] |= bool(confirmed)
        if (
            track.get("pose_available")
            and not supported
            and timestamp - history.get("last_pose_support", history["first"]) > 1500
        ):
            steps = np.diff(np.array([s[3] for s in samples]), axis=0)
            if not len(steps) or np.median(np.linalg.norm(steps, axis=1)) < 0.002:
                return "unverified_static"
        if history["confirmed"]:
            return "confirmed"
        return (
            "unverified_static"
            if timestamp - history["first"] >= self.cfg["probation_ms"]
            else "probation"
        )

    def _merge(self, left, right, timestamp, similarity, reason):
        left, right = self.resolve_global_id(left), self.resolve_global_id(right)
        if left == right:
            return None
        components = lambda gid: (
            {gid} | {x for x in self.aliases if self.resolve_global_id(x) == gid}
        )
        if any(
            tuple(sorted((a, b))) in self.conflicts
            for a in components(left)
            for b in components(right)
        ):
            return None
        canonical, source = sorted(
            (left, right), key=lambda g: (self.identities[g]["first"], g)
        )
        self.aliases[source] = canonical
        self.sequence += 1
        self.metrics["aliases"] += 1
        self.filters.pop(source, None)
        self.trails.pop(source, None)
        return {
            "schema_version": "1.0",
            "event": "global_id_alias",
            "site_id": self.site["site_id"],
            "session_id": self.session_id,
            "sequence": self.sequence,
            "timestamp_ms": timestamp,
            "source_global_id": source,
            "canonical_global_id": canonical,
            "similarity": similarity,
            "reason": reason,
        }

    def _fuse(self, timestamp):
        rows = [
            r
            for r in self.latest.values()
            if timestamp - r["timestamp_ms"] <= self.cfg["synchronization_ms"]
            and r.get("point_m") is not None
        ]
        evidence = {}
        for i, a in enumerate(rows):
            for b in rows[i + 1 :]:
                ga, gb = (
                    self.resolve_global_id(a["global_id"]),
                    self.resolve_global_id(b["global_id"]),
                )
                if a["camera_id"] == b["camera_id"] or ga == gb:
                    continue
                distance = float(np.linalg.norm(np.array(a["point_m"]) - b["point_m"]))
                if distance > self.cfg["pair_distance_m"]:
                    continue
                va, vb = (
                    self.identities[ga]["embedding"],
                    self.identities[gb]["embedding"],
                )
                if va is None or vb is None or len(va) != len(vb):
                    continue
                sim = float(np.asarray(va) @ np.asarray(vb))
                if (
                    sim
                    < min(
                        self.cfg["appearance_minimum"],
                        self.cfg["occluded_appearance_minimum"],
                    )
                    - self.cfg["ambiguity_margin"]
                ):
                    continue
                pair = tuple(sorted((ga, gb)))
                if pair not in evidence or distance < evidence[pair][0]:
                    evidence[pair] = (distance, sim)
        alternatives = {}
        for pair, (_, score) in evidence.items():
            for gid in pair:
                alternatives.setdefault(gid, []).append((score, pair))
        alternatives = {
            gid: sorted(values, reverse=True)[:2]
            for gid, values in alternatives.items()
        }
        updates = []
        for pair, (distance, sim) in evidence.items():
            competitors = [
                score
                for gid in pair
                for score, other_pair in alternatives[gid]
                if other_pair != pair
            ]
            if competitors and sim - max(competitors) < self.cfg["ambiguity_margin"]:
                continue
            records = self.pairs.setdefault(pair, deque(maxlen=120))
            if not records or records[-1][0] != timestamp:
                records.append((timestamp, distance, sim))
            while (
                records
                and timestamp - records[0][0] > self.cfg["evidence_retention_ms"]
            ):
                records.popleft()
            span = records[-1][0] - records[0][0]
            gaps = np.diff([r[0] for r in records])
            if len(records) < 3 or (len(gaps) and max(gaps) > 2500):
                continue
            d = float(np.median([r[1] for r in records]))
            s = float(np.median([r[2] for r in records]))
            strong = (
                span >= self.cfg["merge_support_ms"]
                and d <= self.cfg["merge_distance_m"]
                and s >= self.cfg["appearance_minimum"]
            )
            occluded = (
                span >= self.cfg["occluded_support_ms"]
                and d <= self.cfg["merge_distance_m"]
                and s >= self.cfg["occluded_appearance_minimum"]
            )
            if strong or occluded:
                update = self._merge(
                    *pair, timestamp, s, "sustained_spatial_appearance_evidence"
                )
                if update:
                    updates.append(update)
        return updates

    def process_batch(self, packets):
        start = time.perf_counter()
        with self.lock:
            observations = []
            for packet in sorted(
                packets, key=lambda p: (p["timestamp_ms"], p["camera_id"])
            ):
                cid = packet["camera_id"]
                timestamp = int(packet["timestamp_ms"])
                if cid not in self.projectors:
                    raise ValueError(f"Unknown camera: {cid}")
                if (
                    timestamp < 0
                    or timestamp <= self.last_camera_time.get(cid, -1)
                    or timestamp < self.clock - self.cfg["synchronization_ms"]
                ):
                    self.metrics["stale_packets"] += 1
                    continue
                for update in packet.get("upstream_identity_updates", []):
                    if int(update["timestamp_ms"]) > timestamp:
                        raise ValueError("Identity update is from the future")
                    try:
                        self.apply_identity_alias(update)
                    except ValueError as error:
                        self.metrics["rejected_identity_updates"] += 1
                        self.sequence += 1
                        self.pending_updates.append(
                            {
                                "schema_version": "1.0",
                                "event": "identity_update_rejected",
                                "site_id": self.site["site_id"],
                                "session_id": self.session_id,
                                "sequence": self.sequence,
                                "timestamp_ms": timestamp,
                                "source_global_id": str(update.get("source_global_id")),
                                "canonical_global_id": str(
                                    update.get("canonical_global_id")
                                ),
                                "reason": str(error),
                            }
                        )
                        continue
                    self.sequence += 1
                    self.pending_updates.append(
                        {
                            **update,
                            "source_global_id": str(update["source_global_id"]),
                            "canonical_global_id": self.resolve_global_id(
                                update["canonical_global_id"]
                            ),
                            "session_id": self.session_id,
                            "sequence": self.sequence,
                            "reason": "upstream_"
                            + update.get("reason", "reconciliation"),
                        }
                    )
                size = packet["frame_size_wh"]
                self.projectors[cid].undistort([0, 0], size)
                self.last_camera_time[cid] = timestamp
                self.clock = max(self.clock, timestamp)
                self.metrics["packets"] += 1
                epoch = str(packet.get("camera_epoch", "0"))
                for k in [k for k, r in self.latest.items() if r["camera_id"] == cid]:
                    del self.latest[k]
                active = []
                seen_local = set()
                for t in packet["tracks"]:
                    key = f"{cid}\x1f{epoch}\x1f{t['local_id']}"
                    try:
                        if key in seen_local:
                            raise ValueError("Duplicate local ID in one packet")
                        seen_local.add(key)
                        human = self._human(t, key, timestamp, size)
                        supplied = t.get("global_id")
                        gid = self.local[key]["gid"]
                        if supplied is not None:
                            supplied = str(supplied)
                            if gid is not None and self.resolve_global_id(
                                gid
                            ) != self.resolve_global_id(supplied):
                                previous = self.resolve_global_id(gid)
                                target = self.resolve_global_id(supplied)
                                self._record(target, timestamp)
                                if (
                                    tuple(sorted((previous, target)))
                                    not in self.conflicts
                                ):
                                    self.aliases[previous] = target
                                    self.filters.pop(previous, None)
                                    self.trails.pop(previous, None)
                                    self.sequence += 1
                                    self.pending_updates.append(
                                        {
                                            "schema_version": "1.0",
                                            "event": "global_id_alias",
                                            "site_id": self.site["site_id"],
                                            "session_id": self.session_id,
                                            "sequence": self.sequence,
                                            "timestamp_ms": timestamp,
                                            "source_global_id": previous,
                                            "canonical_global_id": target,
                                            "similarity": None,
                                            "reason": "upstream_assignment_same_local_track",
                                        }
                                    )
                            gid = supplied
                        if gid is None:
                            gid = f"new:{self.next_id}"
                            self.next_id += 1
                        if self.resolve_global_id(gid) in active:
                            gid = f"collision:{cid}:{epoch}:{t['local_id']}"
                        gid = self._record(gid, timestamp)
                        self.local[key]["gid"] = gid
                        active.append(gid)
                        embedding = unit_vector(t.get("embedding"))
                        if embedding is not None:
                            old = self.identities[gid]["embedding"]
                            if old is not None and len(old) != len(embedding):
                                raise ValueError("Appearance model dimension changed")
                            self.identities[gid]["embedding"] = unit_vector(
                                embedding
                                if old is None
                                else np.asarray(old) * 0.85 + embedding * 0.15
                            ).tolist()
                        point, reliability, method = footpoint(
                            t, size, self.cfg, self.projectors[cid]
                        )
                        history = self.local[key]
                        if point is not None and method in (
                            "ankles",
                            "torso_estimate",
                            "gravity_torso_estimate",
                        ):
                            history["ground_contact"] = {
                                "timestamp": timestamp,
                                "normalized": (point / np.asarray(size)).tolist(),
                            }
                        elif (
                            (
                                method == "bbox_estimate"
                                or (
                                    t.get("pose_available")
                                    and method
                                    in (
                                        "ground_contact_unobserved",
                                        "insufficient_ground_contact",
                                    )
                                )
                            )
                            and history.get("ground_contact")
                            and timestamp - history["ground_contact"]["timestamp"]
                            <= 800
                        ):
                            point = np.array(
                                history["ground_contact"]["normalized"]
                            ) * np.asarray(size)
                            reliability = 0.25
                            method = "recent_contact_estimate"
                        floor = None
                        status = human if human != "confirmed" else method
                        if point is not None and human == "confirmed":
                            floor = self.projectors[cid].project(point, size)
                        r = {
                            "camera_id": cid,
                            "local_id": t["local_id"],
                            "global_id": gid,
                            "timestamp_ms": timestamp,
                            "bbox_xyxy": t["bbox_xyxy"],
                            "status": status,
                            "footpoint_source": method,
                            "reliability": reliability,
                            "footpoint_px": point.tolist()
                            if point is not None
                            else None,
                            "floor_px": floor.tolist() if floor is not None else None,
                            "point_m": (floor / self.ppm).tolist()
                            if floor is not None
                            else None,
                        }
                        self.latest[key] = r
                        observations.append(r)
                    except (ValueError, KeyError, IndexError, TypeError) as error:
                        self.metrics["invalid_observations"] += 1
                        observations.append(
                            {
                                "camera_id": cid,
                                "local_id": t.get("local_id"),
                                "global_id": None,
                                "bbox_xyxy": t.get("bbox_xyxy"),
                                "status": "projection_rejected",
                                "reason": str(error),
                                "floor_px": None,
                                "point_m": None,
                            }
                        )
                for i, a in enumerate(active):
                    for b in active[i + 1 :]:
                        if a != b:
                            self.conflicts[tuple(sorted((a, b)))] = timestamp
            timestamp = self.clock
            updates = self.pending_updates + self._fuse(timestamp)
            self.pending_updates = []
            grouped = {}
            for r in self.latest.values():
                if (
                    timestamp - r["timestamp_ms"] > self.cfg["active_ms"]
                    or r["point_m"] is None
                ):
                    continue
                gid = self.resolve_global_id(r["global_id"])
                grouped.setdefault(gid, []).append(r)
            tracks = []
            for gid, rows in grouped.items():
                newest = max(r["timestamp_ms"] for r in rows)
                rows = [
                    r
                    for r in rows
                    if newest - r["timestamp_ms"] <= self.cfg["synchronization_ms"]
                ]
                positions = np.array([r["point_m"] for r in rows])
                weights = np.array([max(0.05, r["reliability"]) for r in rows])
                if (
                    len(rows) > 1
                    and np.max(
                        np.linalg.norm(positions[:, None] - positions[None, :], axis=-1)
                    )
                    > self.cfg["pair_distance_m"]
                ):
                    idx = int(np.argmax(weights))
                    measurement = positions[idx]
                    reliability = weights[idx]
                else:
                    measurement = np.average(positions, axis=0, weights=weights)
                    reliability = float(np.average(weights))
                filt = self.filters.setdefault(gid, FloorFilter(self.cfg))
                point = filt.update(newest, measurement, reliability)
                trail = self.trails.setdefault(gid, deque(maxlen=2000))
                if filt.discontinuity:
                    trail.clear()
                if not trail or newest > trail[-1][0]:
                    trail.append((newest, point.tolist()))
                while (
                    trail and timestamp - trail[0][0] > self.cfg["trail_seconds"] * 1000
                ):
                    trail.popleft()
                tracks.append(
                    {
                        "global_id": gid,
                        "position_status": "reacquired"
                        if filt.discontinuity
                        else "continuous",
                        "floor_px": (point * self.ppm).tolist(),
                        "position_m": point.tolist()
                        if self.site["floor"].get("pixels_per_metre")
                        else None,
                        "camera_ids": sorted({r["camera_id"] for r in rows}),
                        "reliability": reliability,
                        "trail_px": [
                            [ts, *(np.array(p) * self.ppm).tolist()] for ts, p in trail
                        ],
                    }
                )
            for r in observations:
                if r.get("global_id") is not None:
                    r["global_id"] = self.resolve_global_id(r["global_id"])
            self._expire(timestamp)
            return {
                "schema_version": "1.0",
                "site_id": self.site["site_id"],
                "session_id": self.session_id,
                "timestamp_ms": timestamp,
                "preview": self.preview,
                "tracks": tracks,
                "observations": observations,
                "identity_updates": updates,
                "processing_ms": (time.perf_counter() - start) * 1000,
                "metrics": dict(self.metrics),
            }

    def _expire(self, timestamp):
        self.local = {
            k: v
            for k, v in self.local.items()
            if timestamp - v["last"] <= self.cfg["track_retention_ms"]
        }
        self.latest = {
            k: v
            for k, v in self.latest.items()
            if timestamp - v["timestamp_ms"] <= self.cfg["active_ms"]
        }
        self.pairs = {
            k: v
            for k, v in self.pairs.items()
            if v and timestamp - v[-1][0] <= self.cfg["evidence_retention_ms"]
        }
        self.conflicts = {
            k: v
            for k, v in self.conflicts.items()
            if timestamp - v <= self.cfg["retention_ms"]
        }
        dead = {
            k
            for k, v in self.identities.items()
            if timestamp - v["last"] > self.cfg["retention_ms"]
        }
        for k in dead:
            self.identities.pop(k)
            self.filters.pop(k, None)
            self.trails.pop(k, None)
        self.aliases = {
            a: b
            for a, b in self.aliases.items()
            if self.resolve_global_id(b) not in dead
        }

    def save_state(self, path):
        with self.lock:
            value = {
                "site_fingerprint": digest(self.site),
                "session_id": self.session_id,
                "clock": self.clock,
                "sequence": self.sequence,
                "next_id": self.next_id,
                "identities": self.identities,
                "aliases": self.aliases,
                "local": {
                    k: {**v, "center": v["center"].tolist()}
                    for k, v in self.local.items()
                },
                "last_camera_time": self.last_camera_time,
                "conflicts": [[list(k), v] for k, v in self.conflicts.items()],
            }
            p = Path(path)
            p.parent.mkdir(parents=True, exist_ok=True)
            tmp = p.with_suffix(p.suffix + ".tmp")
            tmp.write_text(json.dumps(value, allow_nan=False))
            tmp.replace(p)

    def restore_state(self, path):
        with self.lock:
            d = json.loads(Path(path).read_text())
            if d["site_fingerprint"] != digest(self.site):
                raise ValueError("State belongs to another site/calibration")
            for key in (
                "session_id",
                "clock",
                "sequence",
                "next_id",
                "identities",
                "aliases",
                "last_camera_time",
            ):
                setattr(self, key, d[key])
            self.local = {
                k: {**v, "center": np.asarray(v["center"])}
                for k, v in d["local"].items()
            }
            self.conflicts = {tuple(k): v for k, v in d["conflicts"]}
            for gid in self.aliases:
                self.resolve_global_id(gid)

    def apply_identity_alias(self, event):
        """Accept an upstream alias after checking the shared site and active collisions."""
        with self.lock:
            if event["site_id"] != self.site["site_id"]:
                raise ValueError("Alias belongs to another site")
            a, b = str(event["source_global_id"]), str(event["canonical_global_id"])
            ts = int(event["timestamp_ms"])
            self._record(a, ts)
            self._record(b, ts)
            a, b = self.resolve_global_id(a), self.resolve_global_id(b)
            members = lambda gid: (
                {gid} | {x for x in self.aliases if self.resolve_global_id(x) == gid}
            )
            if a != b and any(
                tuple(sorted((x, y))) in self.conflicts
                for x in members(a)
                for y in members(b)
            ):
                raise ValueError("Alias conflicts with observed simultaneous tracks")
            if self.resolve_global_id(a) != self.resolve_global_id(b):
                self.aliases[self.resolve_global_id(a)] = self.resolve_global_id(b)
