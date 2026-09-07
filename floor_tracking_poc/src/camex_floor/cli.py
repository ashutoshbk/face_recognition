import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import uuid

import cv2

from .geometry import calibration_fingerprint, load_site, validate_site
from .replay import replay
from .server import atomic_json, serve


def import_camera(
    site_path, camera_id, reference, distortion=None, metadata=None, point_layer=None
):
    path = Path(site_path)
    site = json.loads(path.read_text())
    if camera_id in site["cameras"]:
        raise ValueError(
            "Camera ID already exists; use a new ID or an explicit migration"
        )
    im = cv2.imread(str(reference))
    if im is None:
        raise ValueError("Cannot read reference image")
    name = f"assets/{uuid.uuid4().hex}.png"
    p = path.parent / name
    p.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(p), im)
    camera = {
        "label": camera_id,
        "reference": name,
        "reference_sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
        "image_size_wh": [im.shape[1], im.shape[0]],
        "distortion": distortion or {"model": "none"},
        "anchors": [],
        "image_to_floor_h": None,
    }
    if metadata and point_layer:
        import numpy as np

        m = json.loads(Path(metadata).read_text())
        name = f"assets/{uuid.uuid4().hex}.png"
        shutil.copy2(point_layer, path.parent / name)
        camera["point_layer"] = name
        camera["layer_to_image_h"] = np.linalg.inv(
            np.asarray(m["ground_to_layer_H"])
            @ np.asarray(m["rectified_image_to_ground_H"])
        ).tolist()
    site["cameras"][camera_id] = camera
    site["revision"] = site.get("revision", 0) + 1
    atomic_json(path, site)


def main():
    parser = argparse.ArgumentParser(prog="camex-floor")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("import-vggt")
    p.add_argument("--site", type=Path, required=True)
    p.add_argument("--camera-id", required=True)
    p.add_argument("--calibration-dir", type=Path, required=True)
    p = sub.add_parser("init")
    p.add_argument("--site", required=True)
    p.add_argument("--floor", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--pixels-per-metre", type=float)
    p = sub.add_parser("add-camera")
    p.add_argument("--site", required=True, type=Path)
    p.add_argument("--camera-id", required=True)
    p.add_argument("--image", required=True, type=Path)
    p.add_argument("--geocalib", type=Path)
    p.add_argument("--metadata", type=Path)
    p.add_argument("--point-layer", type=Path)
    p = sub.add_parser("calibrate-camera")
    p.add_argument("--engine-root", required=True, type=Path)
    p.add_argument("--input", required=True, type=Path)
    p.add_argument("--camera-id", required=True)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--floor-polygon", type=Path)
    p = sub.add_parser("editor")
    p.add_argument("--site", required=True, type=Path)
    p.add_argument("--port", default=8766, type=int)
    p = sub.add_parser("validate")
    p.add_argument("--site", required=True, type=Path)
    p.add_argument("--approve", action="store_true")
    p = sub.add_parser("replay")
    p.add_argument("--site", required=True, type=Path)
    p.add_argument("--events", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--allow-draft", action="store_true")
    p.add_argument("--fps", type=int, default=10)
    p.add_argument("--realtime", action="store_true")
    p.add_argument("--no-video", action="store_true")
    p.add_argument("--progress-file", type=Path)
    p.add_argument("--expected-duration", type=float)
    sub.add_parser("doctor")
    args = parser.parse_args()
    try:
        if args.command == "import-vggt":
            from .vggt_import import import_vggt

            print(
                json.dumps(
                    import_vggt(args.site, args.camera_id, args.calibration_dir),
                    indent=2,
                )
            )
        elif args.command == "init":
            if args.output.exists():
                raise ValueError("Output exists; existing site will not be overwritten")
            im = cv2.imread(str(args.floor))
            if im is None:
                raise ValueError("Floor must be a PNG or JPEG image")
            args.output.parent.mkdir(parents=True, exist_ok=True)
            asset = args.output.parent / "assets"
            asset.mkdir(exist_ok=True)
            filename = uuid.uuid4().hex + ".png"
            cv2.imwrite(str(asset / filename), im)
            atomic_json(
                args.output,
                {
                    "schema_version": "1.0",
                    "revision": 0,
                    "site_id": args.site,
                    "floor": {
                        "asset": "assets/" + filename,
                        "asset_sha256": hashlib.sha256(
                            (asset / filename).read_bytes()
                        ).hexdigest(),
                        "size_wh": [im.shape[1], im.shape[0]],
                        "pixels_per_metre": args.pixels_per_metre,
                    },
                    "cameras": {},
                },
            )
        elif args.command == "add-camera":
            d = None
            if args.geocalib:
                g = json.loads(args.geocalib.read_text())
                if g["camera_model"] != "simple_divisional":
                    raise ValueError(
                        "This importer supports GeoCalib simple_divisional only"
                    )
                d = {
                    "model": "simple_divisional",
                    "focal_xy": g["focal_xy_pixels"][0],
                    "principal_xy": g["principal_point_xy_pixels"][0],
                    "k1": g["distortion_parameters"][0][0],
                }
            import_camera(
                args.site,
                args.camera_id,
                args.image,
                d,
                args.metadata,
                args.point_layer,
            )
        elif args.command == "calibrate-camera":
            command = [
                "bash",
                str(args.engine_root.resolve() / "calibrate.sh"),
                "--input",
                str(args.input.resolve()),
                "--camera-id",
                args.camera_id,
                "--output",
                str(args.output.resolve()),
            ]
            if args.floor_polygon:
                command += ["--floor-polygon", str(args.floor_polygon.resolve())]
            subprocess.run(command, check=True)
        elif args.command == "editor":
            serve(args.site, args.port)
        elif args.command == "validate":
            site = load_site(args.site)
            reports = validate_site(site, False)
            print(json.dumps(reports, indent=2))
            if not reports or not all(r["ready"] for r in reports.values()):
                return 2
            if args.approve:
                for c in site["cameras"].values():
                    c["approved_fingerprint"] = calibration_fingerprint(
                        c, site["floor"]
                    )
                site["revision"] = site.get("revision", 0) + 1
                atomic_json(args.site, site)
        elif args.command == "replay":
            print(
                json.dumps(
                    replay(
                        args.site,
                        args.events,
                        args.output,
                        args.allow_draft,
                        args.fps,
                        args.realtime,
                        not args.no_video,
                        args.progress_file,
                        args.expected_duration,
                    ),
                    indent=2,
                )
            )
        elif args.command == "doctor":
            import numpy, scipy

            print(
                json.dumps(
                    {
                        "python": sys.version.split()[0],
                        "numpy": numpy.__version__,
                        "scipy": scipy.__version__,
                        "opencv": cv2.__version__,
                        "ffmpeg": bool(shutil.which("ffmpeg")),
                        "runtime_gpu_required": False,
                    },
                    indent=2,
                )
            )
        return 0
    except (ValueError, KeyError, OSError, subprocess.CalledProcessError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
