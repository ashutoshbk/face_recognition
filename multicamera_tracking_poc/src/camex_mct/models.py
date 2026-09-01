from __future__ import annotations

import importlib.util
import time
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from .dependencies import configure_third_party

# ByteTrack still uses removed NumPy aliases internally.
if "float" not in np.__dict__:
    np.float = float  # type: ignore[attr-defined]


def yolox_preprocess(image: np.ndarray, input_size: tuple[int, int]) -> np.ndarray:
    """Match the preprocessing contract used by the official YOLOX checkpoints."""
    padded = np.full((input_size[0], input_size[1], 3), 114, dtype=np.uint8)
    ratio = min(input_size[0] / image.shape[0], input_size[1] / image.shape[1])
    resized = cv2.resize(
        image,
        (int(image.shape[1] * ratio), int(image.shape[0] * ratio)),
        interpolation=cv2.INTER_LINEAR,
    )
    padded[: resized.shape[0], : resized.shape[1]] = resized
    return np.ascontiguousarray(padded.transpose(2, 0, 1))


class YoloXPersonDetector:
    def __init__(
        self,
        experiment_path: str,
        checkpoint_path: str,
        input_size: int = 416,
        confidence: float = 0.25,
        nms: float = 0.45,
        device: str = "cuda",
        dedicated_stream: bool = False,
        torchscript_path: str | None = None,
        preprocess_workers: int = 1,
    ) -> None:
        configure_third_party()
        from yolox.exp import get_exp
        from yolox.utils import fuse_model, postprocess

        self._postprocess = postprocess
        self.device = torch.device(device)
        self.stream = (
            torch.cuda.Stream(device=self.device)
            if dedicated_stream and self.device.type == "cuda"
            else None
        )
        self.test_size = (input_size, input_size)
        exp = get_exp(experiment_path, None)
        exp.test_size = self.test_size
        exp.test_conf = confidence
        exp.nmsthre = nms
        self.num_classes = exp.num_classes
        self.confidence = confidence
        self.nms = nms

        if torchscript_path:
            self.model = torch.jit.load(torchscript_path, map_location=self.device).eval()
            self.uses_torchscript = True
        else:
            model = exp.get_model()
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            model.load_state_dict(checkpoint["model"])
            model.eval()
            self.model = fuse_model(model).to(self.device)
            self.uses_torchscript = False
        torch.backends.cudnn.benchmark = True
        self.preprocess_executor = (
            ThreadPoolExecutor(max_workers=preprocess_workers) if preprocess_workers > 1 else None
        )

    def _stream_context(self):
        if self.stream is None:
            return nullcontext()
        return torch.cuda.stream(self.stream)

    def _synchronize(self) -> None:
        if self.device.type != "cuda":
            return
        if self.stream is None:
            torch.cuda.synchronize(self.device)
        else:
            self.stream.synchronize()

    def warmup(self, batch_size: int, iterations: int = 3) -> None:
        dummy = torch.zeros(
            (batch_size, 3, self.test_size[0], self.test_size[1]),
            dtype=torch.float32,
            device=self.device,
        )
        with self._stream_context(), torch.inference_mode():
            for _ in range(iterations):
                self.model(dummy)
        self._synchronize()

    def infer(self, frames: Sequence[np.ndarray]) -> tuple[list[np.ndarray], dict[str, float]]:
        preprocess_started = time.perf_counter()
        if self.preprocess_executor is None:
            prepared = [yolox_preprocess(frame, self.test_size) for frame in frames]
        else:
            prepared = list(
                self.preprocess_executor.map(
                    lambda frame: yolox_preprocess(frame, self.test_size),
                    frames,
                )
            )
        with self._stream_context():
            batch = torch.from_numpy(np.stack(prepared)).to(self.device, non_blocking=True)
            batch = batch.float()
        self._synchronize()
        preprocess_ms = (time.perf_counter() - preprocess_started) * 1000.0

        inference_started = time.perf_counter()
        with self._stream_context(), torch.inference_mode():
            raw = self.model(batch)
            outputs = self._postprocess(raw, self.num_classes, self.confidence, self.nms)
        self._synchronize()
        inference_ms = (time.perf_counter() - inference_started) * 1000.0

        person_detections: list[np.ndarray] = []
        postprocess_started = time.perf_counter()
        for output in outputs:
            if output is None or output.numel() == 0:
                person_detections.append(np.empty((0, 5), dtype=np.float32))
                continue
            output = output[output[:, 6] == 0]
            if output.numel() == 0:
                person_detections.append(np.empty((0, 5), dtype=np.float32))
                continue
            scores = output[:, 4] * output[:, 5]
            detections = torch.cat((output[:, :4], scores[:, None]), dim=1)
            person_detections.append(detections.detach().cpu().numpy().astype(np.float32))
        postprocess_ms = (time.perf_counter() - postprocess_started) * 1000.0
        return person_detections, {
            "preprocess_ms": preprocess_ms,
            "detector_ms": inference_ms,
            "postprocess_ms": postprocess_ms,
        }

    def close(self) -> None:
        if self.preprocess_executor is not None:
            self.preprocess_executor.shutdown(wait=True)


class OsNetEmbedder:
    def __init__(
        self,
        source_file: str,
        checkpoint_path: str,
        architecture: str = "osnet_x0_25",
        device: str = "cuda",
        batch_sizes: tuple[int, ...] = (8, 16, 32),
        dedicated_stream: bool = False,
        preprocess_workers: int = 1,
    ) -> None:
        self.device = torch.device(device)
        if not batch_sizes or tuple(sorted(batch_sizes)) != batch_sizes:
            raise ValueError("batch_sizes must be a non-empty sorted tuple")
        self.batch_sizes = batch_sizes
        self.stream = (
            torch.cuda.Stream(device=self.device)
            if dedicated_stream and self.device.type == "cuda"
            else None
        )
        spec = importlib.util.spec_from_file_location("poc_osnet", source_file)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Unable to load OSNet source: {source_file}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        builder = getattr(module, architecture, None)
        if builder is None:
            raise ValueError(f"Unsupported OSNet architecture: {architecture}")
        model = builder(num_classes=1041, pretrained=False)

        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state = checkpoint.get("state_dict", checkpoint)
        clean_state = {key.removeprefix("module."): value for key, value in state.items()}
        current = model.state_dict()
        compatible = {
            key: value
            for key, value in clean_state.items()
            if key in current and current[key].shape == value.shape
        }
        if not compatible:
            raise RuntimeError("No compatible OSNet weights were found in the checkpoint")
        current.update(compatible)
        model.load_state_dict(current)
        self.model = model.to(self.device).eval()
        self.mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)
        self.preprocess_executor = (
            ThreadPoolExecutor(max_workers=preprocess_workers) if preprocess_workers > 1 else None
        )

    def _prepare_crop(self, crop: np.ndarray) -> np.ndarray:
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (128, 256), interpolation=cv2.INTER_LINEAR)
        normalized = (resized.astype(np.float32) / 255.0 - self.mean) / self.std
        return normalized.transpose(2, 0, 1)

    def _stream_context(self):
        if self.stream is None:
            return nullcontext()
        return torch.cuda.stream(self.stream)

    def _synchronize(self) -> None:
        if self.device.type != "cuda":
            return
        if self.stream is None:
            torch.cuda.synchronize(self.device)
        else:
            self.stream.synchronize()

    def warmup(self, iterations: int = 2) -> None:
        with self._stream_context(), torch.inference_mode():
            for batch_size in self.batch_sizes:
                dummy = torch.zeros((batch_size, 3, 256, 128), device=self.device)
                for _ in range(iterations):
                    self.model(dummy)
        self._synchronize()

    def embed(self, crops: Sequence[np.ndarray]) -> tuple[np.ndarray, float]:
        if not crops:
            return np.empty((0, 512), dtype=np.float32), 0.0
        if self.preprocess_executor is None:
            prepared = [self._prepare_crop(crop) for crop in crops]
        else:
            prepared = list(self.preprocess_executor.map(self._prepare_crop, crops))
        feature_chunks: list[np.ndarray] = []
        elapsed_ms = 0.0
        maximum_batch = self.batch_sizes[-1]
        for start in range(0, len(prepared), maximum_batch):
            chunk = prepared[start : start + maximum_batch]
            real_count = len(chunk)
            target_batch = next(size for size in self.batch_sizes if size >= real_count)
            if real_count < target_batch:
                padding = [np.zeros_like(chunk[0]) for _ in range(target_batch - real_count)]
                chunk = chunk + padding
            with self._stream_context():
                batch = torch.from_numpy(np.stack(chunk)).to(self.device, non_blocking=True)
            self._synchronize()
            started = time.perf_counter()
            with self._stream_context(), torch.inference_mode():
                features = F.normalize(self.model(batch), dim=1)
            self._synchronize()
            elapsed_ms += (time.perf_counter() - started) * 1000.0
            feature_chunks.append(features[:real_count].detach().cpu().numpy().astype(np.float32))
        return np.concatenate(feature_chunks, axis=0), elapsed_ms

    def close(self) -> None:
        if self.preprocess_executor is not None:
            self.preprocess_executor.shutdown(wait=True)
