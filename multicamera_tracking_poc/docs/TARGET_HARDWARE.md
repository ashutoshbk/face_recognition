# Target Hardware Acceptance

## Intended Workstation

| Component | Specification |
|---|---|
| System | Dell Pro Max Tower T2 (FCT2250) |
| CPU | Intel Core Ultra 9 285K, 24 cores, up to 5.7 GHz |
| GPU | NVIDIA RTX 5000 Ada Generation, 32 GB VRAM |
| System memory | 32 GB DDR5-5600, non-ECC |
| Storage | 2 x 1 TB TLC M.2 2280 PCIe Gen4 SSD |
| Power supply | 1500 W Platinum |
| Operating system | Windows 11 Pro |

The GPU and CPU make this a strong candidate for the 20-camera PoC, but the
200 FPS aggregate target remains an acceptance result to be measured rather
than an assumed capability. The supplied 32 GB of system memory is workable for
the PoC; 64 GB is preferable if the platform will also decode, buffer, display,
and retain all 20 streams on the same workstation.

## Runtime Preparation

Use WSL2 with a Linux distribution and Docker GPU support, or an equivalent
validated Linux deployment. Before model setup, confirm that both WSL2 and the
container can see the RTX 5000 Ada GPU:

```bash
nvidia-smi
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi
```

## Acceptance Goal

- 20 live camera inputs at 10 FPS each
- 200 accepted and processed frames per second in aggregate
- Independent local tracking state for every camera
- Site-wide global ReID output through the public result schema
- Bounded latency without an accumulating frame queue

Record accepted and processed frames, latest-slot replacements, p50/p95 batch
latency, callback errors, GPU utilization, peak VRAM, CPU utilization, system
memory, and temperature. Run for at least two minutes after warm-up using the
client platform's production decoder.

If the target is missed, profile decode, preprocessing, detector inference,
per-camera tracking, ReID, and callback publication separately before changing
accuracy thresholds. TensorRT and asynchronous pipeline stages are the first
optimization candidates.
