import os
import cv2
import numpy as np
from numpy.linalg import norm
from insightface.app import FaceAnalysis
import sys

# -------------------------------
# Cosine similarity
# -------------------------------
def cosine_similarity(emb1, emb2):
    return np.dot(emb1, emb2) / (norm(emb1) * norm(emb2))


# -------------------------------
# Initialize InsightFace (GPU)
# -------------------------------
def initialize_model(use_gpu=True):
    """
    Initialize InsightFace with GPU support
    
    Args:
        use_gpu: If True, uses CUDA. If False, falls back to CPU
    """
    if use_gpu:
        try:
            # Try GPU first with CUDA provider
            app = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
            app.prepare(ctx_id=0, det_size=(640, 640))  # ctx_id=0 for GPU
            print("✅ GPU (CUDA) initialized successfully!")
        except Exception as e:
            print(f"⚠️ GPU initialization failed: {e}")
            print("📌 Falling back to CPU...")
            app = FaceAnalysis(providers=['CPUExecutionProvider'])
            app.prepare(ctx_id=-1, det_size=(640, 640))  # ctx_id=-1 for CPU
    else:
        print("📌 Using CPU mode...")
        app = FaceAnalysis(providers=['CPUExecutionProvider'])
        app.prepare(ctx_id=-1, det_size=(640, 640))
    
    return app


# -------------------------------
# Load embeddings from images
# -------------------------------
def load_embeddings_from_images(folder, recognizer):
    face_database = {}

    for file in os.listdir(folder):
        if file.lower().endswith(('.jpg', '.jpeg', '.png')):
            name = os.path.splitext(file)[0]
            img_path = os.path.join(folder, file)

            img = cv2.imread(img_path)
            if img is None:
                continue

            faces = recognizer.get(img)
            if len(faces) == 0:
                print(f"No face found in {file}")
                continue

            face = max(
                faces,
                key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])
            )

            face_database[name] = face.embedding
            print(f"Loaded embedding for: {name}")

    return face_database


# -------------------------------
# Face recognition from VIDEO + PROGRESS
# -------------------------------
def face_recognition_from_video(video_path, dataset_folder, output_path, threshold=0.5, use_gpu=True):
    recognizer = initialize_model(use_gpu=use_gpu)
    face_database = load_embeddings_from_images(dataset_folder, recognizer)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("❌ Cannot open video:", video_path)
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    print("\n▶ Processing video...")
    print(f"🎞 Total frames: {total_frames}")
    print(f"📊 Resolution: {width}x{height} @ {fps:.2f} FPS")
    print("💾 Saving output to:", output_path)
    print("-" * 50)

    frame_count = 0
    import time
    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        faces = recognizer.get(frame)

        for face in faces:
            bbox = face.bbox.astype(int)
            embedding = face.embedding

            best_match = "Unknown"
            highest_similarity = 0.0

            for name, db_embedding in face_database.items():
                sim = cosine_similarity(embedding, db_embedding)
                if sim > highest_similarity:
                    highest_similarity = sim
                    best_match = name

            if highest_similarity < threshold:
                best_match = "Unknown"

            label = f"{best_match} ({highest_similarity:.2f})"
            
            # Color: Green for known, Red for unknown
            color = (0, 255, 0) if best_match != "Unknown" else (0, 0, 255)

            cv2.rectangle(
                frame,
                (bbox[0], bbox[1]),
                (bbox[2], bbox[3]),
                color,
                2
            )

            cv2.putText(
                frame,
                label,
                (bbox[0], bbox[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                color,
                2
            )

        out.write(frame)

        # -------- PROGRESS PRINT WITH FPS --------
        progress = (frame_count / total_frames) * 100
        elapsed = time.time() - start_time
        fps_current = frame_count / elapsed if elapsed > 0 else 0
        eta = (total_frames - frame_count) / fps_current if fps_current > 0 else 0
        
        sys.stdout.write(
            f"\r⏳ Processing: {progress:.2f}% "
            f"({frame_count}/{total_frames} frames) | "
            f"⚡ {fps_current:.2f} FPS | "
            f"⏱ ETA: {eta:.1f}s"
        )
        sys.stdout.flush()

    cap.release()
    out.release()

    total_time = time.time() - start_time
    print(f"\n\n✅ Finished processing in {total_time:.2f} seconds")
    print(f"📈 Average FPS: {total_frames / total_time:.2f}")
    print(f"💾 Output video saved: {output_path}")


# -------------------------------
# Test GPU availability
# -------------------------------
def test_gpu():
    """Test if GPU is available for ONNX Runtime"""
    try:
        import onnxruntime as ort
        print("\n🔍 Checking GPU availability...")
        print(f"ONNX Runtime version: {ort.__version__}")
        print(f"Available providers: {ort.get_available_providers()}")
        
        if 'CUDAExecutionProvider' in ort.get_available_providers():
            print("✅ GPU (CUDA) is available!")
            return True
        else:
            print("⚠️ GPU not available. Will use CPU.")
            return False
    except ImportError:
        print("⚠️ ONNX Runtime not installed")
        return False


# -------------------------------
# RUN
# -------------------------------
if __name__ == "__main__":
    # Test GPU first
    gpu_available = test_gpu()
    
    print("\n" + "="*50)
    
    video_path = r"videos\cam1001_test.mp4"
    dataset_folder = "Office_staff_photos"

    # Create output filename: video_output.mp4
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    output_path = f"{base_name}_output_gpu.mp4"

    face_recognition_from_video(
        video_path,
        dataset_folder,
        output_path,
        threshold=0.3,
        use_gpu=gpu_available  # Automatically use GPU if available
    )