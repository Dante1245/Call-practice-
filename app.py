import argparse
import base64
import logging
import os
import queue
import sys
import threading
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np

try:
    import requests
except Exception:  # pragma: no cover
    requests = None

try:
    import pyvirtualcam
except Exception:  # pragma: no cover
    pyvirtualcam = None

try:
    import mediapipe as mp
except Exception:  # pragma: no cover
    mp = None


logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
LOG = logging.getLogger("enhanced-face-swap")


@dataclass
class AppConfig:
    camera_index: int = 0
    mode: str = "face_swap"  # face_swap | prompt
    prompt: str = "anime style portrait"
    intensity: float = 0.7
    quality: str = "balanced"  # high | balanced | low
    output_path: str = "output_recording.mp4"
    enable_virtual_camera: bool = False
    use_decart: bool = True
    model: str = "lucy-2.0"
    task_prompt: str = "video_to_video"
    task_faceswap: str = "real_time_video_editing"


@dataclass
class FaceRegion:
    bbox: Tuple[int, int, int, int]
    landmarks: Optional[np.ndarray] = None  # (N, 2)


class QualityManager:
    QUALITY_SCALES = {
        "high": 1.0,
        "balanced": 0.8,
        "low": 0.6,
    }

    @classmethod
    def resize_for_quality(cls, frame: np.ndarray, quality: str) -> np.ndarray:
        scale = cls.QUALITY_SCALES.get(quality, 0.8)
        h, w = frame.shape[:2]
        return cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)


class Metrics:
    def __init__(self):
        self.fps = 0.0
        self.latency_ms = 0.0
        self._last_t = time.perf_counter()

    def tick(self, latency_ms: float):
        now = time.perf_counter()
        dt = now - self._last_t
        self._last_t = now
        fps_instant = 1.0 / dt if dt > 0 else 0.0
        self.fps = 0.8 * self.fps + 0.2 * fps_instant if self.fps else fps_instant
        self.latency_ms = 0.8 * self.latency_ms + 0.2 * latency_ms if self.latency_ms else latency_ms


class DecartClient:
    """Endpoint-agnostic Decart adapter with graceful local fallback support."""

    def __init__(self, model: str):
        self.api_key = os.getenv("DECART_API_KEY", "")
        self.base_url = os.getenv("DECART_BASE_URL", "https://api.decart.ai")
        self.model = os.getenv("DECART_MODEL", model)
        self.timeout = int(os.getenv("DECART_TIMEOUT_MS", "1200")) / 1000.0
        self.enabled = bool(self.api_key and requests is not None)

    @staticmethod
    def _encode_frame(frame: np.ndarray, quality: int = 90) -> str:
        ok, data = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        if not ok:
            raise RuntimeError("failed to encode image")
        return base64.b64encode(data.tobytes()).decode("utf-8")

    @staticmethod
    def _decode_frame(image_b64: str) -> Optional[np.ndarray]:
        if not image_b64:
            return None
        try:
            raw = base64.b64decode(image_b64)
            arr = np.frombuffer(raw, dtype=np.uint8)
            return cv2.imdecode(arr, cv2.IMREAD_COLOR)
        except Exception:
            return None

    def _post(self, endpoint: str, payload: dict) -> Optional[dict]:
        if not self.enabled:
            return None
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        url = f"{self.base_url.rstrip('/')}{endpoint}"
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=self.timeout)
            if resp.status_code != 200:
                LOG.warning("Decart request failed status=%s endpoint=%s", resp.status_code, endpoint)
                return None
            return resp.json()
        except Exception as exc:
            LOG.warning("Decart request error: %s", exc)
            return None

    def prompt_transform(self, frame: np.ndarray, prompt: str, strength: float, task: str) -> Optional[np.ndarray]:
        payload = {
            "model": self.model,
            "mode": "lucy_edit",
            "task": task,
            "prompt": prompt,
            "strength": float(np.clip(strength, 0.0, 1.0)),
            "frame_base64": self._encode_frame(frame),
        }
        data = self._post("/v1/realtime/transform", payload)
        if not data:
            return None
        return self._decode_frame(data.get("image_base64", ""))

    def reference_faceswap(self, frame: np.ndarray, reference: np.ndarray, strength: float, task: str) -> Optional[np.ndarray]:
        payload = {
            "model": self.model,
            "mode": "face_swap",
            "task": task,
            "strength": float(np.clip(strength, 0.0, 1.0)),
            "frame_base64": self._encode_frame(frame),
            "reference_base64": self._encode_frame(reference),
        }
        data = self._post("/v1/realtime/transform", payload)
        if not data:
            return None
        return self._decode_frame(data.get("image_base64", ""))

    def text_to_video(self, prompt: str, seconds: int = 4) -> Optional[dict]:
        payload = {
            "model": self.model,
            "task": "text_to_video",
            "prompt": prompt,
            "duration_seconds": max(1, int(seconds)),
        }
        return self._post("/v1/generate", payload)


class FaceAnalyzer:
    def __init__(self):
        self.haar = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        self.mesh = None
        if mp is not None:
            self.mesh = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=4,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )

    def detect_faces(self, frame: np.ndarray) -> List[FaceRegion]:
        if self.mesh is not None:
            return self._detect_mediapipe(frame)
        return self._detect_haar(frame)

    def _detect_haar(self, frame: np.ndarray) -> List[FaceRegion]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.haar.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(64, 64))
        return [FaceRegion((int(x), int(y), int(w), int(h)), None) for (x, y, w, h) in faces]

    def _detect_mediapipe(self, frame: np.ndarray) -> List[FaceRegion]:
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.mesh.process(rgb)
        if not result.multi_face_landmarks:
            return []
        out: List[FaceRegion] = []
        for face_lm in result.multi_face_landmarks:
            pts = np.array([(int(lm.x * w), int(lm.y * h)) for lm in face_lm.landmark], dtype=np.int32)
            x, y, bw, bh = cv2.boundingRect(pts)
            out.append(FaceRegion((x, y, bw, bh), pts))
        return out


class LocalPromptTransformer:
    def apply(self, frame: np.ndarray, prompt: str, intensity: float) -> np.ndarray:
        amount = float(np.clip(intensity, 0.0, 1.0))
        p = prompt.lower()
        if "anime" in p:
            out = self._anime(frame)
        elif "cyberpunk" in p:
            out = self._cyberpunk(frame)
        elif "oil" in p or "painting" in p:
            out = self._oil(frame)
        elif "watercolor" in p:
            out = cv2.stylization(frame, sigma_s=120, sigma_r=0.35)
        elif "sketch" in p:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            out = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        elif "night" in p:
            out = self._night_scene(frame)
        else:
            out = cv2.stylization(frame, sigma_s=90, sigma_r=0.3)
        return cv2.addWeighted(frame, 1 - amount, out, amount, 0)

    def _anime(self, frame: np.ndarray) -> np.ndarray:
        smooth = cv2.bilateralFilter(frame, d=9, sigmaColor=120, sigmaSpace=120)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edge = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 3)
        edge = cv2.cvtColor(edge, cv2.COLOR_GRAY2BGR)
        return cv2.bitwise_and(smooth, edge)

    def _cyberpunk(self, frame: np.ndarray) -> np.ndarray:
        b, g, r = cv2.split(frame)
        neon = cv2.merge(
            (
                np.clip(b * 1.4, 0, 255).astype(np.uint8),
                np.clip(g * 0.85, 0, 255).astype(np.uint8),
                np.clip(r * 1.3, 0, 255).astype(np.uint8),
            )
        )
        glow = cv2.GaussianBlur(neon, (0, 0), 2)
        return cv2.addWeighted(neon, 0.8, glow, 0.2, 0)

    def _oil(self, frame: np.ndarray) -> np.ndarray:
        if hasattr(cv2, "xphoto") and hasattr(cv2.xphoto, "oilPainting"):
            return cv2.xphoto.oilPainting(frame, 7, 1)
        return cv2.stylization(frame, sigma_s=80, sigma_r=0.25)

    def _night_scene(self, frame: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.25, 0, 255).astype(np.uint8)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * 0.55, 0, 255).astype(np.uint8)
        out = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        return cv2.applyColorMap(out, cv2.COLORMAP_OCEAN)


class LocalFaceSwapper:
    FACE_INDICES = [33, 263, 1, 61, 291, 199]  # stable facial anchors in mediapipe mesh

    def __init__(self, analyzer: FaceAnalyzer):
        self.analyzer = analyzer
        self.reference_image: Optional[np.ndarray] = None
        self.reference_face: Optional[FaceRegion] = None

    def set_reference(self, image: np.ndarray) -> bool:
        faces = self.analyzer.detect_faces(image)
        if not faces:
            return False
        # Largest face as reference
        self.reference_face = max(faces, key=lambda f: f.bbox[2] * f.bbox[3])
        self.reference_image = image.copy()
        return True

    def swap(self, frame: np.ndarray, intensity: float) -> np.ndarray:
        if self.reference_image is None or self.reference_face is None:
            return frame
        targets = self.analyzer.detect_faces(frame)
        if not targets:
            return frame

        output = frame.copy()
        for target in targets:
            output = self._swap_one(output, target, intensity)
        return output

    def _swap_one(self, frame: np.ndarray, target: FaceRegion, intensity: float) -> np.ndarray:
        x, y, w, h = target.bbox
        if target.landmarks is None or self.reference_face.landmarks is None:
            return self._bbox_swap(frame, target, intensity)

        src_pts = self._pick_landmarks(self.reference_face.landmarks)
        dst_pts = self._pick_landmarks(target.landmarks)
        if src_pts is None or dst_pts is None:
            return self._bbox_swap(frame, target, intensity)

        matrix, _ = cv2.estimateAffinePartial2D(src_pts, dst_pts)
        if matrix is None:
            return self._bbox_swap(frame, target, intensity)

        warped = cv2.warpAffine(
            self.reference_image,
            matrix,
            (frame.shape[1], frame.shape[0]),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT,
        )

        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        hull = cv2.convexHull(target.landmarks.astype(np.int32))
        cv2.fillConvexPoly(mask, hull, 255)

        center = (x + w // 2, y + h // 2)
        try:
            cloned = cv2.seamlessClone(warped, frame, mask, center, cv2.NORMAL_CLONE)
            return cv2.addWeighted(frame, 1 - intensity, cloned, intensity, 0)
        except cv2.error:
            return self._bbox_swap(frame, target, intensity)

    def _bbox_swap(self, frame: np.ndarray, target: FaceRegion, intensity: float) -> np.ndarray:
        x, y, w, h = target.bbox
        rx, ry, rw, rh = self.reference_face.bbox
        ref = self.reference_image[ry:ry + rh, rx:rx + rw]
        if ref.size == 0:
            return frame

        ref_resized = cv2.resize(ref, (w, h), interpolation=cv2.INTER_LINEAR)
        mask = 255 * np.ones((h, w), dtype=np.uint8)
        center = (x + w // 2, y + h // 2)

        try:
            blended = cv2.seamlessClone(ref_resized, frame, mask, center, cv2.NORMAL_CLONE)
            return cv2.addWeighted(frame, 1 - intensity, blended, intensity, 0)
        except cv2.error:
            roi = frame[y:y + h, x:x + w]
            if roi.shape[:2] != ref_resized.shape[:2]:
                return frame
            frame[y:y + h, x:x + w] = cv2.addWeighted(roi, 1 - intensity, ref_resized, intensity, 0)
            return frame

    def _pick_landmarks(self, points: np.ndarray) -> Optional[np.ndarray]:
        try:
            anchors = np.array([points[idx] for idx in self.FACE_INDICES], dtype=np.float32)
            return anchors
        except Exception:
            return None


class ProcessingPipeline:
    def __init__(self, config: AppConfig):
        self.config = config
        self.metrics = Metrics()
        self.analyzer = FaceAnalyzer()
        self.face_swapper = LocalFaceSwapper(self.analyzer)
        self.prompt_transformer = LocalPromptTransformer()
        self.decart = DecartClient(config.model)

    def load_reference(self, path: str) -> Tuple[bool, str]:
        if not os.path.exists(path):
            return False, f"Path not found: {path}"
        image = cv2.imread(path)
        if image is None:
            return False, "Unable to read reference image"
        ok = self.face_swapper.set_reference(image)
        if not ok:
            return False, "No detectable face found in reference"
        return True, "Reference loaded"

    def process(self, frame: np.ndarray) -> np.ndarray:
        start = time.perf_counter()

        if self.config.mode == "face_swap":
            processed = self._face_swap(frame)
        else:
            processed = self._prompt_transform(frame)

        latency_ms = (time.perf_counter() - start) * 1000
        self.metrics.tick(latency_ms)
        return processed

    def _face_swap(self, frame: np.ndarray) -> np.ndarray:
        if self.config.use_decart and self.face_swapper.reference_image is not None:
            remote = self.decart.reference_faceswap(
                frame=frame,
                reference=self.face_swapper.reference_image,
                strength=self.config.intensity,
                task=self.config.task_faceswap,
            )
            if remote is not None:
                return remote
        return self.face_swapper.swap(frame, self.config.intensity)

    def _prompt_transform(self, frame: np.ndarray) -> np.ndarray:
        if self.config.use_decart:
            remote = self.decart.prompt_transform(
                frame=frame,
                prompt=self.config.prompt,
                strength=self.config.intensity,
                task=self.config.task_prompt,
            )
            if remote is not None:
                return remote
        return self.prompt_transformer.apply(frame, self.config.prompt, self.config.intensity)


class LiveVideoApp:
    def __init__(self, config: AppConfig):
        self.config = config
        self.pipeline = ProcessingPipeline(config)
        self.cap: Optional[cv2.VideoCapture] = None
        self.writer: Optional[cv2.VideoWriter] = None
        self.virtual_cam = None
        self.recording = False
        self.running = False
        self.frame_queue: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=4)

    def run(self):
        self.cap = cv2.VideoCapture(self.config.camera_index)
        if not self.cap.isOpened():
            LOG.error("Unable to open camera index %s", self.config.camera_index)
            sys.exit(1)

        self.running = True
        threading.Thread(target=self._capture_loop, daemon=True).start()

        LOG.info("Controls: m=toggle mode, r=load ref, t=prompt, [/] intensity, 1/2/3 quality, v=record, g=t2v, q=quit")

        while self.running:
            try:
                frame = self.frame_queue.get(timeout=0.2)
            except queue.Empty:
                continue

            frame = QualityManager.resize_for_quality(frame, self.config.quality)
            processed = self.pipeline.process(frame)
            processed = self._draw_overlay(processed)

            cv2.imshow("Enhanced Face Swap + Decart Prompt Transform", processed)

            if self.recording and self.writer is not None:
                self.writer.write(processed)

            self._send_virtual_camera(processed)
            self._handle_key(cv2.waitKey(1) & 0xFF, processed)

        self.shutdown()

    def _capture_loop(self):
        while self.running and self.cap is not None:
            ok, frame = self.cap.read()
            if not ok:
                continue
            try:
                self.frame_queue.put_nowait(frame)
            except queue.Full:
                _ = self.frame_queue.get_nowait()
                self.frame_queue.put_nowait(frame)

    def _draw_overlay(self, frame: np.ndarray) -> np.ndarray:
        text = (
            f"mode={self.config.mode} | q={self.config.quality} | intensity={self.config.intensity:.2f} | "
            f"fps={self.pipeline.metrics.fps:.1f} | latency={self.pipeline.metrics.latency_ms:.1f}ms"
        )
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 34), (0, 0, 0), -1)
        cv2.putText(frame, text, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1, cv2.LINE_AA)
        return frame

    def _toggle_recording(self, shape: Tuple[int, int]):
        if not self.recording:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self.writer = cv2.VideoWriter(self.config.output_path, fourcc, 24.0, shape)
            self.recording = True
            LOG.info("Recording started -> %s", self.config.output_path)
            return
        self.recording = False
        if self.writer is not None:
            self.writer.release()
            self.writer = None
        LOG.info("Recording stopped")

    def _send_virtual_camera(self, frame: np.ndarray):
        if not self.config.enable_virtual_camera or pyvirtualcam is None:
            return
        if self.virtual_cam is None:
            self.virtual_cam = pyvirtualcam.Camera(width=frame.shape[1], height=frame.shape[0], fps=24)
        self.virtual_cam.send(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        self.virtual_cam.sleep_until_next_frame()

    def _handle_key(self, key: int, processed: np.ndarray):
        if key == ord("q"):
            self.running = False
        elif key == ord("m"):
            self.config.mode = "prompt" if self.config.mode == "face_swap" else "face_swap"
            LOG.info("Mode -> %s", self.config.mode)
        elif key == ord("r"):
            path = input("Reference image path: ").strip()
            ok, msg = self.pipeline.load_reference(path)
            LOG.info("Reference load: %s", msg)
            if ok:
                self.config.mode = "face_swap"
        elif key == ord("t"):
            prompt = input("Prompt: ").strip()
            if prompt:
                self.config.prompt = prompt
                self.config.mode = "prompt"
                LOG.info("Prompt -> %s", self.config.prompt)
        elif key == ord("["):
            self.config.intensity = max(0.0, self.config.intensity - 0.05)
        elif key == ord("]"):
            self.config.intensity = min(1.0, self.config.intensity + 0.05)
        elif key == ord("1"):
            self.config.quality = "high"
        elif key == ord("2"):
            self.config.quality = "balanced"
        elif key == ord("3"):
            self.config.quality = "low"
        elif key == ord("v"):
            self._toggle_recording((processed.shape[1], processed.shape[0]))
        elif key == ord("g"):
            self._trigger_text_to_video()

    def _trigger_text_to_video(self):
        if not self.config.use_decart:
            LOG.info("Text-to-video requires Decart mode")
            return
        prompt = input("Text-to-video prompt (blank=use current): ").strip() or self.config.prompt
        seconds_raw = input("Duration seconds (default=4): ").strip()
        seconds = int(seconds_raw) if seconds_raw.isdigit() else 4
        data = self.pipeline.decart.text_to_video(prompt, seconds)
        if data is None:
            LOG.warning("Text-to-video request failed")
            return
        LOG.info("Text-to-video response: %s", data)

    def shutdown(self):
        self.running = False
        if self.cap is not None:
            self.cap.release()
        if self.writer is not None:
            self.writer.release()
        if self.virtual_cam is not None:
            self.virtual_cam.close()
        cv2.destroyAllWindows()


def parse_args() -> AppConfig:
    parser = argparse.ArgumentParser(description="Enhanced Face Swap + Prompt-driven Video Transform App")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--mode", choices=["face_swap", "prompt"], default="face_swap")
    parser.add_argument("--prompt", default="anime style portrait")
    parser.add_argument("--quality", choices=["high", "balanced", "low"], default="balanced")
    parser.add_argument("--intensity", type=float, default=0.7)
    parser.add_argument("--no-decart", action="store_true")
    parser.add_argument("--virtual-camera", action="store_true")
    parser.add_argument("--model", default="lucy-2.0")
    parser.add_argument("--task-prompt", default="video_to_video")
    parser.add_argument("--task-faceswap", default="real_time_video_editing")
    parser.add_argument("--output", default="output_recording.mp4")

    args = parser.parse_args()
    return AppConfig(
        camera_index=args.camera,
        mode=args.mode,
        prompt=args.prompt,
        intensity=float(np.clip(args.intensity, 0.0, 1.0)),
        quality=args.quality,
        output_path=args.output,
        enable_virtual_camera=args.virtual_camera,
        use_decart=not args.no_decart,
        model=args.model,
        task_prompt=args.task_prompt,
        task_faceswap=args.task_faceswap,
    )


def main():
    config = parse_args()
    app = LiveVideoApp(config)
    app.run()


if __name__ == "__main__":
    main()
