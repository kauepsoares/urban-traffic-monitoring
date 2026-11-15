"""Fase 2 — Detecção (YOLO) + rastreamento (ByteTrack).

Gera, para qualquer vídeo:
- outputs/<nome>/tracks.csv  : uma linha por objeto por quadro
  (frame, time_s, track_id, class_id, class_name, conf, x1, y1, x2, y2)
- outputs/<nome>/annotated.mp4 : vídeo com caixas, IDs e trajetórias
- resumo de FPS no terminal (e em outputs/<nome>/run_info.yaml)

Uso:
    python -m src.detect_track --source data/clips/video.mp4
    python -m src.detect_track --source video.mp4 --model yolo11s.pt --config configs/cena_01.yaml --show
"""

from __future__ import annotations

import argparse
import csv
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import supervision as sv
import yaml
from ultralytics import YOLO

from src.config import COCO_CLASSES, SceneConfig

TRACKED_CLASS_IDS = list(COCO_CLASSES.keys())


@dataclass
class FrameResult:
    index: int
    time_s: float
    frame: np.ndarray
    detections: sv.Detections


def track_video(
    source: str | Path,
    model_path: str = "yolo11n.pt",
    conf: float = 0.3,
    imgsz: int = 640,
    device: str | int = 0,
    classes: list[int] | None = None,
    tracker: str = "bytetrack.yaml",
    max_frames: int | None = None,
) -> Iterator[FrameResult]:
    """Roda YOLO + ByteTrack quadro a quadro e devolve as detecções rastreadas.

    É um gerador: os módulos de contagem, semáforo e infração consomem os
    quadros um a um, sem carregar o vídeo inteiro na memória.
    """
    model = YOLO(model_path)
    fps = sv.VideoInfo.from_video_path(str(source)).fps or 30
    results = model.track(
        source=str(source),
        stream=True,
        persist=True,
        tracker=tracker,
        classes=classes if classes is not None else TRACKED_CLASS_IDS,
        conf=conf,
        imgsz=imgsz,
        device=device,
        verbose=False,
    )
    for i, result in enumerate(results):
        if max_frames is not None and i >= max_frames:
            break
        detections = sv.Detections.from_ultralytics(result)
        # Sem IDs (nenhum objeto confirmado pelo ByteTrack neste quadro)
        if detections.tracker_id is None:
            detections = sv.Detections.empty()
            detections.tracker_id = np.empty(0, dtype=int)
        yield FrameResult(index=i, time_s=i / fps, frame=result.orig_img, detections=detections)


@dataclass
class Annotators:
    box: sv.BoxAnnotator
    label: sv.LabelAnnotator
    trace: sv.TraceAnnotator
    thickness: int
    text_scale: float

    def draw(self, img: np.ndarray, det: sv.Detections) -> np.ndarray:
        labels = [f"#{t} {COCO_CLASSES.get(int(c), c)}" for t, c in zip(det.tracker_id, det.class_id)]
        img = self.trace.annotate(img, det)
        img = self.box.annotate(img, det)
        return self.label.annotate(img, det, labels=labels)


def make_annotators(video_info: sv.VideoInfo) -> Annotators:
    """Espessura e fonte proporcionais à resolução (legível de 540p a 4K)."""
    thickness = sv.calculate_optimal_line_thickness(video_info.resolution_wh)
    text_scale = sv.calculate_optimal_text_scale(video_info.resolution_wh)
    return Annotators(
        box=sv.BoxAnnotator(thickness=thickness),
        label=sv.LabelAnnotator(text_scale=text_scale, text_thickness=thickness),
        trace=sv.TraceAnnotator(thickness=thickness, trace_length=40),
        thickness=thickness,
        text_scale=text_scale,
    )


def draw_scene(frame: np.ndarray, scene: SceneConfig) -> np.ndarray:
    """Desenha as linhas/zonas configuradas da cena (apenas visualização)."""
    if scene.counting_line:
        a, b = scene.counting_line
        cv2.line(frame, tuple(a), tuple(b), (255, 200, 0), 2)
        cv2.putText(frame, "contagem", tuple(a), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 2)
    if scene.stop_line:
        a, b = scene.stop_line
        cv2.line(frame, tuple(a), tuple(b), (0, 0, 255), 2)
        cv2.putText(frame, "retencao", tuple(a), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    if scene.crosswalk_zone:
        pts = np.array(scene.crosswalk_zone, dtype=np.int32)
        cv2.polylines(frame, [pts], True, (0, 255, 255), 2)
    for name, (a, b) in (scene.pedestrian_lines or {}).items():
        cv2.line(frame, tuple(a), tuple(b), (0, 165, 255), 2)
        cv2.putText(frame, name, tuple(a), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
    if scene.traffic_light_roi:
        x1, y1, x2, y2 = scene.traffic_light_roi
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)
    return frame


def main() -> None:
    ap = argparse.ArgumentParser(description="YOLO + ByteTrack em um vídeo")
    ap.add_argument("--source", required=True, help="caminho do vídeo")
    ap.add_argument("--model", default="yolo11n.pt", help="pesos YOLO (n, s, m...)")
    ap.add_argument("--config", help="configs/cena_XX.yaml (opcional, só desenha as regiões)")
    ap.add_argument("--out", default="outputs", help="pasta de saída")
    ap.add_argument("--conf", type=float, default=0.3)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="0", help="0 = GPU, cpu = CPU")
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--no-video", action="store_true", help="não grava o vídeo anotado")
    ap.add_argument("--show", action="store_true", help="mostra a janela durante o processamento")
    args = ap.parse_args()

    source = Path(args.source)
    out_dir = Path(args.out) / f"{source.stem}_{Path(args.model).stem}"
    out_dir.mkdir(parents=True, exist_ok=True)
    scene = SceneConfig.load(args.config) if args.config else None
    device = int(args.device) if args.device.isdigit() else args.device

    video_info = sv.VideoInfo.from_video_path(str(source))
    annotators = make_annotators(video_info)

    sink = None if args.no_video else sv.VideoSink(str(out_dir / "annotated.mp4"), video_info)
    csv_file = open(out_dir / "tracks.csv", "w", newline="", encoding="utf-8")
    writer = csv.writer(csv_file)
    writer.writerow(["frame", "time_s", "track_id", "class_id", "class_name", "conf", "x1", "y1", "x2", "y2"])

    n_frames, track_ids = 0, set()
    t0 = time.perf_counter()
    try:
        if sink:
            sink.__enter__()
        for fr in track_video(
            source, args.model, args.conf, args.imgsz, device, max_frames=args.max_frames
        ):
            n_frames += 1
            det = fr.detections
            for xyxy, cls, tid, cf in zip(det.xyxy, det.class_id, det.tracker_id, det.confidence):
                track_ids.add(int(tid))
                writer.writerow([
                    fr.index, f"{fr.time_s:.3f}", int(tid), int(cls),
                    COCO_CLASSES.get(int(cls), str(cls)), f"{cf:.3f}",
                    *(f"{v:.1f}" for v in xyxy),
                ])

            if sink or args.show:
                img = fr.frame.copy()
                if scene:
                    img = draw_scene(img, scene)
                img = annotators.draw(img, det)
                if sink:
                    sink.write_frame(img)
                if args.show:
                    cv2.imshow("detect_track (q para sair)", img)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
    finally:
        if sink:
            sink.__exit__(None, None, None)
        csv_file.close()
        cv2.destroyAllWindows()

    elapsed = time.perf_counter() - t0
    fps = n_frames / elapsed if elapsed > 0 else 0.0
    info = {
        "source": str(source),
        "model": args.model,
        "frames": n_frames,
        "seconds": round(elapsed, 2),
        "fps": round(fps, 2),
        "video_fps": video_info.fps,
        "resolution": [video_info.width, video_info.height],
        "unique_tracks": len(track_ids),
        "saved_video": not args.no_video,
    }
    (out_dir / "run_info.yaml").write_text(yaml.safe_dump(info, sort_keys=False), encoding="utf-8")
    extra = "inclui gravação do vídeo" if sink else "sem gravação do vídeo"
    print(f"\n{n_frames} quadros em {elapsed:.1f}s -> {fps:.1f} FPS ({extra})")
    print(f"{len(track_ids)} IDs únicos. Saída em {out_dir}")


if __name__ == "__main__":
    main()
