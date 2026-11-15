"""Fase 6 — Contagem de pedestres.

Rastreia pessoas (YOLO + ByteTrack, classe 0) e acompanha o ponto inferior
central da caixa (os pés). Dois modos, que podem ser usados juntos:

1. Linhas (`pedestrian_lines` da cena, por nome): cada ID que cruza a linha é
   contado uma vez, com o sentido (in/out, mesma convenção da counting_line).
   É a mesma lógica do count.py (Fase 3). Usos típicos:
     - linha no eixo da rua: quem a cruza atravessou a rua (com ou sem faixa);
     - linha atravessando a calçada: fluxo de pedestres na calçada.
2. Faixa (`crosswalk_zone`, polígono): para cenas com faixa marcada. Para cada
   ID que entra na faixa:
     - presença: tempo dentro da zona (só conta com pelo menos `--min-dwell` s);
     - travessia: deslocamento dentro da zona ao longo do eixo maior da faixa de
       pelo menos `--min-travel` vezes o comprimento dela; quem só espera na
       esquina, dentro do polígono, conta como presença e não como travessia;
     - sentido "ab" ou "ba" (A = ponta esquerda, ou de cima, do eixo maior).
   O instante do evento é a entrada na faixa.

Saídas em outputs/<video>_pedestres/:
    crossings.csv         um evento por pessoa e linha (coluna `line`)
    counts_total.csv      totais por linha e sentido
    counts_per_window.csv totais por janela, linha e sentido
    zone_events.csv       eventos da faixa (se houver crosswalk_zone)
    occupancy.csv         pessoas dentro da faixa por quadro (idem)
    annotated.mp4         (opcional) vídeo anotado
    run_info.yaml         parâmetros, FPS e totais

Uso:
    python -m src.pedestrians --config configs/aic21_cam_19.yaml
    python -m src.pedestrians --config configs/aic21_cam_19.yaml --video
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import supervision as sv
import yaml

from src.config import PERSON_CLASS_ID, SceneConfig
from src.count import VehicleCounter
from src.detect_track import draw_scene, make_annotators, track_video


# ---------------------------------------------------------------- linhas

def line_counter(points: list[list[int]]) -> VehicleCounter:
    (x1, y1), (x2, y2) = points
    line = sv.LineZone(start=sv.Point(x1, y1), end=sv.Point(x2, y2),
                       triggering_anchors=[sv.Position.BOTTOM_CENTER])
    return VehicleCounter(line=line, direction_names=("in", "out"))


def in_arrow(points: list[list[int]], size: float = 40) -> tuple[tuple[int, int], tuple[int, int]]:
    """Seta a partir do meio da linha apontando para o lado "in"."""
    a, b = np.array(points[0], float), np.array(points[1], float)
    d = (b - a) / np.linalg.norm(b - a)
    n = np.array([d[1], -d[0]])  # "in" = lado esquerdo de quem olha de A para B
    m = (a + b) / 2
    return tuple(m.astype(int)), tuple((m + n * size).astype(int))


# ---------------------------------------------------------------- faixa

def crosswalk_axis(polygon: list[list[int]]) -> tuple[np.ndarray, np.ndarray]:
    """Pontas A e B do eixo maior do retângulo mínimo que contém a faixa."""
    (cx, cy), (w, h), ang = cv2.minAreaRect(np.array(polygon, dtype=np.float32))
    length = max(w, h)
    theta = np.deg2rad(ang if w >= h else ang + 90)
    u = np.array([np.cos(theta), np.sin(theta)])
    if abs(u[0]) >= abs(u[1]):
        u = u if u[0] > 0 else -u   # A à esquerda
    else:
        u = u if u[1] > 0 else -u   # A em cima
    c = np.array([cx, cy])
    return c - u * length / 2, c + u * length / 2


@dataclass
class ZoneTrack:
    first_frame: int
    first_point: np.ndarray
    last_frame: int
    last_point: np.ndarray
    frames_inside: int = 1


@dataclass
class CrosswalkCounter:
    zone: np.ndarray
    a: np.ndarray
    b: np.ndarray
    fps: float
    min_dwell_s: float = 0.5
    min_travel: float = 0.4
    lost_s: float = 2.0
    tracks: dict[int, ZoneTrack] = field(default_factory=dict)
    last_seen: dict[int, int] = field(default_factory=dict)
    events: list[dict] = field(default_factory=list)
    occupancy: list[tuple[int, int]] = field(default_factory=list)

    @property
    def length(self) -> float:
        return float(np.linalg.norm(self.b - self.a))

    def update(self, frame: int, det: sv.Detections) -> int:
        feet = det.get_anchors_coordinates(sv.Position.BOTTOM_CENTER) if len(det) else np.empty((0, 2))
        n_in = 0
        for tid, p in zip(det.tracker_id, feet):
            tid = int(tid)
            self.last_seen[tid] = frame
            if cv2.pointPolygonTest(self.zone, (float(p[0]), float(p[1])), False) < 0:
                continue
            n_in += 1
            t = self.tracks.get(tid)
            if t is None:
                self.tracks[tid] = ZoneTrack(frame, p.copy(), frame, p.copy())
            else:
                t.last_frame, t.last_point = frame, p.copy()
                t.frames_inside += 1
        self.occupancy.append((frame, n_in))
        lost = int(self.lost_s * self.fps)
        for tid in [t for t, f in self.last_seen.items() if frame - f > lost]:
            self._finish(tid)
        return n_in

    def _finish(self, tid: int) -> None:
        self.last_seen.pop(tid, None)
        t = self.tracks.pop(tid, None)
        if t is None or t.frames_inside / self.fps < self.min_dwell_s:
            return
        axis = (self.b - self.a) / self.length
        travel = float(np.dot(t.last_point - t.first_point, axis)) / self.length
        self.events.append({
            "track_id": tid, "frame": t.first_frame, "time_s": round(t.first_frame / self.fps, 3),
            "exit_time_s": round(t.last_frame / self.fps, 3),
            "dwell_s": round(t.frames_inside / self.fps, 2), "travel_frac": round(travel, 3),
            "direction": "ab" if travel > 0 else "ba", "crossed": abs(travel) >= self.min_travel,
            "class": "pessoa",
        })

    def flush(self) -> None:
        for tid in list(self.tracks):
            self._finish(tid)


# ---------------------------------------------------------------- vídeo

def draw_overlay(img: np.ndarray, scene: SceneConfig, lines: dict[str, VehicleCounter],
                 zone: CrosswalkCounter | None, n_in: int, scale: float) -> np.ndarray:
    rows = []
    for name, pts in (scene.pedestrian_lines or {}).items():
        p, q = in_arrow(pts, 40 * scale)
        cv2.arrowedLine(img, p, q, (0, 165, 255), 2, tipLength=0.3)
        cv2.putText(img, "in", q, cv2.FONT_HERSHEY_SIMPLEX, 0.6 * scale, (0, 165, 255), 2)
        c = lines[name].crossings
        rows.append(f"{name}: in {sum(x.direction == 'in' for x in c)}  out {sum(x.direction == 'out' for x in c)}")
    if zone is not None:
        a, b = tuple(zone.a.astype(int)), tuple(zone.b.astype(int))
        cv2.arrowedLine(img, a, b, (0, 255, 255), 2, tipLength=0.04)
        cv2.putText(img, "A", a, cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
        cv2.putText(img, "B", b, cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
        done = [e for e in zone.events if e["crossed"]]
        rows += [f"na faixa: {n_in}",
                 f"faixa A->B {sum(e['direction'] == 'ab' for e in done)}  "
                 f"B->A {sum(e['direction'] == 'ba' for e in done)}"]
    s = max(0.6, scale)
    cv2.rectangle(img, (0, 0), (int(460 * s), int(16 * s + 32 * s * len(rows))), (0, 0, 0), -1)
    for i, t in enumerate(rows):
        cv2.putText(img, t, (10, int((36 + 32 * i) * s)), cv2.FONT_HERSHEY_SIMPLEX, 0.8 * s, (255, 255, 255), 2)
    return img


def main() -> None:
    ap = argparse.ArgumentParser(description="Contagem de pedestres (linhas e faixa)")
    ap.add_argument("--config", required=True, help="cena com pedestrian_lines e/ou crosswalk_zone")
    ap.add_argument("--source", help="vídeo (padrão: o da configuração)")
    ap.add_argument("--model", default="models/yolo11s.pt")
    ap.add_argument("--imgsz", type=int, default=1280, help="pessoas são pequenas: use resolução alta")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--device", default="0")
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--window", type=float, default=60.0)
    ap.add_argument("--min-dwell", type=float, default=0.5, help="faixa: tempo mínimo dentro (s)")
    ap.add_argument("--min-travel", type=float, default=0.4, help="faixa: fração do comprimento percorrida")
    ap.add_argument("--lost", type=float, default=2.0, help="faixa: segundos sem ver o ID para encerrá-lo")
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--video", action="store_true", help="grava vídeo anotado")
    args = ap.parse_args()

    scene = SceneConfig.load(args.config)
    if not scene.pedestrian_lines and not scene.crosswalk_zone:
        raise SystemExit(f"A cena '{scene.name}' não tem pedestrian_lines nem crosswalk_zone "
                         "(src.scene_editor, teclas w e f).")
    source = Path(args.source or scene.video)
    info = sv.VideoInfo.from_video_path(str(source))
    fps = info.fps or 10.0
    lines = {name: line_counter(pts) for name, pts in (scene.pedestrian_lines or {}).items()}
    zone = None
    if scene.crosswalk_zone:
        a, b = crosswalk_axis(scene.crosswalk_zone)
        zone = CrosswalkCounter(np.array(scene.crosswalk_zone, dtype=np.int32), a, b, fps,
                                args.min_dwell, args.min_travel, args.lost)
    out_dir = Path(args.out) / f"{source.stem}_pedestres"
    out_dir.mkdir(parents=True, exist_ok=True)
    device = int(args.device) if args.device.isdigit() else args.device
    ann = make_annotators(info)
    sink = sv.VideoSink(str(out_dir / "annotated.mp4"), info) if args.video else None

    n, ids = 0, set()
    t0 = time.perf_counter()
    try:
        if sink:
            sink.__enter__()
        for fr in track_video(source, args.model, args.conf, args.imgsz, device,
                              classes=[PERSON_CLASS_ID], max_frames=args.max_frames):
            n += 1
            ids.update(int(t) for t in fr.detections.tracker_id)
            for lc in lines.values():
                lc.update(fr.detections, fr.index, fr.time_s)
            n_in = zone.update(fr.index, fr.detections) if zone else 0
            if sink:
                img = draw_scene(fr.frame.copy(), scene)
                img = ann.draw(img, fr.detections)
                sink.write_frame(draw_overlay(img, scene, lines, zone, n_in, ann.text_scale))
    finally:
        if sink:
            sink.__exit__(None, None, None)
    elapsed = time.perf_counter() - t0

    cols = ["frame", "time_s", "track_id", "direction", "class_at_crossing", "class", "line"]
    parts = [lc.to_dataframe().assign(line=name) for name, lc in lines.items()]
    df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=cols)
    df = df.reindex(columns=cols).sort_values("frame")
    df.to_csv(out_dir / "crossings.csv", index=False)
    total = df.groupby(["line", "direction"]).size().unstack(fill_value=0) if len(df) else pd.DataFrame()
    total.to_csv(out_dir / "counts_total.csv")
    win = df.assign(window_start_s=(df.time_s // args.window * args.window).astype(int))
    win.groupby(["window_start_s", "line", "direction"]).size().rename("count").reset_index().to_csv(
        out_dir / "counts_per_window.csv", index=False)

    run = {"source": str(source), "config": args.config, "model": args.model, "imgsz": args.imgsz,
           "frames": n, "seconds": round(elapsed, 1), "fps": round(n / elapsed, 1) if elapsed else 0.0,
           "person_ids": len(ids),
           "lines": {name: {d: int(v) for d, v in g.direction.value_counts().items()}
                     for name, g in df.groupby("line")}}
    if zone:
        zone.flush()
        z = pd.DataFrame(zone.events, columns=["track_id", "frame", "time_s", "exit_time_s", "dwell_s",
                                               "travel_frac", "direction", "crossed", "class"])
        z.to_csv(out_dir / "zone_events.csv", index=False)
        occ = pd.DataFrame(zone.occupancy, columns=["frame", "pessoas_na_faixa"])
        occ.to_csv(out_dir / "occupancy.csv", index=False)
        cr = z[z.crossed]
        run["faixa"] = {"travessias_ab": int((cr.direction == "ab").sum()),
                        "travessias_ba": int((cr.direction == "ba").sum()),
                        "presenca_sem_travessia": int((~z.crossed).sum()),
                        "max_simultaneos": int(occ.pessoas_na_faixa.max()) if len(occ) else 0}
    (out_dir / "run_info.yaml").write_text(yaml.safe_dump(run, sort_keys=False, allow_unicode=True),
                                           encoding="utf-8")
    print(f"\n{n} quadros em {elapsed:.1f}s -> {run['fps']} FPS, {len(ids)} IDs de pessoa")
    print(total.to_string() if len(total) else "Nenhum cruzamento de linha.")
    if zone:
        print("Faixa:", run["faixa"])
    print(f"Saída em {out_dir}")


if __name__ == "__main__":
    main()
