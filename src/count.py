"""Fase 3 — Contagem de veículos por linha virtual.

Cada ID de veículo que cruza a `counting_line` da cena é contado uma única vez,
por classe e por sentido, e agregado em janelas de tempo.

A classe final de cada veículo é a mais votada em todos os quadros em que ele
apareceu (o YOLO às vezes alterna carro/caminhão no mesmo objeto). A classe no
momento do cruzamento também é salva, para comparação.

Saídas em outputs/<video>_<modelo>_contagem/:
    crossings.csv         um evento por veículo contado
    counts_total.csv      totais por classe e sentido
    counts_per_window.csv totais por janela de tempo, classe e sentido
    annotated.mp4         vídeo com a linha e o placar
    run_info.yaml         parâmetros, FPS e totais

Uso:
    python -m src.count --source data/clips/video.mp4 --config configs/cena_01.yaml
"""

from __future__ import annotations

import argparse
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import supervision as sv
import yaml

from src.config import COCO_CLASSES, VEHICLE_CLASS_IDS, SceneConfig
from src.detect_track import draw_scene, make_annotators, track_video


@dataclass
class Crossing:
    frame: int
    time_s: float
    track_id: int
    direction: str
    class_at_crossing: str


@dataclass
class VehicleCounter:
    """Conta cruzamentos da linha virtual, uma vez por ID."""

    line: sv.LineZone
    direction_names: tuple[str, str]
    class_votes: dict[int, Counter] = field(default_factory=lambda: defaultdict(Counter))
    counted_ids: set[int] = field(default_factory=set)
    crossings: list[Crossing] = field(default_factory=list)

    @classmethod
    def from_scene(cls, scene: SceneConfig) -> "VehicleCounter":
        if not scene.counting_line:
            raise SystemExit(f"A cena '{scene.name}' não tem counting_line. Use src.scene_editor.")
        (x1, y1), (x2, y2) = scene.counting_line
        line = sv.LineZone(
            start=sv.Point(x1, y1),
            end=sv.Point(x2, y2),
            # Ponto de contato com o chão: menos sensível à perspectiva que a caixa inteira
            triggering_anchors=[sv.Position.BOTTOM_CENTER],
        )
        names = tuple(scene.direction_names[:2]) if len(scene.direction_names) >= 2 else ("in", "out")
        return cls(line=line, direction_names=names)

    def majority_class(self, track_id: int) -> str:
        votes = self.class_votes.get(track_id)
        if not votes:
            return "desconhecido"
        return COCO_CLASSES.get(votes.most_common(1)[0][0], "desconhecido")

    def update(self, det: sv.Detections, frame: int, time_s: float) -> None:
        for tid, cls in zip(det.tracker_id, det.class_id):
            self.class_votes[int(tid)][int(cls)] += 1
        if len(det) == 0:
            return
        crossed_in, crossed_out = self.line.trigger(det)
        for i in np.flatnonzero(crossed_in | crossed_out):
            tid = int(det.tracker_id[i])
            if tid in self.counted_ids:  # já contado (ex.: oscilou sobre a linha)
                continue
            self.counted_ids.add(tid)
            direction = self.direction_names[0] if crossed_in[i] else self.direction_names[1]
            self.crossings.append(Crossing(
                frame=frame,
                time_s=round(time_s, 3),
                track_id=tid,
                direction=direction,
                class_at_crossing=COCO_CLASSES.get(int(det.class_id[i]), "desconhecido"),
            ))

    def live_totals(self) -> Counter:
        """Totais (classe, sentido) usando a classe mais votada até agora."""
        return Counter((self.majority_class(c.track_id), c.direction) for c in self.crossings)

    def to_dataframe(self) -> pd.DataFrame:
        cols = ["frame", "time_s", "track_id", "direction", "class_at_crossing", "class"]
        if not self.crossings:
            return pd.DataFrame(columns=cols)
        df = pd.DataFrame([c.__dict__ for c in self.crossings])
        df["class"] = df["track_id"].map(self.majority_class)
        return df[cols]


def summarize(df: pd.DataFrame, window_s: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Totais por classe/sentido e por janela de tempo."""
    total = (
        df.groupby(["class", "direction"]).size().unstack(fill_value=0)
        if len(df) else pd.DataFrame()
    )
    if len(total):
        total["total"] = total.sum(axis=1)
        total.loc["total"] = total.sum()
    windows = df.assign(window_start_s=(df["time_s"] // window_s * window_s).astype(int))
    per_window = (
        windows.groupby(["window_start_s", "class", "direction"]).size().rename("count").reset_index()
        if len(df) else pd.DataFrame(columns=["window_start_s", "class", "direction", "count"])
    )
    return total, per_window


def draw_scoreboard(img: np.ndarray, totals: Counter, names: tuple[str, str], scale: float) -> np.ndarray:
    """Tabela no canto superior esquerdo: linhas = classes, colunas = sentidos."""
    rows: list[list[str]] = [["", names[0], names[1]]]
    for c in (COCO_CLASSES[i] for i in VEHICLE_CLASS_IDS):
        rows.append([c, str(totals[(c, names[0])]), str(totals[(c, names[1])])])
    rows.append(["total"] + [str(sum(v for (_, d), v in totals.items() if d == n)) for n in names])

    font, fs = cv2.FONT_HERSHEY_SIMPLEX, scale * 1.2
    th = max(1, int(round(fs * 2)))
    size = lambda s: cv2.getTextSize(s, font, fs, th)[0]  # noqa: E731
    pad = size("M")[0]
    col_w = [max(size(r[j])[0] for r in rows) + 2 * pad for j in range(3)]
    step = int(size("Ag")[1] * 1.9)
    w, h = sum(col_w) + pad, step * len(rows) + step // 2

    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
    img = cv2.addWeighted(overlay, 0.6, img, 0.4, 0)
    for i, row in enumerate(rows):
        x = pad
        for j, text in enumerate(row):
            # nomes à esquerda, números alinhados à direita da coluna
            tx = x if j == 0 else x + col_w[j] - pad - size(text)[0]
            cv2.putText(img, text, (tx, step * (i + 1)), font, fs, (255, 255, 255), th, cv2.LINE_AA)
            x += col_w[j]
    return img


def main() -> None:
    ap = argparse.ArgumentParser(description="Contagem de veículos por linha virtual")
    ap.add_argument("--source", required=True)
    ap.add_argument("--config", required=True, help="configs/cena_XX.yaml com counting_line")
    ap.add_argument("--model", default="models/yolo11n.pt")
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--window", type=float, default=60.0, help="janela de agregação em segundos")
    ap.add_argument("--conf", type=float, default=0.3)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="0")
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()

    source = Path(args.source)
    scene = SceneConfig.load(args.config)
    counter = VehicleCounter.from_scene(scene)
    out_dir = Path(args.out) / f"{source.stem}_{Path(args.model).stem}_contagem"
    out_dir.mkdir(parents=True, exist_ok=True)
    device = int(args.device) if args.device.isdigit() else args.device

    video_info = sv.VideoInfo.from_video_path(str(source))
    ann = make_annotators(video_info)
    line_ann = sv.LineZoneAnnotator(
        thickness=ann.thickness, text_thickness=ann.thickness, text_scale=ann.text_scale,
        custom_in_text=counter.direction_names[0], custom_out_text=counter.direction_names[1],
    )
    sink = None if args.no_video else sv.VideoSink(str(out_dir / "annotated.mp4"), video_info)

    n_frames = 0
    t0 = time.perf_counter()
    try:
        if sink:
            sink.__enter__()
        for fr in track_video(
            source, args.model, args.conf, args.imgsz, device,
            classes=VEHICLE_CLASS_IDS, max_frames=args.max_frames,
        ):
            n_frames += 1
            counter.update(fr.detections, fr.index, fr.time_s)
            if sink or args.show:
                img = draw_scene(fr.frame.copy(), scene)
                img = ann.draw(img, fr.detections)
                img = line_ann.annotate(img, counter.line)
                img = draw_scoreboard(img, counter.live_totals(), counter.direction_names, ann.text_scale)
                if sink:
                    sink.write_frame(img)
                if args.show:
                    cv2.imshow("contagem (q para sair)", img)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
    finally:
        if sink:
            sink.__exit__(None, None, None)
        cv2.destroyAllWindows()
    elapsed = time.perf_counter() - t0

    df = counter.to_dataframe()
    total, per_window = summarize(df, args.window)
    df.to_csv(out_dir / "crossings.csv", index=False)
    total.to_csv(out_dir / "counts_total.csv")
    per_window.to_csv(out_dir / "counts_per_window.csv", index=False)

    changed = int((df["class"] != df["class_at_crossing"]).sum()) if len(df) else 0
    info = {
        "source": str(source),
        "config": args.config,
        "model": args.model,
        "frames": n_frames,
        "seconds": round(elapsed, 2),
        "fps": round(n_frames / elapsed, 2) if elapsed else 0.0,
        "window_s": args.window,
        "vehicles_counted": len(df),
        "class_corrected_by_vote": changed,
        "totals": {f"{c}/{d}": int(v) for (c, d), v in df.groupby(["class", "direction"]).size().items()},
    }
    (out_dir / "run_info.yaml").write_text(yaml.safe_dump(info, sort_keys=False, allow_unicode=True), encoding="utf-8")

    print(f"\n{n_frames} quadros em {elapsed:.1f}s -> {info['fps']} FPS")
    print(f"{len(df)} veículos contados ({changed} com classe corrigida pela votação)\n")
    print(total.to_string() if len(total) else "Nenhum cruzamento.")
    print(f"\nSaída em {out_dir}")


if __name__ == "__main__":
    main()
