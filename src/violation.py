"""Fase 5 — Detecção de avanço de sinal vermelho.

Junta o rastreamento (YOLO + ByteTrack) com o estado do semáforo (Fase 4):

1. Cruzamento: para cada ID de veículo, acompanha o ponto inferior central da
   caixa (contato com o chão). Há cruzamento quando esse ponto passa da linha
   de retenção (`stop_line`) no sentido de entrada do cruzamento
   (`stop_line_direction`) dentro do segmento marcado. O instante do
   cruzamento é interpolado entre os dois quadros (a 8-10 FPS um carro anda
   ~1,5 m entre quadros). Dois filtros descartam cruzamentos falsos:
     - pulo maior que a altura da caixa entre quadros seguidos (troca de ID
       do rastreador: o ID de um carro parado passa para outro que passava);
     - movimento quase paralelo à linha (ângulo > 60° com a normal), típico
       de quem atravessa pela rua transversal na ponta da linha;
     - trajetória depois da linha: ~1 s após o cruzamento o veículo precisa
       estar andando para dentro do cruzamento (até 60° da normal). Pega a
       troca de ID após uma lacuna: o ByteTrack guarda o ID de um veículo
       perdido por ~30 quadros e pode entregá-lo a outro que passa pela
       transversal; a "trajetória" resultante segue paralela à linha.
       (Descartar toda lacuna > 0,5 s também resolve, mas na cam_7 eliminou
       9% dos cruzamentos reais, de veículos encobertos por galhos e poste.)
2. Balanço da câmera: o deslocamento medido no semáforo a cada quadro é
   aplicado também à linha de retenção.
3. Estado do sinal no cruzamento: a votação temporal da Fase 4 atrasa a troca
   de cor alguns quadros (mais ainda quando os quadros da transição ficam
   "desconhecidos"). Como a decisão já espera o veículo andar, a troca é
   reposicionada em retrospectiva: a nova cor começa logo depois do último
   quadro bruto que ainda mostrava a cor antiga.
4. Decisão, `decide_after_s` segundos depois do cruzamento (ou quando o ID some):
     - o veículo precisa ter avançado além da linha (`min_travel`). Quem para
       em cima da linha fica em espera (até `max_wait_s`); se depois avançar,
       vale a cor do instante em que avançou; se não, é "parado_na_linha";
     - se entrou na `exempt_zone`, é "conversao_permitida";
     - senão, o veredito é a cor do sinal no instante do cruzamento:
       vermelho -> "infracao", amarelo -> "amarelo", verde -> "verde",
       sem leitura -> "indeterminado". Com `red_grace_s` > 0, quem cruza no
       vermelho antes dessa tolerância vira "vermelho_tolerancia".
5. Evidência: para cada infração, uma imagem com 3 quadros (antes, no
   cruzamento e depois), com pessoas e a faixa da placa desfocadas.

Saídas em outputs/<video>_infracoes/:
    crossings.csv   todos os cruzamentos da linha de retenção com o veredito
    violations.csv  só as infrações
    evidence/       imagens de evidência
    light_states.csv estado do semáforo por quadro
    annotated.mp4   (opcional) vídeo anotado
    run_info.yaml   parâmetros, FPS e totais

Uso:
    python -m src.violation --config configs/aic21_cam_7.yaml
    python -m src.violation --config configs/aic21_cam_7.yaml --model models/yolo11s.pt --video
"""

from __future__ import annotations

import argparse
import time
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import supervision as sv
import yaml

from src.config import COCO_CLASSES, PERSON_CLASS_ID, VEHICLE_CLASS_IDS, SceneConfig
from src.detect_track import make_annotators, track_video
from src.traffic_light import (GREEN, RED, STATE_BGR, UNKNOWN, YELLOW, TrafficLightClassifier,
                               draw_light)

VERDICT_BY_STATE = {RED: "infracao", YELLOW: "amarelo", GREEN: "verde", UNKNOWN: "indeterminado"}


def _side(a: np.ndarray, b: np.ndarray, p: np.ndarray) -> float:
    """Produto vetorial: > 0 de um lado da reta a->b, < 0 do outro."""
    return float((b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0]))


@dataclass
class Candidate:
    track_id: int
    frame: int                 # quadro em que o cruzamento foi visto
    frame_float: float         # instante interpolado do cruzamento (em quadros)
    point: tuple[float, float]  # ponto de cruzamento sobre a linha
    travel_needed: float       # avanço mínimo além da linha para confirmar (px)
    max_travel: float = 0.0    # maior avanço além da linha (px)
    travel_frame: int | None = None  # quadro em que o avanço mínimo foi atingido
    last_seen: int = 0
    last_point: tuple[float, float] | None = None
    post_ok: bool | None = None  # trajetória ~1 s depois aponta para dentro do cruzamento?
    entered_exempt: bool = False


@dataclass
class LightTimeline:
    """Histórico de estados do semáforo para consultar a cor num instante passado."""

    lag: int
    fps: float
    raw: dict[int, str] = field(default_factory=dict)
    smoothed: dict[int, str] = field(default_factory=dict)

    def add(self, frame: int, raw: str, state: str) -> None:
        self.raw[frame] = raw
        self.smoothed[frame] = state

    def state_at(self, frame: int) -> str:
        """Estado no quadro, com as trocas de cor reposicionadas em retrospectiva.

        A votação temporal troca de cor alguns quadros depois da troca real (mais
        ainda se os quadros da transição ficarem "desconhecidos"). Como a decisão
        só é tomada depois, dá para olhar os quadros seguintes: o início da nova
        cor é o quadro após o último quadro bruto que ainda mostrava a cor antiga.
        """
        if frame not in self.smoothed:
            return UNKNOWN
        cur = self.smoothed[frame]
        w = 4 * self.lag + 4
        for g in range(frame + 1, frame + w + 1):  # troca logo depois deste quadro?
            if g not in self.smoothed:
                break
            if self.smoothed[g] != cur:
                return self.smoothed[g] if frame >= self._onset(cur, g, w) else cur
        for g in range(frame, frame - w, -1):  # troca logo antes (suavização adiantada)?
            if g - 1 not in self.smoothed:
                break
            if self.smoothed[g - 1] != cur:
                return cur if frame >= self._onset(self.smoothed[g - 1], g, w) else self.smoothed[g - 1]
        return cur

    def _onset(self, old: str, switch: int, w: int) -> int:
        """Primeiro quadro da nova cor: depois do último quadro bruto com a cor antiga."""
        olds = [k for k in range(switch - w, switch + self.lag + 1) if self.raw.get(k) == old]
        return max(olds) + 1 if olds else switch

    def phase_start(self, frame: int) -> int:
        """Primeiro quadro da fase (cor contínua) que contém `frame`."""
        s = self.state_at(frame)
        f = frame
        while f - 1 in self.smoothed and self.state_at(f - 1) == s:
            f -= 1
        return f

    def prune(self, keep_from: int) -> None:
        for d in (self.raw, self.smoothed):
            for k in [k for k in d if k < keep_from]:
                del d[k]


@dataclass
class ViolationDetector:
    stop_line: np.ndarray            # [[x1, y1], [x2, y2]] no quadro de referência
    direction: str                   # "in" ou "out"
    fps: float
    lag: int
    exempt_zone: np.ndarray | None = None
    decide_after_s: float = 2.0
    max_wait_s: float = 60.0         # espera máxima por quem parou em cima da linha
    red_grace_s: float = 0.0         # tolerância após o início do vermelho
    min_travel_frac: float = 0.3     # avanço mínimo além da linha, em alturas de caixa
    min_travel_px: float = 6.0
    segment_margin: float = 0.05     # tolerância nas pontas do segmento (fração do comprimento)
    post_check_s: float = 1.0        # quando conferir a trajetória depois da linha
    max_step_frac: float = 1.0       # deslocamento máx. por quadro, em alturas de caixa
    min_cos_normal: float = 0.5      # cos do ângulo entre o movimento e a normal da linha (60°)
    rejected: Counter = field(default_factory=Counter)  # cruzamentos descartados por filtro
    timeline: LightTimeline = field(init=False)
    prev: dict[int, tuple[int, np.ndarray]] = field(default_factory=dict)
    class_votes: dict[int, Counter] = field(default_factory=lambda: defaultdict(Counter))
    pending: dict[int, Candidate] = field(default_factory=dict)
    decided_ids: set[int] = field(default_factory=set)
    records: list[dict] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.timeline = LightTimeline(self.lag, self.fps)
        self._sign = 1.0 if self.direction == "in" else -1.0
        self._last_offset = (0, 0)

    def line_at(self, offset: tuple[int, int]) -> np.ndarray:
        return self.stop_line + np.array(offset, dtype=float)

    def _beyond(self, a: np.ndarray, b: np.ndarray, p: np.ndarray) -> float:
        """Distância (px) além da linha no sentido de entrada; negativa antes dela."""
        # "in" = lado positivo -> negativo (subindo para linha marcada da esquerda p/ direita)
        return -self._sign * _side(a, b, p) / (np.linalg.norm(b - a) + 1e-9)

    def update(self, frame: int, det: sv.Detections, offset: tuple[int, int],
               raw_state: str, state: str) -> list[dict]:
        """Processa um quadro; devolve as decisões tomadas neste quadro."""
        self.timeline.add(frame, raw_state, state)
        self._last_offset = offset
        a, b = self.line_at(offset)
        ab = b - a
        zone = None if self.exempt_zone is None else self.exempt_zone + np.array(offset, dtype=float)

        for xyxy, cls, tid in zip(det.xyxy, det.class_id, det.tracker_id):
            tid = int(tid)
            self.class_votes[tid][int(cls)] += 1
            p = np.array([(xyxy[0] + xyxy[2]) / 2, xyxy[3]], dtype=float)
            if tid in self.pending:
                c = self.pending[tid]
                c.max_travel = max(c.max_travel, self._beyond(a, b, p))
                if c.travel_frame is None and c.max_travel >= c.travel_needed:
                    c.travel_frame = frame
                c.last_seen = frame
                c.last_point = (float(p[0]), float(p[1]))
                if c.post_ok is None and frame - c.frame >= self.post_check_s * self.fps:
                    c.post_ok = self._post_direction_ok(c, a, b)
                if zone is not None and cv2.pointPolygonTest(zone.astype(np.float32), tuple(p), False) >= 0:
                    c.entered_exempt = True
            elif tid not in self.decided_ids and tid in self.prev:
                pf, pp = self.prev[tid]
                d0, d1 = self._beyond(a, b, pp), self._beyond(a, b, p)
                if d0 < 0 <= d1:  # passou da linha no sentido de entrada
                    alpha = -d0 / (d1 - d0)
                    x = pp + (p - pp) * alpha
                    t = float(np.dot(x - a, ab) / (np.dot(ab, ab) + 1e-9))
                    move = p - pp
                    step = float(np.linalg.norm(move)) / max(1, frame - pf)
                    cos_n = (d1 - d0) / (float(np.linalg.norm(move)) + 1e-9)
                    box_h = float(xyxy[3] - xyxy[1])
                    if not (-self.segment_margin <= t <= 1 + self.segment_margin):
                        pass  # fora do segmento marcado
                    elif step > max(20.0, self.max_step_frac * box_h):
                        self.rejected["pulo_de_id"] += 1
                    elif cos_n < self.min_cos_normal:
                        self.rejected["movimento_transversal"] += 1
                    else:
                        need = max(self.min_travel_px, self.min_travel_frac * box_h)
                        self.pending[tid] = Candidate(
                            track_id=tid, frame=frame, frame_float=pf + (frame - pf) * alpha,
                            point=(float(x[0]), float(x[1])), travel_needed=need,
                            max_travel=d1, travel_frame=frame if d1 >= need else None, last_seen=frame,
                            last_point=(float(p[0]), float(p[1])))
            self.prev[tid] = (frame, p)

        decided = []
        horizon = int(round(self.decide_after_s * self.fps))
        for tid, c in list(self.pending.items()):
            lost = frame - c.last_seen > int(self.fps)  # ID sumiu há mais de 1 s
            waited = frame - c.frame
            ready = c.travel_frame is not None and frame - c.travel_frame >= min(horizon, waited)
            if (ready and waited >= horizon) or lost or waited >= self.max_wait_s * self.fps:
                decided.append(self._decide(c, frame))
                del self.pending[tid]
                self.decided_ids.add(tid)
        # memória limitada: guarda 5 min de estados (fases passam de 1 min)
        self.timeline.prune(frame - int(300 * self.fps))
        return decided

    def _post_direction_ok(self, c: Candidate, a: np.ndarray, b: np.ndarray) -> bool | None:
        """O deslocamento desde o ponto de cruzamento aponta para dentro do cruzamento?"""
        if c.last_point is None:
            return None
        v = np.array(c.last_point) - np.array(c.point)
        dist = float(np.linalg.norm(v))
        if dist < max(4.0, 0.25 * c.travel_needed):  # quase parado: decide pelo avanço
            return None
        return bool(self._beyond(a, b, np.array(c.last_point)) / dist >= self.min_cos_normal)

    def _decide(self, c: Candidate, now: int) -> dict:
        horizon = int(round(self.decide_after_s * self.fps))
        waited_on_line = c.travel_frame is not None and c.travel_frame - c.frame > horizon
        # quem parou sobre a linha e avançou depois "entra" no cruzamento quando avança
        t_frame = float(c.travel_frame) if waited_on_line else c.frame_float
        cross_frame = int(np.floor(t_frame))
        state = self.timeline.state_at(cross_frame)
        start = self.timeline.phase_start(cross_frame)
        since = (t_frame - (start - 0.5)) / self.fps  # início da fase ~ meio do quadro
        if c.post_ok is None and c.travel_frame is not None:  # decidido antes de 1 s
            a, b = self.line_at(self._last_offset)
            c.post_ok = self._post_direction_ok(c, a, b)
        if c.post_ok is False:
            verdict = "trajetoria_incompativel"
        elif c.travel_frame is None:
            verdict = "parado_na_linha"
        elif c.entered_exempt and state == RED:
            verdict = "conversao_permitida"
        elif state == RED and since < self.red_grace_s:
            verdict = "vermelho_tolerancia"
        else:
            verdict = VERDICT_BY_STATE[state]
        votes = self.class_votes.get(c.track_id)
        cls = COCO_CLASSES.get(votes.most_common(1)[0][0], "?") if votes else "?"
        rec = {
            "track_id": c.track_id,
            "class": cls,
            "cross_frame": cross_frame,
            "cross_time_s": round(t_frame / self.fps, 3),
            "light_state": state,
            "time_in_state_s": round(since, 2),
            "travel_px": round(c.max_travel, 1),
            "waited_on_line": waited_on_line,
            "verdict": verdict,
            "decided_frame": now,
            "x": round(c.point[0], 1),
            "y": round(c.point[1], 1),
        }
        self.records.append(rec)
        return rec

    def flush(self, frame: int) -> list[dict]:
        """Decide os candidatos pendentes no fim do vídeo."""
        out = [self._decide(c, frame) for c in self.pending.values()]
        self.pending.clear()
        return out


# ----------------------------------------------------------------------------- evidência

def blur_privacy(img: np.ndarray, det: sv.Detections) -> np.ndarray:
    """Desfoca pessoas (caixa inteira) e a faixa provável da placa dos veículos."""
    out = img.copy()
    H, W = out.shape[:2]
    for (x1, y1, x2, y2), cls in zip(det.xyxy.astype(int), det.class_id):
        if int(cls) == PERSON_CLASS_ID:
            r = (x1, y1, x2, y2)
        else:  # placa: faixa central no terço inferior da caixa (frente ou traseira)
            w, h = x2 - x1, y2 - y1
            r = (x1 + w // 4, y1 + int(h * 0.62), x2 - w // 4, y2)
        rx1, ry1, rx2, ry2 = max(0, r[0]), max(0, r[1]), min(W, r[2]), min(H, r[3])
        if rx2 - rx1 > 2 and ry2 - ry1 > 2:
            k = max(3, ((rx2 - rx1) // 3) | 1)
            out[ry1:ry2, rx1:rx2] = cv2.GaussianBlur(out[ry1:ry2, rx1:rx2], (k, k), 0)
    return out


@dataclass
class BufferedFrame:
    index: int
    frame: np.ndarray
    det: sv.Detections
    offset: tuple[int, int]
    state: str
    light_box: tuple[int, int, int, int]


def evidence_panel(bf: BufferedFrame, tid: int, line: np.ndarray, title: str, fps: float,
                   state: str) -> np.ndarray:
    """Um quadro da evidência; `state` é a cor em retrospectiva (a mesma da decisão)."""
    img = blur_privacy(bf.frame, bf.det)
    a, b = line.astype(int)
    cv2.line(img, tuple(a), tuple(b), STATE_BGR[state], 3)
    x1, y1, x2, y2 = bf.light_box
    cv2.rectangle(img, (x1 - 3, y1 - 3), (x2 + 3, y2 + 3), STATE_BGR[state], 2)
    for xyxy, t in zip(bf.det.xyxy.astype(int), bf.det.tracker_id):
        if int(t) == tid:
            cv2.rectangle(img, tuple(xyxy[:2]), tuple(xyxy[2:]), (0, 0, 255), 3)
            cv2.putText(img, f"#{tid}", (xyxy[0], xyxy[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    # recorte em volta da linha e do semáforo, para a evidência ficar legível
    xs = [a[0], b[0], x1, x2]
    ys = [a[1], b[1], y1, y2]
    pad = 120
    H, W = img.shape[:2]
    cx1, cy1 = max(0, min(xs) - pad), max(0, min(ys) - pad)
    cx2, cy2 = min(W, max(xs) + pad), min(H, max(ys) + pad)
    crop = img[cy1:cy2, cx1:cx2].copy()
    label = f"{title}  t={bf.index / fps:.2f}s  sinal: {state}"
    cv2.rectangle(crop, (0, 0), (crop.shape[1], 30), (0, 0, 0), -1)
    cv2.putText(crop, label, (6, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return crop


def save_evidence(buffer: deque, rec: dict, detector: ViolationDetector, fps: float, path: Path) -> bool:
    by_idx = {bf.index: bf for bf in buffer}
    cf = rec["cross_frame"]
    wanted = [("antes", cf - int(round(0.5 * fps))), ("cruzamento", cf + 1), ("depois", rec["decided_frame"])]
    panels = []
    for title, idx in wanted:
        bf = by_idx.get(idx) or by_idx.get(min(by_idx, key=lambda k: abs(k - idx)))
        panels.append(evidence_panel(bf, rec["track_id"], detector.line_at(bf.offset), title, fps,
                                     detector.timeline.state_at(bf.index)))
    h = min(p.shape[0] for p in panels)
    panels = [cv2.resize(p, (int(p.shape[1] * h / p.shape[0]), h)) for p in panels]
    sheet = np.hstack(panels)
    footer = np.full((40, sheet.shape[1], 3), 255, np.uint8)
    cv2.putText(footer, (f"Infracao: ID {rec['track_id']} ({rec['class']}) cruzou a linha de retencao "
                         f"{rec['time_in_state_s']:.1f}s apos o inicio do vermelho"),
                (8, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 180), 2, cv2.LINE_AA)
    return cv2.imwrite(str(path), np.vstack([sheet, footer]))


# ----------------------------------------------------------------------------- execução

def main() -> None:
    ap = argparse.ArgumentParser(description="Detecção de avanço de sinal vermelho")
    ap.add_argument("--config", required=True, help="cena com stop_line e traffic_light_roi")
    ap.add_argument("--source", help="vídeo (padrão: o da configuração)")
    ap.add_argument("--model", default="models/yolo11s.pt")
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--device", default="0")
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--decide-after", type=float, default=2.0, help="segundos até decidir cada cruzamento")
    ap.add_argument("--red-grace", type=float, default=0.0,
                    help="tolerância (s) após o início do vermelho antes de contar infração")
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--video", action="store_true", help="grava vídeo anotado")
    args = ap.parse_args()

    scene = SceneConfig.load(args.config)
    if not scene.stop_line:
        raise SystemExit(f"A cena '{scene.name}' não tem stop_line. Use src.scene_editor (tecla r).")
    source = Path(args.source or scene.video)
    info = sv.VideoInfo.from_video_path(str(source))
    fps = info.fps or 10
    device = int(args.device) if args.device.isdigit() else args.device

    light = TrafficLightClassifier.from_scene(scene, source, force_after=int(round(3 * fps)))
    detector = ViolationDetector(
        stop_line=np.array(scene.stop_line, dtype=float),
        direction=scene.stop_line_direction,
        fps=fps,
        lag=light.lag,
        exempt_zone=np.array(scene.exempt_zone, dtype=float) if scene.exempt_zone else None,
        decide_after_s=args.decide_after,
        red_grace_s=args.red_grace,
    )

    out_dir = Path(args.out) / f"{source.stem}_infracoes"
    (out_dir / "evidence").mkdir(parents=True, exist_ok=True)
    for old in (out_dir / "evidence").glob("*.jpg"):
        old.unlink()
    ann = make_annotators(info)
    sink = sv.VideoSink(str(out_dir / "annotated.mp4"), info) if args.video else None
    # cobre o atraso da decisão (até decide_after + 1 s de ID perdido) mais 0,5 s antes do cruzamento
    buffer: deque[BufferedFrame] = deque(maxlen=int(np.ceil((args.decide_after + 4.0) * fps)))
    light_rows = []
    violators: set[int] = set()

    t0 = time.perf_counter()
    n = 0
    try:
        if sink:
            sink.__enter__()
        for fr in track_video(source, args.model, args.conf, args.imgsz, device,
                              classes=VEHICLE_CLASS_IDS + [PERSON_CLASS_ID], max_frames=args.max_frames):
            n += 1
            reading = light.update(fr.frame)
            light_rows.append((fr.index, round(fr.time_s, 3), reading.raw, reading.state))
            det = fr.detections
            vehicles = det[np.isin(det.class_id, VEHICLE_CLASS_IDS)] if len(det) else det
            buffer.append(BufferedFrame(fr.index, fr.frame, det, light.offset, reading.state, reading.box))

            for rec in detector.update(fr.index, vehicles, light.offset, reading.raw, reading.state):
                if rec["verdict"] == "infracao":
                    violators.add(rec["track_id"])
                    name = f"infracao_{rec['cross_time_s']:07.1f}s_id{rec['track_id']}.jpg"
                    save_evidence(buffer, rec, detector, fps, out_dir / "evidence" / name)
                    print(f"  infração: ID {rec['track_id']} ({rec['class']}) em {rec['cross_time_s']:.1f}s, "
                          f"{rec['time_in_state_s']:.1f}s após o vermelho")

            if sink:
                img = blur_privacy(fr.frame, det)
                a, b = detector.line_at(light.offset).astype(int)
                cv2.line(img, tuple(a), tuple(b), STATE_BGR[reading.state], ann.thickness + 1)
                img = ann.draw(img, vehicles)
                for xyxy, t in zip(vehicles.xyxy.astype(int), vehicles.tracker_id):
                    if int(t) in violators:
                        cv2.rectangle(img, tuple(xyxy[:2]), tuple(xyxy[2:]), (0, 0, 255), ann.thickness + 2)
                img = draw_light(img, reading)
                sink.write_frame(img)
    finally:
        if sink:
            sink.__exit__(None, None, None)
    for rec in detector.flush(n):
        if rec["verdict"] == "infracao":
            print(f"  infração (fim do vídeo, sem evidência): ID {rec['track_id']}")
    elapsed = time.perf_counter() - t0

    cols = ["track_id", "class", "cross_frame", "cross_time_s", "light_state", "time_in_state_s",
            "travel_px", "waited_on_line", "verdict", "decided_frame", "x", "y"]
    df = pd.DataFrame(detector.records, columns=cols).sort_values("cross_frame")
    df.to_csv(out_dir / "crossings.csv", index=False)
    df[df.verdict == "infracao"].to_csv(out_dir / "violations.csv", index=False)
    pd.DataFrame(light_rows, columns=["frame", "time_s", "raw", "state"]).to_csv(
        out_dir / "light_states.csv", index=False)

    summary = df["verdict"].value_counts().to_dict()
    run = {
        "source": str(source), "config": args.config, "model": args.model, "imgsz": args.imgsz,
        "frames": n, "seconds": round(elapsed, 1), "fps": round(n / elapsed, 1) if elapsed else 0,
        "decide_after_s": args.decide_after, "red_grace_s": args.red_grace, "crossings": len(df),
        "verdicts": {k: int(v) for k, v in summary.items()},
        "rejected_crossings": dict(detector.rejected),
    }
    (out_dir / "run_info.yaml").write_text(yaml.safe_dump(run, sort_keys=False, allow_unicode=True),
                                           encoding="utf-8")
    print(f"\n{n} quadros em {elapsed:.0f}s -> {run['fps']} FPS (YOLO + rastreamento + semáforo)")
    print(f"{len(df)} cruzamentos da linha de retenção: " +
          ", ".join(f"{k}={v}" for k, v in summary.items()))
    if detector.rejected:
        print("descartados: " + ", ".join(f"{k}={v}" for k, v in detector.rejected.items()))
    print(f"Saída em {out_dir}")


if __name__ == "__main__":
    main()
