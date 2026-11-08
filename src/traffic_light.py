"""Fase 4 — Estado do semáforo por quadro (versão 1: regras em HSV).

Método:
1. Localização: a câmera balança alguns pixels (vento). Em cada quadro o grupo
   focal é reencontrado por correlação de template (cv2.matchTemplate) numa
   janela pequena em volta da posição do quadro anterior. A correlação é
   feita num mapa de "silhueta" (pixel escuro ou colorido = 1, fundo cinza =
   0), em que a caixa do semáforo tem a mesma forma qualquer que seja a
   lâmpada acesa; na imagem em tons de cinza o halo de uma lâmpada acesa
   mudaria o template e deslocaria a caixa.
2. Cor acesa: dentro da caixa, contam-se os pixels saturados e claros
   (S >= 90, V >= 110) em cada faixa de matiz (H, escala 0-179 do OpenCV):
       vermelho: H <= 12 ou H >= 160   amarelo: 20 <= H <= 40
       verde (LED verde-azulado): 60 <= H <= 105
   A cor com mais pixels vence, se tiver um mínimo de pixels e pelo menos o
   dobro da segunda colocada; senão o quadro é "desconhecido".
   Por que não usar só o brilho/posição: com sol baixo as lentes apagadas
   refletem branco quase tão claro quanto a lâmpada acesa, e na chuva o halo
   do vermelho vaza para a posição do amarelo. Reflexo branco tem S baixo e
   não conta; o halo vermelho continua vermelho. O miolo da lâmpada acesa
   costuma estourar para branco, então é a borda colorida que decide.
3. Fundo claro sem cor (ex.: caçamba branca de caminhão passando atrás do
   semáforo) não gera pixels coloridos e resulta em "desconhecido".
4. Suavização temporal: o candidato é a maioria dos últimos N quadros
   (ignorando "desconhecido"). Ele só substitui o estado atual se a troca
   respeitar o ciclo verde -> amarelo -> vermelho -> verde; uma troca fora
   do ciclo só é aceita se persistir por `force_after` quadros (recupera de
   erros sem ficar preso).

Uso (não precisa do YOLO, roda só o semáforo):
    python -m src.traffic_light --config configs/aic21_cam_7.yaml
    python -m src.traffic_light --config configs/aic21_cam_7.yaml --max-frames 2000 --video
"""

from __future__ import annotations

import argparse
import time
from collections import Counter, deque
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from src.config import SceneConfig

RED, YELLOW, GREEN, UNKNOWN = "vermelho", "amarelo", "verde", "desconhecido"
LAMP_ORDER = (RED, YELLOW, GREEN)  # mesma ordem das contagens de pixels
NEXT_STATE = {GREEN: YELLOW, YELLOW: RED, RED: GREEN}  # ciclo normal (padrão dos EUA/Brasil)
STATE_BGR = {RED: (0, 0, 255), YELLOW: (0, 200, 255), GREEN: (0, 200, 0), UNKNOWN: (160, 160, 160)}


@dataclass
class LightReading:
    raw: str                      # estado do quadro, sem suavização
    state: str                    # estado suavizado
    box: tuple[int, int, int, int]  # grupo focal localizado neste quadro
    match: float                  # correlação do template (0 a 1)
    counts: tuple[int, int, int]  # pixels coloridos (vermelho, amarelo, verde)


@dataclass
class TrafficLightClassifier:
    roi: tuple[int, int, int, int]      # caixa do grupo focal no quadro de referência
    reference: np.ndarray               # quadro de referência (BGR)
    search: int = 25                    # deslocamento máximo em relação à referência (px)
    step: int = 6                       # deslocamento máximo entre quadros seguidos (px)
    min_match: float = 0.6              # abaixo disso mantém a última posição
    min_sat: int = 90                   # saturação mínima de um pixel "colorido"
    min_val: int = 110                  # brilho mínimo de um pixel "colorido"
    min_frac: float = 0.005             # pixels da cor vencedora / área da caixa (mín. 2 px)
    min_ratio: float = 2.0              # vencedora / segunda colocada
    smooth: int = 5                     # quadros na votação temporal
    force_after: int = 30               # quadros para aceitar troca fora do ciclo
    _template: np.ndarray = field(init=False, repr=False)
    _tpl_xy: tuple[int, int] = field(init=False)  # canto do template no quadro de referência
    _offset: tuple[int, int] = field(init=False, default=(0, 0))
    _history: deque = field(init=False, repr=False)
    _last_state: str = field(init=False, default=UNKNOWN)
    _pending: str = field(init=False, default=UNKNOWN)
    _pending_count: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        x1, y1, x2, y2 = self.roi
        # template = grupo focal + margem (inclui braço e fundo, melhora o encaixe);
        # recortado na borda da imagem quando o semáforo está encostado nela
        pad = max(4, (y2 - y1) // 3)
        self._tpl_xy = (max(0, x1 - pad), max(0, y1 - pad))
        self._template = self._gray(self._crop(self.reference, x1 - pad, y1 - pad, x2 + pad, y2 + pad))
        self._history = deque(maxlen=self.smooth)

    @staticmethod
    def _gray(img: np.ndarray) -> np.ndarray:
        """Mapa de silhueta: max(escuridão, saturação), suavizado."""
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
        m = np.maximum(255.0 - hsv[..., 2], hsv[..., 1])
        return cv2.GaussianBlur(m, (3, 3), 0)

    @staticmethod
    def _crop(img: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> np.ndarray:
        h, w = img.shape[:2]
        return img[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]

    def locate(self, frame: np.ndarray) -> tuple[tuple[int, int, int, int], float]:
        """Procura o grupo focal perto da posição do quadro anterior."""
        tx, ty = self._tpl_xy
        th, tw = self._template.shape
        ox, oy, st = *self._offset, self.step
        wx, wy = max(0, tx + ox - st), max(0, ty + oy - st)  # canto real da janela
        window = self._gray(frame[wy:ty + oy + th + st, wx:tx + ox + tw + st])
        if window.shape[0] < th or window.shape[1] < tw:
            return self._shifted(), 0.0
        res = cv2.matchTemplate(window, self._template, cv2.TM_CCOEFF_NORMED)
        _, score, _, (mx, my) = cv2.minMaxLoc(res)
        nx, ny = wx + mx - tx, wy + my - ty
        if score >= self.min_match and abs(nx) <= self.search and abs(ny) <= self.search:
            self._offset = (nx, ny)
        return self._shifted(), float(score)

    @property
    def offset(self) -> tuple[int, int]:
        """Deslocamento atual da câmera em relação ao quadro de referência (px)."""
        return self._offset

    @property
    def lag(self) -> int:
        """Atraso aproximado (quadros) introduzido pela votação temporal."""
        return self.smooth // 2

    def _shifted(self) -> tuple[int, int, int, int]:
        dx, dy = self._offset
        x1, y1, x2, y2 = self.roi
        return x1 + dx, y1 + dy, x2 + dx, y2 + dy

    def _classify(self, frame: np.ndarray, box: tuple[int, int, int, int]) -> tuple[str, tuple]:
        x1, y1, x2, y2 = box
        m = max(1, (y2 - y1) // 8)  # folga para pequenos erros de alinhamento
        hsv = cv2.cvtColor(self._crop(frame, x1 - 1, y1 - m, x2 + 1, y2 + m), cv2.COLOR_BGR2HSV)
        if hsv.size == 0:
            return UNKNOWN, (0, 0, 0)
        H, S, V = hsv[..., 0], hsv[..., 1], hsv[..., 2]
        colored = (S >= self.min_sat) & (V >= self.min_val)
        counts = (
            int(np.count_nonzero(colored & ((H <= 12) | (H >= 160)))),
            int(np.count_nonzero(colored & (H >= 20) & (H <= 40))),
            int(np.count_nonzero(colored & (H >= 60) & (H <= 105))),
        )
        order = np.argsort(counts)[::-1]
        top, second = counts[order[0]], counts[order[1]]
        min_px = max(2, int(self.min_frac * H.size))
        if top < min_px or top < self.min_ratio * second:
            return UNKNOWN, counts
        return LAMP_ORDER[order[0]], counts

    def _smooth(self, raw: str) -> str:
        self._history.append(raw)
        votes = Counter(s for s in self._history if s != UNKNOWN)
        if not votes:
            return self._last_state
        candidate = votes.most_common(1)[0][0]
        if candidate == self._last_state:
            self._pending, self._pending_count = UNKNOWN, 0
        elif self._last_state == UNKNOWN or NEXT_STATE[self._last_state] == candidate:
            self._last_state = candidate
            self._pending, self._pending_count = UNKNOWN, 0
        else:  # troca fora do ciclo: só aceita se persistir
            self._pending_count = self._pending_count + 1 if candidate == self._pending else 1
            self._pending = candidate
            if self._pending_count >= self.force_after:
                self._last_state = candidate
                self._pending, self._pending_count = UNKNOWN, 0
        return self._last_state

    def update(self, frame: np.ndarray) -> LightReading:
        box, score = self.locate(frame)
        raw, counts = self._classify(frame, box)
        return LightReading(raw, self._smooth(raw), box, score, counts)

    @classmethod
    def from_scene(cls, scene: SceneConfig, video: str | Path | None = None, **kw) -> "TrafficLightClassifier":
        if not scene.traffic_light_roi:
            raise SystemExit(f"A cena '{scene.name}' não tem traffic_light_roi. Use src.scene_editor.")
        cap = cv2.VideoCapture(str(video or scene.video))
        cap.set(cv2.CAP_PROP_POS_FRAMES, scene.reference_frame)
        ok, ref = cap.read()
        cap.release()
        if not ok:
            raise SystemExit("Não foi possível ler o quadro de referência do vídeo.")
        return cls(roi=tuple(scene.traffic_light_roi), reference=ref, **kw)


def draw_light(img: np.ndarray, reading: LightReading, inset_scale: int = 4) -> np.ndarray:
    """Desenha a caixa do semáforo e uma ampliação no canto superior direito."""
    x1, y1, x2, y2 = reading.box
    color = STATE_BGR[reading.state]
    cv2.rectangle(img, (x1 - 2, y1 - 2), (x2 + 2, y2 + 2), color, 2)
    crop = img[max(0, y1 - 4):y2 + 4, max(0, x1 - 4):x2 + 4]
    if crop.size:
        z = cv2.resize(crop, None, fx=inset_scale, fy=inset_scale, interpolation=cv2.INTER_NEAREST)
        h, w = z.shape[:2]
        H, W = img.shape[:2]
        if h + 50 < H and w + 10 < W:
            img[10:10 + h, W - w - 10:W - 10] = z
            cv2.rectangle(img, (W - w - 12, 8), (W - 8, 12 + h), color, 3)
            cv2.putText(img, reading.state.upper(), (W - w - 10 - 160, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3, cv2.LINE_AA)
    return img


def phases(df: pd.DataFrame, fps: float) -> pd.DataFrame:
    """Agrupa quadros consecutivos com o mesmo estado suavizado em fases."""
    change = df["state"].ne(df["state"].shift()).cumsum()
    g = df.groupby(change)
    out = pd.DataFrame({
        "state": g["state"].first(),
        "start_frame": g["frame"].first(),
        "end_frame": g["frame"].last(),
    })
    out["start_s"] = (out["start_frame"] / fps).round(2)
    out["duration_s"] = ((out["end_frame"] - out["start_frame"] + 1) / fps).round(2)
    return out.reset_index(drop=True)


def save_timeline(df: pd.DataFrame, fps: float, path: Path, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {RED: "#d62728", YELLOW: "#f2b701", GREEN: "#2ca02c", UNKNOWN: "#bbbbbb"}
    ph = phases(df, fps)
    fig, axes = plt.subplots(2, 1, figsize=(14, 3.2), sharex=True,
                             gridspec_kw={"height_ratios": [1, 1]})
    for ax, col, label in ((axes[0], "raw", "por quadro"), (axes[1], "state", "suavizado")):
        p = ph if col == "state" else phases(df.assign(state=df["raw"]), fps)
        for _, r in p.iterrows():
            ax.broken_barh([(r.start_s, r.duration_s)], (0, 1), color=colors[r.state])
        ax.set_yticks([0.5], [label])
        ax.set_ylim(0, 1)
    axes[1].set_xlim(0, (df["frame"].max() + 1) / fps)
    axes[1].set_xlabel("tempo (s)")
    axes[0].set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def save_samples(samples: list[tuple[int, np.ndarray, LightReading]], path: Path, cols: int = 10) -> None:
    """Mosaico de recortes ampliados com o estado previsto, para conferência visual."""
    tiles = []
    for idx, frame, r in samples:
        x1, y1, x2, y2 = r.box
        pad = (y2 - y1) // 2
        crop = frame[max(0, y1 - pad):y2 + pad, max(0, x1 - pad * 2):x2 + pad * 2]
        tile = cv2.resize(crop, (120, 160), interpolation=cv2.INTER_NEAREST)
        tile = cv2.copyMakeBorder(tile, 0, 34, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))
        cv2.putText(tile, f"{idx}", (3, 175), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
        cv2.putText(tile, r.state[:8], (3, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.45, STATE_BGR[r.state], 2)
        tiles.append(cv2.copyMakeBorder(tile, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=(255, 255, 255)))
    while len(tiles) % cols:
        tiles.append(np.full_like(tiles[0], 255))
    rows = [np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)]
    cv2.imwrite(str(path), np.vstack(rows))


def main() -> None:
    ap = argparse.ArgumentParser(description="Estado do semáforo por quadro (HSV)")
    ap.add_argument("--config", required=True)
    ap.add_argument("--source", help="vídeo (padrão: o da configuração)")
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--smooth", type=int, default=5, help="quadros na votação temporal")
    ap.add_argument("--force-after-s", type=float, default=3.0,
                    help="segundos para aceitar uma troca fora do ciclo verde-amarelo-vermelho")
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--samples", type=int, default=60, help="recortes no mosaico de conferência")
    ap.add_argument("--video", action="store_true", help="grava vídeo anotado")
    args = ap.parse_args()

    scene = SceneConfig.load(args.config)
    source = Path(args.source or scene.video)
    cap = cv2.VideoCapture(str(source))
    fps = cap.get(cv2.CAP_PROP_FPS) or 10
    clf = TrafficLightClassifier.from_scene(
        scene, source, smooth=args.smooth, force_after=int(round(args.force_after_s * fps)))
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    n_target = min(n_total, args.max_frames or n_total)
    sample_at = set(np.linspace(0, n_target - 1, args.samples, dtype=int).tolist())

    out_dir = Path(args.out) / f"{source.stem}_semaforo"
    out_dir.mkdir(parents=True, exist_ok=True)
    writer = None
    if args.video:
        w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(str(out_dir / "annotated.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    rows, samples = [], []
    t0 = time.perf_counter()
    idx = 0
    while idx < n_target:
        ok, frame = cap.read()
        if not ok:
            break
        r = clf.update(frame)
        rows.append((idx, round(idx / fps, 3), r.raw, r.state, round(r.match, 3), *r.counts, *r.box))
        if idx in sample_at:
            samples.append((idx, frame.copy(), r))
        if writer:
            writer.write(draw_light(frame, r))
        idx += 1
    elapsed = time.perf_counter() - t0
    cap.release()
    if writer:
        writer.release()

    df = pd.DataFrame(rows, columns=["frame", "time_s", "raw", "state", "match",
                                     "n_red", "n_yellow", "n_green", "x1", "y1", "x2", "y2"])
    df.to_csv(out_dir / "states.csv", index=False)
    ph = phases(df, fps)
    ph.to_csv(out_dir / "phases.csv", index=False)
    save_timeline(df, fps, out_dir / "timeline.png", f"{scene.name} — {source.name}")
    if samples:
        save_samples(samples, out_dir / "samples.jpg")

    # Resumo: proporção de cada estado e duração típica das fases completas
    inner = ph.iloc[1:-1] if len(ph) > 2 else ph
    print(f"{len(df)} quadros em {elapsed:.1f}s -> {len(df) / elapsed:.0f} FPS (só semáforo)")
    print(f"desconhecido por quadro (antes da suavização): {(df['raw'] == UNKNOWN).mean():.1%}")
    print(f"correlação do template: mediana {df['match'].median():.2f}, mínima {df['match'].min():.2f}")
    print(f"deslocamento da caixa: x {df['x1'].min() - clf.roi[0]}..{df['x1'].max() - clf.roi[0]} px, "
          f"y {df['y1'].min() - clf.roi[1]}..{df['y1'].max() - clf.roi[1]} px\n")
    print("estado        % quadros   fases   duração mediana (s)   mín   máx")
    for s in LAMP_ORDER + (UNKNOWN,):
        d = inner[inner["state"] == s]["duration_s"]
        share = (df["state"] == s).mean()
        if share == 0:
            continue
        print(f"{s:13s} {share:9.1%} {len(d):7d} {d.median() if len(d) else float('nan'):17.1f} "
              f"{d.min() if len(d) else float('nan'):9.1f} {d.max() if len(d) else float('nan'):5.1f}")
    print(f"\nSaída em {out_dir}")


if __name__ == "__main__":
    main()
