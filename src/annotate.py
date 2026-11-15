"""Fase 7 — Anotação manual do gabarito (eventos com instante), para o src.evaluate.

Em vez de desenhar caixas quadro a quadro (CVAT), o anotador assiste ao vídeo e
aperta uma tecla no instante de cada evento. Três tarefas:

    semaforo  início de cada estado do semáforo: [g] verde  [y] amarelo
              [r] vermelho  [k] desconhecido (encoberto, ilegível). Pause e ande
              quadro a quadro para marcar o PRIMEIRO quadro da nova cor; o recorte
              ampliado do semáforo aparece no canto da tela.
    contagem  um evento por veículo que cruza a counting_line, no quadro em que
              a FRENTE do veículo toca a linha: [1] carro [2] moto [3] ônibus
              [4] caminhão. [t] alterna o sentido (in/out, como no count.py).
    retencao  idem para a stop_line (sentido fixo = stop_line_direction). Com
              --light-gt semaforo.yaml e --skip-green, o vídeo pula as fases
              verdes do gabarito (com margem), e só os trechos amarelo/vermelho
              precisam ser assistidos.

Controles: [espaço] play/pausa  [d]/[a] +1/-1 quadro  [l]/[j] +1/-1 s
           [L]/[J] +10/-10 s  [n] próxima fase não verde (retencao)
           [+]/[-] velocidade  [u] desfaz o último evento  [x] apaga o evento
           mais próximo (até 1 s)  [s] salva  [q]/ESC salva e sai

Só contam como "assistidos" os quadros mostrados em sequência (play ou [d]); o
arquivo guarda esses intervalos e a avaliação ignora o que ficou de fora. As
previsões do sistema não são mostradas, para não induzir o anotador.

Saída: data/annotations/<cena>/<tarefa>.yaml

Uso:
    python -m src.annotate --config configs/aic21_cam_7.yaml --task semaforo
    python -m src.annotate --config configs/aic21_cam_7.yaml --task retencao \
        --light-gt data/annotations/aic21_cam_7/semaforo.yaml --skip-green
"""

from __future__ import annotations

import argparse
import datetime as dt
import time
from pathlib import Path

import cv2
import numpy as np
import yaml

from src.config import SceneConfig

WINDOW = "annotate"
LIGHT_KEYS = {ord("g"): "verde", ord("y"): "amarelo", ord("r"): "vermelho", ord("k"): "desconhecido"}
CLASS_KEYS = {ord("1"): "carro", ord("2"): "moto", ord("3"): "onibus", ord("4"): "caminhao"}
COLORS = {"verde": (0, 200, 0), "amarelo": (0, 200, 255), "vermelho": (0, 0, 255),
          "desconhecido": (160, 160, 160)}
SPEEDS = [0.25, 0.5, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0]


# ---------------------------------------------------------------- arquivo de anotação

def frames_to_intervals(mask: np.ndarray) -> list[list[int]]:
    """Máscara booleana por quadro -> intervalos [início, fim] inclusivos."""
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return []
    breaks = np.flatnonzero(np.diff(idx) > 1)
    starts = np.r_[idx[0], idx[breaks + 1]]
    ends = np.r_[idx[breaks], idx[-1]]
    return [[int(s), int(e)] for s, e in zip(starts, ends)]


def intervals_to_mask(intervals: list[list[int]], n: int) -> np.ndarray:
    mask = np.zeros(n, dtype=bool)
    for s, e in intervals or []:
        mask[max(0, s):min(n, e + 1)] = True
    return mask


def load_annotation(path: str | Path) -> dict:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    data.setdefault("events", [])
    data.setdefault("reviewed", [])
    data["events"] = sorted(data["events"], key=lambda e: e["frame"])
    return data


def light_state_per_frame(ann: dict, n: int) -> np.ndarray:
    """Estado do gabarito do semáforo por quadro ('' antes do primeiro evento)."""
    states = np.full(n, "", dtype=object)
    events = sorted(ann["events"], key=lambda e: e["frame"])
    for i, ev in enumerate(events):
        end = events[i + 1]["frame"] if i + 1 < len(events) else n
        states[max(0, ev["frame"]):min(n, end)] = ev["label"]
    return states


# ---------------------------------------------------------------- ferramenta

class Annotator:
    def __init__(self, scene: SceneConfig, task: str, out: Path, annotator: str,
                 light_gt: Path | None, skip_green: bool, margin_s: float):
        self.scene, self.task, self.out = scene, task, out
        self.cap = cv2.VideoCapture(scene.video)
        if not self.cap.isOpened():
            raise SystemExit(f"Não foi possível abrir {scene.video}")
        self.n = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = float(self.cap.get(cv2.CAP_PROP_FPS)) or 10.0

        self.line = None
        self.direction = "in"
        if task == "contagem":
            if not scene.counting_line:
                raise SystemExit("A cena não tem counting_line. Marque com src.scene_editor (tecla c).")
            self.line = scene.counting_line
        elif task == "retencao":
            if not scene.stop_line:
                raise SystemExit("A cena não tem stop_line. Marque com src.scene_editor (tecla r).")
            self.line = scene.stop_line
            self.direction = scene.stop_line_direction
        elif task == "semaforo" and not scene.traffic_light_roi:
            raise SystemExit("A cena não tem traffic_light_roi. Marque com src.scene_editor (tecla t).")

        self.events: list[dict] = []
        self.reviewed = np.zeros(self.n, dtype=bool)
        self.annotator = annotator
        if out.exists():
            ann = load_annotation(out)
            self.events = ann["events"]
            self.reviewed = intervals_to_mask(ann["reviewed"], self.n)
            self.annotator = annotator or ann.get("annotator", "")
            print(f"Continuando {out}: {len(self.events)} eventos, "
                  f"{self.reviewed.mean():.0%} do vídeo assistido")
        self.history: list[dict] = []  # para o [u]

        # fases verdes do gabarito a pular (retencao)
        self.skip = np.zeros(self.n, dtype=bool)
        if light_gt is not None:
            states = light_state_per_frame(load_annotation(light_gt), self.n)
            m = int(round(margin_s * self.fps))
            green = states == "verde"
            for s, e in frames_to_intervals(green):
                if e - s + 1 > 2 * m:
                    self.skip[s + m:e - m + 1] = True
            if not skip_green:
                self.skip[:] = False
        self.light_gt = light_gt

        self.playing = False
        self.speed_i = SPEEDS.index(1.0)
        self.idx = -1
        self.frame = None
        self.seek(int(np.argmin(self.reviewed)) if self.reviewed.any() and not self.reviewed.all() else 0)

    # ------------------------------------------------------------ leitura
    def seek(self, idx: int) -> None:
        idx = max(0, min(idx, self.n - 1))
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = self.cap.read()
        if ok:
            self.frame, self.idx = frame, idx

    def step(self) -> bool:
        """Avança um quadro em sequência (conta como assistido)."""
        target = self.idx + 1
        while target < self.n and self.skip[target]:
            target += 1
        if target >= self.n:
            self.playing = False
            return False
        if target != self.idx + 1:
            self.seek(target)
        else:
            ok, frame = self.cap.read()
            if not ok:
                self.playing = False
                return False
            self.frame, self.idx = frame, target
        self.reviewed[self.idx] = True
        return True

    def next_phase(self) -> None:
        """Pula para o próximo trecho não verde (retencao com --light-gt)."""
        f = self.idx + 1
        while f < self.n and self.skip[f]:
            f += 1
        while f < self.n and not self.skip[f]:  # sai do trecho atual
            f += 1
        while f < self.n and self.skip[f]:
            f += 1
        self.seek(f)

    # ------------------------------------------------------------ eventos
    def add(self, label: str) -> None:
        ev = {"frame": int(self.idx), "time_s": round(self.idx / self.fps, 3), "label": label}
        if self.task in ("contagem", "retencao"):
            ev["direction"] = self.direction
        if self.task == "semaforo":  # um estado por quadro: substitui
            self.events = [e for e in self.events if e["frame"] != self.idx]
        self.events.append(ev)
        self.events.sort(key=lambda e: e["frame"])
        self.history.append(ev)
        self.reviewed[self.idx] = True

    def undo(self) -> None:
        while self.history:
            ev = self.history.pop()
            if ev in self.events:
                self.events.remove(ev)
                return

    def delete_nearest(self) -> None:
        if not self.events:
            return
        ev = min(self.events, key=lambda e: abs(e["frame"] - self.idx))
        if abs(ev["frame"] - self.idx) <= self.fps:
            self.events.remove(ev)

    def save(self) -> None:
        self.out.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "task": self.task,
            "scene": self.scene.name,
            "video": Path(self.scene.video).as_posix(),
            "fps": self.fps,
            "n_frames": self.n,
            "line": self.line,
            "annotator": self.annotator,
            "updated": dt.datetime.now().isoformat(timespec="seconds"),
            "light_gt": Path(self.light_gt).as_posix() if self.light_gt else None,
            "reviewed": frames_to_intervals(self.reviewed),
            "events": self.events,
        }
        self.out.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")
        print(f"Salvo em {self.out}: {len(self.events)} eventos, {self.reviewed.mean():.0%} assistido")

    # ------------------------------------------------------------ desenho
    def current_light(self) -> str:
        past = [e for e in self.events if e["frame"] <= self.idx]
        return past[-1]["label"] if past else ""

    def render(self) -> np.ndarray:
        img = self.frame.copy()
        h, w = img.shape[:2]
        s = max(1.0, w / 1280)  # escala de texto
        if self.line:
            (x1, y1), (x2, y2) = self.line
            cv2.line(img, (x1, y1), (x2, y2), (255, 200, 0), max(1, int(2 * s)))
        if self.task == "semaforo":
            x1, y1, x2, y2 = self.scene.traffic_light_roi
            pad = 30
            X1, Y1, X2, Y2 = max(0, x1 - pad), max(0, y1 - pad), min(w, x2 + pad), min(h, y2 + pad)
            crop = img[Y1:Y2, X1:X2]
            z = max(1, int(min(0.45 * h / crop.shape[0], 0.3 * w / crop.shape[1])))
            big = cv2.resize(crop, None, fx=z, fy=z, interpolation=cv2.INTER_NEAREST)
            bh, bw = big.shape[:2]
            img[40:40 + bh, w - bw - 10:w - 10] = big
            cv2.rectangle(img, (w - bw - 10, 40), (w - 10, 40 + bh), (255, 255, 255), 2)
            cur = self.current_light()
            if cur:
                cv2.putText(img, f"gabarito: {cur}", (w - bw - 10, 40 + bh + int(30 * s)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8 * s, COLORS[cur], max(1, int(2 * s)))

        # eventos próximos (±3 s)
        near = [e for e in self.events if abs(e["frame"] - self.idx) <= 3 * self.fps]
        for i, e in enumerate(near[-8:]):
            d = (e["frame"] - self.idx) / self.fps
            txt = f"{d:+.2f}s {e['label']}" + (f" ({e['direction']})" if "direction" in e else "")
            col = (0, 255, 255) if e["frame"] == self.idx else (255, 255, 255)
            org = (10, int((70 + 26 * i) * s))
            cv2.putText(img, txt, org, cv2.FONT_HERSHEY_SIMPLEX, 0.6 * s, (0, 0, 0), max(1, int(s)) + 3)
            cv2.putText(img, txt, org, cv2.FONT_HERSHEY_SIMPLEX, 0.6 * s, col, max(1, int(s)))

        # cabeçalho
        keys = {"semaforo": "g y r k = estado", "contagem": "1 carro 2 moto 3 onibus 4 caminhao  t = sentido",
                "retencao": "1 carro 2 moto 3 onibus 4 caminhao  n = prox. fase"}[self.task]
        hud = (f"{self.task} | quadro {self.idx}/{self.n - 1}  {self.idx / self.fps:7.2f}s | "
               f"{SPEEDS[self.speed_i]:g}x {'PLAY' if self.playing else 'PAUSA'} | "
               + (f"sentido {self.direction} | " if self.line else "")
               + f"eventos {len(self.events)} | {keys}")
        cv2.rectangle(img, (0, 0), (w, int(32 * s)), (0, 0, 0), -1)
        cv2.putText(img, hud, (8, int(22 * s)), cv2.FONT_HERSHEY_SIMPLEX, 0.55 * s, (255, 255, 255),
                    max(1, int(s)))

        # linha do tempo: cinza = assistido, azul-escuro = pulado, traços = eventos
        bar_h = int(14 * s)
        bar = np.full((bar_h, w, 3), 40, np.uint8)
        cols = (np.arange(w) * self.n // w).clip(0, self.n - 1)
        bar[:, self.reviewed[cols]] = (150, 150, 150)
        bar[:, self.skip[cols] & ~self.reviewed[cols]] = (90, 40, 0)
        for e in self.events:
            x = int(e["frame"] * w / self.n)
            bar[:, x] = COLORS.get(e["label"], (0, 255, 255))
        bar[:, min(w - 1, int(self.idx * w / self.n))] = (0, 0, 255)
        img[h - bar_h:h] = bar
        return img

    # ------------------------------------------------------------ laço
    def run(self) -> None:
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
        last_save = time.time()
        while True:
            t0 = time.perf_counter()
            cv2.imshow(WINDOW, self.render())
            delay = 1000.0 / (self.fps * SPEEDS[self.speed_i]) if self.playing else 30
            spent = (time.perf_counter() - t0) * 1000
            k = cv2.waitKey(max(1, int(delay - spent))) & 0xFF

            if k == ord(" "):
                self.playing = not self.playing
            elif k == ord("d"):
                self.playing = False
                self.step()
            elif k == ord("a"):
                self.playing = False
                self.seek(self.idx - 1)
            elif k in (ord("l"), ord("j"), ord("L"), ord("J")):
                jump = {ord("l"): 1, ord("j"): -1, ord("L"): 10, ord("J"): -10}[k]
                self.seek(self.idx + int(round(jump * self.fps)))
            elif k == ord("n") and self.task == "retencao":
                self.next_phase()
            elif k in (ord("+"), ord("=")):
                self.speed_i = min(self.speed_i + 1, len(SPEEDS) - 1)
            elif k == ord("-"):
                self.speed_i = max(self.speed_i - 1, 0)
            elif k == ord("t") and self.task == "contagem":
                self.direction = "out" if self.direction == "in" else "in"
            elif k == ord("u"):
                self.undo()
            elif k == ord("x"):
                self.delete_nearest()
            elif k == ord("s"):
                self.save()
                last_save = time.time()
            elif k in (ord("q"), 27):
                break
            elif self.task == "semaforo" and k in LIGHT_KEYS:
                self.add(LIGHT_KEYS[k])
            elif self.task != "semaforo" and k in CLASS_KEYS:
                self.add(CLASS_KEYS[k])
            elif self.playing:
                self.step()

            if time.time() - last_save > 60:  # salva sozinho a cada minuto
                self.save()
                last_save = time.time()
            if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                break
        self.save()
        self.cap.release()
        cv2.destroyAllWindows()


def main() -> None:
    ap = argparse.ArgumentParser(description="Anotação manual do gabarito por eventos")
    ap.add_argument("--config", required=True, help="configs/cena.yaml")
    ap.add_argument("--task", required=True, choices=["semaforo", "contagem", "retencao"])
    ap.add_argument("--out", help="padrão: data/annotations/<cena>/<tarefa>.yaml")
    ap.add_argument("--annotator", default="", help="nome de quem anota (fica no arquivo)")
    ap.add_argument("--light-gt", help="gabarito do semáforo (retencao): habilita [n] e --skip-green")
    ap.add_argument("--skip-green", action="store_true", help="pula as fases verdes do gabarito")
    ap.add_argument("--margin", type=float, default=3.0,
                    help="segundos mantidos no início e no fim de cada fase verde pulada")
    args = ap.parse_args()

    scene = SceneConfig.load(args.config)
    out = Path(args.out) if args.out else Path("data/annotations") / scene.name / f"{args.task}.yaml"
    if args.skip_green and not args.light_gt:
        raise SystemExit("--skip-green precisa de --light-gt")
    Annotator(scene, args.task, out, args.annotator,
              Path(args.light_gt) if args.light_gt else None, args.skip_green, args.margin).run()


if __name__ == "__main__":
    main()
