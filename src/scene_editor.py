"""Fase 1 — Ferramenta para marcar as regiões de uma cena clicando no vídeo.

Abre um quadro do vídeo e permite marcar:
    [c] linha de contagem     (2 cliques)
    [r] linha de retenção     (2 cliques)
    [f] zona da faixa         (vários cliques, ENTER fecha o polígono)
    [t] região do semáforo    (2 cliques: cantos opostos da caixa preta do grupo
                               focal, bem justa; as 3 lâmpadas na vertical)
Outras teclas:
    [u] desfaz o último ponto    [x] apaga a região do modo atual
    [n]/[p] avança/volta 30 quadros    [s] salva o YAML    [q]/ESC sai

Uso:
    python -m src.scene_editor --video data/clips/cruzamento.mp4 --out configs/cena_01.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from src.config import SceneConfig

MODES = {
    ord("c"): ("counting_line", "linha de contagem", 2, (255, 200, 0)),
    ord("r"): ("stop_line", "linha de retencao", 2, (0, 0, 255)),
    ord("f"): ("crosswalk_zone", "faixa de pedestres", None, (0, 255, 255)),
    ord("t"): ("traffic_light_roi", "semaforo", 2, (255, 0, 255)),
}
WINDOW = "scene_editor"


class Editor:
    def __init__(self, video: Path, out: Path, frame_idx: int, condition: str):
        self.cap = cv2.VideoCapture(str(video))
        if not self.cap.isOpened():
            raise SystemExit(f"Não foi possível abrir {video}")
        self.n_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.out = out
        if out.exists():
            self.scene = SceneConfig.load(out)
            print(f"Editando configuração existente: {out}")
        else:
            self.scene = SceneConfig(name=out.stem, video=video.as_posix(), condition=condition)
        self.points: dict[str, list[list[int]]] = {
            "counting_line": list(self.scene.counting_line or []),
            "stop_line": list(self.scene.stop_line or []),
            "crosswalk_zone": list(self.scene.crosswalk_zone or []),
            "traffic_light_roi": (
                [self.scene.traffic_light_roi[:2], self.scene.traffic_light_roi[2:]]
                if self.scene.traffic_light_roi else []
            ),
        }
        self.mode = ord("c")
        if out.exists() and frame_idx == 0:
            frame_idx = self.scene.reference_frame  # reabre no quadro em que foi marcado
        self.frame_idx = frame_idx
        self.frame = self._read(frame_idx)

    def _read(self, idx: int) -> np.ndarray:
        idx = max(0, min(idx, max(self.n_frames - 1, 0)))
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = self.cap.read()
        if not ok:
            raise SystemExit("Falha ao ler o quadro do vídeo")
        self.frame_idx = idx
        h, w = frame.shape[:2]
        self.scene.frame_size = [w, h]
        return frame

    def on_mouse(self, event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        key, _, max_pts, _ = MODES[self.mode]
        pts = self.points[key]
        if max_pts is not None and len(pts) >= max_pts:
            pts.clear()  # recomeça a região
        pts.append([int(x), int(y)])

    def render(self) -> np.ndarray:
        img = self.frame.copy()
        for mkey, (key, label, _, color) in MODES.items():
            pts = self.points[key]
            if not pts:
                continue
            thick = 3 if mkey == self.mode else 2
            arr = np.array(pts, dtype=np.int32)
            if key == "traffic_light_roi" and len(pts) == 2:
                cv2.rectangle(img, tuple(pts[0]), tuple(pts[1]), color, thick)
            elif key == "crosswalk_zone" and len(pts) >= 3:
                cv2.polylines(img, [arr], True, color, thick)
            else:
                cv2.polylines(img, [arr], False, color, thick)
            for p in pts:
                cv2.circle(img, tuple(p), 4, color, -1)
            cv2.putText(img, label, tuple(pts[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        _, label, _, color = MODES[self.mode]
        hud = (f"modo: {label} | quadro {self.frame_idx}/{self.n_frames} | "
               "c r f t = modo  u = desfaz  x = apaga  n/p = quadro  s = salva  q = sai")
        cv2.rectangle(img, (0, 0), (img.shape[1], 28), (0, 0, 0), -1)
        cv2.putText(img, hud, (8, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        return img

    def save(self) -> None:
        p = self.points
        self.scene.counting_line = p["counting_line"] if len(p["counting_line"]) == 2 else None
        self.scene.stop_line = p["stop_line"] if len(p["stop_line"]) == 2 else None
        self.scene.crosswalk_zone = p["crosswalk_zone"] if len(p["crosswalk_zone"]) >= 3 else None
        if len(p["traffic_light_roi"]) == 2:
            (x1, y1), (x2, y2) = p["traffic_light_roi"]
            self.scene.traffic_light_roi = [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]
        else:
            self.scene.traffic_light_roi = None
        self.scene.reference_frame = self.frame_idx
        self.scene.save(self.out)
        print(f"Salvo em {self.out}")

    def run(self) -> None:
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(WINDOW, self.on_mouse)
        while True:
            cv2.imshow(WINDOW, self.render())
            k = cv2.waitKey(20) & 0xFF
            if k in MODES:
                self.mode = k
            elif k == ord("u"):
                pts = self.points[MODES[self.mode][0]]
                if pts:
                    pts.pop()
            elif k == ord("x"):
                self.points[MODES[self.mode][0]].clear()
            elif k == ord("n"):
                self.frame = self._read(self.frame_idx + 30)
            elif k == ord("p"):
                self.frame = self._read(self.frame_idx - 30)
            elif k == ord("s"):
                self.save()
            elif k in (ord("q"), 27):
                break
            if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                break
        self.cap.release()
        cv2.destroyAllWindows()


def main() -> None:
    ap = argparse.ArgumentParser(description="Marca linhas e zonas de uma cena")
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", required=True, help="ex.: configs/cena_01.yaml")
    ap.add_argument("--frame", type=int, default=0, help="quadro inicial")
    ap.add_argument("--condition", default="dia", choices=["dia", "noite", "chuva"])
    args = ap.parse_args()
    Editor(Path(args.video), Path(args.out), args.frame, args.condition).run()


if __name__ == "__main__":
    main()
