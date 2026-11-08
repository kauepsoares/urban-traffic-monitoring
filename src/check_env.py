"""Fase 0 — Confere se o PyTorch enxerga a GPU e se as bibliotecas estão instaladas.

Uso: python -m src.check_env
"""

import cv2
import supervision as sv
import torch
import ultralytics


def main() -> None:
    print(f"PyTorch      {torch.__version__} (CUDA {torch.version.cuda})")
    print(f"Ultralytics  {ultralytics.__version__}")
    print(f"supervision  {sv.__version__}")
    print(f"OpenCV       {cv2.__version__}")
    ok = torch.cuda.is_available()
    print(f"CUDA disponível: {ok}")
    if not ok:
        raise SystemExit("GPU não encontrada: reinstale o torch pelo índice cu128.")
    print(f"GPU: {torch.cuda.get_device_name(0)} | capability {torch.cuda.get_device_capability(0)}")
    x = torch.randn(2048, 2048, device="cuda")
    print(f"Teste de multiplicação na GPU: ok ({(x @ x).shape})")


if __name__ == "__main__":
    main()
