"""Leitura e escrita da configuração de cada cena (configs/cena_XX.yaml).

Cada cena guarda, em pixels do vídeo original:
- counting_line: linha virtual de contagem de veículos (2 pontos)
- stop_line: linha de retenção do semáforo (2 pontos)
- crosswalk_zone: polígono sobre a faixa de pedestres (3+ pontos)
- traffic_light_roi: retângulo do semáforo [x1, y1, x2, y2]
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

import yaml

Point = tuple[int, int]

# Classes do COCO usadas no projeto (id -> nome em português)
COCO_CLASSES: dict[int, str] = {
    0: "pessoa",
    2: "carro",
    3: "moto",
    5: "onibus",
    7: "caminhao",
}
VEHICLE_CLASS_IDS: list[int] = [2, 3, 5, 7]
PERSON_CLASS_ID: int = 0


@dataclass
class SceneConfig:
    name: str
    video: str = ""
    frame_size: list[int] = field(default_factory=list)  # [largura, altura]
    condition: str = "dia"  # dia | noite | chuva
    counting_line: list[list[int]] | None = None
    # Nomes dos dois sentidos da linha de contagem: [sentido "in", sentido "out"].
    # Para uma linha marcada da esquerda para a direita na imagem, "in" é quem
    # cruza subindo (de baixo para cima) e "out" quem cruza descendo.
    # Inverter a ordem dos dois cliques inverte os sentidos.
    direction_names: list[str] = field(default_factory=lambda: ["in", "out"])
    stop_line: list[list[int]] | None = None
    # Sentido em que os veículos cruzam a linha de retenção ao entrar no
    # cruzamento, com a mesma convenção de direction_names ("in" = subindo na
    # imagem para uma linha marcada da esquerda para a direita).
    stop_line_direction: str = "in"
    # Polígono opcional onde a conversão é permitida no vermelho (ex.: direita
    # livre). Quem cruza no vermelho e entra nele não é infração.
    exempt_zone: list[list[int]] | None = None
    crosswalk_zone: list[list[int]] | None = None
    # Caixa justa em volta do grupo focal (a "caixa preta" com as 3 lâmpadas),
    # marcada no quadro `reference_frame` do vídeo.
    traffic_light_roi: list[int] | None = None
    # Quadro em que as regiões foram marcadas (usado como referência para
    # compensar o balanço da câmera no semáforo).
    reference_frame: int = 0
    notes: str = ""

    @classmethod
    def load(cls, path: str | Path) -> "SceneConfig":
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in data.items() if k in known})

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            yaml.safe_dump(asdict(self), sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )
