"""Fase 7 — Avaliação contra o gabarito manual (feito com src.annotate).

Três avaliações, cada uma restrita aos quadros que o anotador assistiu:

semaforo  estado por quadro (states.csv da Fase 4 ou light_states.csv da Fase 5)
          contra o gabarito. Avalia três versões do estado: `raw` (quadro a
          quadro), `state` (suavizado, o que o sistema usa em tempo real) e
          `retro` (trocas reposicionadas em retrospectiva, o que a Fase 5 usa
          para decidir). Métricas: acurácia, cobertura, matriz de confusão e
          erro no instante de cada troca de cor.

contagem  eventos de cruzamento (crossings.csv do count.py, ou do violation.py
          para a linha de retenção) contra os eventos do gabarito. Pareamento
          um para um pelo instante (tolerância --tol) dentro do mesmo sentido.
          Métricas: precisão, revocação, F1, erro de contagem por classe e
          sentido, confusão de classe e erro absoluto médio por janela.
          Aceita vários --pred (ex.: yolo11n, yolo11s, yolo11m) para comparar.

infracao  gabarito de cruzamentos da linha de retenção + gabarito do semáforo.
          Um cruzamento do gabarito é infração se acontece no vermelho do
          gabarito (depois de --red-grace). Compara com os vereditos do
          violation.py: precisão e revocação das infrações, detecção dos
          cruzamentos e tabela cor do gabarito x veredito do sistema.

Saídas em outputs/avaliacao/<cena>_<tarefa>/ (metrics.yaml e CSVs).

Uso:
    python -m src.evaluate semaforo --gt data/annotations/aic21_cam_7/semaforo.yaml \
        --pred outputs/cam_7_semaforo/states.csv
    python -m src.evaluate contagem --gt data/annotations/aic21_cam_7/retencao.yaml \
        --pred outputs/full/cam_7_infracoes/crossings.csv
    python -m src.evaluate infracao --gt data/annotations/aic21_cam_7/retencao.yaml \
        --light-gt data/annotations/aic21_cam_7/semaforo.yaml \
        --pred outputs/full/cam_7_infracoes/crossings.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.annotate import intervals_to_mask, light_state_per_frame, load_annotation
from src.traffic_light import GREEN, RED, UNKNOWN, YELLOW

LIGHTS = [GREEN, YELLOW, RED]


# ---------------------------------------------------------------- utilidades

def ratio(a: float, b: float) -> float | None:
    return round(float(a) / b, 4) if b else None


def prf(tp: int, fp: int, fn: int) -> dict:
    p, r = ratio(tp, tp + fp), ratio(tp, tp + fn)
    f1 = round(2 * p * r / (p + r), 4) if p and r else (0.0 if p is not None and r is not None else None)
    return {"tp": int(tp), "fp": int(fp), "fn": int(fn), "precisao": p, "revocacao": r, "f1": f1}


def plain(x):
    """Converte tipos do numpy/pandas para tipos nativos (para o YAML)."""
    if isinstance(x, dict):
        return {str(k): plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [plain(v) for v in x]
    if isinstance(x, np.generic):
        return x.item()
    return x


def write_yaml(path: Path, data: dict) -> None:
    path.write_text(yaml.safe_dump(plain(data), sort_keys=False, allow_unicode=True), encoding="utf-8")


def out_dir_for(args, gt: dict, task: str) -> Path:
    suffix = f"_{gt['line_name']}" if gt.get("line_name") else ""
    d = Path(args.out) / f"{gt.get('scene', 'cena')}_{task}{suffix}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def match_events(gt_t: np.ndarray, pred_t: np.ndarray, tol: float) -> list[tuple[int, int]]:
    """Pareamento um para um, guloso pela menor diferença de tempo (<= tol)."""
    if len(gt_t) == 0 or len(pred_t) == 0:
        return []
    d = np.abs(gt_t[:, None] - pred_t[None, :])
    gi, pj = np.nonzero(d <= tol)
    order = np.argsort(d[gi, pj], kind="stable")
    used_g, used_p, pairs = set(), set(), []
    for k in order:
        g, p = int(gi[k]), int(pj[k])
        if g not in used_g and p not in used_p:
            used_g.add(g)
            used_p.add(p)
            pairs.append((g, p))
    return pairs


# ---------------------------------------------------------------- semáforo

def retro_states(df: pd.DataFrame, fps: float, lag: int) -> np.ndarray:
    """Estado com as trocas reposicionadas, exatamente como a Fase 5 consulta."""
    from src.violation import LightTimeline

    tl = LightTimeline(lag, fps)
    for f, raw, st in zip(df["frame"], df["raw"], df["state"]):
        tl.add(int(f), raw, st)
    return np.array([tl.state_at(int(f)) for f in df["frame"]], dtype=object)


def transitions(states: np.ndarray, frames: np.ndarray) -> list[tuple[int, str, str]]:
    """Trocas entre estados conhecidos: (quadro da nova cor, cor antiga, cor nova)."""
    out, last = [], None
    for f, s in zip(frames, states):
        if s not in LIGHTS:
            continue
        if last is not None and s != last:
            out.append((int(f), last, s))
        last = s
    return out


def eval_light(args) -> None:
    gt = load_annotation(args.gt)
    fps, n = float(gt["fps"]), int(gt["n_frames"])
    pred = pd.read_csv(args.pred)
    pred = pred[pred["frame"] < n]
    gt_state = light_state_per_frame(gt, n)
    reviewed = intervals_to_mask(gt["reviewed"], n)

    frames = pred["frame"].to_numpy()
    g = gt_state[frames]
    valid = reviewed[frames] & np.isin(g, LIGHTS)  # desconhecido no gabarito não entra
    variants = {"raw": pred["raw"].to_numpy(dtype=object), "state": pred["state"].to_numpy(dtype=object),
                "retro": retro_states(pred, fps, args.lag)}

    out = out_dir_for(args, gt, "semaforo")
    metrics = {"gabarito": str(args.gt), "previsao": str(args.pred),
               "quadros_avaliados": int(valid.sum()),
               "segundos_avaliados": round(valid.sum() / fps, 1),
               "quadros_por_cor_gabarito": {c: int((g[valid] == c).sum()) for c in LIGHTS},
               "versoes": {}}
    gt_tr = [t for t in transitions(gt_state, np.arange(n)) if reviewed[t[0]] and reviewed[t[0] - 1]]
    tr_rows = []
    print(f"Semáforo: {valid.sum()} quadros avaliados ({valid.sum() / fps / 60:.1f} min), "
          f"{len(gt_tr)} trocas no gabarito\n")
    print(f"{'versão':8} {'acurácia':>9} {'cobertura':>9} {'acur. conhecidos':>17} "
          f"{'trocas achadas':>15} {'erro médio (s)':>15} {'erro máx (s)':>13} {'trocas falsas':>14}")
    for name, p in variants.items():
        pv, gv = p[valid], g[valid]
        known = pv != UNKNOWN
        cm = pd.crosstab(pd.Series(gv, name="gabarito"), pd.Series(pv, name="previsto"))
        cm = cm.reindex(index=LIGHTS, columns=LIGHTS + [UNKNOWN], fill_value=0)
        cm.to_csv(out / f"confusao_{name}.csv")

        # trocas: cada troca do gabarito pareada com a troca prevista para a mesma cor
        pr_tr = [t for t in transitions(p, frames) if reviewed[t[0]]]
        errs, matched_pred = [], set()
        for f, old, new in gt_tr:
            cands = [(abs(pf - f), i) for i, (pf, _, pn) in enumerate(pr_tr)
                     if pn == new and i not in matched_pred and abs(pf - f) <= args.tr_tol * fps]
            if cands:
                _, i = min(cands)
                matched_pred.add(i)
                e = (pr_tr[i][0] - f) / fps
                errs.append(e)
                tr_rows.append({"versao": name, "quadro_gabarito": f, "de": old, "para": new,
                                "quadro_previsto": pr_tr[i][0], "erro_s": round(e, 3)})
            else:
                tr_rows.append({"versao": name, "quadro_gabarito": f, "de": old, "para": new,
                                "quadro_previsto": None, "erro_s": None})
        errs = np.array(errs)
        m = {
            "acuracia": ratio((pv == gv).sum(), len(gv)),
            "cobertura": ratio(known.sum(), len(gv)),
            "acuracia_conhecidos": ratio((pv[known] == gv[known]).sum(), known.sum()),
            "por_cor": {c: {"revocacao": ratio(((gv == c) & (pv == c)).sum(), (gv == c).sum()),
                            "precisao": ratio(((gv == c) & (pv == c)).sum(), (pv == c).sum())}
                        for c in LIGHTS},
            "trocas": {"gabarito": len(gt_tr), "encontradas": int(len(errs)),
                       "falsas": len(pr_tr) - len(matched_pred),
                       "erro_medio_s": round(float(errs.mean()), 3) if len(errs) else None,
                       "erro_abs_medio_s": round(float(np.abs(errs).mean()), 3) if len(errs) else None,
                       "erro_abs_max_s": round(float(np.abs(errs).max()), 3) if len(errs) else None},
        }
        metrics["versoes"][name] = m
        t = m["trocas"]
        fmt = lambda v: "-" if v is None else f"{v:.3f}"  # noqa: E731
        print(f"{name:8} {m['acuracia']:9.2%} {m['cobertura']:9.2%} {m['acuracia_conhecidos']:17.2%} "
              f"{t['encontradas']:>7}/{t['gabarito']:<7} {fmt(t['erro_medio_s']):>15} "
              f"{fmt(t['erro_abs_max_s']):>13} {t['falsas']:>14}")
    pd.DataFrame(tr_rows).to_csv(out / "trocas.csv", index=False)
    write_yaml(out / "metrics.yaml", metrics)
    print("\nErro da troca > 0: o sistema troca de cor depois do gabarito.")
    print(f"Saída em {out}")


# ---------------------------------------------------------------- contagem

def load_pred_crossings(path: Path, fps: float, direction: str, merge_heavy: bool,
                        line: str | None = None) -> pd.DataFrame:
    """Lê crossings.csv do count.py (time_s, direction) ou do violation.py (cross_time_s)."""
    df = pd.read_csv(path)
    if line is not None and "line" in df.columns:  # pedestrians.py: várias linhas no mesmo CSV
        df = df[df["line"] == line]
    if "cross_time_s" in df.columns:  # violation.py: linha de retenção, sentido único
        df = df.rename(columns={"cross_time_s": "time_s"})
        df["direction"] = direction
    df["frame"] = (df["time_s"] * fps).round().astype(int)
    df["class"] = df["class"].astype(str)
    if merge_heavy:
        df["class"] = df["class"].replace({"onibus": "pesado", "caminhao": "pesado"})
    return df.reset_index(drop=True)


def gt_events(gt: dict, merge_heavy: bool) -> pd.DataFrame:
    df = pd.DataFrame(gt["events"], columns=["frame", "time_s", "label", "direction"])
    df["time_s"] = df["frame"] / float(gt["fps"])
    df = df.rename(columns={"label": "class"})
    if merge_heavy:
        df["class"] = df["class"].replace({"onibus": "pesado", "caminhao": "pesado"})
    return df.reset_index(drop=True)


def match_crossings(g: pd.DataFrame, p: pd.DataFrame, tol: float) -> pd.DataFrame:
    """Tabela com um registro por evento: par (gabarito, previsto), só gabarito (FN) ou só previsto (FP)."""
    rows = []
    for d in sorted(set(g["direction"]) | set(p["direction"])):
        gi = g.index[g["direction"] == d].to_numpy()
        pi = p.index[p["direction"] == d].to_numpy()
        pairs = match_events(g.loc[gi, "time_s"].to_numpy(), p.loc[pi, "time_s"].to_numpy(), tol)
        mg, mp = {a for a, _ in pairs}, {b for _, b in pairs}
        for a, b in pairs:
            rows.append({"direction": d, "gt_idx": gi[a], "pred_idx": pi[b]})
        rows += [{"direction": d, "gt_idx": gi[a], "pred_idx": None} for a in range(len(gi)) if a not in mg]
        rows += [{"direction": d, "gt_idx": None, "pred_idx": pi[b]} for b in range(len(pi)) if b not in mp]
    m = pd.DataFrame(rows, columns=["direction", "gt_idx", "pred_idx"])
    m["gt_time_s"] = m["gt_idx"].map(lambda i: g.at[i, "time_s"] if pd.notna(i) else np.nan)
    m["pred_time_s"] = m["pred_idx"].map(lambda i: p.at[i, "time_s"] if pd.notna(i) else np.nan)
    m["gt_class"] = m["gt_idx"].map(lambda i: g.at[i, "class"] if pd.notna(i) else None)
    m["pred_class"] = m["pred_idx"].map(lambda i: p.at[i, "class"] if pd.notna(i) else None)
    m["dt_s"] = (m["pred_time_s"] - m["gt_time_s"]).round(3)
    m["_t"] = m["gt_time_s"].fillna(m["pred_time_s"])
    return m.sort_values("_t").drop(columns="_t").reset_index(drop=True)


def count_metrics(g: pd.DataFrame, p: pd.DataFrame, m: pd.DataFrame, window_s: float,
                  reviewed_s: float) -> dict:
    tp = int((m["gt_idx"].notna() & m["pred_idx"].notna()).sum())
    fp = int(m["gt_idx"].isna().sum())
    fn = int(m["pred_idx"].isna().sum())
    res = {"eventos": prf(tp, fp, fn), "por_sentido": {}, "totais": {}}
    for d, md in m.groupby("direction"):
        res["por_sentido"][d] = prf(int((md.gt_idx.notna() & md.pred_idx.notna()).sum()),
                                    int(md.gt_idx.isna().sum()), int(md.pred_idx.isna().sum()))
    # totais por classe e sentido (o que um relatório de contagem mostraria)
    tot_g = g.groupby(["direction", "class"]).size()
    tot_p = p.groupby(["direction", "class"]).size()
    tot = pd.concat([tot_g.rename("gabarito"), tot_p.rename("previsto")], axis=1).fillna(0).astype(int)
    for (d, c), r in tot.iterrows():
        res["totais"][f"{d}/{c}"] = {"gabarito": int(r.gabarito), "previsto": int(r.previsto),
                                     "erro_pct": ratio(100 * (r.previsto - r.gabarito), r.gabarito)}
    res["totais"]["todos"] = {"gabarito": len(g), "previsto": len(p),
                              "erro_pct": ratio(100 * (len(p) - len(g)), len(g))}
    # classe entre os pares
    pairs = m[m.gt_idx.notna() & m.pred_idx.notna()]
    res["classe_correta_nos_pares"] = ratio((pairs.gt_class == pairs.pred_class).sum(), len(pairs))
    res["dt_mediano_s"] = round(float(pairs.dt_s.median()), 3) if len(pairs) else None
    # erro absoluto médio por janela (contagem total por janela)
    if window_s > 0 and len(g) + len(p):
        end = max(g.time_s.max() if len(g) else 0, p.time_s.max() if len(p) else 0)
        bins = np.arange(0, end + window_s, window_s)
        cg, _ = np.histogram(g.time_s, bins)
        cp, _ = np.histogram(p.time_s, bins)
        res["janela_s"] = window_s
        res["mae_por_janela"] = round(float(np.abs(cg - cp).mean()), 3)
        res["mae_relativo_por_janela"] = ratio(np.abs(cg - cp).sum(), cg.sum())
    res["minutos_avaliados"] = round(reviewed_s / 60, 2)
    return res


def eval_count(args) -> None:
    gt = load_annotation(args.gt)
    fps, n = float(gt["fps"]), int(gt["n_frames"])
    reviewed = intervals_to_mask(gt["reviewed"], n)
    direction = "in"
    if gt.get("task") == "retencao" and gt["events"]:
        direction = gt["events"][0].get("direction", "in")
    g = gt_events(gt, args.merge_heavy)
    g = g[reviewed[g.frame.clip(0, n - 1)]].reset_index(drop=True)
    names = args.names or [Path(p).parent.name for p in args.pred]
    out = out_dir_for(args, gt, "contagem")
    summary, all_metrics = [], {"gabarito": str(args.gt), "tolerancia_s": args.tol, "previsoes": {}}
    for name, path in zip(names, args.pred):
        p = load_pred_crossings(Path(path), fps, direction, args.merge_heavy, gt.get("line_name"))
        p = p[reviewed[p.frame.clip(0, n - 1)] & (p.frame < n)].reset_index(drop=True)
        m = match_crossings(g, p, args.tol)
        met = count_metrics(g, p, m, args.window, reviewed.sum() / fps)
        met["previsao"] = str(path)
        all_metrics["previsoes"][name] = met
        m.to_csv(out / f"pares_{name}.csv", index=False)
        cls = m[m.gt_idx.notna() & m.pred_idx.notna()]
        pd.crosstab(cls.gt_class.rename("gabarito"), cls.pred_class.rename("previsto")).to_csv(
            out / f"confusao_classe_{name}.csv")
        e = met["eventos"]
        summary.append({"modelo": name, "gabarito": len(g), "previsto": len(p), **{k: e[k] for k in e},
                        "erro_total_pct": met["totais"]["todos"]["erro_pct"],
                        "classe_ok": met["classe_correta_nos_pares"],
                        "mae_janela": met.get("mae_por_janela")})
    write_yaml(out / "metrics.yaml", all_metrics)
    s = pd.DataFrame(summary)
    s.to_csv(out / "resumo.csv", index=False)
    print(f"Contagem: {len(g)} eventos no gabarito, {reviewed.sum() / fps / 60:.1f} min assistidos, "
          f"tolerância {args.tol} s\n")
    print(s.to_string(index=False))
    print(f"\nSaída em {out}")


# ---------------------------------------------------------------- infração

def eval_violation(args) -> None:
    gt = load_annotation(args.gt)
    lgt = load_annotation(args.light_gt)
    fps, n = float(gt["fps"]), int(gt["n_frames"])
    reviewed = intervals_to_mask(gt["reviewed"], n) & intervals_to_mask(lgt["reviewed"], n)
    light = light_state_per_frame(lgt, n)
    direction = gt["events"][0].get("direction", "in") if gt["events"] else "in"

    g = gt_events(gt, False)
    g = g[reviewed[g.frame.clip(0, n - 1)]].reset_index(drop=True)
    g["gt_light"] = light[g.frame.clip(0, n - 1)]
    # tempo desde o início da fase atual do gabarito
    starts = np.zeros(n, dtype=int)
    for i in range(1, n):
        starts[i] = starts[i - 1] if light[i] == light[i - 1] else i
    g["gt_time_in_state_s"] = (g.frame - starts[g.frame.clip(0, n - 1)]) / fps
    g["gt_violation"] = (g.gt_light == RED) & (g.gt_time_in_state_s >= args.red_grace)

    p = load_pred_crossings(Path(args.pred), fps, direction, False)
    p = p[reviewed[p.frame.clip(0, n - 1)] & (p.frame < n)].reset_index(drop=True)
    p["pred_violation"] = p["verdict"] == "infracao"

    m = match_crossings(g, p, args.tol)
    m["gt_light"] = m.gt_idx.map(lambda i: g.at[i, "gt_light"] if pd.notna(i) else None)
    m["gt_time_in_state_s"] = m.gt_idx.map(lambda i: round(g.at[i, "gt_time_in_state_s"], 3)
                                           if pd.notna(i) else None)
    m["gt_violation"] = m.gt_idx.map(lambda i: bool(g.at[i, "gt_violation"]) if pd.notna(i) else False)
    m["verdict"] = m.pred_idx.map(lambda i: p.at[i, "verdict"] if pd.notna(i) else None)
    m["track_id"] = m.pred_idx.map(lambda i: p.at[i, "track_id"] if pd.notna(i) else None)
    m["pred_violation"] = m.verdict == "infracao"

    tp = int((m.gt_violation & m.pred_violation).sum())
    fp = int((~m.gt_violation & m.pred_violation).sum())
    fn = int((m.gt_violation & ~m.pred_violation).sum())
    crossings = prf(int((m.gt_idx.notna() & m.pred_idx.notna()).sum()),
                    int(m.gt_idx.isna().sum()), int(m.pred_idx.isna().sum()))
    pairs = m[m.gt_idx.notna() & m.pred_idx.notna()]
    table = pd.crosstab(pairs.gt_light.rename("cor_gabarito"), pairs.verdict.rename("veredito"))

    out = out_dir_for(args, gt, "infracao")
    m.to_csv(out / "pares.csv", index=False)
    table.to_csv(out / "cor_x_veredito.csv")
    m[m.gt_violation | m.pred_violation].to_csv(out / "infracoes.csv", index=False)
    metrics = {"gabarito_cruzamentos": str(args.gt), "gabarito_semaforo": str(args.light_gt),
               "previsao": str(args.pred), "tolerancia_s": args.tol, "red_grace_s": args.red_grace,
               "minutos_avaliados": round(reviewed.sum() / fps / 60, 2),
               "infracoes_gabarito": int(g.gt_violation.sum()),
               "infracoes_previstas": int(p.pred_violation.sum()),
               "infracoes": prf(tp, fp, fn), "cruzamentos": crossings,
               "cruzamentos_por_cor_gabarito": g.gt_light.value_counts().to_dict()}
    write_yaml(out / "metrics.yaml", metrics)

    print(f"Infração: {metrics['minutos_avaliados']} min avaliados, tolerância {args.tol} s, "
          f"red-grace {args.red_grace} s\n")
    print(f"Cruzamentos  gabarito {len(g)}  previstos {len(p)}  "
          f"precisão {crossings['precisao']}  revocação {crossings['revocacao']}")
    print(f"Infrações    gabarito {metrics['infracoes_gabarito']}  previstas {metrics['infracoes_previstas']}  "
          f"TP {tp}  FP {fp}  FN {fn}  precisão {metrics['infracoes']['precisao']}  "
          f"revocação {metrics['infracoes']['revocacao']}\n")
    print("Cor do gabarito x veredito do sistema (cruzamentos pareados):")
    print(table.to_string() if len(table) else "  (nenhum par)")
    print(f"\nSaída em {out}")


# ---------------------------------------------------------------- CLI

def main() -> None:
    ap = argparse.ArgumentParser(description="Avaliação contra o gabarito manual")
    sub = ap.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("semaforo", help="estado do semáforo por quadro")
    a.add_argument("--gt", required=True, help="data/annotations/<cena>/semaforo.yaml")
    a.add_argument("--pred", required=True, help="states.csv (Fase 4) ou light_states.csv (Fase 5)")
    a.add_argument("--lag", type=int, default=2, help="atraso da votação (smooth // 2) para a versão retro")
    a.add_argument("--tr-tol", type=float, default=3.0, help="janela (s) para parear trocas de cor")

    b = sub.add_parser("contagem", help="eventos de cruzamento de linha")
    b.add_argument("--gt", required=True, help="contagem.yaml, retencao.yaml ou pedestres_<linha>.yaml")
    b.add_argument("--pred", required=True, nargs="+", help="um ou mais crossings.csv")
    b.add_argument("--names", nargs="+", help="nome de cada --pred (padrão: pasta)")
    b.add_argument("--tol", type=float, default=1.0, help="diferença máxima de tempo (s) num par")
    b.add_argument("--window", type=float, default=60.0, help="janela (s) para o erro por janela")
    b.add_argument("--merge-heavy", action="store_true", help="junta ônibus e caminhão em 'pesado'")

    c = sub.add_parser("infracao", help="avanço de sinal vermelho")
    c.add_argument("--gt", required=True, help="retencao.yaml")
    c.add_argument("--light-gt", required=True, help="semaforo.yaml")
    c.add_argument("--pred", required=True, help="crossings.csv do violation.py")
    c.add_argument("--tol", type=float, default=1.0)
    c.add_argument("--red-grace", type=float, default=0.0, help="tolerância (s) após o início do vermelho")

    for p in (a, b, c):
        p.add_argument("--out", default="outputs/avaliacao")
    args = ap.parse_args()
    {"semaforo": eval_light, "contagem": eval_count, "infracao": eval_violation}[args.cmd](args)


if __name__ == "__main__":
    main()
