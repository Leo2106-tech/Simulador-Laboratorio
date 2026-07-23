from __future__ import annotations

"""Modelo de otimização do agendador de férias do CTO."""

import math
import re
import time
import unicodedata
from collections import defaultdict
from itertools import combinations
from typing import Any

import pulp as pl


_CACHE_COMPAT_TAREFA = {}
_CACHE_ROTA = {}
_CACHE_STATS = defaultdict(int)


class _HiGHSComWarmStart(pl.HiGHS):
    """Pequena ponte para enviar initial values do PuLP como user solution no HiGHS."""

    def buildSolverModel(self, lp):
        super().buildSolverModel(lp)
        indices = []
        valores = []
        for var in lp.variables():
            val = getattr(var, "varValue", None)
            if val is None:
                continue
            lb = var.lowBound
            ub = var.upBound
            val = float(val)
            if lb is not None:
                val = max(float(lb), val)
            if ub is not None:
                val = min(float(ub), val)
            indices.append(int(var.index))
            valores.append(val)
        if not indices:
            lp._highs_warm_start_count = 0
            lp._highs_warm_start_status = "sem_valores"
            return
        status = lp.solverModel.setSolution(len(indices), indices, valores)
        lp._highs_warm_start_count = len(indices)
        lp._highs_warm_start_status = str(status)


def _nome(prefixo, *partes):
    texto = "_".join(str(p) for p in (prefixo, *partes))
    return re.sub(r"[^A-Za-z0-9_]+", "_", texto)[:240]


def _valor(var):
    val = var.varValue
    return 0.0 if val is None else float(val)


def _limpar_caches_modelo():
    _CACHE_COMPAT_TAREFA.clear()
    _CACHE_ROTA.clear()
    _CACHE_STATS.clear()


def _snapshot_cache_stats():
    return dict(_CACHE_STATS)


def _delta_cache_stats(antes):
    chaves = set(antes) | set(_CACHE_STATS)
    return {ch: int(_CACHE_STATS.get(ch, 0) - antes.get(ch, 0)) for ch in chaves}


def _remover_acentos(texto):
    return unicodedata.normalize("NFKD", str(texto)).encode("ascii", errors="ignore").decode("ascii")


def _normalizar_cidade(cidade):
    return _remover_acentos(str(cidade).strip().lower())


def _fmt_periodo(dados, ini, fim):
    return f"{dados['t_para_data'][ini].strftime('%d/%m/%Y')} a {dados['t_para_data'][fim].strftime('%d/%m/%Y')}"


# Regra fixa de inicio de ferias: segunda a quinta.
DIAS_SEMANA_INICIO_FERIAS_PERMITIDOS = {0, 1, 2, 3}
DURACOES_BASE_BLOCOS_FERIAS = {5, 10, 14, 15, 20, 30}


def _dias_programados_total(dados: dict[str, Any], i: str, prog_days: set[int] | None = None) -> int:
    """Dias de ferias ja programados na base, incluindo trecho anterior ao horizonte."""
    if "dias_ferias_programadas_total" in dados:
        return max(int(dados["dias_ferias_programadas_total"].get(i, 0)), 0)
    return len(prog_days or set())


def _inicio_ferias_permitido_clt(dados: dict[str, Any], t: int) -> bool:
    data = dados.get("t_para_data", {}).get(t)
    return data is not None and int(data.weekday()) in DIAS_SEMANA_INICIO_FERIAS_PERMITIDOS


def _intervalos_contiguos(dias: set[int]) -> list[tuple[int, int]]:
    if not dias:
        return []
    ordenados = sorted(dias)
    intervalos = []
    ini = fim = ordenados[0]
    for t in ordenados[1:]:
        if t == fim + 1:
            fim = t
        else:
            intervalos.append((ini, fim))
            ini = fim = t
    intervalos.append((ini, fim))
    return intervalos


def _duracoes_candidatas_ferias(
    saldo: int,
    prog_days: set[int],
    max_total_dias: int,
    max_len: int,
    usar_duracoes_discretas: bool = True,
) -> list[int]:
    if max_len <= 0:
        return []
    if not usar_duracoes_discretas:
        return list(range(5, max_len + 1))

    duracoes = {d for d in DURACOES_BASE_BLOCOS_FERIAS if 5 <= d <= max_len}
    for dur in {saldo, max_total_dias, len(prog_days)}:
        if 5 <= dur <= max_len:
            duracoes.add(int(dur))

    for ini_prog, fim_prog in _intervalos_contiguos(prog_days):
        span = fim_prog - ini_prog + 1
        for extra in {0, 1, 2, 3, 5, 7, 10, 14, 15, saldo, max_total_dias - span}:
            dur = span + max(0, int(extra))
            if 5 <= dur <= max_len:
                duracoes.add(dur)
        for dur in range(max(5, span), min(max_len, span + 10) + 1):
            duracoes.add(dur)
    return sorted(duracoes)


def gerar_blocos_ferias_colaborador(
    i: str,
    dados: dict[str, Any],
    max_blocos_por_colaborador: int | None = None,
    max_duracao_bloco: int | None = None,
    usar_duracoes_discretas: bool = True,
) -> list[dict[str, Any]]:
    """
    Gera blocos individuais candidatos de ferias.

    Na formulacao por colunas o mestre escolhe diretamente z[i, bloco_id].
    Ferias programadas continuam abatendo o saldo novo a planejar.
    """
    T = dados["T"]
    H = max(T)
    saldo_total = max(int(dados["b"].get(i, 0)), 0)
    limite = int(dados["L"].get(i, H + 1))
    prog_days = {t for t in T if dados["ferias_programadas"].get((i, t), 0) == 1}
    dias_prog_total = _dias_programados_total(dados, i, prog_days)
    saldo_novo_necessario = max(saldo_total - dias_prog_total, 0)

    if saldo_total == 0 and not prog_days:
        return []

    max_total_dias = max(saldo_total, dias_prog_total, len(prog_days))
    if max_total_dias <= 0:
        return []
    max_len = min(int(max_duracao_bloco or H), 30, max(5, max_total_dias))
    duracoes = _duracoes_candidatas_ferias(
        saldo_total, prog_days, max_total_dias, max_len, usar_duracoes_discretas
    )

    blocos: list[dict[str, Any]] = []
    for ini_prog, fim_prog in _intervalos_contiguos(prog_days):
        dias = tuple(range(ini_prog, fim_prog + 1))
        bloco_prog = {
            "colaborador": i,
            "bloco_id": f"{i}_PROG_{ini_prog}_{fim_prog}",
            "ini": ini_prog,
            "fim": fim_prog,
            "duracao": len(dias),
            "dias": dias,
            "dias_novos": 0,
            "dias_programados": len(dias),
            "tem_bloco_14": int(len(dias) >= 14),
            "programado_fixo": 1,
        }
        blocos.append(bloco_prog)
        if max_blocos_por_colaborador and len(blocos) >= max_blocos_por_colaborador:
            return blocos

    for ini in [t for t in T if _inicio_ferias_permitido_clt(dados, t)]:
        if limite <= H and ini > limite:
            continue
        for dur in duracoes:
            fim = ini + dur - 1
            if fim > H:
                continue
            dias = tuple(range(ini, fim + 1))
            dias_set = set(dias)
            dias_programados = len(dias_set & prog_days)
            dias_novos = dur - dias_programados
            if dias_novos > saldo_novo_necessario:
                continue
            blocos.append({
                "colaborador": i,
                "bloco_id": f"{i}_B{len(blocos) + 1:06d}",
                "ini": ini,
                "fim": fim,
                "duracao": dur,
                "dias": dias,
                "dias_novos": dias_novos,
                "dias_programados": dias_programados,
                "tem_bloco_14": int(dur >= 14),
                "programado_fixo": 0,
            })
            if max_blocos_por_colaborador and len(blocos) >= max_blocos_por_colaborador:
                return blocos
    return blocos


def gerar_blocos_ferias_mestre(dados: dict[str, Any], config: dict[str, Any]):
    """
    Gera todos os blocos candidatos para IA+IE.

    I_S nao entra aqui: suplente potencial e individuo virtual contratavel, sem ferias.
    """
    max_blocos_candidatos = config.get("max_blocos_candidatos_por_colaborador")
    if max_blocos_candidatos is None:
        # O mestre restrito comeca pequeno: poucos blocos de ferias por funcionario.
        # Os demais blocos permanecem em um pool e entram por pricing heuristico.
        max_blocos_candidatos = config.get("max_blocos_mestre_por_colaborador", 9)
    max_blocos_candidatos = int(max_blocos_candidatos or 0)
    max_duracao_bloco = config.get("max_duracao_bloco")
    if max_duracao_bloco == 0:
        max_duracao_bloco = None

    I_F = dados["I_A"] + dados["I_E"]
    blocos_por_i: dict[str, list[dict[str, Any]]] = {}
    blocos_pool_por_i: dict[str, list[dict[str, Any]]] = {}
    bloco_info: dict[tuple[str, str], dict[str, Any]] = {}
    blocos_por_dia = defaultdict(list)
    fallback_colaboradores = []

    for i in I_F:
        blocos = gerar_blocos_ferias_colaborador(
            i, dados, None, max_duracao_bloco, usar_duracoes_discretas=True
        )
        if _precisa_bloco_nao_vazio(i, dados) and not blocos:
            blocos = gerar_blocos_ferias_colaborador(
                i, dados, None, max_duracao_bloco, usar_duracoes_discretas=False
            )
            fallback_colaboradores.append(i)
        blocos_pool_por_i[i] = blocos
        if max_blocos_candidatos > 0:
            blocos = _selecionar_blocos_mestre(i, dados, blocos, max_blocos_candidatos)
        blocos_por_i[i] = blocos
        for bloco in blocos:
            chave = (i, bloco["bloco_id"])
            bloco_info[chave] = bloco
            for t in bloco["dias"]:
                blocos_por_dia[(i, t)].append(chave)

    print("\nBlocos candidatos de ferias para o mestre")
    print("  Formulacao: z[i,b] escolhe blocos, nao planos completos lambda[i,a]")
    print(f"  Colaboradores com ferias (IA+IE): {len(I_F):,}")
    if max_blocos_candidatos > 0:
        print(f"  Blocos iniciais no mestre por colaborador: {max_blocos_candidatos:,}")
    else:
        print("  Blocos no mestre: malha completa gerada pelas regras de ferias")
    print(f"  Blocos ativos iniciais: {sum(len(v) for v in blocos_por_i.values()):,}")
    print(f"  Blocos no pool completo: {sum(len(v) for v in blocos_pool_por_i.values()):,}")
    if fallback_colaboradores:
        print(f"  Fallback de duracoes completas usado para {len(fallback_colaboradores):,} colaborador(es).")
    return blocos_por_i, bloco_info, blocos_por_dia, blocos_pool_por_i


def _precisa_bloco_nao_vazio(i: str, dados: dict[str, Any]) -> bool:
    T = dados["T"]
    H = max(T)
    saldo = max(int(dados["b"].get(i, 0)), 0)
    prog_days = {t for t in T if dados["ferias_programadas"].get((i, t), 0) == 1}
    prog = _dias_programados_total(dados, i, prog_days)
    saldo_novo = max(saldo - prog, 0)
    exige_14 = saldo_novo >= 14 and int(dados["tem_bloco_aprovado_14"].get(i, 0)) == 0
    return bool(prog) or (int(dados["L"].get(i, H + 1)) <= H and saldo_novo > 0) or exige_14


def _selecionar_blocos_mestre(i: str, dados: dict[str, Any], blocos: list[dict[str, Any]], limite_qtd: int):
    """
    Reduz a malha de blocos que entra no mestre.

    A enumeracao completa pode criar dezenas de milhares de tarefas e milhoes de
    rotas iniciais. Para manter o modelo tatico resolvivel na interface, escolhemos
    uma amostra priorizando: ferias programadas, blocos de 14 dias quando exigidos,
    datas antes do prazo de gozo e blocos que ajudam a fechar o saldo novo.
    """
    if limite_qtd <= 0 or len(blocos) <= limite_qtd:
        return blocos

    T = dados["T"]
    H = max(T)
    saldo_total = max(int(dados["b"].get(i, 0)), 0)
    prog_days = {t for t in T if dados["ferias_programadas"].get((i, t), 0) == 1}
    saldo_novo = max(saldo_total - _dias_programados_total(dados, i, prog_days), 0)
    limite = int(dados["L"].get(i, H + 1))
    exige_14 = saldo_novo >= 14 and int(dados["tem_bloco_aprovado_14"].get(i, 0)) == 0

    def score(bloco):
        dias = set(bloco["dias"])
        cobre_prog = len(dias & prog_days)
        atraso = max(0, int(bloco["fim"]) - limite) if limite <= H else 0
        falta_14 = 0 if (not exige_14 or int(bloco["duracao"]) >= 14) else 1
        return (
            -cobre_prog,
            falta_14,
            abs(int(bloco["dias_novos"]) - saldo_novo),
            atraso,
            int(bloco["ini"]),
            int(bloco["duracao"]),
        )

    def combo_valido(combo):
        combo = sorted(combo, key=lambda b: (b["ini"], b["fim"]))
        for b1, b2 in zip(combo, combo[1:]):
            if int(b2["ini"]) <= int(b1["fim"]) + 1:
                return False
        dias_combo = set()
        for b in combo:
            dias_b = set(b["dias"])
            if dias_combo & dias_b:
                return False
            dias_combo |= dias_b
        if not prog_days.issubset(dias_combo):
            return False
        dias_novos = sum(int(b["dias_novos"]) for b in combo)
        if dias_novos < saldo_novo:
            return False
        if exige_14 and not any(int(b["duracao"]) >= 14 for b in combo):
            return False
        return True

    # Gera inicialmente blocos que formem pelo menos tres planos completos.
    selecionados = []
    ids = set()
    candidatos_base = sorted(blocos, key=score)[: min(len(blocos), 60)]
    planos = []
    for qtd in (1, 2, 3):
        for combo in combinations(candidatos_base, qtd):
            if combo_valido(combo):
                planos.append(combo)
                if len(planos) >= int(dados.get("min_planos_iniciais_ferias", 3)):
                    break
        if len(planos) >= int(dados.get("min_planos_iniciais_ferias", 3)):
            break

    for combo in planos:
        for b in combo:
            if b["bloco_id"] not in ids:
                selecionados.append(b)
                ids.add(b["bloco_id"])

    if len(selecionados) < min(limite_qtd, len(blocos)):
        for b in sorted(blocos, key=score):
            if b["bloco_id"] in ids:
                continue
            selecionados.append(b)
            ids.add(b["bloco_id"])
            if len(selecionados) >= limite_qtd:
                break

    ids = {b["bloco_id"] for b in selecionados}

    # Garante pelo menos uma alternativa para cada dia ja programado.
    for t in sorted(prog_days):
        if any(t in b["dias"] for b in selecionados):
            continue
        candidatos = [b for b in blocos if t in b["dias"] and b["bloco_id"] not in ids]
        if candidatos:
            escolhido = min(candidatos, key=score)
            selecionados.append(escolhido)
            ids.add(escolhido["bloco_id"])

    # Garante ao menos uma opcao de bloco longo quando a regra de 14 dias se aplica.
    if exige_14 and not any(int(b["duracao"]) >= 14 for b in selecionados):
        candidatos = [b for b in blocos if int(b["duracao"]) >= 14 and b["bloco_id"] not in ids]
        if candidatos:
            escolhido = min(candidatos, key=score)
            selecionados.append(escolhido)
            ids.add(escolhido["bloco_id"])

    fixos_programados = [b for b in selecionados if int(b.get("programado_fixo", 0)) == 1]
    ids_fixos = {b["bloco_id"] for b in fixos_programados}
    demais = [b for b in selecionados if b["bloco_id"] not in ids_fixos]
    espaco_demais = max(0, limite_qtd - len(fixos_programados))
    selecionados = fixos_programados + sorted(demais, key=score)[:espaco_demais]
    return sorted(selecionados, key=lambda b: (b["ini"], b["fim"], b["duracao"], b["bloco_id"]))


def _blocos_incompativeis(b1: dict[str, Any], b2: dict[str, Any]) -> bool:
    return int(b2["ini"]) <= int(b1["fim"]) + 1 and int(b1["ini"]) <= int(b2["fim"]) + 1


def gerar_tarefas_cobertura_5_dias(dados: dict[str, Any], bloco_info: dict[tuple[str, str], dict[str, Any]]):
    """
    Quebra cada bloco candidato em janelas de ate 5 dias consecutivos.

    A cobertura e binaria por tarefa: a janela inteira e coberta por um unico suplente
    ou vira falta u[j]. Se o bloco nao for escolhido, a tarefa fica inativa.
    """
    tarefas: dict[str, dict[str, Any]] = {}
    tarefas_por_bloco = defaultdict(list)
    for (i, bloco_id), bloco in bloco_info.items():
        dias = list(bloco["dias"])
        for idx in range(0, len(dias), 5):
            janela = tuple(dias[idx:idx + 5])
            tarefa_id = f"T{len(tarefas) + 1:07d}"
            projeto = dados["projeto_original"][i]
            tarefa = {
                "tarefa_id": tarefa_id,
                "colaborador_ferias": i,
                "bloco_id": bloco_id,
                "bloco_chave": (i, bloco_id),
                "ini": min(janela),
                "fim": max(janela),
                "dias": janela,
                "projeto": projeto,
                "cargo": dados["cargo"][i],
                "turno": dados["turno_original"][i],
                "cidade_projeto": dados["cidade_projeto"].get(projeto, ""),
            }
            tarefas[tarefa_id] = tarefa
            tarefas_por_bloco[(i, bloco_id)].append(tarefa_id)
    print(f"  Tarefas de cobertura em janelas de ate 5 dias: {len(tarefas):,}")
    return tarefas, dict(tarefas_por_bloco)


def adicionar_bloco_ativo(
    bloco: dict[str, Any],
    blocos_por_i,
    bloco_info,
    blocos_por_dia,
):
    """Insere um bloco no mestre restrito, se ele ainda nao estiver ativo."""
    chave = (bloco["colaborador"], bloco["bloco_id"])
    if chave in bloco_info:
        return None
    blocos_por_i.setdefault(bloco["colaborador"], []).append(bloco)
    bloco_info[chave] = bloco
    for t in bloco["dias"]:
        blocos_por_dia[(bloco["colaborador"], t)].append(chave)
    return chave


def adicionar_tarefas_para_blocos(dados, tarefas, tarefas_por_bloco, bloco_info, novas_chaves):
    """Cria tarefas de cobertura de ate 5 dias apenas para os novos blocos."""
    novas_tarefas = []
    for chave in novas_chaves:
        if chave is None:
            continue
        if chave in tarefas_por_bloco:
            continue
        i, bloco_id = chave
        bloco = bloco_info[chave]
        dias = list(bloco["dias"])
        for idx in range(0, len(dias), 5):
            janela = tuple(dias[idx:idx + 5])
            tarefa_id = f"T{len(tarefas) + 1:07d}"
            projeto = dados["projeto_original"][i]
            tarefa = {
                "tarefa_id": tarefa_id,
                "colaborador_ferias": i,
                "bloco_id": bloco_id,
                "bloco_chave": chave,
                "ini": min(janela),
                "fim": max(janela),
                "dias": janela,
                "projeto": projeto,
                "cargo": dados["cargo"][i],
                "turno": dados["turno_original"][i],
                "cidade_projeto": dados["cidade_projeto"].get(projeto, ""),
            }
            tarefas[tarefa_id] = tarefa
            tarefas_por_bloco.setdefault(chave, []).append(tarefa_id)
            novas_tarefas.append(tarefa_id)
    return novas_tarefas


def _turno_permitido(dados, suplente, turno_tarefa):
    return bool(int(dados["flex"].get(suplente, 1))) or dados["turno_original"].get(suplente) == turno_tarefa


def _tarefa_compativel(dados, suplente, tarefa, tipo):
    chave = (
        tipo,
        suplente,
        tarefa.get("tarefa_id"),
        tarefa.get("colaborador_ferias"),
        tarefa.get("cargo"),
        tarefa.get("turno"),
    )
    if chave in _CACHE_COMPAT_TAREFA:
        _CACHE_STATS["compat_hit"] += 1
        return _CACHE_COMPAT_TAREFA[chave]
    _CACHE_STATS["compat_miss"] += 1
    if tarefa["colaborador_ferias"] == suplente:
        resultado = False
    elif dados["a"].get((dados["cargo"][suplente], tarefa["cargo"]), 0) != 1:
        resultado = False
    elif not _turno_permitido(dados, suplente, tarefa["turno"]):
        resultado = False
    else:
        resultado = True
    _CACHE_COMPAT_TAREFA[chave] = resultado
    return resultado


def _dist_pessoa_projeto(dados, suplente, projeto):
    return float(dados["dist"].get((suplente, projeto), 999.0))


def _dist_cidade_cidade(dados, cidade_a, cidade_b):
    ca = _normalizar_cidade(cidade_a)
    cb = _normalizar_cidade(cidade_b)
    if ca == cb:
        return 0.0
    matriz = dados.get("dist_cidade", {})
    if (ca, cb) in matriz:
        return float(matriz[(ca, cb)])
    if (cb, ca) in matriz:
        return float(matriz[(cb, ca)])
    print(f"AVISO: distancia cidade-cidade ausente: {cidade_a} -> {cidade_b}. Usando 999 km.")
    return 999.0


def _custo_transicao_tarefas(dados, suplente, t1, t2):
    Cdist = float(dados["Cdist"])
    delta_dias = int(t2["ini"]) - int(t1["fim"])
    mesma_cidade = _normalizar_cidade(t1["cidade_projeto"]) == _normalizar_cidade(t2["cidade_projeto"])
    if delta_dias <= 5 and mesma_cidade:
        return 0.0, "permanece_no_mesmo_projeto"
    if delta_dias <= 5:
        dist = _dist_cidade_cidade(dados, t1["cidade_projeto"], t2["cidade_projeto"])
        return Cdist * dist, f"{t1['projeto']} -> {t2['projeto']}"
    dist_volta = _dist_pessoa_projeto(dados, suplente, t1["projeto"])
    dist_ida = _dist_pessoa_projeto(dados, suplente, t2["projeto"])
    return Cdist * (dist_volta + dist_ida), f"{t1['projeto']} -> casa -> {t2['projeto']}"


def calcular_custo_transporte_rota(dados, suplente, tarefas_rota):
    """Calcula transporte literal da rota: casa -> tarefas -> casa, com regra dos 5 dias."""
    if not tarefas_rota:
        return 0.0, []
    Cdist = float(dados["Cdist"])
    ordenadas = sorted(tarefas_rota, key=lambda j: (j["ini"], j["fim"], j["tarefa_id"]))
    pernas = []

    primeira = ordenadas[0]
    dist_ini = _dist_pessoa_projeto(dados, suplente, primeira["projeto"])
    custo_total = Cdist * dist_ini
    pernas.append({
        "origem": dados["cidade"].get(suplente, ""),
        "destino": primeira["projeto"],
        "regra": "ida_inicial",
        "distancia_km": dist_ini,
        "custo": Cdist * dist_ini,
    })

    for atual, prox in zip(ordenadas, ordenadas[1:]):
        custo, regra = _custo_transicao_tarefas(dados, suplente, atual, prox)
        custo_total += custo
        pernas.append({
            "origem": atual["projeto"],
            "destino": prox["projeto"],
            "regra": regra,
            "distancia_km": custo / Cdist if Cdist else 0.0,
            "custo": custo,
        })

    ultima = ordenadas[-1]
    dist_fim = _dist_pessoa_projeto(dados, suplente, ultima["projeto"])
    custo_total += Cdist * dist_fim
    pernas.append({
        "origem": ultima["projeto"],
        "destino": dados["cidade"].get(suplente, ""),
        "regra": "volta_final",
        "distancia_km": dist_fim,
        "custo": Cdist * dist_fim,
    })
    return custo_total, pernas


def calcular_custo_mobilizacao_rota(dados, suplente, tarefas_rota):
    """Identifica projetos que exigem mobilizacao; o custo anual e cobrado no mestre."""
    projetos = sorted({j["projeto"] for j in tarefas_rota})
    cobrados = [p for p in projetos if int(dados.get("mobilizado", {}).get((suplente, p), 0)) == 0]
    return 0.0, cobrados


def calcular_custo_noturno_rota(dados, suplente, tarefas_rota):
    if dados["turno_original"].get(suplente) != "diurno":
        return 0.0
    cargo_s = dados["cargo"][suplente]
    return sum(
        float(dados["Cnot"][cargo_s]) * len(j["dias"])
        for j in tarefas_rota
        if j["turno"] == "noturno"
    )


def _criar_rota(dados, tipo, suplente, tarefas_rota, tarefas):
    tarefas_objs = [tarefas[j] if isinstance(j, str) else j for j in tarefas_rota]
    tarefas_objs = sorted(tarefas_objs, key=lambda x: (x["ini"], x["fim"], x["tarefa_id"]))
    ids = tuple(j["tarefa_id"] for j in tarefas_objs)
    assinatura_tarefas = tuple(
        (j["tarefa_id"], j["ini"], j["fim"], j["projeto"], j["cargo"], j["turno"])
        for j in tarefas_objs
    )
    chave_cache = (tipo, suplente, assinatura_tarefas)
    if chave_cache in _CACHE_ROTA:
        _CACHE_STATS["rota_hit"] += 1
        return dict(_CACHE_ROTA[chave_cache])
    _CACHE_STATS["rota_miss"] += 1
    custo_transp, pernas = calcular_custo_transporte_rota(dados, suplente, tarefas_objs)
    custo_mob, projetos_mob = calcular_custo_mobilizacao_rota(dados, suplente, tarefas_objs)
    custo_not = calcular_custo_noturno_rota(dados, suplente, tarefas_objs)
    rota = {
        "tipo": tipo,
        "suplente": suplente,
        "tarefas": ids,
        "dias": tuple(sorted({t for j in tarefas_objs for t in j["dias"]})),
        "projetos": tuple(j["projeto"] for j in tarefas_objs),
        "custo_transporte": custo_transp,
        "custo_mobilizacao": custo_mob,
        "custo_noturno": custo_not,
        "custo_total": custo_transp + custo_mob + custo_not,
        "projetos_mobilizados_cobrados": tuple(projetos_mob),
        "pernas_transporte": tuple(pernas),
        "assinatura": (tipo, suplente, ids),
    }
    _CACHE_ROTA[chave_cache] = dict(rota)
    return rota


def rota_vazia(dados, suplente, tarefas):
    rota = _criar_rota(dados, "IE", suplente, (), tarefas)
    rota["rota_id"] = f"RE_{suplente}_VAZIA"
    return rota


def _rotas_temporais_compativeis(tarefas_seq):
    ordenadas = sorted(tarefas_seq, key=lambda j: (j["ini"], j["fim"], j["tarefa_id"]))
    for a, b in zip(ordenadas, ordenadas[1:]):
        if int(a["fim"]) >= int(b["ini"]):
            return False
    return True


def gerar_rotas_iniciais(dados, tarefas, config):
    """
    Cria o conjunto inicial minimo.

    O mestre ja fica viavel pelas variaveis de falta u[j]. Por isso, por padrao,
    nao enumeramos rotas individuais aqui: elas entram na primeira iteracao de
    geracao de colunas pelo criterio de vizinho mais proximo.
    """
    rotas_E: dict[str, list[dict[str, Any]]] = defaultdict(list)
    rotas_S: dict[str, list[dict[str, Any]]] = defaultdict(list)
    vistos = set()

    for h in dados["I_E"]:
        r = rota_vazia(dados, h, tarefas)
        rotas_E[h].append(r)
        vistos.add(r["assinatura"])

    def adicionar(tipo, suplente, ids):
        rota = _criar_rota(dados, tipo, suplente, ids, tarefas)
        if rota["assinatura"] in vistos:
            return
        prefixo = "RE" if tipo == "IE" else "RS"
        rota["rota_id"] = f"{prefixo}_{suplente}_{len(rotas_E[suplente] if tipo == 'IE' else rotas_S[suplente]) + 1:06d}"
        (rotas_E if tipo == "IE" else rotas_S)[suplente].append(rota)
        vistos.add(rota["assinatura"])

    tarefas_lista = list(tarefas.values())
    gerar_individuais = bool(config.get("gerar_rotas_individuais_iniciais", False))
    max_individuais = int(config.get("max_rotas_individuais_iniciais_por_suplente", 250))
    compat_por_suplente = {}
    for tipo, conjunto in [("IE", dados["I_E"]), ("IS", dados["I_S"])]:
        for suplente in conjunto:
            comp = [j for j in tarefas_lista if _tarefa_compativel(dados, suplente, j, tipo)]
            compat_por_suplente[(tipo, suplente)] = comp
            if not gerar_individuais:
                continue
            comp_inicial = sorted(
                comp,
                key=lambda j: (
                    -float(dados["Receita"][j["cargo"]]) * len(j["dias"]),
                    calcular_custo_transporte_rota(dados, suplente, [j])[0],
                    j["ini"],
                ),
            )
            if max_individuais > 0:
                comp_inicial = comp_inicial[:max_individuais]
            for j in comp_inicial:
                adicionar(tipo, suplente, (j["tarefa_id"],))

            # Rotas naturais de duas tarefas no mesmo projeto, com intervalo curto.
            limite_pares = int(config.get("max_rotas_pares_iniciais_por_suplente", 150))
            qtd_pares = 0
            for a_idx, a in enumerate(comp_inicial):
                if qtd_pares >= limite_pares:
                    break
                for b in comp_inicial[a_idx + 1:]:
                    if a["projeto"] != b["projeto"]:
                        continue
                    if int(a["fim"]) < int(b["ini"]) and int(b["ini"]) - int(a["fim"]) <= 5:
                        adicionar(tipo, suplente, (a["tarefa_id"], b["tarefa_id"]))
                        qtd_pares += 1
                        if qtd_pares >= limite_pares:
                            break

    total_rotas = sum(len(v) for v in rotas_E.values()) + sum(len(v) for v in rotas_S.values())
    print(f"  Rotas iniciais geradas: {total_rotas:,}")
    if gerar_individuais:
        print(f"  Limite de rotas individuais iniciais por suplente: {max_individuais:,}")
    else:
        print("  Rotas individuais enumeradas: desativadas; faltas u[j] garantem viabilidade inicial")
    return dict(rotas_E), dict(rotas_S), compat_por_suplente, vistos


def atualizar_compatibilidade_para_tarefas(dados, tarefas, compat_por_suplente, tarefa_ids):
    """Adiciona novas tarefas ao indice de compatibilidade por suplente."""
    if not tarefa_ids:
        return
    for tipo, conjunto in [("IE", dados["I_E"]), ("IS", dados["I_S"])]:
        for suplente in conjunto:
            lista = compat_por_suplente.setdefault((tipo, suplente), [])
            existentes = {j["tarefa_id"] for j in lista}
            for tarefa_id in tarefa_ids:
                if tarefa_id in existentes:
                    continue
                tarefa = tarefas[tarefa_id]
                if _tarefa_compativel(dados, suplente, tarefa, tipo):
                    lista.append(tarefa)


def adicionar_rotas_iniciais_por_tarefa(
    dados,
    tarefas,
    tarefa_ids,
    rotas_E,
    rotas_S,
    compat_por_suplente,
    vistos,
    config,
    contexto="inicial",
):
    """
    Para cada tarefa, cria pelo menos uma rota individual barata, quando houver
    suplente compativel ainda sem conflito na base inicial.
    """
    max_por_tarefa = int(config.get("max_alocacoes_iniciais_por_tarefa", 1))
    if max_por_tarefa <= 0:
        return 0

    ocupacao = defaultdict(set)
    for rotas in list(rotas_E.values()) + list(rotas_S.values()):
        for rota in rotas:
            if not rota.get("tarefas"):
                continue
            ocupacao[(rota["tipo"], rota["suplente"])].update(rota.get("dias", tuple()))

    novas = []
    tarefas_sem_rota = 0
    for tarefa_id in tarefa_ids:
        tarefa = tarefas[tarefa_id]
        custo_falta = float(dados["Receita"][tarefa["cargo"]]) * len(tarefa["dias"])

        candidatos_ie = []
        for suplente in dados["I_E"]:
            if any(t in ocupacao[("IE", suplente)] for t in tarefa["dias"]):
                continue
            comp_ids = {j["tarefa_id"] for j in compat_por_suplente.get(("IE", suplente), [])}
            if tarefa_id not in comp_ids:
                continue
            rota = _criar_rota(dados, "IE", suplente, (tarefa_id,), tarefas)
            if rota["assinatura"] not in vistos:
                candidatos_ie.append((rota["custo_total"], rota))

        candidatos = sorted(candidatos_ie, key=lambda x: x[0])

        if not candidatos:
            candidatos_is = []
            for suplente in dados["I_S"]:
                if any(t in ocupacao[("IS", suplente)] for t in tarefa["dias"]):
                    continue
                comp_ids = {j["tarefa_id"] for j in compat_por_suplente.get(("IS", suplente), [])}
                if tarefa_id not in comp_ids:
                    continue
                rota = _criar_rota(dados, "IS", suplente, (tarefa_id,), tarefas)
                custo_com_contratacao = rota["custo_total"] + float(dados["Ccontrat"][dados["cargo"][suplente]])
                if rota["assinatura"] not in vistos and custo_com_contratacao < custo_falta:
                    candidatos_is.append((custo_com_contratacao, rota))
            candidatos = sorted(candidatos_is, key=lambda x: x[0])

        selecionadas = candidatos[:max_por_tarefa]
        if not selecionadas:
            tarefas_sem_rota += 1
        for _custo, rota in selecionadas:
            novas.append((0.0, rota))
            ocupacao[(rota["tipo"], rota["suplente"])].update(rota.get("dias", tuple()))

    adicionadas = adicionar_rotas_ao_pool(rotas_E, rotas_S, vistos, novas)
    rotulo = "Heuristica inicial por tarefa" if contexto == "inicial" else "Heuristica para tarefas novas"
    print(
        f"  {rotulo} | "
        f"rotas={adicionadas:,} | tarefas_sem_rota={tarefas_sem_rota:,}"
    )
    return adicionadas


def adicionar_rotas_recomendadas_mini_mestre(
    dados,
    tarefas,
    tarefas_por_bloco,
    novas_chaves_blocos,
    rotas_E,
    rotas_S,
    vistos,
    duais,
    duais_disponibilidade,
    bloco_info,
):
    """Materializa no pool global as coberturas sugeridas pelo mini-mestre local."""
    novas = []
    for chave in novas_chaves_blocos:
        if chave not in bloco_info:
            continue
        bloco = bloco_info[chave]
        recomendadas = list(bloco.get("_mini_mestre_rotas", tuple()))
        if not recomendadas:
            continue
        tarefas_bloco = [
            tarefas[tarefa_id]
            for tarefa_id in tarefas_por_bloco.get(chave, [])
            if tarefa_id in tarefas
        ]
        tarefas_bloco.sort(key=lambda j: (j["ini"], j["fim"], j["tarefa_id"]))
        for rec in recomendadas:
            idx = int(rec.get("janela_idx", -1))
            if idx < 0 or idx >= len(tarefas_bloco):
                continue
            tarefa = tarefas_bloco[idx]
            tipo = rec["tipo"]
            suplente = rec["suplente"]
            if not _tarefa_compativel(dados, suplente, tarefa, tipo):
                continue
            rota = _criar_rota(dados, tipo, suplente, (tarefa["tarefa_id"],), tarefas)
            if rota["assinatura"] in vistos:
                continue
            rc = _custo_reduzido_rota(rota, duais, duais_disponibilidade)
            novas.append((rc, rota))
    return adicionar_rotas_ao_pool(rotas_E, rotas_S, vistos, novas)


def _indexar_rotas(rotas_E, rotas_S):
    rotas_por_tarefa = defaultdict(list)
    rotas_por_suplente_dia_E = defaultdict(list)
    rota_info = {}
    for tipo, rotas_dict in [("IE", rotas_E), ("IS", rotas_S)]:
        for suplente, rotas in rotas_dict.items():
            for rota in rotas:
                key = (tipo, suplente, rota["rota_id"])
                rota_info[key] = rota
                for j in rota["tarefas"]:
                    rotas_por_tarefa[j].append(key)
                if tipo == "IE":
                    for t in rota["dias"]:
                        rotas_por_suplente_dia_E[(suplente, t)].append(key)
    return rota_info, rotas_por_tarefa, rotas_por_suplente_dia_E


def _aplicar_fixacoes_e_warm_start(grupos, fixacoes, warm_start):
    for nome, variaveis in grupos.items():
        for chave, var in variaveis.items():
            if (nome, chave) in fixacoes:
                valor = float(fixacoes[(nome, chave)])
                var.lowBound = valor
                var.upBound = valor
            if (nome, chave) in warm_start:
                try:
                    var.setInitialValue(float(warm_start[(nome, chave)]))
                except Exception:
                    pass


def construir_mestre_colunas(
    dados,
    bloco_info,
    blocos_por_i,
    blocos_por_dia,
    tarefas,
    rotas_E,
    rotas_S,
    relaxado,
    fixacoes=None,
    warm_start=None,
):
    """Monta o mestre restrito, usado como LP durante pricing e como MIP no final."""
    model = pl.LpProblem("Ferias_Rotas_Suplentes_Tatico", pl.LpMinimize)
    cat_bin = "Continuous" if relaxado else "Binary"
    I_F = dados["I_A"] + dados["I_E"]
    H = max(dados["T"])
    max_blocos_ferias = int(dados.get("max_blocos_ferias_por_funcionario", 3))

    z = {
        chave: pl.LpVariable(_nome("z", *chave), lowBound=0, upBound=1, cat=cat_bin)
        for chave in bloco_info
    }
    rota_info, rotas_por_tarefa, rotas_por_suplente_dia_E = _indexar_rotas(rotas_E, rotas_S)
    yE = {
        key: pl.LpVariable(_nome("yE", key[1], key[2]), lowBound=0, upBound=1, cat=cat_bin)
        for key in rota_info
        if key[0] == "IE"
    }
    yS = {
        key: pl.LpVariable(_nome("yS", key[1], key[2]), lowBound=0, upBound=1, cat=cat_bin)
        for key in rota_info
        if key[0] == "IS"
    }
    contrataS = {
        s: pl.LpVariable(_nome("contrataS", s), lowBound=0, upBound=1, cat=cat_bin)
        for s in dados["I_S"]
    }
    mobiliza_keys = sorted({
        (key[1], projeto)
        for key, rota in rota_info.items()
        for projeto in rota.get("projetos_mobilizados_cobrados", tuple())
    })
    mobiliza = {
        chave: pl.LpVariable(_nome("mobiliza", *chave), lowBound=0, upBound=1, cat=cat_bin)
        for chave in mobiliza_keys
    }
    deficit_ferias = {
        i: pl.LpVariable(_nome("deficit_ferias", i), lowBound=0, cat="Continuous")
        for i in I_F
    }
    u = {
        j: pl.LpVariable(_nome("u", j), lowBound=0, upBound=1, cat="Continuous")
        for j in tarefas
    }
    _aplicar_fixacoes_e_warm_start(
        {"z": z, "yE": yE, "yS": yS, "contrataS": contrataS, "mobiliza": mobiliza},
        fixacoes or {},
        warm_start or {},
    )

    # Restricoes de ferias por blocos: saldo, incompatibilidade, 14 dias, programadas e fragmentacao.
    ferias_constraints = {
        "saldo": {},
        "expandido": {},
        "bloco14": {},
        "programadas": {},
        "fragmentacao": {},
    }
    for i in I_F:
        chaves_i = [(i, b["bloco_id"]) for b in blocos_por_i.get(i, [])]
        saldo_total = max(int(dados["b"].get(i, 0)), 0)
        prog_days = {t for t in dados["T"] if dados["ferias_programadas"].get((i, t), 0) == 1}
        saldo_novo = max(saldo_total - _dias_programados_total(dados, i, prog_days), 0)
        limite = int(dados["L"].get(i, H + 1))
        expr_saldo = pl.lpSum(int(bloco_info[ch]["dias_novos"]) * z[ch] for ch in chaves_i)
        if limite <= H:
            nome = _nome("saldo_novo_minimo", i)
            model += (expr_saldo + deficit_ferias[i] >= saldo_novo, nome)
        else:
            nome = _nome("saldo_novo_maximo", i)
            model += (expr_saldo <= saldo_novo, nome)
        ferias_constraints["saldo"][i] = model.constraints[nome]

        # Nao sobreposicao e nao adjacencia em forma compacta:
        # cada bloco ocupa seus dias reais e tambem o dia imediatamente posterior.
        # Assim, dois blocos que se sobrepoem ou encostam no dia seguinte disputam
        # pelo menos um mesmo "dia expandido" e nao podem ser escolhidos juntos.
        for t_expandido in range(1, H + 2):
            chaves_expandido = [
                ch for ch in chaves_i
                if int(bloco_info[ch]["ini"]) <= t_expandido <= int(bloco_info[ch]["fim"]) + 1
            ]
            if chaves_expandido:
                nome = _nome("blocos_nao_sobrepostos_nem_adjacentes", i, t_expandido)
                model += (
                    pl.lpSum(z[ch] for ch in chaves_expandido) <= 1,
                    nome,
                )
                ferias_constraints["expandido"][(i, t_expandido)] = model.constraints[nome]

        if saldo_novo >= 14 and int(dados["tem_bloco_aprovado_14"].get(i, 0)) == 0:
            longos = [ch for ch in chaves_i if int(bloco_info[ch]["duracao"]) >= 14]
            if not longos and saldo_novo > 0:
                raise SystemExit(f"Colaborador {i} precisa de bloco de 14 dias, mas nenhum candidato foi gerado.")
            if longos:
                nome = _nome("bloco_14_obrigatorio", i)
                model += (pl.lpSum(z[ch] for ch in longos) >= 1, nome)
                ferias_constraints["bloco14"][i] = model.constraints[nome]

        for t in prog_days:
            cobre_t = [ch for ch in blocos_por_dia.get((i, t), [])]
            if not cobre_t:
                raise SystemExit(f"Ferias programadas de {i} no dia {t} nao sao cobertas por nenhum bloco candidato.")
            nome = _nome("ferias_programadas", i, t)
            model += (pl.lpSum(z[ch] for ch in cobre_t) == 1, nome)
            ferias_constraints["programadas"][(i, t)] = model.constraints[nome]

        nome = _nome("limite_fragmentacao", i)
        model += (pl.lpSum(z[ch] for ch in chaves_i) <= max_blocos_ferias, nome)
        ferias_constraints["fragmentacao"][i] = model.constraints[nome]

    # Cobertura das tarefas: cada janela de ate 5 dias acompanha z do bloco de origem.
    cobertura_constraints = {}
    for tarefa_id, tarefa in tarefas.items():
        lhs = pl.lpSum(
            (yE[key] if key[0] == "IE" else yS[key])
            for key in rotas_por_tarefa.get(tarefa_id, [])
        ) + u[tarefa_id]
        nome = _nome("cobertura_tarefa", tarefa_id)
        model += (lhs == z[tarefa["bloco_chave"]], nome)
        cobertura_constraints[tarefa_id] = model.constraints[nome]

    disponibilidade_constraints = {"IE": {}, "IS": {}}

    # Disponibilidade diaria: como cada coluna de rota e uma alocacao unitária
    # tarefa-suplente, o mestre combina varias colunas desde que nao haja conflito
    # de dias. A disponibilidade vem da propria solucao relaxada/binaria do mestre.
    for h in dados["I_E"]:
        for t in dados["T"]:
            lhs_rota = pl.lpSum(yE[key] for key in rotas_por_suplente_dia_E.get((h, t), []))
            lhs_ferias = pl.lpSum(z[ch] for ch in blocos_por_dia.get((h, t), []))
            nome = _nome("IE_nao_cobre_em_ferias", h, t)
            model += (lhs_rota + lhs_ferias <= 1, nome)
            disponibilidade_constraints["IE"][(h, t)] = model.constraints[nome]

    rotas_por_suplente_dia_S = defaultdict(list)
    for key, rota in rota_info.items():
        if key[0] != "IS":
            continue
        for t in rota.get("dias", tuple()):
            rotas_por_suplente_dia_S[(key[1], t)].append(key)

    for s in dados["I_S"]:
        for t in dados["T"]:
            keys = rotas_por_suplente_dia_S.get((s, t), [])
            if keys:
                nome = _nome("capacidade_diaria_IS", s, t)
                model += (
                    pl.lpSum(yS[key] for key in keys) <= contrataS[s],
                    nome,
                )
                disponibilidade_constraints["IS"][(s, t)] = model.constraints[nome]

    for chave in mobiliza_keys:
        suplente, projeto = chave
        keys = [
            key for key, rota in rota_info.items()
            if key[1] == suplente and projeto in rota.get("projetos_mobilizados_cobrados", tuple())
        ]
        for key in keys:
            y = yE[key] if key[0] == "IE" else yS[key]
            nome = _nome("aciona_mobilizacao", suplente, projeto, key[2])
            model += (y <= mobiliza[chave], nome)

    receita_perdida = pl.lpSum(
        float(dados["Receita"][tarefas[j]["cargo"]]) * len(tarefas[j]["dias"]) * u[j]
        for j in tarefas
    )
    custo_rotas_E = pl.lpSum(rota_info[key]["custo_total"] * yE[key] for key in yE)
    custo_rotas_S = pl.lpSum(rota_info[key]["custo_total"] * yS[key] for key in yS)
    custo_contratacao = pl.lpSum(float(dados["Ccontrat"][dados["cargo"][s]]) * contrataS[s] for s in dados["I_S"])
    custo_mobilizacao = float(dados.get("Cmob", 600.0)) * pl.lpSum(mobiliza[ch] for ch in mobiliza)
    penalidade_deficit_ferias = float(dados.get("penalidade_deficit_ferias", 1_000_000.0))
    custo_deficit_ferias = penalidade_deficit_ferias * pl.lpSum(deficit_ferias[i] for i in I_F)
    model += custo_rotas_E + custo_rotas_S + custo_contratacao + custo_mobilizacao + receita_perdida + custo_deficit_ferias

    variaveis = {
        "z": z,
        "yE": yE,
        "yS": yS,
        "contrataS": contrataS,
        "mobiliza": mobiliza,
        "deficit_ferias": deficit_ferias,
        "u": u,
        "rota_info": rota_info,
        "rotas_E": rotas_E,
        "rotas_S": rotas_S,
        "rotas_por_tarefa": rotas_por_tarefa,
        "rotas_por_suplente_dia_E": rotas_por_suplente_dia_E,
        "cobertura_constraints": cobertura_constraints,
        "ferias_constraints": ferias_constraints,
        "disponibilidade_constraints": disponibilidade_constraints,
    }
    return model, variaveis


def resolver_mestre_lp(model, time_limit=None):
    kwargs = {"msg": False}
    if time_limit is not None and time_limit > 0:
        kwargs["timeLimit"] = time_limit
    return model.solve(pl.HiGHS(**kwargs))


def _custo_reduzido_rota(rota, duais, duais_disponibilidade=None):
    rc = float(rota["custo_total"]) - sum(float(duais.get(j, 0.0)) for j in rota["tarefas"])
    if duais_disponibilidade:
        tipo = rota["tipo"]
        suplente = rota["suplente"]
        for t in rota.get("dias", tuple()):
            rc -= float(duais_disponibilidade.get(tipo, {}).get((suplente, t), 0.0))
    return rc


def _ordenar_tarefas_vizinho_mais_proximo(dados, suplente, tarefas_bloco):
    """
    Ordena as tarefas do bloco por um vizinho mais proximo temporal.

    Como as tarefas possuem janelas de tempo, a busca respeita fim_j < ini_l.
    A escolha gulosa usa o menor custo de transicao como aproximacao operacional
    do caminho mais curto sobre o grafo temporal.
    """
    restantes = sorted(tarefas_bloco, key=lambda j: (j["ini"], j["fim"], j["tarefa_id"]))
    if not restantes:
        return []

    atual = min(restantes, key=lambda j: (_dist_pessoa_projeto(dados, suplente, j["projeto"]), j["ini"]))
    rota = [atual]
    restantes.remove(atual)

    while restantes:
        candidatos = [j for j in restantes if int(j["ini"]) > int(atual["fim"])]
        if not candidatos:
            break
        prox = min(
            candidatos,
            key=lambda j: (
                _custo_transicao_tarefas(dados, suplente, atual, j)[0],
                j["ini"],
                j["fim"],
            ),
        )
        rota.append(prox)
        restantes.remove(prox)
        atual = prox
    return rota


def gerar_rotas_por_blocos_ativos(
    dados,
    tarefas,
    tarefas_por_bloco,
    blocos_ativos,
    compat_por_suplente,
    duais,
    vistos,
    config,
    metodo,
    compat_ids_por_suplente=None,
    tempo_limite_s=None,
    duais_disponibilidade=None,
    max_por_bloco_override=None,
    retornar_diagnostico=False,
):
    """
    Gera ate N colunas para cada bloco de ferias ativo no LP.

    metodo='vizinho' cria as primeiras colunas por proximidade operacional.
    metodo='pricing' ranqueia as colunas pelo custo reduzido calculado com os
    duais de cobertura do mestre.
    """
    novas = []
    diagnostico = {
        "blocos_varridos": 0,
        "blocos_sem_tarefas": 0,
        "tarefas_avaliadas": 0,
        "pares_ie_possiveis": 0,
        "pares_is_possiveis": 0,
        "pares_ie_compativeis": 0,
        "pares_is_compativeis": 0,
        "duplicadas_ie": 0,
        "duplicadas_is": 0,
        "candidatas_ie": 0,
        "candidatas_is": 0,
        "baldes_ie": 0,
        "baldes_is": 0,
        "parou_por_tempo": 0,
    }
    tarefas_por_id = tarefas
    if compat_ids_por_suplente is None:
        compat_ids_por_suplente = {
            chave: {j["tarefa_id"] for j in lista}
            for chave, lista in compat_por_suplente.items()
        }
    inicio = time.time()

    for idx_bloco, bloco_chave in enumerate(blocos_ativos, start=1):
        if tempo_limite_s and time.time() - inicio > tempo_limite_s:
            diagnostico["parou_por_tempo"] = 1
            break
        diagnostico["blocos_varridos"] += 1
        ids_bloco = tarefas_por_bloco.get(bloco_chave, [])
        if not ids_bloco:
            diagnostico["blocos_sem_tarefas"] += 1
            continue
        tarefas_bloco = [tarefas_por_id[j] for j in ids_bloco]
        diagnostico["tarefas_avaliadas"] += len(tarefas_bloco)
        diagnostico["pares_ie_possiveis"] += len(dados["I_E"]) * len(tarefas_bloco)
        diagnostico["pares_is_possiveis"] += len(dados["I_S"]) * len(tarefas_bloco)
        candidatos_por_balde = {}

        for tipo, conjunto in [("IE", dados["I_E"]), ("IS", dados["I_S"])]:
            for suplente in conjunto:
                comp_ids = compat_ids_por_suplente.get((tipo, suplente), set())
                tarefas_compativeis = [j for j in tarefas_bloco if j["tarefa_id"] in comp_ids]
                if not tarefas_compativeis:
                    continue
                chave_compativeis = "pares_ie_compativeis" if tipo == "IE" else "pares_is_compativeis"
                diagnostico[chave_compativeis] += len(tarefas_compativeis)

                for tarefa in tarefas_compativeis:
                    rota = _criar_rota(dados, tipo, suplente, (tarefa["tarefa_id"],), tarefas)
                    if rota["assinatura"] in vistos:
                        chave_duplicadas = "duplicadas_ie" if tipo == "IE" else "duplicadas_is"
                        diagnostico[chave_duplicadas] += 1
                        continue
                    chave_candidatas = "candidatas_ie" if tipo == "IE" else "candidatas_is"
                    diagnostico[chave_candidatas] += 1
                    rc = _custo_reduzido_rota(rota, duais, duais_disponibilidade)
                    if tipo == "IE":
                        # IE representa um funcionario real; nao compactamos
                        # pessoas diferentes no mesmo cargo/cidade.
                        balde = (tarefa["tarefa_id"], tipo, suplente)
                    else:
                        # IS representa potenciais por cargo/cidade; dentro do
                        # grupo fica o potencial com menor custo reduzido.
                        balde = (
                            tarefa["tarefa_id"],
                            tipo,
                            dados["cargo"][suplente],
                            dados["cidade"].get(suplente, ""),
                        )
                    atual = candidatos_por_balde.get(balde)
                    candidato = (rc, rota, bloco_chave)
                    if atual is None or (rc, rota["custo_total"]) < (atual[0], atual[1]["custo_total"]):
                        candidatos_por_balde[balde] = candidato

        selecionadas_bloco = list(candidatos_por_balde.values())
        diagnostico["baldes_ie"] += sum(1 for _rc, rota, _bloco in selecionadas_bloco if rota["tipo"] == "IE")
        diagnostico["baldes_is"] += sum(1 for _rc, rota, _bloco in selecionadas_bloco if rota["tipo"] == "IS")
        novas.extend(selecionadas_bloco)

    if retornar_diagnostico:
        return novas, diagnostico
    return novas


def adicionar_rotas_ao_pool(rotas_E, rotas_S, vistos, novas):
    """Adiciona rotas diferentes das ja presentes e devolve a quantidade real."""
    adicionadas = 0
    for item in novas:
        if len(item) == 3:
            _rc, rota, _bloco = item
        else:
            _rc, rota = item
        if rota["assinatura"] in vistos:
            continue
        tipo = rota["tipo"]
        suplente = rota["suplente"]
        colecao = rotas_E if tipo == "IE" else rotas_S
        colecao.setdefault(suplente, [])
        prefixo = "RE" if tipo == "IE" else "RS"
        rota["rota_id"] = f"{prefixo}_{suplente}_{len(colecao[suplente]) + 1:06d}"
        colecao[suplente].append(rota)
        vistos.add(rota["assinatura"])
        adicionadas += 1
    return adicionadas


def gerar_colunas_alocacao_por_baldes(
    dados,
    tarefas,
    tarefas_por_bloco,
    bloco_info,
    blocos_ativos_base,
    compat_por_suplente,
    compat_ids_por_suplente,
    duais,
    vistos,
    config,
    tempo_limite_s,
    duais_disponibilidade,
):
    """
    Gera colunas de alocacao por baldes de diversidade.

    Para cada tarefa dos blocos considerados, abre baldes por:
    - IE: funcionario especifico existente;
    - IS: cargo e cidade/localidade do suplente potencial.

    Dentro de cada balde entra a melhor coluna ainda nao presente no pool.
    """
    inicio = time.time()
    max_tentativas = int(config.get("max_tentativas_expansao_pricing", 5))
    fator = float(config.get("fator_expansao_pricing", 2.0))
    max_total_iter = int(config.get("max_novas_rotas_total_por_iteracao", 1_000_000))
    max_blocos_base = len(blocos_ativos_base)
    coletadas = {}
    detalhes = []

    if max_blocos_base == 0:
        return [], detalhes

    for tentativa in range(1, max_tentativas + 1):
        if tempo_limite_s and time.time() - inicio > tempo_limite_s:
            break
        multiplicador = fator ** (tentativa - 1)
        qtd_blocos = min(max_blocos_base, max(1, int(math.ceil(len(blocos_ativos_base) * min(1.0, multiplicador)))))
        tempo_restante = None
        if tempo_limite_s:
            tempo_restante = max(0.0, tempo_limite_s - (time.time() - inicio))
            if tempo_restante <= 0:
                break

        candidatas, diagnostico = gerar_rotas_por_blocos_ativos(
            dados,
            tarefas,
            tarefas_por_bloco,
            blocos_ativos_base[:qtd_blocos],
            compat_por_suplente,
            duais,
            vistos,
            config,
            "pricing",
            compat_ids_por_suplente=compat_ids_por_suplente,
            tempo_limite_s=tempo_restante,
            duais_disponibilidade=duais_disponibilidade,
            retornar_diagnostico=True,
        )
        novas_distintas = 0
        for rc, rota, _bloco in candidatas:
            if rota["assinatura"] in vistos:
                continue
            if rota["assinatura"] not in coletadas or rc < coletadas[rota["assinatura"]][0]:
                coletadas[rota["assinatura"]] = (rc, rota)
                novas_distintas += 1

        detalhes.append({
            "tentativa": tentativa,
            "blocos_considerados": qtd_blocos,
            "candidatas": len(candidatas),
            "novas_distintas": novas_distintas,
            "acumuladas_total": len(coletadas),
            **diagnostico,
        })

        # Como cada tentativa amplia blocos considerados, paramos quando a busca
        # ja cobriu todos os blocos ativos ou quando nao surgem novas assinaturas.
        if qtd_blocos >= max_blocos_base or novas_distintas == 0:
            break

    novas = sorted(coletadas.values(), key=lambda x: x[0])
    novas = novas[:max_total_iter]
    return novas, detalhes


def _estimativa_cobertura_bloco(dados, bloco):
    """
    Estima o custo de cobertura do bloco antes de criar suas tarefas de fato.
    Usa uma aproximacao: soma o menor custo casa-projeto-casa para cada janela
    de ate 5 dias, entre suplentes compativeis.
    """
    i = bloco["colaborador"]
    projeto = dados["projeto_original"][i]
    cargo_tarefa = dados["cargo"][i]
    turno_tarefa = dados["turno_original"][i]
    melhor_por_janela = []
    for idx in range(0, len(bloco["dias"]), 5):
        janela = bloco["dias"][idx:idx + 5]
        melhor = None
        tarefa_fake = {
            "colaborador_ferias": i,
            "cargo": cargo_tarefa,
            "turno": turno_tarefa,
            "projeto": projeto,
            "cidade_projeto": dados["cidade_projeto"].get(projeto, ""),
            "dias": janela,
            "ini": min(janela),
            "fim": max(janela),
            "tarefa_id": "_fake",
        }
        for tipo, conjunto in [("IE", dados["I_E"]), ("IS", dados["I_S"])]:
            for suplente in conjunto:
                if not _tarefa_compativel(dados, suplente, tarefa_fake, tipo):
                    continue
                rota = _criar_rota(dados, tipo, suplente, [tarefa_fake], {"_fake": tarefa_fake})
                melhor = rota["custo_total"] if melhor is None else min(melhor, rota["custo_total"])
        melhor_por_janela.append(float(melhor if melhor is not None else dados["Receita"][cargo_tarefa] * len(janela)))
    return sum(melhor_por_janela)


def _duais_ferias(vars_lp):
    """Extrai os duais das restricoes de ferias do mestre LP."""
    duais = {}
    for grupo, constraints in vars_lp.get("ferias_constraints", {}).items():
        duais[grupo] = {
            chave: float(getattr(cons, "pi", None) or 0.0)
            for chave, cons in constraints.items()
        }
    return duais


def _duais_disponibilidade(vars_lp):
    duais = {"IE": {}, "IS": {}}
    for tipo, constraints in vars_lp.get("disponibilidade_constraints", {}).items():
        duais[tipo] = {
            chave: float(getattr(cons, "pi", None) or 0.0)
            for chave, cons in constraints.items()
        }
    return duais


def _custo_coluna_ferias_com_cobertura_inicial(dados, bloco, duais_disponibilidade=None, diagnostico=None):
    """
    Custo primal associado a inserir um bloco de ferias.

    Para cada tarefa de ate 5 dias induzida pelo bloco:
    1. tenta cobrir com suplente existente mais barato;
    2. se nao houver IE compativel, tenta IS quando contratar/cobrir e mais barato
       que deixar a tarefa em falta;
    3. caso contrario, usa custo de falta.

    Esse custo e o custo da coluna composta que o subproblema gera: bloco de
    ferias mais as primeiras colunas de alocacao associadas as suas tarefas.
    """
    i = bloco["colaborador"]
    projeto = dados["projeto_original"][i]
    cargo_tarefa = dados["cargo"][i]
    turno_tarefa = dados["turno_original"][i]
    custo_total = 0.0
    duais_disponibilidade = duais_disponibilidade or {"IE": {}, "IS": {}}

    for idx in range(0, len(bloco["dias"]), 5):
        janela = tuple(bloco["dias"][idx:idx + 5])
        if diagnostico is not None:
            diagnostico["ferias_janelas_cobertura"] += 1
        tarefa_fake = {
            "colaborador_ferias": i,
            "cargo": cargo_tarefa,
            "turno": turno_tarefa,
            "projeto": projeto,
            "cidade_projeto": dados["cidade_projeto"].get(projeto, ""),
            "dias": janela,
            "ini": min(janela),
            "fim": max(janela),
            "tarefa_id": "_fake",
        }
        custo_falta = float(dados["Receita"][cargo_tarefa]) * len(janela)

        melhor_ie = None
        for suplente in dados["I_E"]:
            if not _tarefa_compativel(dados, suplente, tarefa_fake, "IE"):
                continue
            rota = _criar_rota(dados, "IE", suplente, [tarefa_fake], {"_fake": tarefa_fake})
            custo_precificado = float(rota["custo_total"]) - sum(
                float(duais_disponibilidade.get("IE", {}).get((suplente, t), 0.0))
                for t in rota.get("dias", tuple())
            )
            melhor_ie = custo_precificado if melhor_ie is None else min(melhor_ie, custo_precificado)

        if melhor_ie is not None:
            escolhido = min(melhor_ie, custo_falta)
            custo_total += escolhido
            if diagnostico is not None:
                diagnostico["ferias_custo_cobertura_estimado"] += escolhido
                if escolhido >= custo_falta - 1e-9:
                    diagnostico["ferias_janelas_falta_estimadas"] += 1
            continue

        melhor_is = None
        for suplente in dados["I_S"]:
            if not _tarefa_compativel(dados, suplente, tarefa_fake, "IS"):
                continue
            rota = _criar_rota(dados, "IS", suplente, [tarefa_fake], {"_fake": tarefa_fake})
            custo_com_contratacao = rota["custo_total"] + float(dados["Ccontrat"][dados["cargo"][suplente]])
            custo_precificado = custo_com_contratacao - sum(
                float(duais_disponibilidade.get("IS", {}).get((suplente, t), 0.0))
                for t in rota.get("dias", tuple())
            )
            melhor_is = custo_precificado if melhor_is is None else min(melhor_is, custo_precificado)

        if melhor_is is not None and melhor_is < custo_falta:
            custo_total += melhor_is
            if diagnostico is not None:
                diagnostico["ferias_custo_cobertura_estimado"] += melhor_is
        else:
            custo_total += custo_falta
            if diagnostico is not None:
                diagnostico["ferias_custo_cobertura_estimado"] += custo_falta
                diagnostico["ferias_janelas_falta_estimadas"] += 1
                if melhor_is is None:
                    diagnostico["ferias_janelas_sem_cobertura"] += 1

    return custo_total


def _custo_reduzido_bloco_ferias(dados, bloco, duais_ferias, duais_disponibilidade=None, diagnostico=None):
    """Calcula o custo reduzido da coluna de ferias candidata."""
    rc = _custo_coluna_ferias_com_cobertura_inicial(
        dados, bloco, duais_disponibilidade=duais_disponibilidade, diagnostico=diagnostico
    )
    rc -= _credito_dual_bloco_ferias(dados, bloco, duais_ferias)
    return rc


def _credito_dual_bloco_ferias(dados, bloco, duais_ferias):
    """Credito dual das restricoes de ferias associado a um bloco candidato."""
    i = bloco["colaborador"]
    credito = float(duais_ferias.get("saldo", {}).get(i, 0.0)) * int(bloco["dias_novos"])
    for t in range(int(bloco["ini"]), int(bloco["fim"]) + 2):
        credito += float(duais_ferias.get("expandido", {}).get((i, t), 0.0))
    if int(bloco["duracao"]) >= 14:
        credito += float(duais_ferias.get("bloco14", {}).get(i, 0.0))
    for t in bloco["dias"]:
        credito += float(duais_ferias.get("programadas", {}).get((i, t), 0.0))
    credito += float(duais_ferias.get("fragmentacao", {}).get(i, 0.0))
    return credito


def _tarefas_fake_para_bloco(dados, bloco):
    tarefas_fake = []
    i = bloco["colaborador"]
    projeto = dados["projeto_original"][i]
    dias = list(bloco["dias"])
    for idx in range(0, len(dias), 5):
        janela = tuple(dias[idx:idx + 5])
        tarefas_fake.append({
            "tarefa_id": f"_mini_{bloco['bloco_id']}_{idx // 5}",
            "colaborador_ferias": i,
            "bloco_id": bloco["bloco_id"],
            "bloco_chave": (i, bloco["bloco_id"]),
            "janela_idx": idx // 5,
            "ini": min(janela),
            "fim": max(janela),
            "dias": janela,
            "projeto": projeto,
            "cargo": dados["cargo"][i],
            "turno": dados["turno_original"][i],
            "cidade_projeto": dados["cidade_projeto"].get(projeto, ""),
        })
    return tarefas_fake


def _avaliar_plano_ferias_mini_mestre(
    dados,
    plano,
    duais_ferias,
    duais_disponibilidade,
    config,
    diagnostico,
    tempo_limite_s=None,
):
    """
    Avalia um plano completo com um mini-mestre local de cobertura.

    A falta local usa o mesmo custo do mestre: receita perdida do cargo vezes
    quantidade de dias da tarefa. As rotas escolhidas sao apenas recomendacoes
    para entrar no pool global; o mestre continua livre para decidir.
    """
    inicio = time.time()
    relaxado = bool(config.get("mini_mestre_ferias_relaxado", True))
    cat = "Continuous" if relaxado else "Binary"
    max_cand_tarefa = int(config.get("max_candidatos_mini_mestre_por_tarefa", 40))
    tempo_solver = float(config.get("time_limit_mini_mestre_ferias_s", 0.5))
    if tempo_limite_s is not None:
        tempo_solver = min(tempo_solver, max(0.0, tempo_limite_s))
    if tempo_solver <= 0:
        diagnostico["ferias_mini_tempo_estourou"] += 1
        return None

    tarefas_fake = []
    for bloco in plano:
        tarefas_fake.extend(_tarefas_fake_para_bloco(dados, bloco))
    if not tarefas_fake:
        return None

    candidatos_por_tarefa = {}
    rota_por_key = {}
    for tarefa in tarefas_fake:
        candidatos = []
        tarefas_ref = {tarefa["tarefa_id"]: tarefa}
        for tipo, conjunto in [("IE", dados["I_E"]), ("IS", dados["I_S"])]:
            for suplente in conjunto:
                if not _tarefa_compativel(dados, suplente, tarefa, tipo):
                    continue
                if tipo == "IE" and any(
                    int(dados["ferias_programadas"].get((suplente, t), 0)) == 1
                    for t in tarefa["dias"]
                ):
                    continue
                rota = _criar_rota(dados, tipo, suplente, (tarefa["tarefa_id"],), tarefas_ref)
                custo = float(rota["custo_total"]) - sum(
                    float(duais_disponibilidade.get(tipo, {}).get((suplente, t), 0.0))
                    for t in rota.get("dias", tuple())
                )
                candidatos.append((custo, tipo, suplente, rota))
        candidatos.sort(key=lambda x: (x[0], x[3]["custo_total"], x[1], x[2]))
        if max_cand_tarefa > 0:
            candidatos = candidatos[:max_cand_tarefa]
        candidatos_por_tarefa[tarefa["tarefa_id"]] = candidatos
        diagnostico["ferias_mini_candidatos"] += len(candidatos)
        for idx, (_custo, tipo, suplente, rota) in enumerate(candidatos):
            rota_por_key[(tarefa["tarefa_id"], idx)] = (tipo, suplente, rota)

    model = pl.LpProblem("MiniMestreFeriasCobertura", pl.LpMinimize)
    x = {
        (tarefa_id, idx): pl.LpVariable(_nome("mini_x", tarefa_id, idx), lowBound=0, upBound=1, cat=cat)
        for tarefa_id, candidatos in candidatos_por_tarefa.items()
        for idx, _cand in enumerate(candidatos)
    }
    falta = {
        tarefa["tarefa_id"]: pl.LpVariable(_nome("mini_falta", tarefa["tarefa_id"]), lowBound=0, upBound=1, cat=cat)
        for tarefa in tarefas_fake
    }
    contrata_keys = sorted({
        suplente
        for (_tarefa_id, _idx), (tipo, suplente, _rota) in rota_por_key.items()
        if tipo == "IS"
    })
    contrata = {
        s: pl.LpVariable(_nome("mini_contrata", s), lowBound=0, upBound=1, cat=cat)
        for s in contrata_keys
    }
    mobiliza_keys = sorted({
        (suplente, projeto)
        for (_tarefa_id, _idx), (_tipo, suplente, rota) in rota_por_key.items()
        for projeto in rota.get("projetos_mobilizados_cobrados", tuple())
    })
    mobiliza = {
        chave: pl.LpVariable(_nome("mini_mobiliza", *chave), lowBound=0, upBound=1, cat=cat)
        for chave in mobiliza_keys
    }

    for tarefa in tarefas_fake:
        tarefa_id = tarefa["tarefa_id"]
        model += (
            pl.lpSum(x[(tarefa_id, idx)] for idx, _cand in enumerate(candidatos_por_tarefa.get(tarefa_id, [])))
            + falta[tarefa_id]
            == 1,
            _nome("mini_cobre", tarefa_id),
        )

    uso_por_suplente_dia = defaultdict(list)
    for key, (_tipo, suplente, rota) in rota_por_key.items():
        for t in rota.get("dias", tuple()):
            uso_por_suplente_dia[(suplente, t)].append(key)
    for (suplente, t), keys in uso_por_suplente_dia.items():
        model += (pl.lpSum(x[key] for key in keys) <= 1, _nome("mini_disp", suplente, t))

    for key, (tipo, suplente, rota) in rota_por_key.items():
        if tipo == "IS":
            model += (x[key] <= contrata[suplente], _nome("mini_aciona_contrata", suplente, key[0], key[1]))
        for projeto in rota.get("projetos_mobilizados_cobrados", tuple()):
            model += (x[key] <= mobiliza[(suplente, projeto)], _nome("mini_aciona_mob", suplente, projeto, key[0], key[1]))

    custo_rotas = pl.lpSum(
        candidatos_por_tarefa[tarefa_id][idx][0] * x[(tarefa_id, idx)]
        for tarefa_id, candidatos in candidatos_por_tarefa.items()
        for idx, _cand in enumerate(candidatos)
    )
    custo_falta = pl.lpSum(
        float(dados["Receita"][tarefa["cargo"]]) * len(tarefa["dias"]) * falta[tarefa["tarefa_id"]]
        for tarefa in tarefas_fake
    )
    custo_contratacao = pl.lpSum(float(dados["Ccontrat"][dados["cargo"][s]]) * contrata[s] for s in contrata)
    custo_mobilizacao = float(dados.get("Cmob", 600.0)) * pl.lpSum(mobiliza[ch] for ch in mobiliza)
    model += custo_rotas + custo_falta + custo_contratacao + custo_mobilizacao

    status = model.solve(pl.HiGHS(msg=False, timeLimit=tempo_solver))
    diagnostico["ferias_mini_avaliados"] += 1
    if pl.LpStatus[status] not in ("Optimal", "Feasible"):
        diagnostico["ferias_mini_inviaveis"] += 1
        if time.time() - inicio >= tempo_solver:
            diagnostico["ferias_mini_tempo_estourou"] += 1
        return None

    custo_local = float(pl.value(model.objective) or 0.0)
    credito_dual = sum(_credito_dual_bloco_ferias(dados, bloco, duais_ferias) for bloco in plano)
    rc_plano = custo_local - credito_dual
    rotas_escolhidas = []
    for key, (_tipo, _suplente, rota) in rota_por_key.items():
        if _valor(x[key]) <= float(config.get("tolerancia_mini_mestre_rota", 1e-5)):
            continue
        tarefa_id = rota["tarefas"][0]
        sufixo = tarefa_id.split("_mini_", 1)[-1]
        bloco_id, janela_txt = sufixo.rsplit("_", 1)
        rotas_escolhidas.append({
            "bloco_id": bloco_id,
            "janela_idx": int(janela_txt),
            "tipo": rota["tipo"],
            "suplente": rota["suplente"],
            "valor": _valor(x[key]),
        })

    faltas_local = sum(_valor(v) for v in falta.values())
    diagnostico["ferias_mini_rotas_escolhidas"] += len(rotas_escolhidas)
    diagnostico["ferias_mini_faltas"] += faltas_local
    diagnostico["ferias_mini_custo_local"] += custo_local
    atual = diagnostico.get("ferias_mini_menor_cr")
    diagnostico["ferias_mini_menor_cr"] = rc_plano if atual is None else min(float(atual), rc_plano)
    return {
        "rc": rc_plano,
        "custo_local": custo_local,
        "faltas": faltas_local,
        "rotas": rotas_escolhidas,
    }


def _faixa_inicio_ferias(dados, ini):
    data = dados.get("t_para_data", {}).get(int(ini))
    if data is not None:
        return f"{int(data.year):04d}-{int(data.month):02d}"
    return int((int(ini) - 1) // 30)


def _balde_bloco_ferias(dados, bloco):
    return (
        bloco["colaborador"],
        int(bloco["duracao"]),
        _faixa_inicio_ferias(dados, bloco["ini"]),
    )


def _plano_ferias_valido(dados, i, combo):
    T = dados["T"]
    saldo_total = max(int(dados["b"].get(i, 0)), 0)
    prog_days = {t for t in T if dados["ferias_programadas"].get((i, t), 0) == 1}
    saldo_novo = max(saldo_total - _dias_programados_total(dados, i, prog_days), 0)
    exige_14 = saldo_novo >= 14 and int(dados["tem_bloco_aprovado_14"].get(i, 0)) == 0
    combo = sorted(combo, key=lambda b: (b["ini"], b["fim"]))
    for b1, b2 in zip(combo, combo[1:]):
        if int(b2["ini"]) <= int(b1["fim"]) + 1:
            return False
    dias = set()
    dias_novos = 0
    for bloco in combo:
        if int(bloco.get("programado_fixo", 0)) == 1:
            continue
        dias_bloco = set(bloco["dias"])
        if dias & dias_bloco:
            return False
        dias |= dias_bloco
        dias_novos += int(bloco.get("dias_novos", 0))
    if dias & prog_days:
        return False
    if any((t - 1 in prog_days) or (t + 1 in prog_days) for t in dias):
        return False
    if dias_novos != saldo_novo:
        return False
    if exige_14 and not any(int(b["duracao"]) >= 14 for b in combo):
        return False
    return True


def _selecionar_blocos_por_pricing_de_planos(
    dados,
    i,
    candidatos_rc,
    duais_ferias,
    duais_disponibilidade,
    config,
    diagnostico,
    tempo_limite_s=None,
):
    max_blocos_plano = int(config.get("max_blocos_ferias_por_funcionario", 3))
    max_base = int(config.get("max_blocos_base_pricing_planos_ferias", 36))
    max_planos = int(config.get("max_planos_ferias_por_funcionario_iteracao", 36))
    if max_blocos_plano <= 1 or max_base <= 0 or max_planos <= 0:
        return []

    base = sorted(candidatos_rc, key=lambda x: x[0])[:max_base]
    planos = []
    for qtd in range(1, max_blocos_plano + 1):
        for combo in combinations(base, qtd):
            blocos_combo = [bloco for _rc, bloco in combo]
            diagnostico["ferias_planos_testados"] += 1
            if not _plano_ferias_valido(dados, i, blocos_combo):
                continue
            rc_plano = sum(float(rc) for rc, _bloco in combo)
            planos.append((rc_plano, tuple(blocos_combo)))
    planos.sort(key=lambda x: x[0])
    diagnostico["ferias_planos_viaveis"] += len(planos)
    if planos:
        menor = float(planos[0][0])
        atual = diagnostico.get("ferias_menor_cr_plano")
        diagnostico["ferias_menor_cr_plano"] = menor if atual is None else min(float(atual), menor)

    if bool(config.get("usar_mini_mestre_planos_ferias", True)) and planos:
        inicio_mini = time.time()
        max_planos_mini = int(config.get("max_planos_mini_mestre_ferias_por_funcionario", 12))
        planos_reavaliados = []
        for rc_estimado, plano in planos[:max(0, max_planos_mini)]:
            if tempo_limite_s and time.time() - inicio_mini > tempo_limite_s:
                diagnostico["ferias_mini_tempo_estourou"] += 1
                break
            restante = None
            if tempo_limite_s:
                restante = max(0.0, tempo_limite_s - (time.time() - inicio_mini))
            aval = _avaliar_plano_ferias_mini_mestre(
                dados,
                plano,
                duais_ferias,
                duais_disponibilidade or {"IE": {}, "IS": {}},
                config,
                diagnostico,
                restante,
            )
            if aval is None:
                planos_reavaliados.append((rc_estimado, plano, []))
                continue
            planos_reavaliados.append((float(aval["rc"]), plano, aval["rotas"]))
        if planos_reavaliados:
            planos_reavaliados.sort(key=lambda x: x[0])
            planos = planos_reavaliados
            menor = float(planos[0][0])
            atual = diagnostico.get("ferias_menor_cr_plano")
            diagnostico["ferias_menor_cr_plano"] = menor if atual is None else min(float(atual), menor)
        else:
            planos = [(rc, plano, []) for rc, plano in planos]
    else:
        planos = [(rc, plano, []) for rc, plano in planos]

    selecionados = []
    ids = set()
    tolerancia = float(config.get("tolerancia_custo_reduzido_ferias", 1e-6))
    planos_negativos_i = 0
    for rc_plano, plano, rotas_mini in planos:
        if rc_plano >= -tolerancia:
            continue
        diagnostico["ferias_planos_negativos"] += 1
        planos_negativos_i += 1
        rotas_por_bloco = defaultdict(list)
        for rota_mini in rotas_mini:
            rotas_por_bloco[rota_mini["bloco_id"]].append(rota_mini)
        for bloco in plano:
            if bloco["bloco_id"] in ids:
                continue
            bloco["_ultimo_rc_plano_ferias"] = float(rc_plano)
            if rotas_por_bloco.get(bloco["bloco_id"]):
                bloco["_mini_mestre_rotas"] = tuple(rotas_por_bloco[bloco["bloco_id"]])
            selecionados.append(bloco)
            ids.add(bloco["bloco_id"])
        if planos_negativos_i >= max_planos:
            break
    diagnostico["ferias_blocos_via_planos"] += len(selecionados)
    return selecionados


def precificar_blocos_ferias(
    dados,
    blocos_pool_por_i,
    blocos_por_i,
    bloco_info,
    duais_ferias,
    duais_disponibilidade,
    config,
    tempo_limite_s,
):
    """
    Subproblema de ferias por enumeracao parcimoniosa e baldes.

    Ferias programadas nao entram como escolha nova do pricing: elas ja sao regra
    do mestre. O pricing varia apenas o saldo nao programado, escolhendo o melhor
    bloco por funcionario, duracao e faixa de inicio.
    """
    inicio = time.time()
    max_por_func = int(config.get("max_novos_blocos_ferias_por_funcionario_iteracao", 9))
    max_total = int(config.get("max_novos_blocos_ferias_total_iteracao", 5000))
    usar_planos = bool(config.get("usar_pricing_planos_ferias", True))
    diagnostico = {
        "ferias_modo_pricing": "planos" if usar_planos else "baldes",
        "ferias_funcionarios_varridos": 0,
        "ferias_candidatos_fora_mestre": 0,
        "ferias_ignorados_programadas": 0,
        "ferias_avaliados": 0,
        "ferias_baldes": 0,
        "ferias_negativos": 0,
        "ferias_neutros_aceitos": 0,
        "ferias_selecionados": 0,
        "ferias_menor_cr": None,
        "ferias_parou_por_tempo": 0,
        "ferias_janelas_cobertura": 0,
        "ferias_janelas_sem_cobertura": 0,
        "ferias_janelas_falta_estimadas": 0,
        "ferias_custo_cobertura_estimado": 0.0,
        "ferias_planos_testados": 0,
        "ferias_planos_viaveis": 0,
        "ferias_planos_negativos": 0,
        "ferias_blocos_via_planos": 0,
        "ferias_menor_cr_plano": None,
        "ferias_mini_avaliados": 0,
        "ferias_mini_inviaveis": 0,
        "ferias_mini_tempo_estourou": 0,
        "ferias_mini_candidatos": 0,
        "ferias_mini_rotas_escolhidas": 0,
        "ferias_mini_faltas": 0.0,
        "ferias_mini_custo_local": 0.0,
        "ferias_mini_menor_cr": None,
    }
    if max_por_func <= 0 or max_total <= 0:
        return [], diagnostico

    candidatos_rc = []

    for i, pool in blocos_pool_por_i.items():
        if tempo_limite_s and time.time() - inicio > tempo_limite_s:
            diagnostico["ferias_parou_por_tempo"] = 1
            break
        diagnostico["ferias_funcionarios_varridos"] += 1
        ativos = {b["bloco_id"] for b in blocos_por_i.get(i, [])}
        candidatos = [b for b in pool if b["bloco_id"] not in ativos]
        diagnostico["ferias_candidatos_fora_mestre"] += len(candidatos)
        if not candidatos:
            continue

        candidatos_func = []
        melhores_por_balde = {}
        for bloco in candidatos:
            if int(bloco.get("dias_programados", 0)) > 0:
                diagnostico["ferias_ignorados_programadas"] += 1
                continue
            if int(bloco.get("dias_novos", 0)) <= 0:
                continue
            diagnostico["ferias_avaliados"] += 1
            rc = _custo_reduzido_bloco_ferias(
                dados,
                bloco,
                duais_ferias,
                duais_disponibilidade=duais_disponibilidade,
                diagnostico=diagnostico,
            )
            candidatos_func.append((rc, bloco))
            balde = _balde_bloco_ferias(dados, bloco)
            atual = melhores_por_balde.get(balde)
            if atual is None or (rc, bloco["ini"], bloco["fim"]) < (atual[0], atual[1]["ini"], atual[1]["fim"]):
                melhores_por_balde[balde] = (rc, bloco)
        diagnostico["ferias_baldes"] += len(melhores_por_balde)
        blocos_planos = (
            _selecionar_blocos_por_pricing_de_planos(
                dados,
                i,
                candidatos_func,
                duais_ferias,
                duais_disponibilidade,
                config,
                diagnostico,
                max(0.0, tempo_limite_s - (time.time() - inicio)) if tempo_limite_s else None,
            )
            if usar_planos
            else []
        )
        for bloco in blocos_planos:
            rc_bloco = float(bloco.get("_ultimo_rc_ferias", 0.0))
            for rc_cand, bloco_cand in candidatos_func:
                if bloco_cand["bloco_id"] == bloco["bloco_id"]:
                    rc_bloco = float(rc_cand)
                    break
            candidatos_rc.append((rc_bloco, bloco))
        melhores_i = list(melhores_por_balde.values())
        melhores_i.sort(key=lambda x: x[0])
        ids_planos = {b["bloco_id"] for b in blocos_planos}
        usar_complemento_baldes = (not usar_planos) or bool(config.get("usar_complemento_baldes_em_planos_ferias", True))
        if usar_complemento_baldes:
            espaco_restante = max(0, max_por_func - len(ids_planos))
            candidatos_rc.extend([(rc, b) for rc, b in melhores_i if b["bloco_id"] not in ids_planos][:espaco_restante])

    candidatos_rc.sort(key=lambda x: x[0])
    if candidatos_rc:
        diagnostico["ferias_menor_cr"] = float(candidatos_rc[0][0])
    # Se nao houver coluna negativa, ainda permitimos uma pequena entrada quando
    # o mestre esta restrito demais; por padrao, somente rc negativo entra.
    tolerancia = float(config.get("tolerancia_custo_reduzido_ferias", 1e-6))
    selecionados = [(rc, b) for rc, b in candidatos_rc if rc < -tolerancia]
    diagnostico["ferias_negativos"] = len(selecionados)
    if not selecionados and bool(config.get("permitir_bloco_ferias_neutro_se_restrito", False)):
        selecionados = candidatos_rc[:max_total]
        diagnostico["ferias_neutros_aceitos"] = len(selecionados)
    selecionados = selecionados[:max_total]
    for rc, bloco in selecionados:
        bloco["_ultimo_rc_ferias"] = float(rc)
    diagnostico["ferias_selecionados"] = len(selecionados)
    return [b for _rc, b in selecionados], diagnostico


def gerar_pool_inicial_caminho_curto(
    dados,
    tarefas,
    tarefas_por_bloco,
    bloco_info,
    rotas_E,
    rotas_S,
    compat_por_suplente,
    vistos,
    config,
):
    """
    Monta o problema inicial com colunas de cobertura por caminho curto.

    Para cada bloco de ferias candidato priorizado, gera poucas rotas distintas usando uma
    heuristica de vizinho mais proximo no grafo temporal. Isso reduz a dependencia
    das faltas u[j] no LP inicial sem inflar demais o mestre.
    """
    usar_pool = bool(config.get("usar_pool_inicial_caminho_curto", False))
    max_iniciais = int(config.get("max_colunas_iniciais_por_bloco", 0))
    if (not usar_pool) or max_iniciais <= 0:
        print("  Pool inicial por caminho curto: desativado; mestre inicia enxuto com faltas e heuristica por tarefa")
        return {
            "iteracao": 0,
            "metodo": "inicial_caminho_curto",
            "status_lp": "not_solved",
            "fo_lp": None,
            "gap_melhoria": None,
            "tempo_iteracao_s": 0.0,
            "tempo_total_s": 0.0,
            "blocos_ativos": 0,
            "novos_blocos_ferias": 0,
            "novas_tarefas": 0,
            "rotas_iniciais_tarefas_novas": 0,
            "adicionadas": 0,
            "colunas_totais": sum(len(v) for v in rotas_E.values()) + sum(len(v) for v in rotas_S.values()),
            "menor_custo_reduzido": None,
        }

    inicio = time.time()
    compat_ids_por_suplente = {
        chave: {j["tarefa_id"] for j in lista}
        for chave, lista in compat_por_suplente.items()
    }
    config_inicial = dict(config)
    config_inicial["max_colunas_por_bloco_por_iteracao"] = max_iniciais
    max_blocos_iniciais = int(config.get("max_blocos_pool_inicial", 2500))
    blocos = sorted(
        bloco_info,
        key=lambda ch: (
            -int(bloco_info[ch].get("programado_fixo", 0)),
            int(bloco_info[ch].get("ini", 0)),
            -int(bloco_info[ch].get("duracao", 0)),
            ch[0],
        ),
    )
    if max_blocos_iniciais > 0:
        blocos = blocos[:max_blocos_iniciais]
    print("\nGerando pool inicial de rotas por caminho curto")
    print(f"  Blocos candidatos considerados: {len(blocos):,}")
    print(f"  Colunas iniciais por bloco: {max_iniciais:,}")

    novas = gerar_rotas_por_blocos_ativos(
        dados,
        tarefas,
        tarefas_por_bloco,
        blocos,
        compat_por_suplente,
        {},
        vistos,
        config_inicial,
        "vizinho",
        compat_ids_por_suplente=compat_ids_por_suplente,
    )
    novas.sort(key=lambda x: (x[2], x[1]["custo_total"]))
    adicionadas = adicionar_rotas_ao_pool(rotas_E, rotas_S, vistos, novas)
    tempo = time.time() - inicio
    total_rotas = sum(len(v) for v in rotas_E.values()) + sum(len(v) for v in rotas_S.values())
    menor_custo = min((rota["custo_total"] for _rc, rota, _bloco in novas), default=None)
    print(
        f"  Pool inicial concluido | novas={adicionadas:,} | "
        f"colunas_totais={total_rotas:,} | tempo={tempo:,.1f}s"
    )
    return {
        "iteracao": 0,
        "metodo": "inicial_caminho_curto",
        "status_lp": "not_solved",
        "fo_lp": None,
        "gap_melhoria": None,
        "tempo_iteracao_s": tempo,
        "tempo_total_s": tempo,
        "blocos_ativos": len(blocos),
        "novos_blocos_ferias": 0,
        "novas_tarefas": 0,
        "rotas_iniciais_tarefas_novas": 0,
        "adicionadas": adicionadas,
        "colunas_totais": total_rotas,
        "menor_custo_reduzido": menor_custo,
    }


def pricing_rotas_suplente(dados, tipo, suplente, tarefas, compativeis, duais, config):
    """
    Pricing heuristico por programacao dinamica em grafo temporal.

    A DP usa custo de transporte + adicional noturno - duais. A mobilizacao por
    conjunto de projetos visitados e calculada depois, como aproximacao comentada
    no enunciado: a rota so e aceita se o custo reduzido final, ja com mobilizacao,
    continuar negativo.
    """
    max_por_suplente = int(config.get("max_novas_rotas_por_suplente", 3))
    max_tarefas_rota = int(config.get("max_tarefas_por_rota_pricing", 6))
    max_tarefas_pricing = int(config.get("max_tarefas_pricing_por_suplente", 1200))
    candidatos = sorted(
        compativeis,
        key=lambda j: (
            -(float(duais.get(j["tarefa_id"], 0.0)) + float(dados["Receita"][j["cargo"]]) * len(j["dias"])),
            j["ini"],
            j["fim"],
        ),
    )
    if max_tarefas_pricing > 0:
        candidatos = candidatos[:max_tarefas_pricing]
    candidatos = sorted(candidatos, key=lambda j: (j["ini"], j["fim"], -float(duais.get(j["tarefa_id"], 0.0))))
    melhores = []

    # Caminhos gulosos iniciados pelas tarefas com maior dual liquido.
    starts = sorted(
        candidatos,
        key=lambda j: float(duais.get(j["tarefa_id"], 0.0)) - float(dados["Receita"][j["cargo"]]),
        reverse=True,
    )[:80]
    for inicio in starts:
        caminho = [inicio]
        atual = inicio
        while len(caminho) < max_tarefas_rota:
            possiveis = [
                j for j in candidatos
                if int(j["ini"]) > int(atual["fim"]) and j["tarefa_id"] not in {x["tarefa_id"] for x in caminho}
            ]
            if not possiveis:
                break
            def ganho(j):
                c_trans, _ = _custo_transicao_tarefas(dados, suplente, atual, j)
                c_not = calcular_custo_noturno_rota(dados, suplente, [j])
                return float(duais.get(j["tarefa_id"], 0.0)) - c_trans - c_not
            prox = max(possiveis, key=ganho)
            if ganho(prox) <= 0:
                break
            caminho.append(prox)
            atual = prox

        rota = _criar_rota(dados, tipo, suplente, [j["tarefa_id"] for j in caminho], tarefas)
        rc = _custo_reduzido_rota(rota, duais)
        if rc < -1e-6:
            melhores.append((rc, rota))

    melhores.sort(key=lambda x: x[0])
    rotas = []
    vistos = set()
    for rc, rota in melhores:
        if rota["assinatura"] in vistos:
            continue
        vistos.add(rota["assinatura"])
        rotas.append((rc, rota))
        if len(rotas) >= max_por_suplente:
            break
    return rotas


def _fase_geracao_colunas(it, melhoria_gap, config):
    return "normal"


def _config_por_fase(config, fase):
    cfg = dict(config)
    cfg["max_planos_mini_mestre_ferias_por_funcionario"] = int(config.get("normal_planos_mini_por_func", 12))
    cfg["max_candidatos_mini_mestre_por_tarefa"] = int(config.get("normal_candidatos_mini_por_tarefa", 40))
    return cfg


def gerar_colunas(
    dados,
    bloco_info,
    blocos_por_i,
    blocos_pool_por_i,
    blocos_por_dia,
    tarefas,
    tarefas_por_bloco,
    rotas_E,
    rotas_S,
    compat_por_suplente,
    vistos,
    config,
    time_limit,
):
    max_iter = int(config.get("max_iter_colunas", 40))
    max_total_base = int(config.get("max_novas_rotas_total_por_iteracao", 1000000))
    tolerancia = float(config.get("tolerancia_custo_reduzido", 1e-5))
    inicio = time.time()
    historico = []
    fo_anterior = None
    motivo_parada = None
    iteracoes_baixa_melhoria_saturacao = 0
    iteracoes_baixa_melhoria_modo_ferias = 0
    iteracoes_baixo_ganho_marginal = 0
    iteracoes_baixa_melhoria_global = 0
    modo_pricing_ferias = str(config.get("modo_inicial_pricing_ferias", "planos")).lower()
    if modo_pricing_ferias not in ("planos", "baldes"):
        modo_pricing_ferias = "planos"
    limiar_evolucao_ferias = float(config.get("limiar_evolucao_pricing_ferias", 0.0001))
    paciencia_evolucao_ferias = int(config.get("paciencia_evolucao_pricing_ferias", 3))
    compat_ids_por_suplente = {
        chave: {j["tarefa_id"] for j in lista}
        for chave, lista in compat_por_suplente.items()
    }

    def atualizar_estagio_pricing_ferias(melhoria_gap):
        nonlocal iteracoes_baixa_melhoria_modo_ferias, modo_pricing_ferias
        contador_log = iteracoes_baixa_melhoria_modo_ferias
        acao = None
        if not bool(config.get("usar_troca_planos_baldes_por_estagnacao", False)):
            return contador_log, acao
        if melhoria_gap is None or paciencia_evolucao_ferias <= 0:
            return contador_log, acao
        if melhoria_gap < limiar_evolucao_ferias:
            iteracoes_baixa_melhoria_modo_ferias += 1
        else:
            iteracoes_baixa_melhoria_modo_ferias = 0
        contador_log = iteracoes_baixa_melhoria_modo_ferias
        if iteracoes_baixa_melhoria_modo_ferias >= paciencia_evolucao_ferias:
            if modo_pricing_ferias == "planos":
                modo_pricing_ferias = "baldes"
                iteracoes_baixa_melhoria_modo_ferias = 0
                acao = "trocar_para_baldes"
            else:
                acao = "parar_baixa_melhoria_baldes"
        return contador_log, acao

    for it in range(1, max_iter + 1):
        inicio_iter = time.time()
        tempos_etapa = {}
        cache_antes = _snapshot_cache_stats()

        t0 = time.time()
        lp, vars_lp = construir_mestre_colunas(
            dados, bloco_info, blocos_por_i, blocos_por_dia, tarefas, rotas_E, rotas_S, relaxado=True
        )
        status = resolver_mestre_lp(lp, time_limit=None)
        tempos_etapa["lp_s"] = time.time() - t0
        if pl.LpStatus[status] not in ("Optimal", "Feasible"):
            print(f"  LP mestre sem solucao util no pricing: {pl.LpStatus[status]}")
            motivo_parada = f"LP mestre sem solucao util ({pl.LpStatus[status]})"
            break
        fo_lp = float(pl.value(lp.objective) or 0.0)
        melhoria_gap = None
        if fo_anterior is not None and abs(fo_anterior) > 1e-9:
            melhoria_gap = (fo_anterior - fo_lp) / abs(fo_anterior)
        fase_pricing = _fase_geracao_colunas(it, melhoria_gap, config)
        config_iter = _config_por_fase(config, fase_pricing)
        max_total_iter = int(config_iter.get("max_novas_rotas_total_por_iteracao", max_total_base))
        tempo_pricing_iter = float(config_iter.get("time_limit_pricing_por_iteracao", time_limit or 30.0))

        duais = {}
        for tarefa_id, cons in vars_lp["cobertura_constraints"].items():
            pi = getattr(cons, "pi", None)
            duais[tarefa_id] = float(pi or 0.0)
        if not any(abs(v) > 1e-9 for v in duais.values()):
            print("  AVISO: solver nao retornou duais confiaveis; pricing heuristico sem duais efetivos.")
        duais_ferias = _duais_ferias(vars_lp)
        duais_disp = _duais_disponibilidade(vars_lp)
        inicio_pricing = time.time()
        config_ferias_iter = dict(config_iter)
        config_ferias_iter["usar_pricing_planos_ferias"] = modo_pricing_ferias == "planos"
        t0 = time.time()
        novos_blocos, diag_ferias = precificar_blocos_ferias(
            dados,
            blocos_pool_por_i,
            blocos_por_i,
            bloco_info,
            duais_ferias,
            duais_disp,
            config_ferias_iter,
            tempo_pricing_iter,
        )
        tempos_etapa["pricing_ferias_s"] = time.time() - t0
        ferias_cr_atual = diag_ferias.get("ferias_menor_cr")
        limiar_cr_ferias_enxuto = float(config.get("limiar_cr_ferias_enxuto", -1000.0))
        limite_blocos_ferias_enxuto = int(config.get("max_novos_blocos_ferias_enxuto", 150))
        enxugar_ferias = (
            ferias_cr_atual is not None
            and float(ferias_cr_atual) > limiar_cr_ferias_enxuto
            and limite_blocos_ferias_enxuto > 0
        )
        blocos_ferias_filtrados = 0
        if enxugar_ferias and len(novos_blocos) > limite_blocos_ferias_enxuto:
            antes = len(novos_blocos)
            novos_blocos = sorted(
                novos_blocos,
                key=lambda b: (
                    float(b.get("_ultimo_rc_ferias", 0.0)),
                    int(b.get("ini", 0)),
                    int(b.get("fim", 0)),
                ),
            )[:limite_blocos_ferias_enxuto]
            blocos_ferias_filtrados = antes - len(novos_blocos)
        t0 = time.time()
        novas_chaves_blocos = [
            adicionar_bloco_ativo(bloco, blocos_por_i, bloco_info, blocos_por_dia)
            for bloco in novos_blocos
        ]
        novas_chaves_blocos = [ch for ch in novas_chaves_blocos if ch is not None]
        novas_tarefas = adicionar_tarefas_para_blocos(
            dados,
            tarefas,
            tarefas_por_bloco,
            bloco_info,
            novas_chaves_blocos,
        )
        tempos_etapa["cria_tarefas_s"] = time.time() - t0
        t0 = time.time()
        atualizar_compatibilidade_para_tarefas(dados, tarefas, compat_por_suplente, novas_tarefas)
        tempos_etapa["compat_s"] = time.time() - t0
        t0 = time.time()
        rotas_tarefas_novas = adicionar_rotas_iniciais_por_tarefa(
            dados,
            tarefas,
            novas_tarefas,
            rotas_E,
            rotas_S,
            compat_por_suplente,
            vistos,
            config_iter,
            contexto="tarefas_novas",
        )
        tempos_etapa["rotas_iniciais_s"] = time.time() - t0
        t0 = time.time()
        rotas_mini_mestre = adicionar_rotas_recomendadas_mini_mestre(
            dados,
            tarefas,
            tarefas_por_bloco,
            novas_chaves_blocos,
            rotas_E,
            rotas_S,
            vistos,
            duais,
            duais_disp,
            bloco_info,
        )
        tempos_etapa["rotas_mini_s"] = time.time() - t0
        compat_ids_por_suplente = {
            chave: {j["tarefa_id"] for j in lista}
            for chave, lista in compat_por_suplente.items()
        }
        tempo_restante_pricing = max(0.0, tempo_pricing_iter - (time.time() - inicio_pricing))

        z_lp = vars_lp["z"]
        max_blocos_ativos = int(config_iter.get("max_blocos_ativos_por_iteracao", 500))
        blocos_ativos = [
            chave for chave, var in z_lp.items()
            if _valor(var) > float(config.get("tolerancia_bloco_ativo", 1e-4))
        ]
        blocos_ativos.sort(
            key=lambda ch: (
                -_valor(z_lp[ch]),
                bloco_info[ch]["ini"],
                bloco_info[ch]["fim"],
            )
        )
        if max_blocos_ativos > 0:
            blocos_ativos = blocos_ativos[:max_blocos_ativos]
        if not blocos_ativos:
            blocos_ativos = sorted(bloco_info, key=lambda ch: (ch[0], bloco_info[ch]["ini"]))[:50]

        metodo = "pricing"
        t0 = time.time()
        novas, detalhes_expansao = gerar_colunas_alocacao_por_baldes(
            dados,
            tarefas,
            tarefas_por_bloco,
            bloco_info,
            blocos_ativos,
            compat_por_suplente,
            compat_ids_por_suplente,
            duais,
            vistos,
            config_iter,
            tempo_restante_pricing,
            duais_disp,
        )
        tempos_etapa["pricing_aloc_s"] = time.time() - t0
        novas.sort(key=lambda x: x[0])
        limiar_gap_so_negativas = float(config.get("limiar_gap_colunas_so_negativas", 0.05))
        so_colunas_negativas = (
            melhoria_gap is not None
            and melhoria_gap < limiar_gap_so_negativas
        )
        colunas_complementares_filtradas = 0
        if so_colunas_negativas:
            antes = len(novas)
            novas = [(rc, rota) for rc, rota in novas if rc < -tolerancia]
            colunas_complementares_filtradas = antes - len(novas)
        novas = novas[:max_total_iter]
        diag_aloc = _somar_detalhes_pricing_alocacao(detalhes_expansao)
        cobertura_saturacao_acionada = 0
        cobertura_saturacao_freada = 0
        blocos_novos_cobertura_saturacao = 0
        colunas_cobertura_saturacao = 0
        duplicadas_total = diag_aloc["duplicadas_ie"] + diag_aloc["duplicadas_is"]
        candidatas_total = diag_aloc["candidatas_ie"] + diag_aloc["candidatas_is"]
        saturou_cobertura = (
            not novas
            and bool(novas_chaves_blocos)
            and candidatas_total == 0
            and duplicadas_total > 0
        )
        gap_min_saturacao = float(config_iter.get("freio_saturacao_gap_min", 0.0005))
        paciencia_saturacao = int(config_iter.get("freio_saturacao_paciencia", 3))
        freio_saturacao_ativo = (
            paciencia_saturacao > 0
            and iteracoes_baixa_melhoria_saturacao >= paciencia_saturacao
        )
        if saturou_cobertura and freio_saturacao_ativo:
            cobertura_saturacao_freada = 1
        if saturou_cobertura and not freio_saturacao_ativo:
            t0 = time.time()
            max_blocos_saturacao = int(config_iter.get("max_blocos_novos_cobertura_saturacao", 50))
            max_colunas_saturacao = int(config_iter.get("max_colunas_cobertura_saturacao", 1500))
            blocos_saturacao = sorted(
                novas_chaves_blocos,
                key=lambda ch: (
                    float(bloco_info[ch].get("_ultimo_rc_ferias", 0.0)),
                    bloco_info[ch]["ini"],
                    bloco_info[ch]["fim"],
                ),
            )
            if max_blocos_saturacao > 0:
                blocos_saturacao = blocos_saturacao[:max_blocos_saturacao]
            tempo_restante_saturacao = max(0.0, tempo_pricing_iter - (time.time() - inicio_pricing))
            if blocos_saturacao and tempo_restante_saturacao > 0:
                novas_sat, detalhes_sat = gerar_colunas_alocacao_por_baldes(
                    dados,
                    tarefas,
                    tarefas_por_bloco,
                    bloco_info,
                    blocos_saturacao,
                    compat_por_suplente,
                    compat_ids_por_suplente,
                    duais,
                    vistos,
                    config_iter,
                    tempo_restante_saturacao,
                    duais_disp,
                )
                novas_sat.sort(key=lambda x: x[0])
                limite_sat = max_total_iter
                if max_colunas_saturacao > 0:
                    limite_sat = min(limite_sat, max_colunas_saturacao)
                novas = novas_sat[:limite_sat]
                detalhes_expansao.extend(detalhes_sat)
                diag_sat = _somar_detalhes_pricing_alocacao(detalhes_sat)
                diag_aloc = _somar_detalhes_pricing_alocacao(detalhes_expansao)
                cobertura_saturacao_acionada = 1
                blocos_novos_cobertura_saturacao = len(blocos_saturacao)
                colunas_cobertura_saturacao = len(novas)
            tempos_etapa["saturacao_s"] = time.time() - t0
        else:
            tempos_etapa["saturacao_s"] = 0.0

        menor_rc = min((rc for rc, _rota in novas), default=0.0)
        colunas_negativas = sum(1 for rc, _rota in novas if rc < -tolerancia)
        colunas_nao_negativas = max(0, len(novas) - colunas_negativas)
        baldes_colunas = len(novas)
        baixa_melhoria_modo_ferias_log, acao_pricing_ferias = atualizar_estagio_pricing_ferias(melhoria_gap)
        ganho_marginal_coluna = None
        cache_delta = _delta_cache_stats(cache_antes)
        limiar_baixa_melhoria_global = float(config.get("limiar_parada_gap_melhoria", 0.01))
        paciencia_baixa_melhoria_global = int(config.get("paciencia_parada_gap_melhoria", 5))
        if melhoria_gap is not None:
            if melhoria_gap < limiar_baixa_melhoria_global:
                iteracoes_baixa_melhoria_global += 1
            else:
                iteracoes_baixa_melhoria_global = 0

        if not novas:
            tempo_iter = time.time() - inicio_iter
            tempo_total = time.time() - inicio
            total_rotas = sum(len(v) for v in rotas_E.values()) + sum(len(v) for v in rotas_S.values())
            hist = {
                "iteracao": it,
                "metodo": metodo,
                "fase_pricing": fase_pricing,
                "status_lp": pl.LpStatus[status],
                "fo_lp": fo_lp,
                "gap_melhoria": melhoria_gap,
                "tempo_iteracao_s": tempo_iter,
                "tempo_total_s": tempo_total,
                "blocos_ativos": len(blocos_ativos),
                "novos_blocos_ferias": len(novas_chaves_blocos),
                "blocos_ferias_filtrados": blocos_ferias_filtrados,
                "novas_tarefas": len(novas_tarefas),
                "rotas_iniciais_tarefas_novas": rotas_tarefas_novas,
                "rotas_mini_mestre_ferias": rotas_mini_mestre,
                **diag_ferias,
                "meta_colunas_alocacao": None,
                "tentativas_expansao_pricing": len(detalhes_expansao),
                "baldes_colunas": baldes_colunas,
                "colunas_negativas": colunas_negativas,
                "colunas_complementares": colunas_nao_negativas,
                "colunas_complementares_filtradas": colunas_complementares_filtradas,
                "so_colunas_negativas": int(so_colunas_negativas),
                "tarefas_shake_falta": 0,
                "tarefas_shake_dual": 0,
                "baixa_melhoria_global": iteracoes_baixa_melhoria_global,
                "baixa_melhoria_modo_ferias": baixa_melhoria_modo_ferias_log,
                "limiar_evolucao_pricing_ferias": limiar_evolucao_ferias,
                "cobertura_saturacao_acionada": cobertura_saturacao_acionada,
                "cobertura_saturacao_freada": cobertura_saturacao_freada,
                "baixa_melhoria_saturacao": iteracoes_baixa_melhoria_saturacao,
                "blocos_novos_cobertura_saturacao": blocos_novos_cobertura_saturacao,
                "colunas_cobertura_saturacao": colunas_cobertura_saturacao,
                **diag_aloc,
                "adicionadas": 0,
                "colunas_totais": total_rotas,
                "menor_custo_reduzido": menor_rc,
                "ganho_marginal_coluna": ganho_marginal_coluna,
                "baixo_ganho_marginal": iteracoes_baixo_ganho_marginal,
                **{f"tempo_{k}": v for k, v in tempos_etapa.items()},
                **{f"cache_{k}": v for k, v in cache_delta.items()},
            }
            historico.append(hist)
            print(
                f"  Iteracao {it:03d} | metodo={metodo} | FO LP={fo_lp:,.2f} | "
                f"gap_melhoria={(melhoria_gap if melhoria_gap is not None else 0.0):.4%} | "
                f"tempo_iter={tempo_iter:,.1f}s | tempo_total={tempo_total:,.1f}s | "
                f"fase={fase_pricing} | "
                f"blocos_ativos={len(blocos_ativos):,} | novos_blocos={len(novas_chaves_blocos):,} | "
                f"blocos_filtrados={blocos_ferias_filtrados:,} | "
                f"ferias_modo={diag_ferias['ferias_modo_pricing']} | "
                f"baixa_ferias={baixa_melhoria_modo_ferias_log} | "
                f"ferias_baldes={diag_ferias['ferias_baldes']:,} | "
                f"ferias_neg={diag_ferias['ferias_negativos']:,} | "
                f"ferias_CR={diag_ferias['ferias_menor_cr'] if diag_ferias['ferias_menor_cr'] is not None else 0.0:,.2f} | "
                f"prog_fixas_ignoradas={diag_ferias['ferias_ignorados_programadas']:,} | "
                f"ferias_falta_est={diag_ferias['ferias_janelas_falta_estimadas']:,} | "
                f"ferias_sem_cob={diag_ferias['ferias_janelas_sem_cobertura']:,} | "
                f"planos_neg={diag_ferias['ferias_planos_negativos']:,} | "
                f"blocos_planos={diag_ferias['ferias_blocos_via_planos']:,} | "
                f"plano_CR={diag_ferias['ferias_menor_cr_plano'] if diag_ferias['ferias_menor_cr_plano'] is not None else 0.0:,.2f} | "
                f"mini_av={diag_ferias['ferias_mini_avaliados']:,} | "
                f"mini_CR={diag_ferias['ferias_mini_menor_cr'] if diag_ferias['ferias_mini_menor_cr'] is not None else 0.0:,.2f} | "
                f"mini_rotas={rotas_mini_mestre:,} | "
                f"mini_falta={diag_ferias['ferias_mini_faltas']:,.2f} | "
                f"novas_tarefas={len(novas_tarefas):,} | novas=0 | "
                f"colunas_totais={total_rotas:,} | menor_CR={menor_rc:,.2f} | "
                f"baldes={baldes_colunas:,} | negativas={colunas_negativas:,} | "
                f"nao_negativas={colunas_nao_negativas:,} | "
                f"compl_filtradas={colunas_complementares_filtradas:,} | "
                f"baixa_gap={iteracoes_baixa_melhoria_global} | "
                f"IE cand={diag_aloc['candidatas_ie']:,}/dup={diag_aloc['duplicadas_ie']:,} | "
                f"IS cand={diag_aloc['candidatas_is']:,}/dup={diag_aloc['duplicadas_is']:,} | "
                f"tarefas_prec={diag_aloc['tarefas_avaliadas']:,} | "
                f"baldes_IE={diag_aloc['baldes_ie']:,} | baldes_IS={diag_aloc['baldes_is']:,} | "
                f"sat_cob={cobertura_saturacao_acionada} | "
                f"freio_sat={cobertura_saturacao_freada} | "
                f"baixa_sat={iteracoes_baixa_melhoria_saturacao} | "
                f"blocos_sat={blocos_novos_cobertura_saturacao:,} | "
                f"cols_sat={colunas_cobertura_saturacao:,} | "
                f"t_lp={tempos_etapa.get('lp_s', 0.0):.1f}s | "
                f"t_ferias={tempos_etapa.get('pricing_ferias_s', 0.0):.1f}s | "
                f"t_aloc={tempos_etapa.get('pricing_aloc_s', 0.0):.1f}s | "
                f"t_sat={tempos_etapa.get('saturacao_s', 0.0):.1f}s | "
                f"cache_rota={cache_delta.get('rota_hit', 0):,}/{cache_delta.get('rota_miss', 0):,} | "
                f"cache_comp={cache_delta.get('compat_hit', 0):,}/{cache_delta.get('compat_miss', 0):,} | "
                f"tempo_pricing_estourou={diag_aloc['parou_por_tempo']} | "
                f"tentativas_pricing={len(detalhes_expansao):,}"
            )
            if novas_chaves_blocos:
                if cobertura_saturacao_acionada and melhoria_gap is not None and melhoria_gap < gap_min_saturacao:
                    iteracoes_baixa_melhoria_saturacao += 1
                elif melhoria_gap is not None and melhoria_gap >= gap_min_saturacao:
                    iteracoes_baixa_melhoria_saturacao = 0
                if (
                    paciencia_baixa_melhoria_global > 0
                    and iteracoes_baixa_melhoria_global >= paciencia_baixa_melhoria_global
                ):
                    motivo_parada = (
                        f"gap_melhoria abaixo de {limiar_baixa_melhoria_global:.4%} "
                        f"por {iteracoes_baixa_melhoria_global} iteracoes"
                    )
                    print(f"  Geracao de colunas parada: {motivo_parada}.")
                    break
                if acao_pricing_ferias == "trocar_para_baldes":
                    print(
                        "  Pricing de ferias: 3 melhorias consecutivas abaixo de "
                        f"{limiar_evolucao_ferias:.4%}; proxima iteracao usara baldes."
                    )
                elif acao_pricing_ferias == "parar_baixa_melhoria_baldes":
                    motivo_parada = (
                        f"baixa melhoria por {baixa_melhoria_modo_ferias_log} iteracoes "
                        f"em modo baldes (< {limiar_evolucao_ferias:.4%})"
                    )
                    print(f"  Geracao de colunas parada: {motivo_parada}.")
                    break
                fo_anterior = fo_lp
                continue
            print("  Geracao de colunas parada: nenhuma coluna nova elegivel.")
            motivo_parada = "nenhuma coluna nova elegivel"
            break

        adicionadas_reais = adicionar_rotas_ao_pool(rotas_E, rotas_S, vistos, novas)
        if fo_anterior is not None:
            novas_decisoes = max(1, adicionadas_reais + len(novas_chaves_blocos))
            ganho_marginal_coluna = max(0.0, fo_anterior - fo_lp) / novas_decisoes

        tempo_iter = time.time() - inicio_iter
        tempo_total = time.time() - inicio
        total_rotas = sum(len(v) for v in rotas_E.values()) + sum(len(v) for v in rotas_S.values())
        hist = {
            "iteracao": it,
            "metodo": metodo,
            "fase_pricing": fase_pricing,
            "status_lp": pl.LpStatus[status],
            "fo_lp": fo_lp,
            "gap_melhoria": melhoria_gap,
            "tempo_iteracao_s": tempo_iter,
            "tempo_total_s": tempo_total,
            "blocos_ativos": len(blocos_ativos),
            "novos_blocos_ferias": len(novas_chaves_blocos),
            "blocos_ferias_filtrados": blocos_ferias_filtrados,
            "novas_tarefas": len(novas_tarefas),
            "rotas_iniciais_tarefas_novas": rotas_tarefas_novas,
            "rotas_mini_mestre_ferias": rotas_mini_mestre,
            **diag_ferias,
            "meta_colunas_alocacao": None,
            "tentativas_expansao_pricing": len(detalhes_expansao),
            "baldes_colunas": baldes_colunas,
            "colunas_negativas": colunas_negativas,
            "colunas_complementares": colunas_nao_negativas,
            "colunas_complementares_filtradas": colunas_complementares_filtradas,
            "so_colunas_negativas": int(so_colunas_negativas),
            "tarefas_shake_falta": 0,
            "tarefas_shake_dual": 0,
            "baixa_melhoria_global": iteracoes_baixa_melhoria_global,
            "baixa_melhoria_modo_ferias": baixa_melhoria_modo_ferias_log,
            "limiar_evolucao_pricing_ferias": limiar_evolucao_ferias,
            "cobertura_saturacao_acionada": cobertura_saturacao_acionada,
            "cobertura_saturacao_freada": cobertura_saturacao_freada,
            "baixa_melhoria_saturacao": iteracoes_baixa_melhoria_saturacao,
            "blocos_novos_cobertura_saturacao": blocos_novos_cobertura_saturacao,
            "colunas_cobertura_saturacao": colunas_cobertura_saturacao,
            **diag_aloc,
            "adicionadas": adicionadas_reais,
            "colunas_totais": total_rotas,
            "menor_custo_reduzido": menor_rc,
            "ganho_marginal_coluna": ganho_marginal_coluna,
            "baixo_ganho_marginal": iteracoes_baixo_ganho_marginal,
            **{f"tempo_{k}": v for k, v in tempos_etapa.items()},
            **{f"cache_{k}": v for k, v in cache_delta.items()},
        }
        historico.append(hist)
        print(
            f"  Iteracao {it:03d} | metodo={metodo} | FO LP={fo_lp:,.2f} | "
            f"gap_melhoria={(melhoria_gap if melhoria_gap is not None else 0.0):.4%} | "
            f"tempo_iter={tempo_iter:,.1f}s | tempo_total={tempo_total:,.1f}s | "
            f"fase={fase_pricing} | "
            f"blocos_ativos={len(blocos_ativos):,} | novos_blocos={len(novas_chaves_blocos):,} | "
            f"blocos_filtrados={blocos_ferias_filtrados:,} | "
            f"ferias_modo={diag_ferias['ferias_modo_pricing']} | "
            f"baixa_ferias={baixa_melhoria_modo_ferias_log} | "
            f"ferias_baldes={diag_ferias['ferias_baldes']:,} | "
            f"ferias_neg={diag_ferias['ferias_negativos']:,} | "
            f"ferias_CR={diag_ferias['ferias_menor_cr'] if diag_ferias['ferias_menor_cr'] is not None else 0.0:,.2f} | "
            f"prog_fixas_ignoradas={diag_ferias['ferias_ignorados_programadas']:,} | "
            f"ferias_falta_est={diag_ferias['ferias_janelas_falta_estimadas']:,} | "
            f"ferias_sem_cob={diag_ferias['ferias_janelas_sem_cobertura']:,} | "
            f"planos_neg={diag_ferias['ferias_planos_negativos']:,} | "
            f"blocos_planos={diag_ferias['ferias_blocos_via_planos']:,} | "
            f"plano_CR={diag_ferias['ferias_menor_cr_plano'] if diag_ferias['ferias_menor_cr_plano'] is not None else 0.0:,.2f} | "
            f"mini_av={diag_ferias['ferias_mini_avaliados']:,} | "
            f"mini_CR={diag_ferias['ferias_mini_menor_cr'] if diag_ferias['ferias_mini_menor_cr'] is not None else 0.0:,.2f} | "
            f"mini_rotas={rotas_mini_mestre:,} | "
            f"mini_falta={diag_ferias['ferias_mini_faltas']:,.2f} | "
            f"novas_tarefas={len(novas_tarefas):,} | novas={adicionadas_reais:,} | "
            f"colunas_totais={total_rotas:,} | menor_CR={menor_rc:,.2f} | "
            f"ganho_col={ganho_marginal_coluna if ganho_marginal_coluna is not None else 0.0:,.4f} | "
            f"baldes={baldes_colunas:,} | negativas={colunas_negativas:,} | "
            f"nao_negativas={colunas_nao_negativas:,} | "
            f"compl_filtradas={colunas_complementares_filtradas:,} | "
            f"baixa_gap={iteracoes_baixa_melhoria_global} | "
            f"IE cand={diag_aloc['candidatas_ie']:,}/dup={diag_aloc['duplicadas_ie']:,} | "
            f"IS cand={diag_aloc['candidatas_is']:,}/dup={diag_aloc['duplicadas_is']:,} | "
            f"tarefas_prec={diag_aloc['tarefas_avaliadas']:,} | "
            f"baldes_IE={diag_aloc['baldes_ie']:,} | baldes_IS={diag_aloc['baldes_is']:,} | "
            f"sat_cob={cobertura_saturacao_acionada} | "
            f"freio_sat={cobertura_saturacao_freada} | "
            f"baixa_sat={iteracoes_baixa_melhoria_saturacao} | "
            f"blocos_sat={blocos_novos_cobertura_saturacao:,} | "
            f"cols_sat={colunas_cobertura_saturacao:,} | "
            f"t_lp={tempos_etapa.get('lp_s', 0.0):.1f}s | "
            f"t_ferias={tempos_etapa.get('pricing_ferias_s', 0.0):.1f}s | "
            f"t_aloc={tempos_etapa.get('pricing_aloc_s', 0.0):.1f}s | "
            f"t_sat={tempos_etapa.get('saturacao_s', 0.0):.1f}s | "
            f"cache_rota={cache_delta.get('rota_hit', 0):,}/{cache_delta.get('rota_miss', 0):,} | "
            f"cache_comp={cache_delta.get('compat_hit', 0):,}/{cache_delta.get('compat_miss', 0):,} | "
            f"tempo_pricing_estourou={diag_aloc['parou_por_tempo']} | "
            f"tentativas_pricing={len(detalhes_expansao):,}"
        )
        if adicionadas_reais == 0 and not novas_chaves_blocos:
            print("  Geracao de colunas parada: candidatas encontradas ja estavam no mestre.")
            motivo_parada = "candidatas encontradas ja estavam no mestre"
            break
        limiar_ganho_col = float(config.get("limiar_ganho_marginal_coluna", 0.05))
        paciencia_ganho_col = int(config.get("paciencia_ganho_marginal_coluna", 0))
        min_iter_ganho_col = int(config.get("min_iter_ganho_marginal_coluna", 8))
        if ganho_marginal_coluna is not None and it >= min_iter_ganho_col:
            if ganho_marginal_coluna < limiar_ganho_col:
                iteracoes_baixo_ganho_marginal += 1
            else:
                iteracoes_baixo_ganho_marginal = 0
        if (
            paciencia_ganho_col > 0
            and iteracoes_baixo_ganho_marginal >= paciencia_ganho_col
        ):
            motivo_parada = (
                f"baixo ganho marginal por coluna por {iteracoes_baixo_ganho_marginal} iteracoes "
                f"(< {limiar_ganho_col:,.4f})"
            )
            print(f"  Geracao de colunas parada: {motivo_parada}.")
            break
        if cobertura_saturacao_acionada and melhoria_gap is not None and melhoria_gap < gap_min_saturacao:
            iteracoes_baixa_melhoria_saturacao += 1
        elif melhoria_gap is not None and melhoria_gap >= gap_min_saturacao:
            iteracoes_baixa_melhoria_saturacao = 0
        if (
            paciencia_baixa_melhoria_global > 0
            and iteracoes_baixa_melhoria_global >= paciencia_baixa_melhoria_global
        ):
            motivo_parada = (
                f"gap_melhoria abaixo de {limiar_baixa_melhoria_global:.4%} "
                f"por {iteracoes_baixa_melhoria_global} iteracoes"
            )
            print(f"  Geracao de colunas parada: {motivo_parada}.")
            break
        if acao_pricing_ferias == "trocar_para_baldes":
            print(
                "  Pricing de ferias: 3 melhorias consecutivas abaixo de "
                f"{limiar_evolucao_ferias:.4%}; proxima iteracao usara baldes."
            )
        elif acao_pricing_ferias == "parar_baixa_melhoria_baldes":
            motivo_parada = (
                f"baixa melhoria por {baixa_melhoria_modo_ferias_log} iteracoes "
                f"em modo baldes (< {limiar_evolucao_ferias:.4%})"
            )
            print(f"  Geracao de colunas parada: {motivo_parada}.")
            break
        fo_anterior = fo_lp
    if motivo_parada is None:
        motivo_parada = f"max_iter_colunas atingido ({max_iter})"
    print(f"  Geracao de colunas encerrada: {motivo_parada}")
    return historico


def _somar_detalhes_pricing_alocacao(detalhes_expansao):
    campos = [
        "blocos_varridos",
        "blocos_sem_tarefas",
        "tarefas_avaliadas",
        "pares_ie_possiveis",
        "pares_is_possiveis",
        "pares_ie_compativeis",
        "pares_is_compativeis",
        "duplicadas_ie",
        "duplicadas_is",
        "candidatas_ie",
        "candidatas_is",
        "baldes_ie",
        "baldes_is",
        "parou_por_tempo",
    ]
    return {
        campo: sum(int(item.get(campo, 0) or 0) for item in detalhes_expansao)
        for campo in campos
    }


def auditar_pricing_final(
    dados,
    bloco_info,
    blocos_por_i,
    blocos_pool_por_i,
    blocos_por_dia,
    tarefas,
    tarefas_por_bloco,
    rotas_E,
    rotas_S,
    compat_por_suplente,
    vistos,
    config,
):
    print("\nAuditando LP restrito e pricing final")
    lp, vars_lp = construir_mestre_colunas(
        dados, bloco_info, blocos_por_i, blocos_por_dia, tarefas, rotas_E, rotas_S, relaxado=True
    )
    status = resolver_mestre_lp(lp, time_limit=None)
    fo_lp = float(pl.value(lp.objective) or 0.0) if pl.LpStatus[status] in ("Optimal", "Feasible") else None
    auditoria = {
        "status_lp_restrito": pl.LpStatus[status],
        "fo_lp_restrito": fo_lp,
        "menor_cr_ferias_auditado": None,
        "negativas_ferias_auditadas": None,
        "menor_cr_cobertura_auditado": None,
        "negativas_cobertura_auditadas": None,
    }
    if pl.LpStatus[status] not in ("Optimal", "Feasible"):
        print(f"  LP restrito sem solucao util na auditoria: {pl.LpStatus[status]}")
        return auditoria

    duais = {
        tarefa_id: float(getattr(cons, "pi", None) or 0.0)
        for tarefa_id, cons in vars_lp["cobertura_constraints"].items()
    }
    duais_ferias = _duais_ferias(vars_lp)
    duais_disp = _duais_disponibilidade(vars_lp)
    tempo_auditoria = float(config.get("time_limit_auditoria_pricing", 30.0))
    candidatos_ferias, diag_ferias = precificar_blocos_ferias(
        dados,
        blocos_pool_por_i,
        blocos_por_i,
        bloco_info,
        duais_ferias,
        duais_disp,
        config,
        tempo_auditoria,
    )
    auditoria["menor_cr_ferias_auditado"] = diag_ferias.get("ferias_menor_cr")
    auditoria["negativas_ferias_auditadas"] = diag_ferias.get("ferias_negativos")
    auditoria["ferias_janelas_falta_estimadas_auditadas"] = diag_ferias.get("ferias_janelas_falta_estimadas")
    auditoria["ferias_janelas_sem_cobertura_auditadas"] = diag_ferias.get("ferias_janelas_sem_cobertura")

    z_lp = vars_lp["z"]
    max_blocos_ativos = int(config.get("max_blocos_ativos_por_iteracao", 500))
    blocos_ativos = [
        chave for chave, var in z_lp.items()
        if _valor(var) > float(config.get("tolerancia_bloco_ativo", 1e-4))
    ]
    blocos_ativos.sort(key=lambda ch: (-_valor(z_lp[ch]), bloco_info[ch]["ini"], bloco_info[ch]["fim"]))
    if max_blocos_ativos > 0:
        blocos_ativos = blocos_ativos[:max_blocos_ativos]
    compat_ids_por_suplente = {
        chave: {j["tarefa_id"] for j in lista}
        for chave, lista in compat_por_suplente.items()
    }
    candidatas_cob, detalhes_cob = gerar_colunas_alocacao_por_baldes(
        dados,
        tarefas,
        tarefas_por_bloco,
        bloco_info,
        blocos_ativos,
        compat_por_suplente,
        compat_ids_por_suplente,
        duais,
        vistos,
        config,
        tempo_auditoria,
        duais_disp,
    )
    tolerancia = float(config.get("tolerancia_custo_reduzido", 1e-5))
    auditoria["menor_cr_cobertura_auditado"] = min((rc for rc, _rota in candidatas_cob), default=None)
    auditoria["negativas_cobertura_auditadas"] = sum(1 for rc, _rota in candidatas_cob if rc < -tolerancia)
    diag_cob = _somar_detalhes_pricing_alocacao(detalhes_cob)
    auditoria["candidatas_cobertura_auditadas"] = len(candidatas_cob)
    auditoria["duplicadas_cobertura_auditadas"] = diag_cob["duplicadas_ie"] + diag_cob["duplicadas_is"]

    print(f"  FO LP restrito: {fo_lp:,.2f}")
    print(
        "  Auditoria ferias | "
        f"menor_CR={auditoria['menor_cr_ferias_auditado'] if auditoria['menor_cr_ferias_auditado'] is not None else 0.0:,.2f} | "
        f"negativas={auditoria['negativas_ferias_auditadas']:,} | "
        f"janelas_falta_estimadas={auditoria['ferias_janelas_falta_estimadas_auditadas']:,}"
    )
    print(
        "  Auditoria cobertura | "
        f"menor_CR={auditoria['menor_cr_cobertura_auditado'] if auditoria['menor_cr_cobertura_auditado'] is not None else 0.0:,.2f} | "
        f"negativas={auditoria['negativas_cobertura_auditadas']:,} | "
        f"candidatas={auditoria['candidatas_cobertura_auditadas']:,} | "
        f"duplicadas={auditoria['duplicadas_cobertura_auditadas']:,}"
    )
    return auditoria


def _valores_solucao(grupos):
    valores = {}
    for nome, variaveis in grupos.items():
        for chave, var in variaveis.items():
            val = getattr(var, "varValue", None)
            if val is not None:
                valores[(nome, chave)] = float(val)
    return valores


def _fixacoes_por_confianca(valores_lp, config):
    if not bool(config.get("usar_fixacao_confianca", True)):
        return {}
    lim_1 = float(config.get("fixar_confianca_um", 0.995))
    fixar_zero = bool(config.get("fixar_confianca_zero", False))
    lim_0 = float(config.get("fixar_confianca_zero_limiar", 1e-6))
    fix = {}
    for (nome, chave), val in valores_lp.items():
        if nome not in {"z", "yE", "yS"}:
            continue
        if val >= lim_1:
            fix[(nome, chave)] = 1.0
        elif fixar_zero and val <= lim_0:
            fix[(nome, chave)] = 0.0
    return fix


def _montar_pool_enxuto_por_lp(dados, bloco_info, blocos_por_i, blocos_por_dia, tarefas, rotas_E, rotas_S, vars_lp, config):
    eps_z = float(config.get("eps_z_mip_enxuto", 1e-4))
    eps_y = float(config.get("eps_y_mip_enxuto", 1e-5))
    top_blocos = int(config.get("top_blocos_por_func_mip_enxuto", 12))
    top_rotas = int(config.get("top_rotas_por_tarefa_mip_enxuto", 12))
    z_lp = vars_lp["z"]
    rota_info = vars_lp["rota_info"]

    keep_blocos = {
        ch for ch, var in z_lp.items()
        if _valor(var) > eps_z or int(bloco_info[ch].get("programado_fixo", 0)) == 1
    }
    for i, blocos in blocos_por_i.items():
        candidatos = [(i, b["bloco_id"]) for b in blocos if (i, b["bloco_id"]) in bloco_info]
        candidatos.sort(
            key=lambda ch: (
                -_valor(z_lp[ch]) if ch in z_lp else 0.0,
                float(bloco_info[ch].get("_ultimo_rc_ferias", 0.0)),
                bloco_info[ch]["ini"],
            )
        )
        keep_blocos.update(candidatos[:top_blocos])
        prog_days = {t for t in dados["T"] if dados["ferias_programadas"].get((i, t), 0) == 1}
        saldo_novo = max(int(dados["b"].get(i, 0)) - _dias_programados_total(dados, i, prog_days), 0)
        if saldo_novo >= 14 and int(dados["tem_bloco_aprovado_14"].get(i, 0)) == 0:
            longos = [ch for ch in candidatos if int(bloco_info[ch]["duracao"]) >= 14]
            if longos:
                keep_blocos.add(min(longos, key=lambda ch: (bloco_info[ch]["ini"], bloco_info[ch]["fim"])))

    bloco_info_red = {ch: bloco_info[ch] for ch in keep_blocos if ch in bloco_info}
    blocos_por_i_red = defaultdict(list)
    blocos_por_dia_red = defaultdict(list)
    for ch, bloco in bloco_info_red.items():
        blocos_por_i_red[ch[0]].append(bloco)
        for t in bloco["dias"]:
            blocos_por_dia_red[(ch[0], t)].append(ch)

    tarefas_red = {
        tid: tarefa for tid, tarefa in tarefas.items()
        if tarefa["bloco_chave"] in bloco_info_red
    }
    tarefas_por_bloco_red = defaultdict(list)
    for tid, tarefa in tarefas_red.items():
        tarefas_por_bloco_red[tarefa["bloco_chave"]].append(tid)

    y_vals = {}
    for nome in ("yE", "yS"):
        for key, var in vars_lp.get(nome, {}).items():
            y_vals[key] = _valor(var)
    keep_rotas = {
        key for key, rota in rota_info.items()
        if (not rota.get("tarefas")) or (
            all(t in tarefas_red for t in rota.get("tarefas", tuple())) and y_vals.get(key, 0.0) > eps_y
        )
    }
    rotas_por_tarefa = defaultdict(list)
    for key, rota in rota_info.items():
        if not all(t in tarefas_red for t in rota.get("tarefas", tuple())):
            continue
        for tid in rota.get("tarefas", tuple()):
            rotas_por_tarefa[tid].append((rota["custo_total"], key))
    for tid, candidatos in rotas_por_tarefa.items():
        candidatos.sort(key=lambda x: x[0])
        keep_rotas.update(key for _custo, key in candidatos[:top_rotas])

    rotas_E_red = defaultdict(list)
    rotas_S_red = defaultdict(list)
    for key in keep_rotas:
        if key not in rota_info:
            continue
        rota = dict(rota_info[key])
        if key[0] == "IE":
            rotas_E_red[key[1]].append(rota)
        else:
            rotas_S_red[key[1]].append(rota)
    return dict(blocos_por_i_red), bloco_info_red, dict(blocos_por_dia_red), tarefas_red, dict(tarefas_por_bloco_red), dict(rotas_E_red), dict(rotas_S_red)


def resolver_mip_final(dados, bloco_info, blocos_por_i, blocos_por_dia, tarefas, rotas_E, rotas_S, time_limit, gap, config=None):
    config = dict(config or {})
    warm_start = {}
    warm_start_obj = None
    fixacoes_full = {}
    if bool(config.get("resolver_mip_enxuto_antes", True)):
        print("\nPreparando MIP enxuto por LP restrito")
        lp, vars_lp = construir_mestre_colunas(
            dados, bloco_info, blocos_por_i, blocos_por_dia, tarefas, rotas_E, rotas_S, relaxado=True
        )
        st_lp = resolver_mestre_lp(lp, time_limit=None)
        if pl.LpStatus[st_lp] in ("Optimal", "Feasible"):
            valores_lp = _valores_solucao({
                "z": vars_lp["z"],
                "yE": vars_lp["yE"],
                "yS": vars_lp["yS"],
                "contrataS": vars_lp["contrataS"],
                "mobiliza": vars_lp["mobiliza"],
            })
            fixacoes = _fixacoes_por_confianca(valores_lp, config)
            if bool(config.get("usar_fixacao_confianca_no_mip_completo", False)):
                fixacoes_full = fixacoes
            red = _montar_pool_enxuto_por_lp(
                dados, bloco_info, blocos_por_i, blocos_por_dia, tarefas, rotas_E, rotas_S, vars_lp, config
            )
            b_i_r, b_info_r, b_dia_r, tarefas_r, _tpb_r, rE_r, rS_r = red
            total_rotas_r = sum(len(v) for v in rE_r.values()) + sum(len(v) for v in rS_r.values())
            print(
                f"  MIP enxuto | z={len(b_info_r):,} | tarefas={len(tarefas_r):,} | "
                f"rotas={total_rotas_r:,} | fixacoes={len(fixacoes):,}"
            )
            mip_r, vars_r = construir_mestre_colunas(
                dados, b_info_r, b_i_r, b_dia_r, tarefas_r, rE_r, rS_r,
                relaxado=False, fixacoes=fixacoes, warm_start=valores_lp
            )
            tempo_r = float(config.get("time_limit_mip_enxuto", 600.0))
            st_r = mip_r.solve(pl.HiGHS(msg=True, timeLimit=tempo_r, gapRel=gap, options={"presolve": "on", "parallel": "on", "threads": 8}))
            print(f"  MIP enxuto encerrado com status: {pl.LpStatus[st_r]}")
            if pl.LpStatus[st_r] in ("Optimal", "Feasible"):
                warm_start_obj = float(pl.value(mip_r.objective) or 0.0)
                warm_start = _valores_solucao({
                    "z": vars_r["z"],
                    "yE": vars_r["yE"],
                    "yS": vars_r["yS"],
                    "contrataS": vars_r["contrataS"],
                    "mobiliza": vars_r["mobiliza"],
                })

    model, variaveis_mestre = construir_mestre_colunas(
        dados, bloco_info, blocos_por_i, blocos_por_dia, tarefas, rotas_E, rotas_S,
        relaxado=False, fixacoes=fixacoes_full, warm_start=warm_start
    )
    total_rotas = sum(len(v) for v in rotas_E.values()) + sum(len(v) for v in rotas_S.values())
    print("\nResolvendo MIP final com as colunas geradas")
    print(f"  Variaveis de blocos z: {len(bloco_info):,}")
    print(f"  Tarefas de cobertura u: {len(tarefas):,}")
    print(f"  Rotas disponiveis yE/yS: {total_rotas:,}")
    if warm_start:
        print(f"  Warm start recebido do MIP enxuto: {len(warm_start):,} valor(es)")
        if warm_start_obj is not None:
            print(f"  FO do warm start enxuto: {warm_start_obj:,.2f}")
    if fixacoes_full:
        print(f"  Fixacoes por confianca no MIP completo: {len(fixacoes_full):,}")
    print("  Limite de tempo do MIP final: sem limite; corte global feito por numero de iteracoes")
    solver_kwargs = {
        "msg": True,
        "gapRel": gap,
        "options": {"presolve": "on", "parallel": "on", "threads": 8},
    }
    solver_cls = _HiGHSComWarmStart if warm_start else pl.HiGHS
    status = model.solve(solver_cls(**solver_kwargs))
    if warm_start:
        print(
            "  Warm start enviado ao HiGHS: "
            f"{getattr(model, '_highs_warm_start_count', 0):,} variaveis | "
            f"status={getattr(model, '_highs_warm_start_status', 'indisponivel')}"
        )
    print(f"  MIP final encerrado com status: {pl.LpStatus[status]}")
    return model, status, variaveis_mestre


def construir_modelo_tatico(dados: dict[str, Any], config: dict[str, Any] | None = None, time_limit=None, gap=0.01):
    _limpar_caches_modelo()
    config = dict(config or {})
    dados["max_blocos_ferias_por_funcionario"] = int(config.get("max_blocos_ferias_por_funcionario", 3))

    print("\nConstruindo modelo tatico por geracao de colunas de rotas")
    print("  Transporte: literal por rota, sem amortizacao por pessoa-dia")
    print("  Mobilizacao: R$ 600 por suplente-projeto nao mobilizado, no maximo uma vez no ano")
    print("  Suplentes existentes e potenciais tratados nominalmente")

    blocos_por_i, bloco_info, blocos_por_dia, blocos_pool_por_i = gerar_blocos_ferias_mestre(dados, config)
    tarefas, tarefas_por_bloco = gerar_tarefas_cobertura_5_dias(dados, bloco_info)
    rotas_E, rotas_S, compat_por_suplente, vistos = gerar_rotas_iniciais(dados, tarefas, config)
    rotas_iniciais_tarefas = adicionar_rotas_iniciais_por_tarefa(
        dados,
        tarefas,
        list(tarefas),
        rotas_E,
        rotas_S,
        compat_por_suplente,
        vistos,
        config,
    )
    print(f"  Rotas iniciais por tarefa adicionadas: {rotas_iniciais_tarefas:,}")
    historico_inicial = gerar_pool_inicial_caminho_curto(
        dados,
        tarefas,
        tarefas_por_bloco,
        bloco_info,
        rotas_E,
        rotas_S,
        compat_por_suplente,
        vistos,
        config,
    )
    historico_colunas = gerar_colunas(
        dados, bloco_info, blocos_por_i, blocos_pool_por_i, blocos_por_dia, tarefas,
        tarefas_por_bloco, rotas_E, rotas_S, compat_por_suplente, vistos, config, time_limit
    )
    historico_colunas = [historico_inicial] + historico_colunas
    auditoria_pricing = auditar_pricing_final(
        dados,
        bloco_info,
        blocos_por_i,
        blocos_pool_por_i,
        blocos_por_dia,
        tarefas,
        tarefas_por_bloco,
        rotas_E,
        rotas_S,
        compat_por_suplente,
        vistos,
        config,
    )
    model, status, variaveis_mestre = resolver_mip_final(
        dados, bloco_info, blocos_por_i, blocos_por_dia, tarefas, rotas_E, rotas_S, time_limit, gap, config
    )
    fo_mip = float(pl.value(model.objective) or 0.0) if pl.LpStatus[status] in ("Optimal", "Feasible") else None
    fo_lp = auditoria_pricing.get("fo_lp_restrito")
    if fo_mip is not None and fo_lp is not None and abs(fo_mip) > 1e-9:
        auditoria_pricing["gap_mip_vs_lp_restrito"] = (fo_mip - fo_lp) / abs(fo_mip)
        print(
            "  Baseline operacional | "
            f"FO_LP={fo_lp:,.2f} | FO_MIP={fo_mip:,.2f} | "
            f"gap={(auditoria_pricing['gap_mip_vs_lp_restrito']):.4%}"
        )

    variaveis_mestre.update({
        "blocos_por_i": blocos_por_i,
        "bloco_info": bloco_info,
        "blocos_por_dia": blocos_por_dia,
        "tarefas": tarefas,
        "tarefas_por_bloco": tarefas_por_bloco,
        "historico_colunas": historico_colunas,
        "auditoria_pricing": auditoria_pricing,
        "transporte_granularidade": "rota_suplente_literal",
        "formulacao": "blocos_z_e_rotas_y",
        # Aliases vazios para evitar dependencias antigas acidentais.
        "lambda": {},
        "v": {},
        "xE": {},
        "xS": {},
        "nS": {},
        "s": variaveis_mestre["u"],
        "S_KEYS": [],
        "X_E_KEYS": [],
        "X_S_KEYS": [],
        "G_S": [],
    })
    return model, status, variaveis_mestre


def resolver_modelo_tatico(dados, time_limit=None, gap=0.05, config: dict[str, Any] | None = None):
    model, status, variaveis = construir_modelo_tatico(dados, config=config, time_limit=time_limit, gap=gap)
    print("\nStatus:", pl.LpStatus[status])
    print("Custo decisorio tatico:", pl.value(model.objective))

    rota_info = variaveis.get("rota_info", {})
    yE = variaveis.get("yE", {})
    yS = variaveis.get("yS", {})
    contrataS = variaveis.get("contrataS", {})
    mobiliza = variaveis.get("mobiliza", {})
    u = variaveis.get("u", {})
    tarefas = variaveis.get("tarefas", {})
    rotas_usadas = [
        rota_info[key] for key, var in {**yE, **yS}.items()
        if _valor(var) > 0.5 and rota_info.get(key, {}).get("tarefas")
    ]
    print(f"Potenciais contratados: {sum(1 for v in contrataS.values() if _valor(v) > 0.5):,}")
    print(f"Custo total de transporte: R$ {sum(r['custo_transporte'] for r in rotas_usadas):,.2f}")
    print(f"Custo total de mobilizacao: R$ {float(dados.get('Cmob', 600.0)) * sum(1 for v in mobiliza.values() if _valor(v) > 0.5):,.2f}")
    print(
        "Custo total de falta: R$ "
        f"{sum(float(dados['Receita'][tarefas[j]['cargo']]) * len(tarefas[j]['dias']) * _valor(var) for j, var in u.items()):,.2f}"
    )
    return model, status, variaveis
