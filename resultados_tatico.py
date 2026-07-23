from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import pandas as pd
import pulp as pl


def _valor(var):
    val = var.varValue
    return 0.0 if val is None else float(val)


def _fmt_data(dados, t):
    return dados["t_para_data"][int(t)].strftime("%d/%m/%Y")


def _fmt_periodo(dados, ini, fim):
    return f"{_fmt_data(dados, ini)} a {_fmt_data(dados, fim)}"


def _reais(v):
    try:
        s = f"{abs(float(v)):,.0f}".replace(",", ".")
    except (TypeError, ValueError):
        return v
    return f"-R$ {s}" if float(v) < 0 else f"R$ {s}"


def _linha_rota_transporte(dados, tipo, suplente, rota):
    tarefas = rota.get("tarefas", tuple())
    periodo = ""
    if rota.get("dias"):
        periodo = _fmt_periodo(dados, min(rota["dias"]), max(rota["dias"]))
    pernas = []
    for perna in rota.get("pernas_transporte", tuple()):
        custo_fmt = _reais(perna.get("custo", 0))
        pernas.append(
            f"{perna.get('origem', '')} -> {perna.get('destino', '')} "
            f"({perna.get('regra', '')}; {perna.get('distancia_km', 0):.1f} km; "
            f"{custo_fmt})"
        )
    return {
        "tipo": tipo,
        "suplente": suplente,
        "rota_id": rota["rota_id"],
        "origem": dados["cidade"].get(suplente, ""),
        "sequencia_projetos": " -> ".join(str(p) for p in rota.get("projetos", tuple())),
        "periodo_rota": periodo,
        "qtd_tarefas": len(tarefas),
        "custo_transporte": rota.get("custo_transporte", 0.0),
        "custo_mobilizacao": 0.0,
        "custo_noturno": rota.get("custo_noturno", 0.0),
        "custo_total_rota": rota.get("custo_total", 0.0),
        "projetos_mobilizados_cobrados": "; ".join(rota.get("projetos_mobilizados_cobrados", tuple())),
        "pernas_transporte": " | ".join(pernas),
    }


def extrair_resultados_tatico(dados, model, status, variaveis):
    I_A = dados["I_A"]
    I_E = dados["I_E"]
    I_S = dados["I_S"]
    R = dados["R"]
    cargo = dados["cargo"]
    cidade = dados["cidade"]
    turno_original = dados["turno_original"]
    Receita = dados["Receita"]
    Csal = dados["Csal"]
    Ccontrat = dados["Ccontrat"]
    d = dados["d"]

    z = variaveis.get("z", {})
    yE = variaveis.get("yE", {})
    yS = variaveis.get("yS", {})
    contrataS = variaveis.get("contrataS", {})
    mobiliza = variaveis.get("mobiliza", {})
    u = variaveis.get("u", variaveis.get("s", {}))
    bloco_info = variaveis.get("bloco_info", {})
    blocos_por_i = variaveis.get("blocos_por_i", {})
    tarefas = variaveis.get("tarefas", {})
    rota_info = variaveis.get("rota_info", {})
    historico_colunas = variaveis.get("historico_colunas", [])

    demanda_por_funcao = defaultdict(float)
    falta_por_funcao = defaultdict(float)
    coberta_por_funcao = defaultdict(float)
    receita_por_funcao = defaultdict(float)

    # A demanda economica continua sendo a demanda diaria original da operacao.
    for (p, r, t, k), qtd in d.items():
        demanda_por_funcao[r] += qtd

    # As faltas agora existem por tarefa de cobertura, nao por pessoa-dia agregado.
    linhas_faltas = []
    for tarefa_id, var in u.items():
        falta = _valor(var)
        if falta <= 1e-6 or tarefa_id not in tarefas:
            continue
        tarefa = tarefas[tarefa_id]
        dias = len(tarefa["dias"])
        valor = Receita[tarefa["cargo"]] * dias * falta
        falta_por_funcao[tarefa["cargo"]] += dias * falta
        linhas_faltas.append({
            "colaborador": tarefa["colaborador_ferias"],
            "projeto": tarefa["projeto"],
            "cargo": tarefa["cargo"],
            "data_inicio": _fmt_data(dados, tarefa["ini"]),
            "data_fim": _fmt_data(dados, tarefa["fim"]),
            "dias": dias,
            "turno": tarefa["turno"],
            "falta": falta,
            "receita_nao_gerada": valor,
        })

    linhas_alocacoes = []
    linhas_transporte = []
    custo_transporte_por_funcao = defaultdict(float)
    custo_noturno_por_funcao = defaultdict(float)
    tarefas_cobertas = set()
    nome_por_matricula = dados.get("nome_por_matricula", {})
    coberturas_por_colaborador = defaultdict(list)

    def processar_rota(tipo, key, var):
        if _valor(var) <= 0.5:
            return
        rota = rota_info.get(key)
        if not rota or not rota.get("tarefas"):
            return
        suplente = key[1]
        nome_suplente = nome_por_matricula.get(suplente, "")
        if not nome_suplente and tipo == "IS":
            nome_suplente = f"Suplente potencial ({suplente})"
        linhas_transporte.append(_linha_rota_transporte(dados, tipo, suplente, rota))
        for tarefa_id in rota["tarefas"]:
            tarefa = tarefas[tarefa_id]
            dias = len(tarefa["dias"])
            tarefas_cobertas.add(tarefa_id)
            coberta_por_funcao[tarefa["cargo"]] += dias
            receita_por_funcao[tarefa["cargo"]] += Receita[tarefa["cargo"]] * dias
            custo_transporte_por_funcao[tarefa["cargo"]] += rota["custo_transporte"] / max(1, len(rota["tarefas"]))
            custo_noturno_por_funcao[cargo[suplente]] += rota["custo_noturno"] / max(1, len(rota["tarefas"]))
            linhas_alocacoes.append({
                "tipo": tipo,
                "suplente": suplente,
                "nome_suplente": nome_suplente,
                "rota_id": rota["rota_id"],
                "colaborador_substituido": tarefa["colaborador_ferias"],
                "projeto": tarefa["projeto"],
                "cargo_demanda": tarefa["cargo"],
                "data_inicio": _fmt_data(dados, tarefa["ini"]),
                "data_fim": _fmt_data(dados, tarefa["fim"]),
                "turno": tarefa["turno"],
                "quantidade_dias": dias,
            })
            coberturas_por_colaborador[tarefa["colaborador_ferias"]].append({
                "matricula": suplente,
                "nome": nome_suplente,
                "ini": int(tarefa["ini"]),
                "fim": int(tarefa["fim"]),
            })

    for key, var in yE.items():
        processar_rota("IE", key, var)
    for key, var in yS.items():
        processar_rota("IS", key, var)

    custo_sal_existente_por_funcao = defaultdict(float)
    for i in I_A + I_E:
        custo_sal_existente_por_funcao[cargo[i]] += Csal[cargo[i]] * 22 * 12

    contratacoes = []
    custo_contratacao_por_funcao = defaultdict(float)
    for s, var in contrataS.items():
        if _valor(var) <= 0.5:
            continue
        custo = Ccontrat[cargo[s]]
        custo_contratacao_por_funcao[cargo[s]] += custo
        contratacoes.append({
            "suplente": s,
            "cargo": cargo[s],
            "cidade": cidade[s],
            "turno_original": turno_original.get(s, ""),
            "flex": dados["flex"].get(s, 1),
            "quantidade_contratada": 1,
            "custo_total": custo,
        })

    # Ferias escolhidas por colaborador, usando z diretamente.
    set_IA = set(I_A)
    escolhidos_por_i = defaultdict(list)
    for chave, var in z.items():
        if _valor(var) > 0.5:
            escolhidos_por_i[chave[0]].append(bloco_info[chave])

    max_blocos = max((len(v) for v in escolhidos_por_i.values()), default=0)
    linhas_ferias = []
    violacoes_bloco14 = []
    for i in I_A + I_E:
        blocos = sorted(escolhidos_por_i.get(i, []), key=lambda b: (b["ini"], b["fim"]))
        total_dias = sum(int(b["duracao"]) for b in blocos)
        dias_programados = sum(int(b["dias_programados"]) for b in blocos)
        dias_novos = sum(int(b["dias_novos"]) for b in blocos)
        if total_dias >= 14 and blocos and max(int(b["duracao"]) for b in blocos) < 14:
            violacoes_bloco14.append(nome_por_matricula.get(i, str(i)))

        # Consolida tarefas consecutivas cobertas pela mesma pessoa. A aba Ferias
        # passa a mostrar quem cobre e o intervalo quando a cobertura e parcial.
        coberturas = sorted(
            coberturas_por_colaborador.get(i, []),
            key=lambda c: (c["ini"], c["fim"], str(c["matricula"])),
        )
        coberturas_consolidadas = []
        for cobertura in coberturas:
            if (
                coberturas_consolidadas
                and coberturas_consolidadas[-1]["matricula"] == cobertura["matricula"]
                and cobertura["ini"] <= coberturas_consolidadas[-1]["fim"] + 1
            ):
                coberturas_consolidadas[-1]["fim"] = max(
                    coberturas_consolidadas[-1]["fim"], cobertura["fim"]
                )
            else:
                coberturas_consolidadas.append(dict(cobertura))

        # Agrupa todos os periodos da mesma pessoa em uma unica descricao:
        # Nome, matricula - periodo 1 | periodo 2
        coberturas_por_pessoa = {}
        for cobertura in coberturas_consolidadas:
            matricula = str(cobertura["matricula"])
            grupo = coberturas_por_pessoa.setdefault(matricula, {
                "nome": cobertura["nome"] or matricula,
                "periodos": [],
            })
            grupo["periodos"].append(
                _fmt_periodo(dados, cobertura["ini"], cobertura["fim"])
            )
        descricoes_cobertura = [
            f"{grupo['nome']}, {matricula} - " + " | ".join(grupo["periodos"])
            for matricula, grupo in coberturas_por_pessoa.items()
        ]
        quem_cobre_periodos = "\n".join(descricoes_cobertura)
        if blocos and not coberturas_consolidadas:
            quem_cobre_periodos = "Sem cobertura"

        linha = {
            "Matrícula": i,
            "Nome": nome_por_matricula.get(i, ""),
            "Cargo": cargo.get(i, ""),
            "Tipo": "Fixo" if i in set_IA else "Suplente",
            "Cidade": cidade.get(i, ""),
            "Turno original": turno_original.get(i, ""),
            "Quantidade de blocos": len(blocos),
            "Quantidade total de férias": total_dias,
            "Dias já programados": dias_programados,
            "Dias novos": dias_novos,
        }
        for idx in range(max_blocos):
            bloco = blocos[idx] if idx < len(blocos) else None
            linha[f"Início bloco {idx + 1}"] = _fmt_data(dados, bloco["ini"]) if bloco else ""
            linha[f"Fim bloco {idx + 1}"] = _fmt_data(dados, bloco["fim"]) if bloco else ""
        linha["Quem cobre e períodos"] = quem_cobre_periodos
        linhas_ferias.append(linha)

    cols_ferias = [
        "Matrícula", "Nome", "Cargo", "Tipo", "Cidade", "Turno original",
        "Quantidade de blocos", "Quantidade total de férias",
        "Dias já programados", "Dias novos",
    ]
    for idx in range(max_blocos):
        cols_ferias += [f"Início bloco {idx + 1}", f"Fim bloco {idx + 1}"]
    cols_ferias.append("Quem cobre e períodos")

    receita_total = sum(receita_por_funcao.values())
    receita_potencial = sum(Receita[r] * qtd for (_p, r, _t, _k), qtd in d.items())
    receita_perdida = sum(float(row["receita_nao_gerada"]) for row in linhas_faltas)
    custo_existentes = sum(custo_sal_existente_por_funcao.values())
    custo_contratacao = sum(custo_contratacao_por_funcao.values())
    custo_noturno = sum(row["custo_noturno"] for row in linhas_transporte)
    custo_transporte = sum(row["custo_transporte"] for row in linhas_transporte)
    Cmob = float(dados.get("Cmob", 600.0))
    mobilizacoes_acionadas = [ch for ch, var in mobiliza.items() if _valor(var) > 0.5]
    custo_mobilizacao = Cmob * len(mobilizacoes_acionadas)
    receita_gerada_total = receita_potencial - receita_perdida
    lucro_calculado = (
        receita_gerada_total
        - custo_existentes
        - custo_contratacao
        - custo_noturno
        - custo_transporte
        - custo_mobilizacao
    )
    fo_valor = pl.value(model.objective)

    linhas_demanda = []
    for r in R:
        linhas_demanda.append({
            "cargo": r,
            "demanda": demanda_por_funcao[r],
            "coberta": coberta_por_funcao[r],
            "nao_atendida": falta_por_funcao[r],
            "receita_coberta": receita_por_funcao[r],
            "salarios_IA_IE": custo_sal_existente_por_funcao[r],
            "contratacao_IS": custo_contratacao_por_funcao[r],
            "adicional_noturno": custo_noturno_por_funcao[r],
            "transporte_mudanca": custo_transporte_por_funcao[r],
        })

    linhas_planos_diag = []
    for i in I_A + I_E:
        blocos = blocos_por_i.get(i, [])
        linhas_planos_diag.append({
            "matricula": i,
            "tipo": "IA" if i in set_IA else "IE",
            "cargo": cargo[i],
            "cidade": cidade[i],
            "qtd_blocos_candidatos": len(blocos),
            "qtd_blocos_escolhidos": len(escolhidos_por_i.get(i, [])),
            "min_dias_novos_bloco": min((b["dias_novos"] for b in blocos), default=None),
            "max_dias_novos_bloco": max((b["dias_novos"] for b in blocos), default=None),
        })
    for item in historico_colunas:
        linhas_planos_diag.append({
            "matricula": f"pricing_iter_{item['iteracao']}",
            "tipo": "pricing",
            "cargo": "",
            "cidade": "",
            "qtd_blocos_candidatos": "",
            "qtd_blocos_escolhidos": item["adicionadas"],
            "min_dias_novos_bloco": "",
            "max_dias_novos_bloco": item["menor_custo_reduzido"],
        })

    linhas_evolucao = []
    for item in historico_colunas:
        linhas_evolucao.append({
            "iteracao": item.get("iteracao"),
            "metodo": item.get("metodo"),
            "fase_pricing": item.get("fase_pricing"),
            "status_lp": item.get("status_lp"),
            "fo_lp": item.get("fo_lp"),
            "gap_melhoria": item.get("gap_melhoria"),
            "tempo_iteracao_s": item.get("tempo_iteracao_s"),
            "tempo_total_s": item.get("tempo_total_s"),
            "blocos_ativos": item.get("blocos_ativos"),
            "novos_blocos_ferias": item.get("novos_blocos_ferias"),
            "blocos_ferias_filtrados": item.get("blocos_ferias_filtrados"),
            "pricing_ferias_modo": item.get("ferias_modo_pricing"),
            "pricing_ferias_funcionarios_varridos": item.get("ferias_funcionarios_varridos"),
            "pricing_ferias_candidatos_fora_mestre": item.get("ferias_candidatos_fora_mestre"),
            "pricing_ferias_ignorados_programadas": item.get("ferias_ignorados_programadas"),
            "pricing_ferias_avaliados": item.get("ferias_avaliados"),
            "pricing_ferias_baldes": item.get("ferias_baldes"),
            "pricing_ferias_negativos": item.get("ferias_negativos"),
            "pricing_ferias_neutros_aceitos": item.get("ferias_neutros_aceitos"),
            "pricing_ferias_selecionados": item.get("ferias_selecionados"),
            "pricing_ferias_menor_cr": item.get("ferias_menor_cr"),
            "pricing_ferias_parou_por_tempo": item.get("ferias_parou_por_tempo"),
            "pricing_ferias_janelas_cobertura": item.get("ferias_janelas_cobertura"),
            "pricing_ferias_janelas_sem_cobertura": item.get("ferias_janelas_sem_cobertura"),
            "pricing_ferias_janelas_falta_estimadas": item.get("ferias_janelas_falta_estimadas"),
            "pricing_ferias_custo_cobertura_estimado": item.get("ferias_custo_cobertura_estimado"),
            "pricing_ferias_planos_testados": item.get("ferias_planos_testados"),
            "pricing_ferias_planos_viaveis": item.get("ferias_planos_viaveis"),
            "pricing_ferias_planos_negativos": item.get("ferias_planos_negativos"),
            "pricing_ferias_blocos_via_planos": item.get("ferias_blocos_via_planos"),
            "pricing_ferias_menor_cr_plano": item.get("ferias_menor_cr_plano"),
            "pricing_ferias_mini_avaliados": item.get("ferias_mini_avaliados"),
            "pricing_ferias_mini_inviaveis": item.get("ferias_mini_inviaveis"),
            "pricing_ferias_mini_tempo_estourou": item.get("ferias_mini_tempo_estourou"),
            "pricing_ferias_mini_candidatos": item.get("ferias_mini_candidatos"),
            "pricing_ferias_mini_rotas_escolhidas": item.get("ferias_mini_rotas_escolhidas"),
            "pricing_ferias_mini_faltas": item.get("ferias_mini_faltas"),
            "pricing_ferias_mini_custo_local": item.get("ferias_mini_custo_local"),
            "pricing_ferias_mini_menor_cr": item.get("ferias_mini_menor_cr"),
            "novas_tarefas": item.get("novas_tarefas"),
            "rotas_iniciais_tarefas_novas": item.get("rotas_iniciais_tarefas_novas"),
            "rotas_mini_mestre_ferias": item.get("rotas_mini_mestre_ferias"),
            "meta_colunas_alocacao": item.get("meta_colunas_alocacao"),
            "tentativas_expansao_pricing": item.get("tentativas_expansao_pricing"),
            "baldes_colunas": item.get("baldes_colunas"),
            "colunas_negativas": item.get("colunas_negativas"),
            "colunas_complementares": item.get("colunas_complementares"),
            "colunas_complementares_filtradas": item.get("colunas_complementares_filtradas"),
            "so_colunas_negativas": item.get("so_colunas_negativas"),
            "tarefas_shake_falta": item.get("tarefas_shake_falta"),
            "tarefas_shake_dual": item.get("tarefas_shake_dual"),
            "baixa_melhoria_global": item.get("baixa_melhoria_global"),
            "baixa_melhoria_modo_ferias": item.get("baixa_melhoria_modo_ferias"),
            "limiar_evolucao_pricing_ferias": item.get("limiar_evolucao_pricing_ferias"),
            "cobertura_saturacao_acionada": item.get("cobertura_saturacao_acionada"),
            "cobertura_saturacao_freada": item.get("cobertura_saturacao_freada"),
            "baixa_melhoria_saturacao": item.get("baixa_melhoria_saturacao"),
            "blocos_novos_cobertura_saturacao": item.get("blocos_novos_cobertura_saturacao"),
            "colunas_cobertura_saturacao": item.get("colunas_cobertura_saturacao"),
            "pricing_blocos_varridos": item.get("blocos_varridos"),
            "pricing_blocos_sem_tarefas": item.get("blocos_sem_tarefas"),
            "pricing_tarefas_avaliadas": item.get("tarefas_avaliadas"),
            "pricing_pares_ie_possiveis": item.get("pares_ie_possiveis"),
            "pricing_pares_is_possiveis": item.get("pares_is_possiveis"),
            "pricing_pares_ie_compativeis": item.get("pares_ie_compativeis"),
            "pricing_pares_is_compativeis": item.get("pares_is_compativeis"),
            "pricing_duplicadas_ie": item.get("duplicadas_ie"),
            "pricing_duplicadas_is": item.get("duplicadas_is"),
            "pricing_candidatas_ie": item.get("candidatas_ie"),
            "pricing_candidatas_is": item.get("candidatas_is"),
            "pricing_baldes_ie": item.get("baldes_ie"),
            "pricing_baldes_is": item.get("baldes_is"),
            "pricing_parou_por_tempo": item.get("parou_por_tempo"),
            "colunas_adicionadas": item.get("adicionadas"),
            "colunas_totais": item.get("colunas_totais"),
            "menor_custo_reduzido": item.get("menor_custo_reduzido"),
            "ganho_marginal_coluna": item.get("ganho_marginal_coluna"),
            "baixo_ganho_marginal": item.get("baixo_ganho_marginal"),
            "tempo_lp_s": item.get("tempo_lp_s"),
            "tempo_pricing_ferias_s": item.get("tempo_pricing_ferias_s"),
            "tempo_cria_tarefas_s": item.get("tempo_cria_tarefas_s"),
            "tempo_compat_s": item.get("tempo_compat_s"),
            "tempo_rotas_iniciais_s": item.get("tempo_rotas_iniciais_s"),
            "tempo_rotas_mini_s": item.get("tempo_rotas_mini_s"),
            "tempo_pricing_aloc_s": item.get("tempo_pricing_aloc_s"),
            "tempo_saturacao_s": item.get("tempo_saturacao_s"),
            "cache_rota_hit": item.get("cache_rota_hit"),
            "cache_rota_miss": item.get("cache_rota_miss"),
            "cache_compat_hit": item.get("cache_compat_hit"),
            "cache_compat_miss": item.get("cache_compat_miss"),
        })

    resumo_itens = [
        ("Custo decisório", fo_valor, "money"),
        ("Receita potencial total", receita_potencial, "money"),
        ("Receita não gerada", -receita_perdida, "money"),
        ("Receita dos postos cobertos", receita_total, "money"),
        ("Contratação IS", -custo_contratacao, "money"),
        ("Adicional noturno", -custo_noturno, "money"),
        ("Transporte por rota", -custo_transporte, "money"),
        ("Mobilização", -custo_mobilizacao, "money"),
        ("Mobilizações cobradas", len(mobilizacoes_acionadas), "number"),
        ("Receita total gerada", receita_gerada_total, "money"),
    ]
    resumo = []
    for indicador, valor, tipo in resumo_itens:
        # Mantem os valores numericos para que o Excel aplique formato monetario
        # real, em vez de gravar textos que apenas parecem valores.
        resumo.append({"indicador": indicador, "valor": valor})

    return {
        "resumo": pd.DataFrame(resumo),
        "demanda": pd.DataFrame(linhas_demanda),
        "contratacoes": pd.DataFrame(contratacoes),
        "ferias": pd.DataFrame(linhas_ferias, columns=cols_ferias),
        "faltas": pd.DataFrame(linhas_faltas),
        "alocacoes": pd.DataFrame(linhas_alocacoes),
        "transporte": pd.DataFrame(linhas_transporte),
        "lucro_calculado": lucro_calculado,
        "custo_total_valor": fo_valor,
        "violacoes_bloco14": violacoes_bloco14,
    }


def imprimir_resultados_terminal_tatico(dados, model, status, variaveis, gerar_excel=True):
    print("\n================ RESULTADOS TATICO ================")
    print(f"Status final: {pl.LpStatus[status]}")
    print(f"Custo total das decisoes (FO): R$ {pl.value(model.objective):,.2f}")

    if pl.LpStatus[status] not in ("Optimal", "Feasible"):
        print("Modelo sem solucao viavel/otima para extrair resultados.")
        return

    resultados = extrair_resultados_tatico(dados, model, status, variaveis)
    print(f"Lucro bruto tatico calculado: R$ {resultados['lucro_calculado']:,.2f}")

    if gerar_excel:
        caminho = Path("Resultados_Tatico.xlsx")
        with pd.ExcelWriter(caminho, engine="openpyxl") as writer:
            resultados["resumo"].to_excel(writer, sheet_name="Resumo", index=False)
            resultados["demanda"].to_excel(writer, sheet_name="Demanda", index=False)
            resultados["contratacoes"].to_excel(writer, sheet_name="Contratacoes", index=False)
            resultados["ferias"].to_excel(writer, sheet_name="Ferias", index=False)
            resultados["faltas"].to_excel(writer, sheet_name="Faltas", index=False)
            resultados["alocacoes"].to_excel(writer, sheet_name="Alocacoes", index=False)
            resultados["transporte"].to_excel(writer, sheet_name="Transporte", index=False)
        print(f"\nArquivo gerado: {caminho.resolve()}")
