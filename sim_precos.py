# sim_precos.py

import io
import os
import re
import math
import uuid
import pickle
from pathlib import Path
from itertools import combinations

import streamlit as st
import pandas as pd
import numpy as np
import altair as alt

# ============================================================
# TABELA DE CUSTOS MÉDIOS
# ============================================================

TABELA_CUSTOS_MEDIA = {
    "CIUSAT": 610.9087613,
    "CIDSAT": 724.4449053,
    "CIUSAT LEC": 916.35,
    "CIDSAT LEC": 1086.6649053,
    "DSS": 2079.130163,
    "CDSS": 2785.001179,
    "GPS": 170.9717246,
    "ADNP": 1105.388247,
    "CADSAT": 1649.771379,
    "LL/LP": 266.2399137,
    "LP": 133.2399137,
    "LL": 133.2399137,
    "ADN": 1099.704441,
    "RET": 4161.791201,
    "MES": 77.78586387,
    "PCT": 465.4403074,
    "CAUSAT": 1042.606098,
    "CPN": 131.2679224,
    "DS": 353.1909116,
    "CIUSAT/GD": 1664.102261,
    "PHL": 374.7956849,
    "BE": 1152.120654,
    "PCV": 338.8997038,
    "CIU": 472.914893,
    "W": 33.7700523,
    "PN": 1158.870974,
    "CIDSAT/GD": 1715.875962,
    "IV": 175.4319856,
    "CID": 729.8759364,
    "MO": 266.5886782,
    "CBR5": 164.5969771,
    "BEP": 1447.142842,
    "UUsat": 480.3904131,
    "DRX": 3611.295256,
    "CPM": 155.0708784,
    "BH": 115.6460048,
    "SCS": 278.2101145,
    "CPI": 151.9327061,
    "PCC": 208.3116049,
    "AEQ": 354.1394795,
    "CAU": 451.1963474,
    "CBR3": 117.7057317,
    "CCIUSAT": 1857.391861,
    "CK0": 3824.504312,
    "FRX": 2124.180592,
    "GPA": 121.6809806,
    "HILF": 208.1029814,
    "RCD": 850.3410435,
    "RCP": 344.498126,
    "RCU": 761.0379884,
    "RDAN": 323.0401776,
    "RDAS": 403.800222,
    "RDP": 403.800222,
    "RTN": 1506.969894,
    "UU": 442.7906458,
    "ADICIONAL": 588.00737,
    "CAMPO": 3365.11735,
}


TABELA_CUSTOS_MEDIA_NORMALIZADA = {
    str(k).strip().upper(): float(v)
    for k, v in TABELA_CUSTOS_MEDIA.items()
}


def obter_custo_unitario(sigla, tabela_custos_global):
    """
    Busca custo unitário de forma robusta, mas preserva o valor exato quando
    a sigla bate diretamente com a tabela.
    """
    sigla_str = str(sigla).strip()

    if sigla_str in tabela_custos_global:
        return float(tabela_custos_global[sigla_str])

    sigla_upper = sigla_str.upper()

    tabela_normalizada = {
        str(k).strip().upper(): float(v)
        for k, v in tabela_custos_global.items()
    }

    if sigla_upper in tabela_normalizada:
        return float(tabela_normalizada[sigla_upper])

    if sigla_upper in TABELA_CUSTOS_MEDIA_NORMALIZADA:
        return float(TABELA_CUSTOS_MEDIA_NORMALIZADA[sigla_upper])

    return 0.0


# ============================================================
# FUNÇÕES AUXILIARES GERAIS
# ============================================================

def limpar_moeda_br(valor):
    if pd.isna(valor):
        return np.nan
    if isinstance(valor, (int, float)):
        return float(valor)
    v = str(valor).replace("R$", "").replace(" ", "").strip()
    if v == "" or v == "-":
        return np.nan
    if "," in v:
        v = v.replace(".", "").replace(",", ".")
    try:
        return float(v)
    except Exception:
        return np.nan

def normalizar_nome_coluna(texto):
    texto = str(texto).strip().upper()
    texto = texto.replace("/", "_")
    texto = texto.replace("-", "_")
    texto = texto.replace(" ", "_")
    texto = texto.replace(".", "")
    texto = texto.replace("(", "")
    texto = texto.replace(")", "")
    return texto

def padronizar_categorica(valor):
    if pd.isna(valor):
        return "OUTROS"
    valor = str(valor).strip()
    if valor == "":
        return "OUTROS"
    return valor

def limpar_prazo(valor):
    """
    Converte prazos como '30 dias', '45', '2 meses' para dias.
    Retorna NaN quando o prazo não é válido.
    """
    if pd.isna(valor):
        return np.nan
    if isinstance(valor, (int, float)):
        if float(valor) <= 0:
            return np.nan
        return int(math.ceil(float(valor)))
    v = str(valor).strip().lower()
    if v in ["", " ", "-", "não definido", "nao definido", "contrato por demanda", "nan", "none"]:
        return np.nan
    if "ano" in v:
        return np.nan
    numeros = [float(n) for n in re.findall(r"\d+(?:\.\d+)?", v)]
    if not numeros:
        return np.nan
    numero = max(numeros)
    if "mes" in v or "mês" in v:
        if numero == 12:
            return np.nan
        return int(math.ceil(numero * 30))
    return int(math.ceil(numero))

def criar_mapa_prazos(ensaios):
    mapa = {}
    for e in ensaios:
        sigla = str(e.get("Sigla", "")).strip()
        qtd = float(e.get("Quantidade", 0))
        prazo = e.get("Prazo")
        mapa[(sigla, qtd)] = prazo
    return mapa

def transformar_com_preprocessador(preprocessor_fit, X_transform, nomes_features):
    Xt = preprocessor_fit.transform(X_transform)
    Xt = pd.DataFrame(
        Xt,
        columns=nomes_features,
        index=X_transform.index,
    )
    for c in Xt.columns:
        Xt[c] = pd.to_numeric(Xt[c], errors="coerce").fillna(0)
    return Xt.reindex(columns=nomes_features, fill_value=0)


# ============================================================
# FUNÇÃO PRINCIPAL CACHEADA - CARREGAMENTO DO MODELO FIXO
# ============================================================

def _caminhos_possiveis_artefato():
    """
    Procura o artefato em caminhos comuns do projeto.
    Preferência: artefatos_modelo.pkl na raiz.
    Também aceita artefatos_modelo/modelo_precos_bundle.pkl se você optar por pasta.
    """
    base_dir = Path(__file__).resolve().parent

    return [
        base_dir / "artefatos_modelo.pkl",
        base_dir / "artefatos_modelo" / "artefatos_modelo.pkl",
        base_dir / "artefatos_modelo" / "modelo_precos_bundle.pkl",

        # Compatibilidade com o nome digitado anteriormente.
        # Mantenho como fallback para não quebrar, mas o correto é .pkl.
        base_dir / "artefatos_modelo.ppkl",
    ]


def _encontrar_artefato_modelo():
    for caminho in _caminhos_possiveis_artefato():
        if caminho.exists():
            return caminho

    caminhos_txt = "\n".join([str(c) for c in _caminhos_possiveis_artefato()])

    raise FileNotFoundError(
        "Arquivo de artefatos do modelo não encontrado. "
        "Gere o arquivo com treinar_modelo.py e envie para o GitHub junto com o app.\n\n"
        "Caminhos procurados:\n"
        f"{caminhos_txt}"
    )


def _validar_chaves_artefato(dados_ml):
    chaves_obrigatorias = [
        "modelo_heuristica",
        "threshold_heuristica",
        "tabela_precos_global",
        "siglas",
        "lista_clientes_bd",
        "lista_setores_bd",
        "lista_minerais_bd",
        "preprocessador_output_final",
        "features_output_final",
        "colunas_categoricas_output_final",
        "colunas_numericas_output_final",
    ]

    faltantes = [k for k in chaves_obrigatorias if k not in dados_ml]

    if faltantes:
        raise KeyError(
            "O arquivo artefatos_modelo.pkl não contém todas as chaves obrigatórias. "
            "Gere novamente o pkl com a versão atualizada do treinar_modelo.py. "
            f"Chaves faltantes: {faltantes}"
        )


def _executar_sanity_check(dados_ml):
    """
    Confere se o modelo carregado no Streamlit reproduz a probabilidade salva
    durante o treino local. Se não reproduzir, é sinal de diferença de ambiente
    ou incompatibilidade de versões.
    """
    if "sanity_raw" in dados_ml and "sanity_prob" in dados_ml:
        sanity_raw = dados_ml["sanity_raw"]
        preproc = dados_ml["preprocessador_output_final"]
        features = dados_ml["features_output_final"]
        modelo = dados_ml["modelo_heuristica"]

        sanity_transform = preproc.transform(sanity_raw)

        sanity_df = pd.DataFrame(
            sanity_transform,
            columns=features,
            index=sanity_raw.index,
        )

        for col in sanity_df.columns:
            sanity_df[col] = pd.to_numeric(sanity_df[col], errors="coerce").fillna(0)

        sanity_df = sanity_df.reindex(columns=features, fill_value=0)

        prob_atual = float(modelo.predict_proba(sanity_df)[0, 1])
        prob_treino = float(dados_ml["sanity_prob"])
        diff = abs(prob_atual - prob_treino)

        return {
            "ok": diff <= 1e-10,
            "prob_treino": prob_treino,
            "prob_atual": prob_atual,
            "diff": diff,
        }

    if "sanity_enc" in dados_ml and "sanity_prob" in dados_ml:
        modelo = dados_ml["modelo_heuristica"]
        sanity_enc = dados_ml["sanity_enc"]
        prob_atual = float(modelo.predict_proba(sanity_enc)[0, 1])
        prob_treino = float(dados_ml["sanity_prob"])
        diff = abs(prob_atual - prob_treino)

        return {
            "ok": diff <= 1e-10,
            "prob_treino": prob_treino,
            "prob_atual": prob_atual,
            "diff": diff,
        }

    return None


@st.cache_resource(show_spinner="Carregando modelo XGBoost pré-treinado...")
def inicializar_modelo_e_dados():
    """
    Esta versão NÃO baixa planilha e NÃO treina modelo dentro do Streamlit.
    Ela apenas carrega o artefato gerado localmente por treinar_modelo.py.
    """
    caminho_artefato = _encontrar_artefato_modelo()

    with open(caminho_artefato, "rb") as f:
        dados_ml = pickle.load(f)

    _validar_chaves_artefato(dados_ml)

    sanity = _executar_sanity_check(dados_ml)

    if sanity is not None and not sanity["ok"]:
        st.warning(
            "⚠️ O modelo carregado não reproduziu exatamente o sanity check salvo no treino. "
            "Isso indica diferença de ambiente/versões ou artefato incompatível.\n\n"
            f"Prob treino: {sanity['prob_treino']:.15f}\n"
            f"Prob atual: {sanity['prob_atual']:.15f}\n"
            f"Diff: {sanity['diff']:.15f}"
        )

    dados_ml["caminho_artefato_modelo"] = str(caminho_artefato)
    dados_ml["sanity_check_streamlit"] = sanity

    return dados_ml


# =========================================================================
# FUNÇÕES DE PROBABILIDADE E MONTAGEM DA NOVA PROPOSTA
# =========================================================================

def montar_linha_modelo(
    novo_negocio_info,
    ensaios_propostos,
    tabela_precos_global,
    preprocessador_output_final,
    features_output_final,
    colunas_categoricas_output_final,
    colunas_numericas_output_final,
    siglas,
):
    linhas_calculadas = []
    mapa_prazos = novo_negocio_info.get("mapa_prazos", {})

    for ensaio in ensaios_propostos:
        sigla = str(ensaio.get("Sigla", "")).strip()
        qtd = float(ensaio.get("Quantidade", 0))

        if sigla == "" or qtd <= 0:
            raise ValueError("Cada ensaio precisa ter Sigla e Quantidade válidas.")

        preco_ref = tabela_precos_global.get(sigla, np.nan)

        preco_unit_proposto = ensaio.get("Preco_Unitario", preco_ref)
        preco_unit_proposto = limpar_moeda_br(preco_unit_proposto)

        if pd.isna(preco_unit_proposto):
            preco_unit_proposto = preco_ref

        if pd.isna(preco_unit_proposto):
            preco_unit_proposto = 0.0

        preco_unit_proposto = float(preco_unit_proposto)
        valor_total_item = preco_unit_proposto * qtd

        if pd.notna(preco_ref) and preco_ref > 0:
            preco_relativo = preco_unit_proposto / preco_ref
        else:
            preco_relativo = 1.0

        preco_relativo = float(np.clip(preco_relativo, 0.1, 5.0))

        prazo_bruto = ensaio.get("Prazo")

        if pd.isna(prazo_bruto) or str(prazo_bruto).strip() == "":
            prazo_bruto = mapa_prazos.get((sigla, qtd))

        prazo_item = limpar_prazo(prazo_bruto)

        if pd.isna(prazo_item):
            raise ValueError(
                f"O ensaio '{sigla}' (Quantidade: {qtd}) está sem prazo válido. "
                f"Informe um número de dias positivo."
            )

        linhas_calculadas.append({
            "Sigla": sigla,
            "Quantidade": qtd,
            "Preco_Unitario": preco_unit_proposto,
            "Valor_Total_Item": valor_total_item,
            "Preco_Relativo": preco_relativo,
            "Prazo": int(prazo_item),
        })

    df_prop = pd.DataFrame(linhas_calculadas)

    if df_prop.empty:
        raise ValueError("Nenhum ensaio foi informado.")

    valor_total_contrato = df_prop["Valor_Total_Item"].sum()
    qtd_total = df_prop["Quantidade"].sum()
    n_siglas = df_prop["Sigla"].nunique()

    p_rel = df_prop["Preco_Relativo"]

    preco_relativo_medio = p_rel.mean()
    preco_relativo_mediano = p_rel.median()
    preco_relativo_min = p_rel.min()
    preco_relativo_max = p_rel.max()
    preco_relativo_std = p_rel.std(ddof=0) if len(df_prop) > 1 else 0.0
    amplitude_preco_relativo = preco_relativo_max - preco_relativo_min

    idx_maior = df_prop["Valor_Total_Item"].idxmax()
    maior_item = str(df_prop.loc[idx_maior, "Sigla"]).strip()
    valor_maior_item = df_prop.loc[idx_maior, "Valor_Total_Item"]

    participacao_maior_item = (
        valor_maior_item / valor_total_contrato
        if valor_total_contrato > 0
        else 0.0
    )

    prazo_medio = int(math.ceil(df_prop["Prazo"].mean()))
    prazo_max = int(math.ceil(df_prop["Prazo"].max()))
    prazo_min = int(math.ceil(df_prop["Prazo"].min()))

    linha = {
        "cliente": padronizar_categorica(novo_negocio_info.get("cliente")),
        "mes": padronizar_categorica(novo_negocio_info.get("mes")),
        "setor": padronizar_categorica(novo_negocio_info.get("setor")),
        "mineral": padronizar_categorica(novo_negocio_info.get("mineral")),
        "prazo_medio": prazo_medio,
        "prazo_max": prazo_max,
        "prazo_min": prazo_min,
        "valor_total_contrato": valor_total_contrato,
        "qtd_total": qtd_total,
        "n_siglas": n_siglas,
        "preco_relativo_medio": preco_relativo_medio,
        "preco_relativo_mediano": preco_relativo_mediano,
        "preco_relativo_min": preco_relativo_min,
        "preco_relativo_max": preco_relativo_max,
        "preco_relativo_std": preco_relativo_std,
        "amplitude_preco_relativo": amplitude_preco_relativo,
        "participacao_maior_item": participacao_maior_item,
        "maior_item": padronizar_categorica(maior_item),
    }

    valor_total_por_sigla = (
        df_prop.groupby("Sigla")["Valor_Total_Item"]
        .sum()
        .to_dict()
    )

    preco_rel_sigla = {}

    for sigla in siglas:
        grupo_sigla = df_prop[df_prop["Sigla"] == sigla]

        if len(grupo_sigla) == 0:
            preco_rel_sigla[sigla] = 0.0
        else:
            pesos = grupo_sigla["Valor_Total_Item"].values
            valores = grupo_sigla["Preco_Relativo"].values

            if pesos.sum() > 0:
                preco_rel_sigla[sigla] = float(np.average(valores, weights=pesos))
            else:
                preco_rel_sigla[sigla] = float(grupo_sigla["Preco_Relativo"].mean())

    for sigla in siglas:
        nome_sigla = normalizar_nome_coluna(sigla)
        valor_sigla = valor_total_por_sigla.get(sigla, 0.0)

        linha[f"preco_relativo_{nome_sigla}"] = preco_rel_sigla.get(sigla, 0.0)
        linha[f"part_valor_{nome_sigla}"] = (
            valor_sigla / valor_total_contrato
            if valor_total_contrato > 0
            else 0.0
        )

    colunas_raw_esperadas = (
        colunas_categoricas_output_final +
        colunas_numericas_output_final
    )

    dado_raw = pd.DataFrame([linha])
    dado_raw = dado_raw.reindex(columns=colunas_raw_esperadas, fill_value=0)

    for col in colunas_categoricas_output_final:
        if col in dado_raw.columns:
            dado_raw[col] = dado_raw[col].apply(padronizar_categorica)

    for col in colunas_numericas_output_final:
        if col in dado_raw.columns:
            dado_raw[col] = pd.to_numeric(dado_raw[col], errors="coerce")

    dado_raw[colunas_numericas_output_final] = (
        dado_raw[colunas_numericas_output_final]
        .fillna(0)
    )

    dado_final_modelo = transformar_com_preprocessador(
        preprocessador_output_final,
        dado_raw,
        features_output_final,
    )

    return dado_final_modelo


def obter_probabilidade_heuristica(
    novo_negocio_info,
    ensaios_propostos,
    modelo_heuristica,
    tabela_precos_global,
    preprocessador_output_final,
    features_output_final,
    colunas_categoricas_output_final,
    colunas_numericas_output_final,
    siglas,
):
    dado_final_modelo = montar_linha_modelo(
        novo_negocio_info=novo_negocio_info,
        ensaios_propostos=ensaios_propostos,
        tabela_precos_global=tabela_precos_global,
        preprocessador_output_final=preprocessador_output_final,
        features_output_final=features_output_final,
        colunas_categoricas_output_final=colunas_categoricas_output_final,
        colunas_numericas_output_final=colunas_numericas_output_final,
        siglas=siglas,
    )

    return float(modelo_heuristica.predict_proba(dado_final_modelo)[0, 1])


# =========================================================================
# HEURÍSTICA DE PREÇOS
# =========================================================================

def heuristica_precos_prob_margem(
    novo_negocio_info,
    ensaios,
    func_probabilidade,
    tabela_precos_global,
    tabela_custos_global,
    omega=0.01,
    fator_inicial=0.70,
    margem_minima=0.50,
    aliquota_imposto=0.1175,
    tolerancia_prob=1e-12,
    max_iter=500,
    debug=False,
):
    ensaios_base = []

    for ensaio in ensaios:
        sigla = str(ensaio.get("Sigla", "")).strip()
        qtd = float(ensaio.get("Quantidade", 0))
        prazo = ensaio.get("Prazo")

        if sigla == "" or qtd <= 0:
            raise ValueError("Cada ensaio precisa ter Sigla e Quantidade válidas.")

        prazo_dias = limpar_prazo(prazo)
        if pd.isna(prazo_dias):
            raise ValueError(
                f"O ensaio '{sigla}' está sem prazo válido. Informe um prazo em dias (número positivo)."
            )

        preco_base = float(tabela_precos_global.get(sigla, 0))
        custo_unitario = obter_custo_unitario(sigla, tabela_custos_global)

        if preco_base <= 0:
            raise ValueError(f"Preço base não encontrado ou inválido para '{sigla}'.")

        if custo_unitario <= 0:
            raise ValueError(f"Custo unitário não encontrado ou inválido para '{sigla}'.")

        ensaios_base.append({
            "Sigla": sigla,
            "Quantidade": qtd,
            "Prazo": str(prazo),
            "P_BASE": preco_base,
            "CUSTO_UNITARIO": custo_unitario,
        })

    n = len(ensaios_base)
    fatores_iniciais = np.full(n, fator_inicial, dtype=float)

    def montar_ensaios_precificados(fatores):
        linhas = []

        for i, item in enumerate(ensaios_base):
            preco_unit = fatores[i] * item["P_BASE"]
            custo_unit = item["CUSTO_UNITARIO"]
            qtd = item["Quantidade"]

            receita_bruta_item = qtd * preco_unit
            receita_liquida_item = receita_bruta_item * (1 - aliquota_imposto)
            custo_total_item = qtd * custo_unit
            margem_total_item = receita_liquida_item - custo_total_item

            linhas.append({
                "Sigla": item["Sigla"],
                "Quantidade": qtd,
                "Prazo": item.get("Prazo"),
                "Preco_Unitario": preco_unit,
                "Custo_Unitario": custo_unit,
                "Receita_Bruta_Item": receita_bruta_item,
                "Receita_Liquida_Item": receita_liquida_item,
                "Custo_Total_Item": custo_total_item,
                "Margem_Total_Item": margem_total_item,
                "Margem_Unitario_Liquida": preco_unit * (1 - aliquota_imposto) - custo_unit,
            })

        return linhas

    def avaliar_estado(fatores):
        ensaios_precificados = montar_ensaios_precificados(fatores)
        prob = float(func_probabilidade(novo_negocio_info, ensaios_precificados))

        receita_bruta = sum(x["Receita_Bruta_Item"] for x in ensaios_precificados)
        receita_liquida = sum(x["Receita_Liquida_Item"] for x in ensaios_precificados)
        custo_total = sum(x["Custo_Total_Item"] for x in ensaios_precificados)
        margem_total = receita_liquida - custo_total

        if receita_liquida > 0:
            margem_percentual = margem_total / receita_liquida
        else:
            margem_percentual = float("-inf")

        return {
            "fatores": fatores.copy(),
            "probabilidade": prob,
            "receita_bruta": receita_bruta,
            "receita_liquida": receita_liquida,
            "custo_total": custo_total,
            "margem_total": margem_total,
            "margem_percentual": margem_percentual,
            "atingiu_margem_minima": margem_percentual >= margem_minima,
            "ensaios_precificados": ensaios_precificados,
        }

    def copiar_estado(estado):
        return {
            "fatores": estado["fatores"].copy(),
            "probabilidade": float(estado["probabilidade"]),
            "receita_bruta": float(estado["receita_bruta"]),
            "receita_liquida": float(estado["receita_liquida"]),
            "custo_total": float(estado["custo_total"]),
            "margem_total": float(estado["margem_total"]),
            "margem_percentual": float(estado["margem_percentual"]),
            "atingiu_margem_minima": bool(estado["atingiu_margem_minima"]),
            "ensaios_precificados": [dict(x) for x in estado["ensaios_precificados"]],
        }

    def melhor_prob_e_margem(a, b):
        if b is None:
            return True

        if a["probabilidade"] > b["probabilidade"] + tolerancia_prob:
            return True

        if abs(a["probabilidade"] - b["probabilidade"]) <= tolerancia_prob:
            return a["margem_total"] > b["margem_total"]

        return False

    def melhor_para_bater_margem(a, b):
        if b is None:
            return True

        if a["atingiu_margem_minima"] and not b["atingiu_margem_minima"]:
            return True

        if not a["atingiu_margem_minima"] and b["atingiu_margem_minima"]:
            return False

        if a["margem_percentual"] > b["margem_percentual"]:
            return True

        if a["margem_percentual"] < b["margem_percentual"]:
            return False

        return a["probabilidade"] > b["probabilidade"]

    def empacotar_cenario(estado):
        return {
            "probabilidade": estado["probabilidade"],
            "receita_bruta": estado["receita_bruta"],
            "receita_liquida": estado["receita_liquida"],
            "custo_total": estado["custo_total"],
            "margem_total": estado["margem_total"],
            "margem_percentual": estado["margem_percentual"],
            "atingiu_margem_minima": estado["atingiu_margem_minima"],
            "precos": [x["Preco_Unitario"] for x in estado["ensaios_precificados"]],
            "fatores": estado["fatores"].tolist(),
            "ensaios": estado["ensaios_precificados"],
        }

    estado_atual = avaliar_estado(fatores_iniciais)
    cenario_max_prob_margem = copiar_estado(estado_atual)

    cenario_margem_minima = (
        copiar_estado(estado_atual)
        if estado_atual["atingiu_margem_minima"]
        else None
    )

    fase = "prob_sem_cair"
    motivo_parada = None

    for iteracao in range(1, max_iter + 1):
        estados_individuais = []

        for i in range(n):
            fatores_teste = estado_atual["fatores"].copy()
            fatores_teste[i] += omega
            estado_teste = avaliar_estado(fatores_teste)
            estados_individuais.append((i, estado_teste))

        if fase == "prob_sem_cair":
            candidatos_idxs = [
                i for i, est in estados_individuais
                if est["probabilidade"] + tolerancia_prob >= estado_atual["probabilidade"]
            ]

            melhor_estado_iter = None

            for tam in range(len(candidatos_idxs), 0, -1):
                melhor_tam = None

                for comb in combinations(candidatos_idxs, tam):
                    fatores_comb = estado_atual["fatores"].copy()

                    for idx in comb:
                        fatores_comb[idx] += omega

                    estado_comb = avaliar_estado(fatores_comb)

                    if estado_comb["probabilidade"] + tolerancia_prob < estado_atual["probabilidade"]:
                        continue

                    if melhor_prob_e_margem(estado_comb, melhor_tam):
                        melhor_tam = copiar_estado(estado_comb)

                if melhor_tam is not None:
                    melhor_estado_iter = melhor_tam
                    break

            if melhor_estado_iter is None:
                if estado_atual["atingiu_margem_minima"]:
                    motivo_parada = "probabilidade_passaria_a_cair_apos_bater_margem_minima"
                    break

                fase = "forcar_margem_minima"
                continue

            estado_atual = copiar_estado(melhor_estado_iter)

            if melhor_prob_e_margem(estado_atual, cenario_max_prob_margem):
                cenario_max_prob_margem = copiar_estado(estado_atual)

            if estado_atual["atingiu_margem_minima"] and cenario_margem_minima is None:
                cenario_margem_minima = copiar_estado(estado_atual)

        else:
            candidatos_idxs = [
                i for i, est in estados_individuais
                if est["margem_total"] > estado_atual["margem_total"]
                or est["margem_percentual"] > estado_atual["margem_percentual"]
            ]

            melhor_estado_iter = None

            for tam in range(len(candidatos_idxs), 0, -1):
                melhor_tam = None

                for comb in combinations(candidatos_idxs, tam):
                    fatores_comb = estado_atual["fatores"].copy()

                    for idx in comb:
                        fatores_comb[idx] += omega

                    estado_comb = avaliar_estado(fatores_comb)

                    if melhor_para_bater_margem(estado_comb, melhor_tam):
                        melhor_tam = copiar_estado(estado_comb)

                if melhor_tam is not None:
                    melhor_estado_iter = melhor_tam
                    break

            if melhor_estado_iter is None:
                motivo_parada = "nao_foi_possivel_aumentar_margem_ate_o_minimo"
                break

            estado_atual = copiar_estado(melhor_estado_iter)

            if melhor_prob_e_margem(estado_atual, cenario_max_prob_margem):
                cenario_max_prob_margem = copiar_estado(estado_atual)

            if estado_atual["atingiu_margem_minima"]:
                cenario_margem_minima = copiar_estado(estado_atual)
                motivo_parada = "margem_minima_atingida_apos_queda_de_probabilidade"
                break

    if motivo_parada is None:
        motivo_parada = "max_iter"

    return {
        "motivo_parada": motivo_parada,
        "cenario_max_prob_margem": empacotar_cenario(cenario_max_prob_margem),
        "cenario_margem_minima": (
            None
            if cenario_margem_minima is None
            else empacotar_cenario(cenario_margem_minima)
        ),
    }


# =========================================================================
# FUNÇÃO PARA ANÁLISE DO CENÁRIO MANUAL DO USUÁRIO
# =========================================================================

def analisar_cenario_manual(
    novo_negocio_info,
    ensaios_com_preco_manual,
    func_probabilidade,
    tabela_custos_global,
    aliquota_imposto=0.1175,
):
    """
    Calcula a probabilidade e as métricas financeiras para um cenário de preços
    definido manualmente pelo usuário.
    """
    prob = float(func_probabilidade(novo_negocio_info, ensaios_com_preco_manual))

    receita_bruta_total = 0
    receita_liquida_total = 0
    custo_total_geral = 0
    ensaios_calculados = []

    for ensaio in ensaios_com_preco_manual:
        sigla = ensaio["Sigla"]
        qtd = ensaio["Quantidade"]
        preco_unit = ensaio["Preco_Unitario"]
        custo_unit = obter_custo_unitario(sigla, tabela_custos_global)

        if custo_unit <= 0:
            raise ValueError(f"Custo não encontrado para o ensaio '{sigla}'.")

        receita_bruta_item = qtd * preco_unit
        receita_liquida_item = receita_bruta_item * (1 - aliquota_imposto)
        custo_total_item = qtd * custo_unit

        receita_bruta_total += receita_bruta_item
        receita_liquida_total += receita_liquida_item
        custo_total_geral += custo_total_item

        ensaio_calculado = ensaio.copy()
        ensaio_calculado.update({
            "Custo_Unitario": custo_unit,
            "Receita_Bruta_Item": receita_bruta_item,
        })
        ensaios_calculados.append(ensaio_calculado)

    margem_total = receita_liquida_total - custo_total_geral
    margem_percentual = margem_total / receita_liquida_total if receita_liquida_total > 0 else float("-inf")

    return {
        "probabilidade": prob,
        "receita_bruta": receita_bruta_total,
        "receita_liquida": receita_liquida_total,
        "custo_total": custo_total_geral,
        "margem_total": margem_total,
        "margem_percentual": margem_percentual,
        "ensaios": ensaios_calculados,
    }

# =========================================================================
# CALLBACKS DA UI
# =========================================================================

def on_change_preco():
    st.session_state.resultado_precos = None

def adicionar_ensaio():
    st.session_state.ensaios_sim_preco.append({
        "id": str(uuid.uuid4()),
        "Sigla": st.session_state.opcoes_siglas[0],
        "Quantidade": 1,
        "Prazo": 30,
    })
    on_change_preco()

def remover_ensaio(idx):
    st.session_state.ensaios_sim_preco.pop(idx)
    on_change_preco()

def limpar_simulacao():
    st.session_state.ensaios_sim_preco = [{
        "id": str(uuid.uuid4()),
        "Sigla": st.session_state.opcoes_siglas[0],
        "Quantidade": 1,
        "Prazo": 30,
    }]

    st.session_state.resultado_precos = None
    st.session_state.resultado_cenario_usuario = None
    st.toast("🗑️ Tela limpa.", icon="🗑️")


# =========================================================================
# RENDERIZAÇÃO DA PÁGINA
# =========================================================================

def render():
    st.header("Simulador de Preços de Novas Propostas")
    st.markdown(
        "Insira os dados da nova proposta para otimizar os preços cobrados, "
        "maximizando a chance de ganho e a receita."
    )

    try:
        dados_ml = inicializar_modelo_e_dados()

        modelo_heuristica = dados_ml["modelo_heuristica"]
        tabela_precos_global = dados_ml["tabela_precos_global"]
        opcoes_siglas = dados_ml["siglas"]
        lista_clientes_bd = dados_ml["lista_clientes_bd"]
        lista_setores_bd = dados_ml["lista_setores_bd"]
        lista_minerais_bd = dados_ml["lista_minerais_bd"]

        preprocessador_output_final = dados_ml["preprocessador_output_final"]
        features_output_final = dados_ml["features_output_final"]
        colunas_categoricas_output_final = dados_ml["colunas_categoricas_output_final"]
        colunas_numericas_output_final = dados_ml["colunas_numericas_output_final"]

        # Atualiza sempre para evitar reaproveitar opções antigas do session_state
        # depois de trocar o arquivo artefatos_modelo.pkl.
        st.session_state.opcoes_siglas = sorted([s for s in opcoes_siglas if str(s) != "nan"])

        with st.sidebar.expander("Debug do modelo", expanded=False):
            st.write("Artefato:", dados_ml.get("caminho_artefato_modelo", "-"))

            metadata = dados_ml.get("metadata")
            if metadata:
                st.write("Arquivo treino:", metadata.get("arquivo_drive_name", metadata.get("arquivo_excel", "-")))
                st.write("ID arquivo:", metadata.get("arquivo_drive_id", "-"))
                st.write("Shape X_enc:", metadata.get("shape_X_enc", "-"))
                st.write("Qtd features:", metadata.get("qtd_features", "-"))
                st.write("XGBoost:", metadata.get("xgboost", "-"))
                st.write("scikit-learn:", metadata.get("sklearn", "-"))

            sanity = dados_ml.get("sanity_check_streamlit")
            if sanity:
                st.write("Sanity prob treino:", sanity.get("prob_treino"))
                st.write("Sanity prob atual:", sanity.get("prob_atual"))
                st.write("Sanity diff:", sanity.get("diff"))

    except Exception as e:
        st.error(f"Erro ao carregar o arquivo do modelo (artefatos_modelo.pkl): {e}")
        return

    if "ensaios_sim_preco" not in st.session_state:
        st.session_state.ensaios_sim_preco = [{
            "id": str(uuid.uuid4()),
            "Sigla": st.session_state.opcoes_siglas[0],
            "Quantidade": 1,
            "Prazo": 30,
        }]

    st.session_state.setdefault("resultado_precos", None)
    st.session_state.setdefault("resultado_cenario_usuario", None)

    col1, col2 = st.columns([3, 1])

    with col1:
        st.subheader("Parâmetros do Cliente")

    with col2:
        st.button(
            "🗑️ Limpar Tudo",
            on_click=limpar_simulacao,
            use_container_width=True,
        )

    with st.container(border=True):
        c_cliente, c_mes = st.columns(2)

        opcoes_cliente = lista_clientes_bd + ["➕ Adicionar novo cliente..."]

        cliente_selecionado = c_cliente.selectbox(
            "Cliente",
            options=opcoes_cliente,
            on_change=on_change_preco,
        )

        if cliente_selecionado == "➕ Adicionar novo cliente...":
            cliente = c_cliente.text_input(
                "Nome do novo cliente",
                placeholder="Digite o nome...",
                on_change=on_change_preco,
            )
        else:
            cliente = cliente_selecionado

        mes = c_mes.selectbox(
            "Mês Previsto",
            options=list(range(1, 13)),
            index=9,
            on_change=on_change_preco,
        )

        c_setor, c_mineral = st.columns(2)

        opcoes_setor = lista_setores_bd + ["➕ Adicionar novo setor..."]

        setor_selecionado = c_setor.selectbox(
            "Setor",
            options=opcoes_setor,
            on_change=on_change_preco,
        )

        if setor_selecionado == "➕ Adicionar novo setor...":
            setor = c_setor.text_input(
                "Nome do novo setor",
                placeholder="Digite o setor...",
                on_change=on_change_preco,
            )
        else:
            setor = setor_selecionado

        opcoes_mineral = lista_minerais_bd + ["➕ Adicionar novo mineral..."]

        mineral_selecionado = c_mineral.selectbox(
            "Mineral",
            options=opcoes_mineral,
            on_change=on_change_preco,
        )

        if mineral_selecionado == "➕ Adicionar novo mineral...":
            mineral = c_mineral.text_input(
                "Nome do novo mineral",
                placeholder="Digite o mineral...",
                on_change=on_change_preco,
            )
        else:
            mineral = mineral_selecionado

    with st.container(border=True):
        st.markdown("##### Parâmetros da Otimização")

        c_margem, c_desc = st.columns(2)

        margem_minima_input = c_margem.number_input(
            "Margem Mínima Alvo (%)",
            min_value=0.0,
            max_value=100.0,
            value=40.0,
            step=1.0,
            help="Margem mínima de lucro desejada para a proposta.",
        )

        desconto_maximo_input = c_desc.number_input(
            "Desconto Máximo Inicial (%)",
            min_value=0.0,
            max_value=99.0,
            value=30.0,
            step=1.0,
            help=(
                "Define o desconto máximo autorizado como ponto de partida. "
                "Exemplo: 30% significa começar em 70% do preço base."
            ),
        )

    st.markdown("---")
    st.subheader("Ensaios da Proposta")

    with st.container(border=True):
        c1, c2, c3, c4 = st.columns([4, 2, 2, 1])

        c1.markdown("**Sigla do Ensaio**")
        c2.markdown("**Quantidade**")
        c3.markdown("**Prazo (dias)**")

        for idx, ensaio in enumerate(st.session_state.ensaios_sim_preco):
            if "id" not in ensaio:
                ensaio["id"] = str(uuid.uuid4())

            ensaio_id = ensaio["id"]

            col1, col2, col3, col4 = st.columns([4, 2, 2, 1])

            idx_sigla = (
                st.session_state.opcoes_siglas.index(ensaio["Sigla"])
                if ensaio.get("Sigla") in st.session_state.opcoes_siglas
                else 0
            )

            nova_sigla = col1.selectbox(
                "Sigla",
                st.session_state.opcoes_siglas,
                index=idx_sigla,
                key=f"sigla_{ensaio_id}",
                label_visibility="collapsed",
                on_change=on_change_preco,
            )

            st.session_state.ensaios_sim_preco[idx]["Sigla"] = nova_sigla

            qtd_str = col2.text_input(
                "Qtd",
                value=str(ensaio.get("Quantidade", 1)),
                key=f"qtd_{ensaio_id}",
                label_visibility="collapsed",
                on_change=on_change_preco,
            )

            st.session_state.ensaios_sim_preco[idx]["Quantidade"] = (
                int(qtd_str)
                if qtd_str.isdigit() and int(qtd_str) > 0
                else 1
            )

            prazo_bruto = ensaio.get("Prazo", 30)
            prazo_numerico = limpar_prazo(prazo_bruto)
            valor_input_prazo = (
                int(prazo_numerico) if pd.notna(prazo_numerico) and prazo_numerico > 0 else 30
            )

            prazo_dias = col3.number_input(
                "Prazo (dias)",
                value=valor_input_prazo,
                min_value=1,
                step=1,
                key=f"prazo_{ensaio_id}",
                label_visibility="collapsed",
                help="Prazo de execução do ensaio em dias.",
                on_change=on_change_preco,
            )

            st.session_state.ensaios_sim_preco[idx]["Prazo"] = prazo_dias

            col4.button(
                "➖",
                key=f"rem_{ensaio_id}",
                on_click=remover_ensaio,
                args=(idx,),
            )

        st.button(
            "➕ Adicionar Ensaio",
            on_click=adicionar_ensaio,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # Define as informações e a função de probabilidade que serão usadas em ambos os botões
    info = {
        "cliente": cliente,
        "mes": mes,
        "setor": setor,
        "mineral": mineral,
        "mapa_prazos": criar_mapa_prazos(st.session_state.ensaios_sim_preco),
    }

    func_prob = lambda inf, ens: obter_probabilidade_heuristica(
        novo_negocio_info=inf, ensaios_propostos=ens,
        modelo_heuristica=modelo_heuristica, tabela_precos_global=tabela_precos_global,
        preprocessador_output_final=preprocessador_output_final, features_output_final=features_output_final,
        colunas_categoricas_output_final=colunas_categoricas_output_final, colunas_numericas_output_final=colunas_numericas_output_final,
        siglas=opcoes_siglas)

    if st.button(
        "🚀 Executar Otimização de Preços",
        use_container_width=True,
        type="primary",
    ):
        if not st.session_state.ensaios_sim_preco:
            st.warning("Adicione pelo menos um ensaio.")
            return

        with st.spinner("Rodando ML e Heurística Matemática..."):

            try:
                resultado = heuristica_precos_prob_margem(
                    novo_negocio_info=info,
                    ensaios=st.session_state.ensaios_sim_preco,
                    func_probabilidade=func_prob,
                    tabela_precos_global=tabela_precos_global,
                    tabela_custos_global=TABELA_CUSTOS_MEDIA,
                    omega=0.01,
                    fator_inicial=(1.0 - (desconto_maximo_input / 100.0)),
                    margem_minima=(margem_minima_input / 100.0),
                    aliquota_imposto=0.1175,
                    debug=False,
                )

                st.session_state.resultado_precos = resultado
                st.session_state.resultado_cenario_usuario = None  # Reseta o cenário manual
                st.success("✅ Otimização Concluída!")

            except Exception as e:
                st.error(f"Erro ao processar a otimização: {e}")

    if st.session_state.resultado_precos is not None:
        res = st.session_state.resultado_precos

        st.markdown("---")
        st.subheader("Resultados da Otimização do Contrato")

        # --- Texto Interpretativo dos Cenários ---
        res_otimizado = res.get("cenario_max_prob_margem")
        res_margem_min = res.get("cenario_margem_minima")

        if res_otimizado:
            st.markdown("##### Análise dos Cenários Sugeridos:")
            if res_otimizado["atingiu_margem_minima"]:
                if res_margem_min and res_otimizado["margem_percentual"] > res_margem_min["margem_percentual"] + 0.0001:
                    st.info(
                        "**Cenário Otimizado (Max Probabilidade & Margem):** Este cenário não apenas maximizou a probabilidade de ganho, "
                        "mas também superou a margem mínima desejada e, inclusive, obteve uma margem percentual maior que o cenário "
                        "focado apenas em atingir a margem mínima. É a melhor combinação de probabilidade e rentabilidade."
                    )
                else:
                    st.info(
                        "**Cenário Otimizado (Max Probabilidade & Margem):** Este cenário maximizou a probabilidade de ganho "
                        "e conseguiu atingir a margem mínima desejada. Representa um bom equilíbrio entre chance de conversão e rentabilidade."
                    )
            else:
                st.warning(
                    "**Cenário Otimizado (Max Probabilidade & Margem):** Este cenário maximiza a probabilidade de ganho, "
                    "mas **não atingiu a margem mínima desejada**. Sugerimos analisar criticamente a perda de probabilidade "
                    "no 'Cenário: Margem Mínima Atingida' e avaliar se vale a pena trabalhar com margens mais baixas "
                    "para aumentar a probabilidade de conversão, ou se a margem mínima é inegociável."
                )
            st.markdown("---")
        # --- Fim do Texto Interpretativo ---

        tab1, tab2 = st.tabs([
            "Cenário: Max Probabilidade & Margem",
            "Cenário: Margem Mínima Atingida",
        ])

        def renderizar_cenario(cenario):
            if cenario is None:
                st.info("Este cenário não foi alcançado ou é idêntico ao cenário principal.")
                return

            m1, m2, m3 = st.columns(3)

            with m1:
                st.metric(
                    "🎯 Probabilidade Estimada",
                    f"{cenario.get('probabilidade', 0) * 100:.2f}%",
                )

            with m2:
                st.metric(
                    "💵 Receita Bruta",
                    f"R$ {cenario.get('receita_bruta', 0):,.2f}"
                    .replace(",", "X")
                    .replace(".", ",")
                    .replace("X", "."),
                )

            with m3:
                st.metric(
                    "💰 Receita Líquida",
                    f"R$ {cenario.get('receita_liquida', 0):,.2f}"
                    .replace(",", "X")
                    .replace(".", ",")
                    .replace("X", "."),
                )

            m4, m5, m6 = st.columns(3)

            with m4:
                st.metric(
                    "🧾 Custos",
                    f"R$ -{cenario.get('custo_total', 0):,.2f}"
                    .replace(",", "X")
                    .replace(".", ",")
                    .replace("X", "."),
                )

            with m5:
                st.metric(
                    "📊 Contribuição",
                    f"R$ {cenario.get('margem_total', 0):,.2f}"
                    .replace(",", "X")
                    .replace(".", ",")
                    .replace("X", "."),
                )

            with m6:
                st.metric(
                    "📈 Margem Percentual",
                    f"{cenario.get('margem_percentual', 0) * 100:.2f}%",
                )

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("##### 🛒 Tabela de Preços Sugeridos por Ensaio")

            df_resultados = pd.DataFrame(cenario.get("ensaios", []))

            if df_resultados.empty:
                st.info("Nenhum ensaio encontrado neste cenário.")
                return

            df_resultados = df_resultados.rename(columns={
                "Sigla": "Ensaio",
                "Quantidade": "Qtd",
                "Prazo": "Prazo",
                "Preco_Unitario": "Preço Otimizado (R$)",
                "Custo_Unitario": "Custo Unitário (R$)",
                "Receita_Bruta_Item": "Valor Total (R$)",
            })

            if "Ensaio" not in df_resultados.columns:
                df_resultados["Ensaio"] = "-"
            if "Qtd" not in df_resultados.columns:
                df_resultados["Qtd"] = 0
            if "Prazo" not in df_resultados.columns:
                df_resultados["Prazo"] = "-"
            if "Preço Otimizado (R$)" not in df_resultados.columns:
                df_resultados["Preço Otimizado (R$)"] = 0.0
            if "Custo Unitário (R$)" not in df_resultados.columns:
                df_resultados["Custo Unitário (R$)"] = 0.0
            if "Valor Total (R$)" not in df_resultados.columns:
                df_resultados["Valor Total (R$)"] = (
                    pd.to_numeric(df_resultados["Qtd"], errors="coerce").fillna(0)
                    *
                    pd.to_numeric(df_resultados["Preço Otimizado (R$)"], errors="coerce").fillna(0)
                )

            df_resultados["Preço Base (R$)"] = df_resultados["Ensaio"].apply(
                lambda x: tabela_precos_global.get(x, 0.0)
            )

            df_resultados["Qtd"] = (
                pd.to_numeric(df_resultados["Qtd"], errors="coerce")
                .fillna(0)
                .astype(int)
            )

            for col in [
                "Preço Base (R$)",
                "Custo Unitário (R$)",
                "Preço Otimizado (R$)",
                "Valor Total (R$)",
            ]:
                df_resultados[col] = pd.to_numeric(
                    df_resultados[col],
                    errors="coerce",
                ).fillna(0)

            def formatar_variacao(row):
                otimizado = row["Preço Otimizado (R$)"]
                base = row["Preço Base (R$)"]

                if base <= 0:
                    return "➖"

                variacao = round(((otimizado / base) - 1) * 100, 2)

                if variacao > 0:
                    return f"🔺 +{variacao:.1f}%"
                if variacao < 0:
                    return f"🔻 {variacao:.1f}%"
                return "➖ 0.0%"

            df_resultados["Variação Aplicada"] = df_resultados.apply(
                formatar_variacao,
                axis=1,
            )

            colunas_ordem = [
                "Ensaio",
                "Qtd",
                "Prazo",
                "Preço Base (R$)",
                "Custo Unitário (R$)",
                "Preço Otimizado (R$)",
                "Variação Aplicada",
                "Valor Total (R$)",
            ]

            df_resultados = df_resultados[colunas_ordem]

            df_formatado = df_resultados.style.format({
                "Preço Base (R$)": "{:.2f}",
                "Custo Unitário (R$)": "-{:.2f}",
                "Preço Otimizado (R$)": "{:.2f}",
                "Valor Total (R$)": "{:.2f}",
            })

            st.dataframe(
                df_formatado,
                use_container_width=True,
                hide_index=True,
            )

        with tab1:
            renderizar_cenario(res.get("cenario_max_prob_margem"))

        with tab2:
            renderizar_cenario(res.get("cenario_margem_minima"))

        # --- SEÇÃO PARA O USUÁRIO SIMULAR O PRÓPRIO CENÁRIO ---
        st.markdown("---")
        st.subheader("Simule seu Próprio Cenário de Preços")
        st.markdown(
            "Ajuste os preços otimizados abaixo e clique em 'Analisar' para ver o impacto "
            "na probabilidade de ganho e na margem."
        )

        cenario_base = res.get("cenario_max_prob_margem")
        if cenario_base and cenario_base.get("ensaios"):
            ensaios_base_para_manual = cenario_base["ensaios"]

            with st.form(key="form_cenario_manual"):
                c1, c2, c3, c4 = st.columns([4, 1, 2, 3])
                c1.markdown("**Ensaio**")
                c2.markdown("**Qtd**")
                c3.markdown("**Prazo**")
                c4.markdown("**Seu Preço Unitário (R$)**")

                ensaios_com_preco_manual = []

                for i, ensaio_base in enumerate(ensaios_base_para_manual):
                    col1, col2, col3, col4 = st.columns([4, 1, 2, 3])

                    sigla = ensaio_base.get("Sigla", "-")
                    qtd = int(ensaio_base.get("Quantidade", 0))
                    prazo = str(ensaio_base.get("Prazo", "-"))
                    preco_otimizado = ensaio_base.get("Preco_Unitario", 0.0)

                    col1.text_input("sigla_manual_disp", value=sigla, key=f"disp_sigla_{i}", disabled=True, label_visibility="collapsed")
                    col2.text_input("qtd_manual_disp", value=str(qtd), key=f"disp_qtd_{i}", disabled=True, label_visibility="collapsed")
                    col3.text_input("prazo_manual_disp", value=prazo, key=f"disp_prazo_{i}", disabled=True, label_visibility="collapsed")

                    preco_manual = col4.number_input(
                        "preco_manual_input", value=float(preco_otimizado), min_value=0.0,
                        step=10.0, format="%.2f", key=f"preco_manual_{i}", label_visibility="collapsed"
                    )

                    ensaio_manual = ensaio_base.copy()
                    ensaio_manual["Preco_Unitario"] = preco_manual
                    ensaios_com_preco_manual.append(ensaio_manual)

                submitted = st.form_submit_button("📊 Analisar Meu Cenário", use_container_width=True)

                if submitted:
                    with st.spinner("Calculando probabilidade e métricas para o seu cenário..."):
                        try:
                            resultado_manual = analisar_cenario_manual(
                                novo_negocio_info=info,
                                ensaios_com_preco_manual=ensaios_com_preco_manual,
                                func_probabilidade=func_prob,
                                tabela_custos_global=TABELA_CUSTOS_MEDIA,
                                aliquota_imposto=0.1175,
                            )
                            st.session_state.resultado_cenario_usuario = resultado_manual
                        except Exception as e:
                            st.error(f"Erro ao analisar cenário manual: {e}")

        # --- EXIBIÇÃO DO RESULTADO MANUAL E COMPARATIVO ---
        if st.session_state.get("resultado_cenario_usuario"):
            st.markdown("---")
            st.subheader("Resultado do Seu Cenário Manual")
            renderizar_cenario(st.session_state.resultado_cenario_usuario)

            st.markdown("---")
            st.subheader("Comparativo dos Cenários")

            res_otimizado = res.get("cenario_max_prob_margem")
            res_margem_min = res.get("cenario_margem_minima")
            res_manual = st.session_state.resultado_cenario_usuario

            cenarios_data = []
            if res_otimizado:
                cenarios_data.append({
                    "Cenário": "Otimizado (Max Prob)",
                    "Probabilidade (%)": res_otimizado['probabilidade'] * 100,
                    "Receita Bruta (R$)": res_otimizado['receita_bruta'],
                    "Margem Total (R$)": res_otimizado['margem_total'],
                    "Margem (%)": res_otimizado['margem_percentual'] * 100,
                })
            if res_margem_min:
                cenarios_data.append({
                    "Cenário": "Margem Mínima",
                    "Probabilidade (%)": res_margem_min['probabilidade'] * 100,
                    "Receita Bruta (R$)": res_margem_min['receita_bruta'],
                    "Margem Total (R$)": res_margem_min['margem_total'],
                    "Margem (%)": res_margem_min['margem_percentual'] * 100,
                })
            if res_manual:
                cenarios_data.append({
                    "Cenário": "Seu Cenário",
                    "Probabilidade (%)": res_manual['probabilidade'] * 100,
                    "Receita Bruta (R$)": res_manual['receita_bruta'],
                    "Margem Total (R$)": res_manual['margem_total'],
                    "Margem (%)": res_manual['margem_percentual'] * 100,
                })

            if cenarios_data:
                df_comp = pd.DataFrame(cenarios_data)

                st.markdown("##### Tabela Comparativa")
                df_comp_display = df_comp.set_index("Cenário").T
                st.dataframe(
                    df_comp_display.style.format({
                        "Probabilidade (%)": "{:.2f}%",
                        "Receita Bruta (R$)": "R$ {:,.2f}",
                        "Margem Total (R$)": "R$ {:,.2f}",
                        "Margem (%)": "{:.2f}%",
                    }),
                    use_container_width=True
                )

                st.markdown("##### Gráficos Comparativos")
                
                col_g1, col_g2 = st.columns(2)
                with col_g1:
                    st.markdown("**Probabilidade de Ganho**")
                    st.bar_chart(
                        df_comp,
                        x="Cenário",
                        y="Probabilidade (%)",
                        color="#054D8B"
                    )
                
                with col_g2:
                    st.markdown("**Margem Percentual**")
                    st.bar_chart(
                        df_comp,
                        x="Cenário",
                        y="Margem (%)",
                        color="#3E5060"
                    )
                st.markdown("##### Receita Bruta e Margem Total por Cenário")

                df_comp_long = df_comp.melt(
                    id_vars="Cenário",
                    value_vars=["Receita Bruta (R$)", "Margem Total (R$)"],
                    var_name="Métrica",
                    value_name="Valor"
                )

                grafico = (
                    alt.Chart(df_comp_long)
                    .mark_bar()
                    .encode(
                        x=alt.X("Cenário:N", title="Cenário"),
                        xOffset=alt.XOffset("Métrica:N"),
                        y=alt.Y("Valor:Q", title="Valor (R$)"),
                        color=alt.Color(
                            "Métrica:N",
                            scale=alt.Scale(
                                domain=["Receita Bruta (R$)", "Margem Total (R$)"],
                                range=["#054D8B", "#3E5060"]
                            ),
                            title="Métrica"
                        ),
                        tooltip=[
                            alt.Tooltip("Cenário:N", title="Cenário"),
                            alt.Tooltip("Métrica:N", title="Métrica"),
                            alt.Tooltip("Valor:Q", title="Valor", format=",.2f")
                        ]
                    )
                    .properties(height=400)
                )

                st.altair_chart(grafico, use_container_width=True)
