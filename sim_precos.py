# sim_precos.py

import io
import streamlit as st
import pandas as pd
import numpy as np
from itertools import combinations
from copy import deepcopy
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
import warnings
import uuid

warnings.filterwarnings('ignore')

# ============================================================
# TABELA DE CUSTOS MÉDIOS
# ============================================================
TABELA_CUSTOS_MEDIA = {
    "CIUSAT": 610.9087613,
    "CIDSAT": 724.4449053,
    "DSS": 2079.130163,
    "CDSS": 2785.001179,
    "GPS": 170.9717246,
    "ADNP": 1105.388247,
    "CADSAT": 1649.771379,
    "LL/LP": 133.2399137,
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
    "UUSAT": 480.3904131,
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
    "CAMPO": 3365.11735
}

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
    except:
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


def remover_outliers_iqr(dados_series):
    q1 = dados_series.quantile(0.25)
    q3 = dados_series.quantile(0.75)
    iqr = q3 - q1
    limite_inferior = q1 - 1.5 * iqr
    limite_superior = q3 + 1.5 * iqr
    return dados_series[(dados_series >= limite_inferior) & (dados_series <= limite_superior)]


def calcular_referencia_limpa(serie):
    serie = serie.dropna().astype(float)

    if len(serie) == 0:
        return np.nan

    if len(serie) > 5:
        serie = remover_outliers_iqr(serie)

    if len(serie) == 0:
        return np.nan

    tolerancia = 0.10

    melhor_contagem = 0
    melhor_media = 0
    menor_variancia = float('inf')

    for valor_base in serie:
        limite_inferior = valor_base * (1 - tolerancia)
        limite_superior = valor_base * (1 + tolerancia)

        valores_na_faixa = serie[(serie >= limite_inferior) & (serie <= limite_superior)]
        contagem = len(valores_na_faixa)

        if contagem > melhor_contagem:
            melhor_contagem = contagem
            melhor_media = valores_na_faixa.mean()
            menor_variancia = valores_na_faixa.std() if contagem > 1 else 0

        elif contagem == melhor_contagem:
            variancia_atual = valores_na_faixa.std() if contagem > 1 else 0
            if variancia_atual < menor_variancia:
                melhor_media = valores_na_faixa.mean()
                menor_variancia = variancia_atual

    return float(max(melhor_media, 0.01))




# =========================================================================
#             FUNÇÕES DE CACHE: ML E PREPARAÇÃO DE DADOS
# =========================================================================

def criar_servico_drive():
    scopes = ["https://www.googleapis.com/auth/drive.readonly"]

    creds = Credentials.from_service_account_info(
        dict(st.secrets["gcp_service_account"]),
        scopes=scopes
    )

    return build("drive", "v3", credentials=creds)


def buscar_arquivo_na_pasta(service, folder_id: str, file_name: str):
    query = (
        f"'{folder_id}' in parents "
        f"and name = '{file_name}' "
        f"and trashed = false"
    )

    response = service.files().list(
        q=query,
        fields="files(id, name, mimeType)",
        pageSize=10,
        supportsAllDrives=True,
        includeItemsFromAllDrives=True
    ).execute()

    files = response.get("files", [])
    if not files:
        raise FileNotFoundError(
            f"Arquivo '{file_name}' não encontrado na pasta '{folder_id}'."
        )

    return files[0]


def baixar_arquivo_drive_em_memoria(service, file_id: str) -> io.BytesIO:
    request = service.files().get_media(fileId=file_id, supportsAllDrives=True)
    buffer = io.BytesIO()
    downloader = MediaIoBaseDownload(buffer, request)

    done = False
    while not done:
        _, done = downloader.next_chunk()

    buffer.seek(0)
    return buffer


# ============================================================
# GERAÇÃO DA BASE DO MODELO A PARTIR DA BASE ENSAIOS
# ============================================================
def gerar_base_modelo(df_ensaios: pd.DataFrame):
    col_contrato = "Négocio"
    col_cliente = "Cliente"
    col_sigla = "Sigla"
    col_qtd = "Quantidade"
    col_valor_unit = "Valor Unitario"
    col_status = "Status"
    col_mes = "Mês Ganho/Perdido"

    df = df_ensaios.copy()

    df[col_valor_unit] = df[col_valor_unit].apply(limpar_moeda_br)
    df[col_sigla] = df[col_sigla].astype(str).str.strip()
    df[col_cliente] = df[col_cliente].astype(str).str.strip()
    df[col_mes] = df[col_mes].astype(str).str.strip()

    df = df.dropna(
        subset=[col_contrato, col_cliente, col_sigla, col_qtd, col_valor_unit, col_status, col_mes]
    ).copy()

    df = df[df[col_valor_unit] > 0].copy()

    df["target"] = (
        df[col_status]
        .astype(str)
        .str.strip()
        .str.upper()
        .map(lambda x: 1 if x == "GANHO" else 0)
    )

    df["valor_total_item"] = df[col_qtd] * df[col_valor_unit]

    df_ref = (
        df.groupby(col_sigla)[col_valor_unit]
          .apply(calcular_referencia_limpa)
          .reset_index()
          .rename(columns={col_valor_unit: "preco_referencia"})
    )

    tabela_ref = dict(zip(df_ref[col_sigla], df_ref["preco_referencia"]))

    df["preco_referencia"] = df[col_sigla].map(tabela_ref)
    df["preco_relativo"] = df[col_valor_unit] / df["preco_referencia"]
    df["preco_relativo"] = df["preco_relativo"].clip(lower=0.1, upper=5.0).fillna(1.0)

    siglas = sorted(df[col_sigla].dropna().unique().tolist())

    linhas_modelo = []

    for id_contrato, grupo in df.groupby(col_contrato):
        grupo = grupo.copy()

        cliente = grupo[col_cliente].iloc[0]
        mes = grupo[col_mes].iloc[0]

        valor_total_contrato = grupo["valor_total_item"].sum()
        qtd_total = grupo[col_qtd].sum()
        n_siglas = grupo[col_sigla].nunique()

        preco_relativo_medio = grupo["preco_relativo"].mean()
        preco_relativo_mediano = grupo["preco_relativo"].median()
        preco_relativo_min = grupo["preco_relativo"].min()
        preco_relativo_max = grupo["preco_relativo"].max()
        preco_relativo_std = grupo["preco_relativo"].std(ddof=0) if len(grupo) > 1 else 0.0
        amplitude_preco_relativo = preco_relativo_max - preco_relativo_min

        target = grupo["target"].max()

        idx_maior = grupo["valor_total_item"].idxmax()
        valor_maior_item = grupo.loc[idx_maior, "valor_total_item"]
        maior_item = grupo.loc[idx_maior, col_sigla]
        participacao_maior_item = (
            valor_maior_item / valor_total_contrato if valor_total_contrato > 0 else 0.0
        )

        linha = {
            "id_contrato": id_contrato,
            "cliente": cliente,
            "mes": mes,
            "valor_total_contrato": valor_total_contrato,
            "qtd_total": qtd_total,
            "n_siglas": n_siglas,
            "preco_relativo_medio": preco_relativo_medio,
            "preco_relativo_mediano": preco_relativo_mediano,
            "preco_relativo_min": preco_relativo_min,
            "preco_relativo_max": preco_relativo_max,
            "preco_relativo_std": preco_relativo_std,
            "amplitude_preco_relativo": amplitude_preco_relativo,
            "target": target,
            "participacao_maior_item": participacao_maior_item,
            "maior_item": maior_item
        }

        valor_total_por_sigla = grupo.groupby(col_sigla)["valor_total_item"].sum().to_dict()

        preco_rel_sigla = {}
        for sigla in siglas:
            grupo_sigla = grupo[grupo[col_sigla] == sigla]
            if len(grupo_sigla) == 0:
                preco_rel_sigla[sigla] = 0.0
            else:
                pesos = grupo_sigla["valor_total_item"].values
                valores = grupo_sigla["preco_relativo"].values

                if pesos.sum() > 0:
                    preco_rel_sigla[sigla] = np.average(valores, weights=pesos)
                else:
                    preco_rel_sigla[sigla] = grupo_sigla["preco_relativo"].mean()

        for sigla in siglas:
            nome_sigla = normalizar_nome_coluna(sigla)
            valor_sigla = valor_total_por_sigla.get(sigla, 0.0)

            linha[f"preco_relativo_{nome_sigla}"] = preco_rel_sigla.get(sigla, 0.0)
            linha[f"part_valor_{nome_sigla}"] = (
                valor_sigla / valor_total_contrato if valor_total_contrato > 0 else 0.0
            )

        linhas_modelo.append(linha)

    df_modelo = pd.DataFrame(linhas_modelo)

    colunas_globais = [
        "id_contrato",
        "cliente",
        "mes",
        "valor_total_contrato",
        "qtd_total",
        "n_siglas",
        "preco_relativo_medio",
        "preco_relativo_mediano",
        "preco_relativo_min",
        "preco_relativo_max",
        "preco_relativo_std",
        "amplitude_preco_relativo",
        "target",
        "participacao_maior_item",
        "maior_item"
    ]

    outras_colunas = [c for c in df_modelo.columns if c not in colunas_globais]
    df_modelo = df_modelo[colunas_globais + sorted(outras_colunas)]

    return df, df_ref, df_modelo, tabela_ref, siglas


# ============================================================
# FUNÇÃO PRINCIPAL
# ============================================================
@st.cache_resource(show_spinner="Carregando base e treinando modelos...")
def inicializar_modelo_e_dados():
    service = criar_servico_drive()

    folder_id = st.secrets["app_config"]["id_pasta_drive"]
    nome_arquivo = st.secrets["app_config"]["nome_arq_xlsx"]
    aba_ensaios = st.secrets["app_config"]["nome_aba_ensaios"]

    arquivo = buscar_arquivo_na_pasta(service, folder_id, nome_arquivo)
    conteudo_excel = baixar_arquivo_drive_em_memoria(service, arquivo["id"])

    # lê somente BASE ENSAIOS do arquivo do secrets
    df_ensaios = pd.read_excel(conteudo_excel, sheet_name=aba_ensaios)

    # gera BASE DE DADOS MODELO a partir da BASE ENSAIOS
    df_ensaios_tratado, df_ref, df_modelo, tabela_precos_global, siglas = gerar_base_modelo(df_ensaios)

    # treino
    df_modelo = df_modelo.dropna(subset=["target"]).copy()

    y = df_modelo["target"].astype(int)
    X = df_modelo.drop(columns=["target", "id_contrato"], errors="ignore")

    colunas_categoricas = ["cliente", "maior_item", "mes"]
    colunas_categoricas = [c for c in colunas_categoricas if c in X.columns]

    X = pd.get_dummies(X, columns=colunas_categoricas, drop_first=True)

    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="coerce")

    X = X.fillna(0)
    colunas_treino_global = X.columns.tolist()

    neg_global = (y == 0).sum()
    pos_global = (y == 1).sum()
    ratio_global = neg_global / pos_global if pos_global > 0 else 1.0

    modelo_xgb = XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=2,
        random_state=42,
        eval_metric="logloss",
        scale_pos_weight=ratio_global
    )

    modelo_logistica = LogisticRegression(
        max_iter=100000,
        class_weight="balanced",
        random_state=42
    )

    smote = SMOTE(random_state=42)
    X_bal, y_bal = smote.fit_resample(X, y)

    modelo_xgb.fit(X_bal, y_bal)
    modelo_logistica.fit(X_bal, y_bal)

    lista_clientes_bd = sorted(
        [str(c).strip() for c in df_modelo["cliente"].dropna().unique() if str(c).strip() != ""]
    )

    return {
        "modelo_heuristica": modelo_xgb,
        "modelo_output_final": modelo_logistica,
        "threshold_heuristica": 0.35,
        "threshold_output_final": 0.45,
        "tabela_precos_global": tabela_precos_global,
        "colunas_treino_global": colunas_treino_global,
        "siglas": siglas,
        "lista_clientes_bd": lista_clientes_bd,
        "df_modelo": df_modelo,
        "df_ref": df_ref,
        "arquivo_drive_usado": arquivo["name"]
    }
# =========================================================================
#             FUNÇÕES DA HEURÍSTICA E PROBABILIDADE (NOVA VERSÃO)
# =========================================================================

def montar_linha_modelo(
    novo_negocio_info,
    ensaios_propostos,
    tabela_precos_global,
    colunas_treino_global,
    siglas
):
    linhas_calculadas = []

    for ensaio in ensaios_propostos:
        sigla = str(ensaio["Sigla"]).strip()
        qtd = float(ensaio["Quantidade"])

        preco_ref = tabela_precos_global.get(sigla, np.nan)

        preco_unit_proposto = ensaio.get("Preco_Unitario", preco_ref)
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

        preco_relativo = np.clip(preco_relativo, 0.1, 5.0)

        linhas_calculadas.append({
            "Sigla": sigla,
            "Quantidade": qtd,
            "Preco_Unitario": preco_unit_proposto,
            "Valor_Total_Item": valor_total_item,
            "Preco_Relativo": preco_relativo
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
    maior_item = df_prop.loc[idx_maior, "Sigla"]
    valor_maior_item = df_prop.loc[idx_maior, "Valor_Total_Item"]
    participacao_maior_item = (
        valor_maior_item / valor_total_contrato if valor_total_contrato > 0 else 0.0
    )

    linha = {
        "cliente": str(novo_negocio_info.get("cliente", "")).strip(),
        "mes": str(novo_negocio_info.get("mes", "")).strip(),
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
        "maior_item": maior_item,
    }

    valor_total_por_sigla = df_prop.groupby("Sigla")["Valor_Total_Item"].sum().to_dict()

    preco_rel_sigla = {}
    for sigla in siglas:
        grupo_sigla = df_prop[df_prop["Sigla"] == sigla]

        if len(grupo_sigla) == 0:
            preco_rel_sigla[sigla] = 0.0
        else:
            pesos = grupo_sigla["Valor_Total_Item"].values
            valores = grupo_sigla["Preco_Relativo"].values

            if pesos.sum() > 0:
                preco_rel_sigla[sigla] = np.average(valores, weights=pesos)
            else:
                preco_rel_sigla[sigla] = grupo_sigla["Preco_Relativo"].mean()

    for sigla in siglas:
        nome_sigla = normalizar_nome_coluna(sigla)
        valor_sigla = valor_total_por_sigla.get(sigla, 0.0)

        linha[f"preco_relativo_{nome_sigla}"] = preco_rel_sigla.get(sigla, 0.0)
        linha[f"part_valor_{nome_sigla}"] = (
            valor_sigla / valor_total_contrato if valor_total_contrato > 0 else 0.0
        )

    dados_input = pd.DataFrame([linha])

    colunas_categoricas = ["cliente", "maior_item", "mes"]
    colunas_categoricas = [c for c in colunas_categoricas if c in dados_input.columns]

    dados_input = pd.get_dummies(
        dados_input,
        columns=colunas_categoricas,
        drop_first=True
    )

    dado_final_modelo = dados_input.reindex(columns=colunas_treino_global, fill_value=0)

    return dado_final_modelo


def obter_probabilidade_heuristica(
    novo_negocio_info,
    ensaios_propostos,
    modelo_heuristica,
    tabela_precos_global,
    colunas_treino_global,
    siglas
):
    dado_final_modelo = montar_linha_modelo(
        novo_negocio_info,
        ensaios_propostos,
        tabela_precos_global,
        colunas_treino_global,
        siglas
    )
    return float(modelo_heuristica.predict_proba(dado_final_modelo)[0, 1])


def obter_probabilidade_logistica(
    novo_negocio_info,
    ensaios_propostos,
    modelo_output_final,
    tabela_precos_global,
    colunas_treino_global,
    siglas
):
    dado_final_modelo = montar_linha_modelo(
        novo_negocio_info,
        ensaios_propostos,
        tabela_precos_global,
        colunas_treino_global,
        siglas
    )
    return float(modelo_output_final.predict_proba(dado_final_modelo)[0, 1])

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
    debug=False
):
    ensaios_base = []
    for ensaio in ensaios:
        sigla = str(ensaio["Sigla"]).strip()
        qtd = float(ensaio["Quantidade"])

        preco_base = float(tabela_precos_global.get(sigla, 0))
        custo_unitario = float(tabela_custos_global.get(sigla.upper(), 0))

        if preco_base <= 0:
            preco_base = 100.0 
        if custo_unitario <= 0:
            custo_unitario = preco_base * 0.5 

        ensaios_base.append({
            "Sigla": sigla,
            "Quantidade": qtd,
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
            receita_liquida_item = receita_bruta_item *(1-aliquota_imposto)
            custo_total_item = qtd * custo_unit
            margem_total_item = receita_liquida_item - custo_total_item

            linhas.append({
                "Sigla": item["Sigla"],
                "Quantidade": qtd,
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
        if b is None: return True
        if a["probabilidade"] > b["probabilidade"] + tolerancia_prob: return True
        if abs(a["probabilidade"] - b["probabilidade"]) <= tolerancia_prob:
            return a["margem_total"] > b["margem_total"]
        return False

    def melhor_para_bater_margem(a, b):
        if b is None: return True
        if a["atingiu_margem_minima"] and not b["atingiu_margem_minima"]: return True
        if not a["atingiu_margem_minima"] and b["atingiu_margem_minima"]: return False
        if a["margem_percentual"] > b["margem_percentual"]: return True
        if a["margem_percentual"] < b["margem_percentual"]: return False
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
    cenario_margem_minima = copiar_estado(estado_atual) if estado_atual["atingiu_margem_minima"] else None

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
        "cenario_margem_minima": None if cenario_margem_minima is None else empacotar_cenario(cenario_margem_minima),
    }

def avaliar_cenario_logistica(
    novo_negocio_info,
    cenario,
    modelo_output_final,
    tabela_precos_global,
    colunas_treino_global,
    siglas
):
    if cenario is None:
        return None

    ensaios_cenario = cenario["ensaios"]

    prob_logistica = obter_probabilidade_logistica(
        novo_negocio_info=novo_negocio_info,
        ensaios_propostos=ensaios_cenario,
        modelo_output_final=modelo_output_final,
        tabela_precos_global=tabela_precos_global,
        colunas_treino_global=colunas_treino_global,
        siglas=siglas
    )

    return {
        "probabilidade_logistica": prob_logistica,
        "receita_bruta": cenario["receita_bruta"],
        "receita_liquida": cenario["receita_liquida"],
        "custo_total": cenario["custo_total"],
        "margem_total": cenario["margem_total"],
        "margem_percentual": cenario["margem_percentual"],
        "ensaios": cenario["ensaios"]
    }

# =========================================================================
#             GERENCIAMENTO DE ESTADO E CALLBACKS DA UI
# =========================================================================

def on_change_preco():
    st.session_state.resultado_precos = None

def adicionar_ensaio():
    st.session_state.ensaios_sim_preco.append({"id": str(uuid.uuid4()), "Sigla": st.session_state.opcoes_siglas[0], "Quantidade": 1})
    on_change_preco()

def remover_ensaio(idx):
    st.session_state.ensaios_sim_preco.pop(idx)
    on_change_preco()

def limpar_simulacao():
    st.session_state.ensaios_sim_preco = [{"id": str(uuid.uuid4()), "Sigla": st.session_state.opcoes_siglas[0], "Quantidade": 1}]
    st.session_state.resultado_precos = None
    st.toast("🗑️ Tela limpa.", icon="🗑️")

# =========================================================================
#             RENDERIZAÇÃO DA PÁGINA
# =========================================================================

def render():
    st.header("Simulador de Preços de Novas Propostas")
    st.markdown("Insira os dados da nova proposta para otimizar os preços cobrados, maximizando a chance de ganho e a receita.")

    # 1. Carrega modelo e dados
    try:
        dados_ml = inicializar_modelo_e_dados()

        modelo_heuristica = dados_ml["modelo_heuristica"]
        modelo_output_final = dados_ml["modelo_output_final"]
        tabela_precos_global = dados_ml["tabela_precos_global"]
        colunas_treino = dados_ml["colunas_treino_global"]
        opcoes_siglas = dados_ml["siglas"]
        lista_clientes_bd = dados_ml["lista_clientes_bd"]

        st.session_state.setdefault('opcoes_siglas', sorted([s for s in opcoes_siglas if str(s) != 'nan']))
    except Exception as e:
        st.error(f"Erro ao carregar dados/modelo. Verifique a planilha: {e}")
        return

    # 2. Inicializa o estado
    if 'ensaios_sim_preco' not in st.session_state:
        st.session_state.ensaios_sim_preco = [{"id": str(uuid.uuid4()), "Sigla": st.session_state.opcoes_siglas[0], "Quantidade": 1}]
    st.session_state.setdefault('resultado_precos', None)

    # 3. Cabeçalho de Ações
    col1, col2 = st.columns([3, 1])
    with col1: st.subheader("Parâmetros do Cliente")
    with col2: st.button("🗑️ Limpar Tudo", on_click=limpar_simulacao, use_container_width=True)

    # 4. Inputs Globais do Contrato
    with st.container(border=True):
        c_cliente, c_mes = st.columns(2)
        
        opcoes_cliente = lista_clientes_bd + ["➕ Adicionar novo cliente..."]
        cliente_selecionado = c_cliente.selectbox("Cliente", options=opcoes_cliente, on_change=on_change_preco)
        
        if cliente_selecionado == "➕ Adicionar novo cliente...":
            cliente = c_cliente.text_input("Nome do novo cliente", placeholder="Digite o nome...", on_change=on_change_preco)
        else:
            cliente = cliente_selecionado
            
        mes = c_mes.selectbox("Mês Previsto", options=list(range(1, 13)), index=9, on_change=on_change_preco)

    with st.container(border=True):
        st.markdown("##### Parâmetros da Otimização")
        c_margem, c_desc = st.columns(2)
        margem_minima_input = c_margem.number_input("Margem Mínima Alvo (%)", min_value=0.0, max_value=100.0, value=40.0, step=1.0, help="Margem mínima de lucro desejada para a proposta.")
        desconto_maximo_input = c_desc.number_input("Desconto Máximo Inicial (%)", min_value=0.0, max_value=99.0, value=30.0, step=1.0, help="Define o desconto máximo autorizado como ponto de partida (ex: 50% significa que a otimização testa a partir de 50% do preço cheio).")


    st.markdown("---")
    st.subheader("Ensaios da Proposta")

    # 5. Entradas Dinâmicas para Ensaios
    with st.container(border=True):
        c1, c2, c3 = st.columns([4, 2, 1])
        c1.markdown("**Sigla do Ensaio**")
        c2.markdown("**Quantidade**")

        for idx, ensaio in enumerate(st.session_state.ensaios_sim_preco):
            if "id" not in ensaio:
                ensaio["id"] = str(uuid.uuid4())
            ensaio_id = ensaio["id"]
            col1, col2, col3 = st.columns([4, 2, 1])
            
            idx_sigla = st.session_state.opcoes_siglas.index(ensaio['Sigla']) if ensaio['Sigla'] in st.session_state.opcoes_siglas else 0
            nova_sigla = col1.selectbox("Sigla", st.session_state.opcoes_siglas, index=idx_sigla, key=f"sigla_{ensaio_id}", label_visibility="collapsed", on_change=on_change_preco)
            st.session_state.ensaios_sim_preco[idx]['Sigla'] = nova_sigla
            
            qtd_str = col2.text_input("Qtd", value=str(ensaio['Quantidade']), key=f"qtd_{ensaio_id}", label_visibility="collapsed", on_change=on_change_preco)
            st.session_state.ensaios_sim_preco[idx]['Quantidade'] = int(qtd_str) if qtd_str.isdigit() and int(qtd_str) > 0 else 1
            
            col3.button("➖", key=f"rem_{ensaio_id}", on_click=remover_ensaio, args=(idx,))

        st.button("➕ Adicionar Ensaio", on_click=adicionar_ensaio)

    # 6. Botão de Execução
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🚀 Executar Otimização de Preços", use_container_width=True, type="primary"):
        if not st.session_state.ensaios_sim_preco:
            st.warning("Adicione pelo menos um ensaio.")
            return
            
        with st.spinner("Rodando ML e Heurística Matemática..."):
            info = { 'cliente': cliente, 'mes': mes}
            func_prob = lambda inf, ens: obter_probabilidade_heuristica(
                inf,
                ens,
                modelo_heuristica,
                tabela_precos_global,
                colunas_treino,
                opcoes_siglas
            )
            
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
                    debug=False
                )
                st.session_state.resultado_precos = resultado
                st.success("✅ Otimização Concluída!")
            except Exception as e:
                st.error(f"Erro ao processar a otimização: {e}")

    # 7. Exibição de Resultados
    if st.session_state.resultado_precos is not None:
        res = st.session_state.resultado_precos
        st.markdown("---")
        st.subheader("Resultados da Otimização do Contrato")

        tab1, tab2 = st.tabs(["Cenário: Max Probabilidade & Margem", "Cenário: Margem Mínima Atingida"])
        
        def renderizar_cenario(cenario):
            if cenario is None:
                st.info("Este cenário não foi alcançado ou é idêntico ao cenário principal.")
                return

            m1, m2, m3 = st.columns(3)
            with m1:
                st.metric("🎯 Probabilidade Estimada", f"{cenario['probabilidade'] * 100:.2f}%")
            with m2:
                st.metric("💵 Receita Bruta", f"R$ {cenario['receita_bruta']:,.2f}".replace(",", "X").replace(".", ",").replace("X", "."))
            with m3:
                st.metric("💰 Receita Líquida", f"R$ {cenario['receita_liquida']:,.2f}".replace(",", "X").replace(".", ",").replace("X", "."))

            m4, m5, m6 = st.columns(3)
            with m4:
                st.metric("🧾 Custos", f"R$ -{cenario['custo_total']:,.2f}".replace(",", "X").replace(".", ",").replace("X", "."))
            with m5:
                st.metric("📊 Contribuição", f"R$ {cenario['margem_total']:,.2f}".replace(",", "X").replace(".", ",").replace("X", "."))
            with m6:
                st.metric("📈 Margem Percentual", f"{cenario['margem_percentual'] * 100:.2f}%")

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("##### 🛒 Tabela de Preços Sugeridos por Ensaio")
            
            df_resultados = pd.DataFrame(cenario['ensaios'])
            
            df_resultados = df_resultados.rename(columns={
                "Sigla": "Ensaio", 
                "Quantidade": "Qtd", 
                "Preco_Unitario": "Preço Otimizado (R$)",
                "Custo_Unitario": "Custo Unitário (R$)",
                "Receita_Bruta_Item": "Valor Total (R$)"
            })

            df_resultados["Preço Base (R$)"] = df_resultados["Ensaio"].apply(lambda x: tabela_precos_global.get(x, 0.0))
            df_resultados["Qtd"] = df_resultados["Qtd"].astype(int)
            
            def formatar_variacao(row):
                otimizado = row["Preço Otimizado (R$)"]
                base = row["Preço Base (R$)"]
                if base <= 0: return "➖"
                variacao = round(((otimizado / base) - 1) * 100, 2)
                if variacao > 0: return f"🔺 +{variacao:.1f}%"
                elif variacao < 0: return f"🔻 {variacao:.1f}%"
                else: return "➖ 0.0%"

            df_resultados["Variação Aplicada"] = df_resultados.apply(formatar_variacao, axis=1)
            
            colunas_ordem = ["Ensaio", "Qtd", "Preço Base (R$)", "Custo Unitário (R$)", "Preço Otimizado (R$)", "Variação Aplicada", "Valor Total (R$)"]
            df_resultados = df_resultados[colunas_ordem]
            
            df_formatado = df_resultados.style.format({
                "Preço Base (R$)": "{:.2f}",
                "Custo Unitário (R$)": "-{:.2f}",
                "Preço Otimizado (R$)": "{:.2f}",
                "Valor Total (R$)": "{:.2f}"
            })
            
            st.dataframe(df_formatado, use_container_width=True, hide_index=True)

        with tab1:
            renderizar_cenario(res.get("cenario_max_prob_margem"))
            
        with tab2:
            renderizar_cenario(res.get("cenario_margem_minima"))