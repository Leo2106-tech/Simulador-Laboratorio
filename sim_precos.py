# sim_precos.py

import io
import re
import math
import uuid
import warnings
from itertools import combinations

import streamlit as st
import pandas as pd
import numpy as np
import altair as alt

from sklearn.model_selection import StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import (
    confusion_matrix,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
)

from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

warnings.filterwarnings("ignore")


# ============================================================
# CONFIGURAÇÕES DO MODELO
# ============================================================

N_FOLDS = 5
RANDOM_STATE = 42
THRESHOLD_XGBOOST = 0.46702912081635545

PARAMS_XGBOOST = {
    "n_estimators": 187,
    "max_depth": 3,
    "learning_rate": 0.09680018169570574,
    "subsample": 0.8840212380369016,
    "colsample_bytree": 0.6400628095628191,
    "min_child_weight": 2,
    "gamma": 0.9946938715788189,
    "reg_alpha": 0.042879353158190935,
    "reg_lambda": 0.6815358078376077,
    "random_state": RANDOM_STATE,
    "verbosity": 0,
    "eval_metric": "logloss",
    "use_label_encoder": False,
    "n_jobs": 1,
}


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


def remover_outliers_iqr(dados_series):
    q1 = dados_series.quantile(0.25)
    q3 = dados_series.quantile(0.75)
    iqr = q3 - q1

    return dados_series[
        (dados_series >= q1 - 1.5 * iqr) &
        (dados_series <= q3 + 1.5 * iqr)
    ]


def calcular_referencia_limpa(serie):
    """
    Calcula a referência de preço como a média da faixa de preços que mais se repete.
    1. Remove nulos.
    2. Remove outliers por IQR quando há amostra suficiente.
    3. Para cada preço base, procura valores dentro de ±10%.
    4. Escolhe a faixa com maior contagem; em empate, menor variância.
    5. Retorna a média da faixa escolhida.
    """
    serie = serie.dropna().astype(float)

    if len(serie) == 0:
        return np.nan

    if len(serie) > 5:
        serie_limpa = remover_outliers_iqr(serie)
        if len(serie_limpa) > 0:
            serie = serie_limpa

    if len(serie) == 0:
        return np.nan

    tolerancia = 0.10
    melhor_contagem = 0
    melhor_media = 0
    menor_variancia = float("inf")

    for valor_base in serie:
        limite_inferior = valor_base * (1 - tolerancia)
        limite_superior = valor_base * (1 + tolerancia)

        valores_na_faixa = serie[
            (serie >= limite_inferior) &
            (serie <= limite_superior)
        ]

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


def calcular_moda_limpa(serie_precos):
    """
    Replica a lógica do Colab para a BASE DE DADOS MODELO:
    - remove outliers por IQR quando há mais de 5 observações;
    - usa a moda da série limpa;
    - se não houver moda válida, usa a média.
    """
    serie_precos = serie_precos.dropna().astype(float)

    if len(serie_precos) == 0:
        return np.nan

    if len(serie_precos) > 5:
        precos_limpos = remover_outliers_iqr(serie_precos)
        valores_para_calc = precos_limpos if not precos_limpos.empty else serie_precos
    else:
        valores_para_calc = serie_precos

    try:
        moda = valores_para_calc.mode()
        if not moda.empty:
            return float(moda.iloc[0])
        return float(valores_para_calc.mean())
    except Exception:
        return np.nan


def encontrar_coluna(df, alternativas):
    mapa = {str(c).strip().lower(): c for c in df.columns}

    for alt in alternativas:
        chave = str(alt).strip().lower()
        if chave in mapa:
            return mapa[chave]

    return None


def criar_onehot_encoder():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def criar_modelo_xgboost():
    return XGBClassifier(**PARAMS_XGBOOST)


def aplicar_smote_seguro(X_treino, y_treino):
    """
    SMOTE no mesmo padrão do Colab: k_neighbors baseado na quantidade
    de positivos do treino. Mantém proteção para bases pequenas.
    """
    if len(y_treino.value_counts()) < 2:
        return X_treino, y_treino

    positivos_treino = int(y_treino.sum())

    if positivos_treino <= 1:
        return X_treino, y_treino

    k_neighbors_ajustado = max(1, min(3, positivos_treino - 1))

    smote = SMOTE(
        random_state=RANDOM_STATE,
        k_neighbors=k_neighbors_ajustado,
    )

    return smote.fit_resample(X_treino, y_treino)


def montar_features_transformadas(preprocessor, X_fit, X_transform, colunas_cat, colunas_num):
    preprocessor_fit = preprocessor.fit(X_fit)
    Xt = preprocessor_fit.transform(X_transform)

    try:
        nomes_cat = (
            preprocessor_fit
            .named_transformers_["cat"]
            .get_feature_names_out(colunas_cat)
            .tolist()
        )
    except Exception:
        nomes_cat = []

    nomes_features = nomes_cat + colunas_num

    Xt = pd.DataFrame(
        Xt,
        columns=nomes_features,
        index=X_transform.index,
    )

    for c in Xt.columns:
        Xt[c] = pd.to_numeric(Xt[c], errors="coerce").fillna(0)

    return Xt, nomes_features, preprocessor_fit


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


def calcular_metricas_modelo(y_real, y_prob, threshold):
    y_pred = (y_prob >= threshold).astype(int)
    matriz = confusion_matrix(y_real, y_pred, labels=[0, 1])

    return {
        "threshold": threshold,
        "accuracy": accuracy_score(y_real, y_pred),
        "precision": precision_score(y_real, y_pred, zero_division=0),
        "recall": recall_score(y_real, y_pred, zero_division=0),
        "f1": f1_score(y_real, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_real, y_prob) if len(np.unique(y_real)) > 1 else np.nan,
        "tn": matriz[0][0],
        "fp": matriz[0][1],
        "fn": matriz[1][0],
        "tp": matriz[1][1],
    }


# =========================================================================
# FUNÇÕES DE DRIVE
# =========================================================================

def criar_servico_drive():
    scopes = ["https://www.googleapis.com/auth/drive.readonly"]

    creds = Credentials.from_service_account_info(
        dict(st.secrets["gcp_service_account"]),
        scopes=scopes,
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
        includeItemsFromAllDrives=True,
    ).execute()

    files = response.get("files", [])

    if not files:
        raise FileNotFoundError(
            f"Arquivo '{file_name}' não encontrado na pasta '{folder_id}'."
        )

    return files[0]


def baixar_arquivo_drive_em_memoria(service, file_id: str) -> io.BytesIO:
    request = service.files().get_media(
        fileId=file_id,
        supportsAllDrives=True,
    )

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
    """
    Gera a BASE DE DADOS MODELO replicando a construção do Colab.

    Pontos importantes para manter compatibilidade:
    - mês vem de "Mês Recebimento";
    - prazo vem de "Prazo de Execução";
    - prazos inválidos são preenchidos pela média por sigla e depois pela média geral sem CAMPO;
    - preço_referencia do modelo usa calcular_moda_limpa;
    - setor/mineral são mantidos como string, como no notebook.
    """
    col_contrato = encontrar_coluna(df_ensaios, ["Négocio", "Negócio", "Negocio"])
    col_cliente = encontrar_coluna(df_ensaios, ["Cliente"])
    col_sigla = encontrar_coluna(df_ensaios, ["Sigla"])
    col_qtd = encontrar_coluna(df_ensaios, ["Quantidade", "Qtd"])
    col_valor_unit = encontrar_coluna(df_ensaios, ["Valor Unitario", "Valor Unitário", "Valor_Unitario"])
    col_status = encontrar_coluna(df_ensaios, ["Status"])
    col_mes = encontrar_coluna(
        df_ensaios,
        [
            "Mês Recebimento",
            "Mes Recebimento",
            "Mês Ganho/Perdido",
            "Mes Ganho/Perdido",
            "Mês",
            "Mes",
        ],
    )
    col_prazo = encontrar_coluna(
        df_ensaios,
        [
            "Prazo de Execução",
            "Prazo de Execucao",
            "Prazo Execução",
            "Prazo Execucao",
            "Prazo",
            "Prazo Entrega",
            "Prazo de Entrega",
        ],
    )
    col_mineral = encontrar_coluna(df_ensaios, ["Mineral"])
    col_setor = encontrar_coluna(df_ensaios, ["Setor"])

    colunas_obrigatorias = {
        "contrato": col_contrato,
        "cliente": col_cliente,
        "sigla": col_sigla,
        "quantidade": col_qtd,
        "valor_unitario": col_valor_unit,
        "status": col_status,
        "mes": col_mes,
    }

    faltantes = [nome for nome, col in colunas_obrigatorias.items() if col is None]

    if faltantes:
        raise ValueError(
            "Colunas obrigatórias não encontradas na BASE ENSAIOS: "
            + ", ".join(faltantes)
        )

    df = df_ensaios.copy()

    if col_setor is None:
        df["_setor_modelo"] = "OUTROS"
        col_setor = "_setor_modelo"

    if col_mineral is None:
        df["_mineral_modelo"] = "OUTROS"
        col_mineral = "_mineral_modelo"

    if col_prazo is None:
        df["_prazo_modelo"] = np.nan
        col_prazo = "_prazo_modelo"

    df[col_valor_unit] = df[col_valor_unit].apply(limpar_moeda_br)
    df[col_qtd] = pd.to_numeric(df[col_qtd], errors="coerce")

    df[col_sigla] = df[col_sigla].astype(str).str.strip()
    df[col_cliente] = df[col_cliente].astype(str).str.strip()
    df[col_mes] = df[col_mes].astype(str).str.strip()
    df[col_mineral] = df[col_mineral].astype(str).str.strip()
    df[col_setor] = df[col_setor].astype(str).str.strip()

    # Limpeza e preenchimento de prazo igual ao Colab
    df[col_prazo] = df[col_prazo].apply(limpar_prazo)

    media_prazo_por_sigla = (
        df.groupby(col_sigla)[col_prazo]
        .transform(lambda x: math.ceil(x.mean()) if x.notna().any() else np.nan)
    )
    df[col_prazo] = df[col_prazo].fillna(media_prazo_por_sigla)

    media_base = df.loc[df[col_sigla] != "CAMPO", col_prazo]
    media_prazo_geral = math.ceil(media_base.mean()) if media_base.notna().any() else 0
    df[col_prazo] = df[col_prazo].fillna(media_prazo_geral)

    df[col_prazo] = df[col_prazo].apply(
        lambda x: math.ceil(x) if pd.notna(x) else 0
    ).astype(int)

    df = df.dropna(
        subset=[
            col_contrato,
            col_cliente,
            col_sigla,
            col_qtd,
            col_valor_unit,
            col_status,
            col_mes,
        ]
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

    # No Colab, a base de treino usa moda limpa como preço de referência
    df_ref_modelo = (
        df.groupby(col_sigla)[col_valor_unit]
        .apply(calcular_moda_limpa)
        .reset_index()
        .rename(columns={col_valor_unit: "preco_referencia"})
    )

    tabela_ref_modelo = dict(zip(df_ref_modelo[col_sigla], df_ref_modelo["preco_referencia"]))

    df["preco_referencia"] = df[col_sigla].map(tabela_ref_modelo)
    df["preco_relativo"] = df[col_valor_unit] / df["preco_referencia"]
    df["preco_relativo"] = (
        df["preco_relativo"]
        .clip(lower=0.1, upper=5.0)
        .fillna(1.0)
    )

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
            valor_maior_item / valor_total_contrato if valor_total_contrato > 0 else 0
        )

        prazo_medio = int(math.ceil(grupo[col_prazo].mean()))
        prazo_max = int(math.ceil(grupo[col_prazo].max()))
        prazo_min = int(math.ceil(grupo[col_prazo].min()))

        mineral = grupo[col_mineral].iloc[0]
        setor = grupo[col_setor].iloc[0]

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
            "maior_item": maior_item,
            "prazo_medio": prazo_medio,
            "prazo_max": prazo_max,
            "prazo_min": prazo_min,
            "mineral": mineral,
            "setor": setor,
        }

        valor_total_por_sigla = (
            grupo.groupby(col_sigla)["valor_total_item"]
            .sum()
            .to_dict()
        )

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
        "mineral",
        "setor",
        "mes",
        "prazo_medio",
        "prazo_max",
        "prazo_min",
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
        "maior_item",
    ]

    outras_colunas = [c for c in df_modelo.columns if c not in colunas_globais]
    df_modelo = df_modelo[colunas_globais + sorted(outras_colunas)]

    return df, df_ref_modelo, df_modelo, tabela_ref_modelo, siglas


def calcular_tabela_precos_global(df_ensaios: pd.DataFrame):
    """
    Replica a tabela de preços usada pela heurística no Colab:
    lê BASE ENSAIOS, limpa Valor Unitario, filtra valores positivos e calcula
    calcular_referencia_limpa por Sigla.
    """
    col_sigla = encontrar_coluna(df_ensaios, ["Sigla"])
    col_valor_unit = encontrar_coluna(df_ensaios, ["Valor Unitario", "Valor Unitário", "Valor_Unitario"])

    if col_sigla is None or col_valor_unit is None:
        raise ValueError("Não foi possível calcular tabela de preços: colunas Sigla/Valor Unitario ausentes.")

    df_precos = df_ensaios.copy()
    df_precos[col_valor_unit] = df_precos[col_valor_unit].apply(limpar_moeda_br)
    df_precos[col_sigla] = df_precos[col_sigla].astype(str).str.strip()

    df_ref = (
        df_precos[df_precos[col_valor_unit] > 0]
        .groupby(col_sigla)[col_valor_unit]
        .apply(calcular_referencia_limpa)
        .reset_index()
    )

    tabela_precos = dict(zip(df_ref[col_sigla], df_ref[col_valor_unit]))

    siglas_preco = sorted(
        df_precos[col_sigla]
        .dropna()
        .astype(str)
        .str.strip()
        .unique()
        .tolist()
    )

    return df_ref, tabela_precos, siglas_preco


# ============================================================
# PREPARAÇÃO E TREINO DO MODELO
# ============================================================

def preparar_dataset_ml(df_modelo: pd.DataFrame):
    df_modelo = df_modelo.dropna(subset=["target"]).copy()

    y = df_modelo["target"].astype(int)

    X_raw = df_modelo.drop(
        columns=["target", "id_contrato"],
        errors="ignore",
    ).copy()

    threshold_nulos = 0.5

    colunas_removidas_por_nulos = [
        c for c in X_raw.columns
        if X_raw[c].isnull().mean() >= threshold_nulos and c != "mineral"
    ]

    X_raw = X_raw.drop(columns=colunas_removidas_por_nulos)

    # Ordem igual ao Colab
    colunas_categoricas_base = [
        "mes",
        "setor",
        "mineral",
        "cliente",
        "maior_item",
    ]

    colunas_categoricas = [
        c for c in colunas_categoricas_base
        if c in X_raw.columns
    ]

    colunas_numericas = [
        c for c in X_raw.columns
        if c not in colunas_categoricas
    ]

    for col in colunas_categoricas:
        X_raw[col] = X_raw[col].apply(padronizar_categorica)

    for col in colunas_numericas:
        X_raw[col] = pd.to_numeric(X_raw[col], errors="coerce")

    X_raw[colunas_numericas] = X_raw[colunas_numericas].fillna(0)

    preprocessador = ColumnTransformer(
        transformers=[
            (
                "cat",
                criar_onehot_encoder(),
                colunas_categoricas,
            ),
            (
                "num",
                "passthrough",
                colunas_numericas,
            ),
        ],
        remainder="drop",
    )

    X_enc, nomes_features, preprocessador_final = montar_features_transformadas(
        preprocessador,
        X_raw,
        X_raw,
        colunas_categoricas,
        colunas_numericas,
    )

    return {
        "X_raw": X_raw,
        "X_enc": X_enc,
        "y": y,
        "preprocessador_final": preprocessador_final,
        "nomes_features": nomes_features,
        "colunas_categoricas": colunas_categoricas,
        "colunas_numericas": colunas_numericas,
        "colunas_removidas_por_nulos": colunas_removidas_por_nulos,
    }


def validar_xgboost_5_folds(X_enc, y):
    contagem_classes = y.value_counts()

    if len(contagem_classes) < 2 or contagem_classes.min() < N_FOLDS:
        return {
            "metricas_globais": {},
            "df_resultados_folds": pd.DataFrame(),
            "df_importancias": pd.DataFrame(columns=["feature", "valor"]),
            "y_real_total": np.array([]),
            "y_prob_total": np.array([]),
        }

    cv = StratifiedKFold(
        n_splits=N_FOLDS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    y_real_total = []
    y_prob_total = []
    resultados_folds = []
    importancias_folds = []

    for fold, (train_idx, val_idx) in enumerate(cv.split(X_enc, y), start=1):
        X_train = X_enc.iloc[train_idx].copy()
        X_val = X_enc.iloc[val_idx].copy()

        y_train = y.iloc[train_idx].copy()
        y_val = y.iloc[val_idx].copy()

        X_train_bal, y_train_bal = aplicar_smote_seguro(X_train, y_train)

        modelo_fold = criar_modelo_xgboost()
        modelo_fold.fit(X_train_bal, y_train_bal)

        y_prob = modelo_fold.predict_proba(X_val)[:, 1]

        y_real_total.extend(y_val.tolist())
        y_prob_total.extend(y_prob.tolist())

        auc_fold = roc_auc_score(y_val, y_prob) if len(np.unique(y_val)) > 1 else np.nan

        y_pred_fold = (y_prob >= THRESHOLD_XGBOOST).astype(int)
        matriz_fold = confusion_matrix(y_val, y_pred_fold, labels=[0, 1])

        resultados_folds.append({
            "fold": fold,
            "roc_auc": auc_fold,
            "accuracy": accuracy_score(y_val, y_pred_fold),
            "precision": precision_score(y_val, y_pred_fold, zero_division=0),
            "recall": recall_score(y_val, y_pred_fold, zero_division=0),
            "f1": f1_score(y_val, y_pred_fold, zero_division=0),
            "tn": matriz_fold[0][0],
            "fp": matriz_fold[0][1],
            "fn": matriz_fold[1][0],
            "tp": matriz_fold[1][1],
        })

        importancias_folds.append(
            pd.DataFrame({
                "feature": X_enc.columns,
                "valor": modelo_fold.feature_importances_,
            })
        )

    y_real_total = np.array(y_real_total)
    y_prob_total = np.array(y_prob_total)

    metricas_globais = calcular_metricas_modelo(
        y_real_total,
        y_prob_total,
        THRESHOLD_XGBOOST,
    )

    df_resultados_folds = pd.DataFrame(resultados_folds)

    if len(importancias_folds) > 0:
        df_importancias = (
            pd.concat(importancias_folds, ignore_index=True)
            .groupby("feature", as_index=False)["valor"]
            .mean()
            .sort_values("valor", ascending=False)
        )
    else:
        df_importancias = pd.DataFrame(columns=["feature", "valor"])

    return {
        "metricas_globais": metricas_globais,
        "df_resultados_folds": df_resultados_folds,
        "df_importancias": df_importancias,
        "y_real_total": y_real_total,
        "y_prob_total": y_prob_total,
    }


def treinar_xgboost_final(X_enc, y):
    X_bal, y_bal = aplicar_smote_seguro(X_enc, y)

    modelo_final = criar_modelo_xgboost()
    modelo_final.fit(X_bal, y_bal)

    return modelo_final


# ============================================================
# FUNÇÃO PRINCIPAL CACHEADA
# ============================================================

@st.cache_resource(show_spinner="Carregando base e treinando modelo XGBoost...")
def inicializar_modelo_e_dados():
    service = criar_servico_drive()

    folder_id = st.secrets["app_config"]["id_pasta_drive"]
    nome_arquivo = st.secrets["app_config"]["nome_arq_xls"]
    aba_ensaios = st.secrets["app_config"]["nome_aba_ensaios"]

    arquivo = buscar_arquivo_na_pasta(service, folder_id, nome_arquivo)
    conteudo_excel = baixar_arquivo_drive_em_memoria(service, arquivo["id"])

    df_ensaios = pd.read_excel(
        conteudo_excel,
        sheet_name=aba_ensaios,
    )

    df_ensaios_tratado, df_ref_modelo, df_modelo, tabela_ref_modelo, siglas_modelo = gerar_base_modelo(
        df_ensaios
    )

    # Tabela de preços da heurística separada, como no Colab.
    df_ref_precos, tabela_precos_global, siglas_precos = calcular_tabela_precos_global(df_ensaios)

    # Para montar as colunas por sigla do modelo, usamos as siglas da base modelo.
    # Para a UI/lista, usamos as siglas disponíveis na tabela histórica.
    siglas = siglas_modelo

    dados_preparo = preparar_dataset_ml(df_modelo)

    X_enc = dados_preparo["X_enc"]
    y = dados_preparo["y"]

    resultado_cv = validar_xgboost_5_folds(X_enc, y)
    modelo_xgboost_final = treinar_xgboost_final(X_enc, y)

    lista_clientes_bd = sorted(
        [
            str(c).strip()
            for c in df_modelo["cliente"].dropna().unique()
            if str(c).strip() != ""
        ]
    )

    lista_setores_bd = sorted(
        [
            str(c).strip()
            for c in df_modelo["setor"].dropna().unique()
            if str(c).strip() != ""
        ]
    )

    lista_minerais_bd = sorted(
        [
            str(c).strip()
            for c in df_modelo["mineral"].dropna().unique()
            if str(c).strip() != ""
        ]
    )

    return {
        "modelo_heuristica": modelo_xgboost_final,
        "threshold_heuristica": THRESHOLD_XGBOOST,
        "tabela_precos_global": tabela_precos_global,
        "siglas": siglas,
        "siglas_precos": siglas_precos,
        "lista_clientes_bd": lista_clientes_bd,
        "lista_setores_bd": lista_setores_bd,
        "lista_minerais_bd": lista_minerais_bd,
        "df_modelo": df_modelo,
        "df_ref": df_ref_precos,
        "df_ref_modelo": df_ref_modelo,
        "df_ensaios_tratado": df_ensaios_tratado,
        "preprocessador_output_final": dados_preparo["preprocessador_final"],
        "features_output_final": dados_preparo["nomes_features"],
        "colunas_categoricas_output_final": dados_preparo["colunas_categoricas"],
        "colunas_numericas_output_final": dados_preparo["colunas_numericas"],
        "colunas_removidas_por_nulos_output_final": dados_preparo["colunas_removidas_por_nulos"],
        "df_resultados_folds": resultado_cv["df_resultados_folds"],
        "metricas_cv_global": resultado_cv["metricas_globais"],
        "df_importancias_xgb": resultado_cv["df_importancias"],
        "arquivo_drive_usado": arquivo["name"],
    }


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
        custo_unitario = float(tabela_custos_global.get(sigla, 0))

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
        custo_unit = float(tabela_custos_global.get(sigla, 0))

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

        st.session_state.setdefault(
            "opcoes_siglas",
            sorted([s for s in opcoes_siglas if str(s) != "nan"]),
        )

    except Exception as e:
        st.error(f"Erro ao carregar dados/modelo. Verifique a planilha: {e}")
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
                if res_margem_min and res_otimizado["margem_percentual"] > res_margem_min["margem_percentual"] + 0.0001: # Adiciona pequena tolerância para float
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
