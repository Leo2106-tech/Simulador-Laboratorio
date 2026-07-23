"""Conexao do Streamlit Cloud com as planilhas do Google.

As tres planilhas de entrada sao exportadas temporariamente como XLSX no
servidor do Streamlit. Isso permite reaproveitar toda a validacao existente em
``dados_ferias_cto.py`` sem depender de arquivos presentes no computador do usuario.

Os resultados sao gravados diretamente nas abas da planilha Google configurada
como ``planilhas.resultados``.
"""

from __future__ import annotations

import json
import math
import tempfile
from datetime import date, datetime
from pathlib import Path
from urllib.parse import quote

import pandas as pd
import streamlit as st
from google.auth.transport.requests import AuthorizedSession
from google.oauth2.service_account import Credentials


SCOPES = [
    "https://www.googleapis.com/auth/drive.readonly",
    "https://www.googleapis.com/auth/spreadsheets",
]
MIME_XLSX = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
MIME_GOOGLE_SHEETS = "application/vnd.google-apps.spreadsheet"

# Nomes exatos que dados_ferias_cto.py procura na pasta temporaria.
NOME_ALOCACAO = "Alocação Atualizada.xlsx"
NOME_FERIAS = "Controle de Férias LAB_CTO.xlsx"
NOME_FLEXIBILIDADE = "Flexibilidade Operacional CTO.xlsx"


def _informacoes_service_account():
    """Le as credenciais, aceitando o formato novo e o modelo antigo."""
    try:
        if "google_credentials_json" in st.secrets:
            conteudo = st.secrets["google_credentials_json"]
            info = json.loads(str(conteudo))
        elif "gcp_service_account" in st.secrets:
            info = dict(st.secrets["gcp_service_account"])
        else:
            raise KeyError("google_credentials_json")
    except (KeyError, TypeError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            "As credenciais do Google nao foram configuradas corretamente em "
            "Advanced settings > Secrets. Cole o JSON completo no campo "
            "google_credentials_json."
        ) from exc

    if not info.get("client_email") or not info.get("private_key"):
        raise RuntimeError(
            "O JSON configurado nos Secrets nao contem client_email e private_key."
        )
    return info


def _credenciais():
    return Credentials.from_service_account_info(
        _informacoes_service_account(), scopes=SCOPES
    )


def _sessao():
    return AuthorizedSession(_credenciais())


def _id_planilha(nome):
    aliases_antigos = {
        "alocacao": "alocacao_id",
        "controle_ferias": "controle_ferias_id",
    }
    try:
        planilhas = st.secrets["planilhas"]
        valor = planilhas.get(nome)
        if not valor and nome in aliases_antigos:
            valor = planilhas.get(aliases_antigos[nome])
    except Exception as exc:
        raise RuntimeError(
            "Falta o bloco [planilhas] nos Secrets do Streamlit."
        ) from exc
    if not valor:
        raise RuntimeError(
            f"Falta o ID '{nome}' no bloco [planilhas] dos Secrets do Streamlit."
        )
    return str(valor).strip()


def _mensagem_erro_google(resp, planilha_id, operacao):
    if resp.status_code == 403:
        email = _informacoes_service_account().get("client_email", "conta de servico")
        return RuntimeError(
            f"Acesso negado ao executar '{operacao}' na planilha {planilha_id}. "
            f"Compartilhe a planilha com {email} e confira a permissao."
        )
    if resp.status_code == 404:
        return RuntimeError(
            f"Planilha {planilha_id} nao encontrada. Confira o ID em [planilhas]."
        )
    try:
        detalhe = resp.json().get("error", {}).get("message", resp.text)
    except Exception:
        detalhe = resp.text
    return RuntimeError(f"Falha no Google ao executar '{operacao}': {detalhe}")


def _verificar_resposta(resp, planilha_id, operacao):
    if not resp.ok:
        raise _mensagem_erro_google(resp, planilha_id, operacao)
    return resp


def _baixar_arquivo_xlsx(session, file_id, destino):
    """Baixa tanto uma planilha Google quanto um arquivo Excel do Drive."""
    url_arquivo = f"https://www.googleapis.com/drive/v3/files/{file_id}"

    resposta_metadados = session.get(
        url_arquivo,
        params={"fields": "mimeType", "supportsAllDrives": "true"},
    )
    _verificar_resposta(
        resposta_metadados,
        file_id,
        "identificar o tipo do arquivo de entrada",
    )
    mime_type = resposta_metadados.json().get("mimeType")

    if mime_type == MIME_GOOGLE_SHEETS:
        resposta_arquivo = session.get(
            f"{url_arquivo}/export",
            params={"mimeType": MIME_XLSX},
        )
        operacao = "exportar planilha Google como XLSX"
    else:
        resposta_arquivo = session.get(
            url_arquivo,
            params={"alt": "media", "supportsAllDrives": "true"},
        )
        operacao = "baixar arquivo XLSX"

    _verificar_resposta(resposta_arquivo, file_id, operacao)
    Path(destino).write_bytes(resposta_arquivo.content)


@st.cache_data(show_spinner="Carregando planilhas do Google...")
def baixar_planilhas():
    """Baixa as tres entradas para uma pasta temporaria do servidor."""
    pasta = Path(tempfile.mkdtemp(prefix="ferias_cto_"))
    session = _sessao()
    arquivos = {
        "alocacao": pasta / NOME_ALOCACAO,
        "controle_ferias": pasta / NOME_FERIAS,
        "flexibilidade_operacional": pasta / NOME_FLEXIBILIDADE,
    }
    for nome, destino in arquivos.items():
        _baixar_arquivo_xlsx(session, _id_planilha(nome), destino)
    return str(pasta)


def _valor_google(valor):
    if valor is None:
        return ""
    if isinstance(valor, (datetime, date, pd.Timestamp)):
        return valor.strftime("%d/%m/%Y")
    if hasattr(valor, "item"):
        try:
            valor = valor.item()
        except Exception:
            pass
    if isinstance(valor, float) and (math.isnan(valor) or math.isinf(valor)):
        return ""
    try:
        if pd.isna(valor):
            return ""
    except (TypeError, ValueError):
        pass
    return valor if isinstance(valor, (str, int, float, bool)) else str(valor)


def _valores_dataframe(df):
    cabecalho = [_valor_google(coluna) for coluna in df.columns]
    linhas = [
        [_valor_google(valor) for valor in linha]
        for linha in df.itertuples(index=False, name=None)
    ]
    return [cabecalho] + linhas


def _metadados_abas(session, planilha_id):
    url = f"https://sheets.googleapis.com/v4/spreadsheets/{planilha_id}"
    resp = session.get(url, params={"fields": "sheets.properties"})
    _verificar_resposta(resp, planilha_id, "ler abas de resultados")
    return {
        item["properties"]["title"]: item["properties"]
        for item in resp.json().get("sheets", [])
    }


def _batch_update(session, planilha_id, requests, operacao):
    if not requests:
        return
    url = f"https://sheets.googleapis.com/v4/spreadsheets/{planilha_id}:batchUpdate"
    resp = session.post(url, json={"requests": requests})
    _verificar_resposta(resp, planilha_id, operacao)


def _garantir_aba(session, planilha_id, nome_aba, linhas, colunas, metadados):
    propriedades = metadados.get(nome_aba)
    if propriedades is None:
        _batch_update(
            session,
            planilha_id,
            [{"addSheet": {"properties": {
                "title": nome_aba,
                "gridProperties": {
                    "rowCount": max(100, linhas),
                    "columnCount": max(20, colunas),
                },
            }}}],
            f"criar aba {nome_aba}",
        )
        propriedades = _metadados_abas(session, planilha_id)[nome_aba]
        metadados[nome_aba] = propriedades
    else:
        grid = propriedades.get("gridProperties", {})
        nova_linhas = max(int(grid.get("rowCount", 0)), linhas, 100)
        nova_colunas = max(int(grid.get("columnCount", 0)), colunas, 20)
        if nova_linhas != grid.get("rowCount") or nova_colunas != grid.get("columnCount"):
            _batch_update(
                session,
                planilha_id,
                [{"updateSheetProperties": {
                    "properties": {
                        "sheetId": propriedades["sheetId"],
                        "gridProperties": {
                            "rowCount": nova_linhas,
                            "columnCount": nova_colunas,
                        },
                    },
                    "fields": "gridProperties.rowCount,gridProperties.columnCount",
                }}],
                f"dimensionar aba {nome_aba}",
            )
    return propriedades


def _gravar_valores(session, planilha_id, nome_aba, valores):
    intervalo_aba = quote(f"'{nome_aba}'", safe="")
    url_clear = (
        f"https://sheets.googleapis.com/v4/spreadsheets/{planilha_id}"
        f"/values/{intervalo_aba}:clear"
    )
    resp = session.post(url_clear, json={})
    _verificar_resposta(resp, planilha_id, f"limpar aba {nome_aba}")
    if not valores or not valores[0]:
        return

    # Divide tabelas grandes para manter requisicoes leves.
    tamanho_lote = 3000
    for inicio in range(0, len(valores), tamanho_lote):
        lote = valores[inicio: inicio + tamanho_lote]
        linha_inicial = inicio + 1
        intervalo = quote(f"'{nome_aba}'!A{linha_inicial}", safe="")
        url = (
            f"https://sheets.googleapis.com/v4/spreadsheets/{planilha_id}"
            f"/values/{intervalo}"
        )
        resp = session.put(
            url,
            params={"valueInputOption": "USER_ENTERED"},
            json={"majorDimension": "ROWS", "values": lote},
        )
        _verificar_resposta(resp, planilha_id, f"gravar aba {nome_aba}")


def _formatar_aba(session, planilha_id, propriedades, df):
    sheet_id = propriedades["sheetId"]
    total_colunas = max(1, len(df.columns))
    total_linhas = max(1, len(df) + 1)
    requests = [
        {"updateSheetProperties": {
            "properties": {
                "sheetId": sheet_id,
                "gridProperties": {"frozenRowCount": 1},
            },
            "fields": "gridProperties.frozenRowCount",
        }},
        {"repeatCell": {
            "range": {
                "sheetId": sheet_id,
                "startRowIndex": 0,
                "endRowIndex": 1,
                "startColumnIndex": 0,
                "endColumnIndex": total_colunas,
            },
            "cell": {"userEnteredFormat": {
                "backgroundColor": {"red": 0.24, "green": 0.32, "blue": 0.39},
                "textFormat": {"bold": True, "foregroundColor": {"red": 1, "green": 1, "blue": 1}},
                "verticalAlignment": "MIDDLE",
            }},
            "fields": "userEnteredFormat(backgroundColor,textFormat,verticalAlignment)",
        }},
        {"setBasicFilter": {"filter": {"range": {
            "sheetId": sheet_id,
            "startRowIndex": 0,
            "endRowIndex": total_linhas,
            "startColumnIndex": 0,
            "endColumnIndex": total_colunas,
        }}}},
        {"autoResizeDimensions": {"dimensions": {
            "sheetId": sheet_id,
            "dimension": "COLUMNS",
            "startIndex": 0,
            "endIndex": total_colunas,
        }}},
    ]

    cabecalhos_monetarios = {
        "Receita coberta", "Salários fixos (IA/IE)", "Contratação de suplentes",
        "Adicional noturno", "Transporte por mudança", "Custo total",
        "Receita não gerada", "Custo de transporte", "Custo de mobilização",
        "Custo total da rota",
    }
    for indice, coluna in enumerate(df.columns):
        if str(coluna) in cabecalhos_monetarios:
            requests.append({"repeatCell": {
                "range": {
                    "sheetId": sheet_id,
                    "startRowIndex": 1,
                    "endRowIndex": total_linhas,
                    "startColumnIndex": indice,
                    "endColumnIndex": indice + 1,
                },
                "cell": {"userEnteredFormat": {"numberFormat": {
                    "type": "CURRENCY", "pattern": "\"R$\" #,##0;-\"R$\" #,##0",
                }}},
                "fields": "userEnteredFormat.numberFormat",
            }})

    if list(df.columns[:2]) == ["Indicador", "Valor"]:
        monetarios = {
            "Custo decisório", "Receita potencial total", "Receita não gerada",
            "Receita dos postos cobertos", "Contratação IS", "Adicional noturno",
            "Transporte por rota", "Mobilização", "Receita total gerada",
        }
        for linha, indicador in enumerate(df.get("Indicador", []), start=1):
            if indicador in monetarios:
                requests.append({"repeatCell": {
                    "range": {
                        "sheetId": sheet_id,
                        "startRowIndex": linha,
                        "endRowIndex": linha + 1,
                        "startColumnIndex": 1,
                        "endColumnIndex": 2,
                    },
                    "cell": {"userEnteredFormat": {"numberFormat": {
                        "type": "CURRENCY", "pattern": "\"R$\" #,##0;-\"R$\" #,##0",
                    }}},
                    "fields": "userEnteredFormat.numberFormat",
                }})

    if "Quem cobre e períodos" in df.columns:
        indice = list(df.columns).index("Quem cobre e períodos")
        requests.extend([
            {"repeatCell": {
                "range": {
                    "sheetId": sheet_id,
                    "startRowIndex": 1,
                    "endRowIndex": total_linhas,
                    "startColumnIndex": indice,
                    "endColumnIndex": indice + 1,
                },
                "cell": {"userEnteredFormat": {
                    "wrapStrategy": "WRAP", "verticalAlignment": "TOP",
                }},
                "fields": "userEnteredFormat(wrapStrategy,verticalAlignment)",
            }},
            {"updateDimensionProperties": {
                "range": {
                    "sheetId": sheet_id,
                    "dimension": "COLUMNS",
                    "startIndex": indice,
                    "endIndex": indice + 1,
                },
                "properties": {"pixelSize": 500},
                "fields": "pixelSize",
            }},
        ])
    _batch_update(session, planilha_id, requests, "formatar resultados")


def salvar_abas_resultados(abas):
    """Substitui o conteudo das abas informadas na planilha Resultados."""
    planilha_id = _id_planilha("resultados")
    session = _sessao()
    metadados = _metadados_abas(session, planilha_id)
    for nome_aba, df in abas.items():
        if not isinstance(df, pd.DataFrame):
            df = pd.DataFrame(df)
        valores = _valores_dataframe(df)
        propriedades = _garantir_aba(
            session,
            planilha_id,
            nome_aba,
            max(2, len(valores)),
            max(1, len(df.columns)),
            metadados,
        )
        _gravar_valores(session, planilha_id, nome_aba, valores)
        _formatar_aba(session, planilha_id, propriedades, df)


def ler_aba_resultados(nome_aba):
    """Le uma aba da planilha Resultados e devolve um DataFrame."""
    planilha_id = _id_planilha("resultados")
    session = _sessao()
    intervalo = quote(f"'{nome_aba}'", safe="")
    url = (
        f"https://sheets.googleapis.com/v4/spreadsheets/{planilha_id}"
        f"/values/{intervalo}"
    )
    resp = session.get(url, params={"valueRenderOption": "UNFORMATTED_VALUE"})
    if resp.status_code == 400 and "Unable to parse range" in resp.text:
        return None
    _verificar_resposta(resp, planilha_id, f"ler aba {nome_aba}")
    valores = resp.json().get("values", [])
    if not valores:
        return pd.DataFrame()
    cabecalho = valores[0]
    largura = len(cabecalho)
    linhas = [linha + [""] * (largura - len(linha)) for linha in valores[1:]]
    return pd.DataFrame([linha[:largura] for linha in linhas], columns=cabecalho)


def limpar_cache():
    """Forca novo download das entradas na proxima chamada."""
    baixar_planilhas.clear()
