from __future__ import annotations

"""Carregamento e validação dos dados do agendador de férias do CTO."""

import math
import time
import unicodedata
from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR

class ErroValidacaoDados(ValueError):
    """Erro de cadastro que deve ser exibido ao usuario antes de rodar o modelo."""


def calcular_matriz_distancias(cidades_funcionarios, cidades_projetos, aba_distancias="Distancias"):
    """
    Calcula distancias entre cidades usando a aba "Distancias" da planilha de Alocacao.
    So consulta o OpenStreetMap para pares ausentes na aba.
    """
    arq_alocacao = PROJECT_DIR / "Alocação Atualizada.xlsx"

    distancias_conhecidas = {}
    try:
        df_cache = pd.read_excel(arq_alocacao, sheet_name=aba_distancias)
        for _, row in df_cache.iterrows():
            distancias_conhecidas[(normalizar_cidade(row["cidade_func"]), normalizar_cidade(row["cidade_proj"]))] = row["distancia_km"]
        print(f"Distancias carregadas da aba '{aba_distancias}': {len(distancias_conhecidas)} pares")
    except Exception as exc:
        print(f"Aviso: nao foi possivel ler a aba '{aba_distancias}' da planilha de Alocacao ({exc}).")

    todas_cidades_func = {
        normalizar_cidade(c)
        for c in cidades_funcionarios
        if pd.notna(c) and str(c).strip() != ""
    }
    todas_cidades_proj = {
        normalizar_cidade(c)
        for c in cidades_projetos
        if pd.notna(c) and str(c).strip() != ""
    }

    # Preserva a grafia original para apresentar mensagens claras ao usuario.
    nomes_originais = {}
    for cidade in list(cidades_funcionarios) + list(cidades_projetos):
        if pd.notna(cidade) and str(cidade).strip():
            nomes_originais.setdefault(normalizar_cidade(cidade), str(cidade).strip())

    pares_faltando = [
        (cf, cp)
        for cf in todas_cidades_func
        for cp in todas_cidades_proj
        if (cf, cp) not in distancias_conhecidas
    ]

    if pares_faltando:
        try:
            from geopy.distance import geodesic
            from geopy.geocoders import Nominatim
        except ModuleNotFoundError as exc:
            raise SystemExit(
                "Faltam pares no cache de distancias e o pacote geopy nao esta instalado. "
                "Instale geopy ou complete a aba 'Distancias' da planilha de Alocacao antes de rodar."
            ) from exc

        print(f"Cidades novas detectadas: consultando API para {len(pares_faltando)} pares...")

        cidades_novas = set()
        for cf, cp in pares_faltando:
            # Nao e necessario geocodificar quando origem e destino sao iguais.
            if cf != cp:
                cidades_novas.add(cf)
                cidades_novas.add(cp)

        geolocator = Nominatim(user_agent="modelo_alocacao_ferias_sem_keys")
        coordenadas = {}
        for cidade_nome in cidades_novas:
            try:
                local = geolocator.geocode(f"{cidade_nome}, Brasil", timeout=10)
                coordenadas[cidade_nome] = (local.latitude, local.longitude) if local else None
                time.sleep(1)
            except Exception:
                coordenadas[cidade_nome] = None

        cidades_nao_localizadas = sorted(
            nomes_originais.get(cidade, cidade)
            for cidade, coordenada in coordenadas.items()
            if coordenada is None
        )
        if cidades_nao_localizadas:
            raise ErroValidacaoDados(
                "Não foi possível localizar as seguintes cidades: "
                + ", ".join(cidades_nao_localizadas)
                + ". Verifique na planilha se elas estão escritas corretamente e tente novamente."
            )

        novas_linhas = []
        for cf, cp in pares_faltando:
            if cf == cp:
                dist = 0.0
            else:
                dist = geodesic(coordenadas[cf], coordenadas[cp]).km

            distancias_conhecidas[(cf, cp)] = dist
            novas_linhas.append({"cidade_func": cf, "cidade_proj": cp, "distancia_km": dist})

        # A fonte de distancias agora e a aba "Distancias" da planilha de Alocacao.
        # Nao gravamos mais um arquivo separado: apenas avisamos quais pares novos
        # foram calculados via API, para que sejam adicionados a aba manualmente.
        print(
            f"ATENCAO: {len(novas_linhas)} par(es) de cidades nao estavam na aba "
            f"'Distancias' e foram calculados via API nesta execucao. "
            "Adicione-os a aba para evitar recalculo:"
        )
        for linha in novas_linhas:
            print(f"  {linha['cidade_func']} -> {linha['cidade_proj']}: {linha['distancia_km']:.2f} km")

    return distancias_conhecidas


def normalizar_sim_nao(valor):
    txt = str(valor).strip().lower()
    if txt in {"sim", "s", "yes", "y", "1", "true", "verdadeiro"}:
        return 1
    if txt in {"nao", "não", "n", "no", "0", "false", "falso"}:
        return 0
    return 1


def remover_acentos(texto):
    return unicodedata.normalize("NFKD", texto).encode("ascii", errors="ignore").decode("ascii")


CARGOS_DETALHADOS_VALIDOS = {
    "auxiliar de laboratorio",
    "auxiliar de servicos gerais",
    "laboratorista junior",
    "laboratorista pleno",
    "laboratorista senior",
    "laboratorista",
    "tecnico em geotecnia pleno",
    "tecnico em geotecnia junior",
    "tecnico em geotecnia senior",
}


def normalizar_cargo_detalhado(valor):
    return remover_acentos(str(valor).strip().lower())


def normalizar_cidade(c):
    return remover_acentos(str(c).strip().lower())


def grupo_cargo(cargo):
    txt = remover_acentos(str(cargo).strip().lower())
    if "tecnico" in txt:
        return "tecnico"
    if "laboratorista" in txt:
        return "laboratorista"
    if "auxiliar" in txt:
        return "auxiliar"
    return txt


def normalizar_matricula(valor):
    if pd.isna(valor):
        return ""
    texto = str(valor).strip()
    if texto.endswith(".0"):
        texto = texto[:-2]
    return texto


def carregar_solicitacoes_ferias_pendentes():
    """Carrega solicitacoes ainda nao aprovadas para selecao no Streamlit."""
    arq_ferias = PROJECT_DIR / "Controle de Férias LAB_CTO.xlsx"
    arq_alocacao = PROJECT_DIR / "Alocação Atualizada.xlsx"
    df = pd.read_excel(
        arq_ferias,
        sheet_name="Respostas ao formulário",
        usecols="B:E,H",
    )
    df_cargos = pd.read_excel(
        arq_alocacao,
        sheet_name="Alocação",
        usecols="A,D",
    )
    df_cargos.columns = ["matricula", "cargo"]
    df_cargos["matricula"] = df_cargos["matricula"].apply(normalizar_matricula)
    df_cargos["cargo_norm"] = df_cargos["cargo"].map(normalizar_cargo_detalhado)
    matriculas_cargos_validos = set(
        df_cargos.loc[
            df_cargos["cargo_norm"].isin(CARGOS_DETALHADOS_VALIDOS),
            "matricula",
        ]
    )
    df.columns = ["nome", "matricula", "inicio", "fim", "aprovacao"]
    df["linha_planilha"] = df.index + 2
    df["matricula"] = df["matricula"].apply(normalizar_matricula)
    df["nome"] = df["nome"].fillna("").astype(str).str.strip()
    df["inicio"] = pd.to_datetime(df["inicio"], errors="coerce", format="mixed", dayfirst=True)
    df["fim"] = pd.to_datetime(df["fim"], errors="coerce", format="mixed", dayfirst=True)

    aprovados = {"sim", "s", "aprovado", "aprovada", "yes", "y", "1", "true", "verdadeiro"}
    df["aprovacao_norm"] = (
        df["aprovacao"]
        .fillna("")
        .astype(str)
        .map(lambda valor: remover_acentos(valor.strip().lower()))
    )
    pendentes = df[~df["aprovacao_norm"].isin(aprovados)].copy()
    pendentes = pendentes[
        (pendentes["matricula"] != "")
        | pendentes["inicio"].notna()
        | pendentes["fim"].notna()
    ].copy()

    invalidas = pendentes[
        (pendentes["matricula"] == "")
        | pendentes["inicio"].isna()
        | pendentes["fim"].isna()
        | (pendentes["fim"] < pendentes["inicio"])
    ].copy()
    consideracoes = []
    for _, row in invalidas.iterrows():
        motivos = []
        if row["matricula"] == "":
            motivos.append("matrícula não informada")
        if pd.isna(row["inicio"]):
            motivos.append("data inicial ausente ou inválida")
        if pd.isna(row["fim"]):
            motivos.append("data final ausente ou inválida")
        if pd.notna(row["inicio"]) and pd.notna(row["fim"]) and row["fim"] < row["inicio"]:
            motivos.append("data final anterior à data inicial")
        consideracoes.append(
            f"Linha {int(row['linha_planilha'])} da aba 'Respostas ao formulário' ignorada: "
            + "; ".join(motivos)
            + "."
        )

    pendentes = pendentes.drop(index=invalidas.index)

    # A lista da simulacao mostra apenas pedidos futuros e dos mesmos cargos
    # considerados pelo modelo. Exclusoes por estes filtros sao silenciosas.
    hoje = (
        pd.Timestamp.now(tz="America/Sao_Paulo")
        .normalize()
        .tz_localize(None)
    )
    pendentes = pendentes[
        (pendentes["inicio"] >= hoje)
        & pendentes["matricula"].isin(matriculas_cargos_validos)
    ].copy()

    duplicadas = pendentes.duplicated(
        subset=["matricula", "inicio", "fim"], keep="first"
    )
    for _, row in pendentes[duplicadas].iterrows():
        consideracoes.append(
            f"Linha {int(row['linha_planilha'])} da aba 'Respostas ao formulário' ignorada: "
            "solicitação duplicada para a mesma matrícula e período."
        )

    pendentes = pendentes[~duplicadas].sort_values(["inicio", "matricula"])

    solicitacoes = []
    for _, row in pendentes.iterrows():
        inicio = row["inicio"].normalize()
        fim = row["fim"].normalize()
        solicitacao_id = (
            f"{row['matricula']}|{inicio:%Y%m%d}|{fim:%Y%m%d}|{int(row['linha_planilha'])}"
        )
        solicitacoes.append({
            "id": solicitacao_id,
            "nome": row["nome"] or f"Matrícula {row['matricula']}",
            "matricula": row["matricula"],
            "inicio": inicio.strftime("%Y-%m-%d"),
            "fim": fim.strftime("%Y-%m-%d"),
        })
    return {"solicitacoes": solicitacoes, "consideracoes": consideracoes}

def normalizar_nome(valor):
    if pd.isna(valor):
        return ""
    txt = remover_acentos(str(valor).strip().lower())
    txt = " ".join(txt.split())
    return txt


def normalizar_projeto(valor):
    if pd.isna(valor):
        return ""

    txt = remover_acentos(str(valor).strip().lower())

    # Ignora variações de hífen/travessão no nome do projeto.
    # Ex.: "5930 - VALE AGUA LIMPA", "5930-VALE AGUA LIMPA"
    # e "5930 VALE AGUA LIMPA" viram a mesma coisa.
    for sep in ["-", "–", "—", "−"]:
        txt = txt.replace(sep, " ")

    txt = " ".join(txt.split())
    return txt


def localizar_coluna(df, candidatos):
    """
    Encontra uma coluna no DataFrame ignorando acento, maiúsculas/minúsculas
    e espaços duplicados.
    """
    mapa = {normalizar_nome(c): c for c in df.columns}
    for candidato in candidatos:
        chave = normalizar_nome(candidato)
        if chave in mapa:
            return mapa[chave]
    return None

def classificar_turno(valor):
    txt = str(valor).strip().lower()
    if txt in {"a", "b", "adm", "nan", ""}:
        return "diurno"
    if txt in {"c", "d", "adm noturno"}:
        return "noturno"
    return "diurno"


def _criar_suplentes_potenciais():
    cidades_principais = [
        "Itabira",
        "Belo Horizonte",
        "Araxa",
        "Barao de Cocais",
        "Nova Lima",
        "Congonhas",
        "Rio Piracicaba",
    ]
    cargos_suplentes = ["tecnico", "laboratorista", "auxiliar"]

    I_S = []
    cargo_suplente = {}
    cidade_suplente = {}

    contador = 1
    for cidade in cidades_principais:
        for cargo in cargos_suplentes:
            cod = f"S{contador}"
            I_S.append(cod)
            cargo_suplente[cod] = cargo
            cidade_suplente[cod] = cidade
            contador += 1

    return I_S, cargo_suplente, cidade_suplente


def carregar_dados(solicitacoes_aprovadas_teste=None):
    """
    Le as planilhas de entrada e monta os conjuntos/parametros da formulacao.
    Nesta release os conjuntos seguem o texto: I_A, I_E, I_S e I.
    """
    arq_alocacao = PROJECT_DIR / "Alocação Atualizada.xlsx"
    arq_ferias = PROJECT_DIR / "Controle de Férias LAB_CTO.xlsx"
    arq_flexibilidade = PROJECT_DIR / "Flexibilidade Operacional CTO.xlsx"

    # Para voltar ao horizonte movel do dia corrente, use:
    # data_inicio = pd.Timestamp.now(tz="America/Sao_Paulo").normalize().tz_localize(None)
    data_inicio = pd.Timestamp("2026-07-16")
    data_fim = data_inicio + pd.Timedelta(days=364)
    datas = pd.date_range(data_inicio, data_fim, freq="D")

    T = list(range(1, len(datas) + 1))
    data_para_t = {data.normalize(): idx + 1 for idx, data in enumerate(datas)}
    t_para_data = {idx + 1: data.normalize() for idx, data in enumerate(datas)}

    df_aloc = pd.read_excel(arq_alocacao, sheet_name="Alocação", usecols="A:B,D,E,G,H,P,T")
    df_aloc.columns = [
        "matricula",
        "nome_alocacao",
        "cargo",
        "projeto_original",
        "turno_original",
        "tipo",
        "cidade",
        "data_demissao",
    ]

    df_localidade = pd.read_excel(arq_alocacao, sheet_name="Localidade", usecols="A:D")
    df_localidade.columns = ["projeto", "cidade_projeto", "inicio_projeto", "fim_projeto"]

    df_disp = pd.read_excel(arq_alocacao, sheet_name="Disponibilidade", usecols="A,F")
    df_disp.columns = ["matricula", "muda_turno"]

    df_controle = pd.read_excel(
        arq_ferias,
        sheet_name="Controle de Férias",
        usecols="A,B,I,L,M,P,R,S,U",
    )
    df_controle.columns = [
        "matricula",
        "nome",
        "fim_aquisitivo",
        "dias_restantes",
        "limite_gozo",
        "inicio_ferias_1",
        "fim_ferias_1",
        "inicio_ferias_2",
        "fim_ferias_2",
    ]

    df_aloc = df_aloc.dropna(subset=["matricula"]).copy()
    df_aloc = df_aloc[df_aloc["data_demissao"].isna()].copy()
    df_aloc["matricula"] = df_aloc["matricula"].apply(normalizar_matricula)
    df_aloc["nome_alocacao"] = df_aloc["nome_alocacao"].fillna("").astype(str).str.strip()

    df_aloc["cargo_raw"] = df_aloc["cargo"].astype(str).str.strip().str.lower()
    df_aloc["cargo_raw_sem_acento"] = df_aloc["cargo_raw"].apply(remover_acentos)
    df_aloc = df_aloc[
        df_aloc["cargo_raw_sem_acento"].isin(CARGOS_DETALHADOS_VALIDOS)
    ].copy()
    df_aloc["cargo"] = df_aloc["cargo_raw"].apply(grupo_cargo)
    df_aloc = df_aloc[df_aloc["cargo"].isin({"tecnico", "laboratorista", "auxiliar"})].copy()

    df_aloc["cidade"] = df_aloc["cidade"].astype(str).str.strip()
    sem_cidade = df_aloc[
        df_aloc["cidade"].str.lower().isin({"", "nan", "none", "nat"})
    ]
    if not sem_cidade.empty:
        matriculas = sorted(sem_cidade["matricula"].astype(str).unique())
        raise ErroValidacaoDados(
            "Os seguintes funcionários não possuem cidade registrada: matrícula(s) "
            + ", ".join(matriculas)
            + ". Preencha a cidade na planilha e tente novamente."
        )

    df_aloc["projeto_original"] = df_aloc["projeto_original"].astype(str).str.strip()
    df_aloc["turno_original"] = df_aloc["turno_original"].apply(classificar_turno)
    df_aloc["tipo"] = df_aloc["tipo"].apply(normalizar_sim_nao)

    df_localidade = df_localidade.dropna(subset=["projeto"]).copy()
    df_localidade["projeto"] = df_localidade["projeto"].astype(str).str.strip()
    df_localidade["cidade_projeto"] = df_localidade["cidade_projeto"].astype(str).str.strip()
    df_localidade["inicio_projeto"] = pd.to_datetime(
        df_localidade["inicio_projeto"], format="%d/%m/%Y", errors="coerce"
    )
    df_localidade["fim_projeto"] = pd.to_datetime(
        df_localidade["fim_projeto"], format="%d/%m/%Y", errors="coerce"
    )

    projetos_validos = set(df_localidade["projeto"])
    df_aloc = df_aloc[df_aloc["projeto_original"].isin(projetos_validos)].copy()

    df_disp = df_disp.dropna(subset=["matricula"]).copy()
    df_disp["matricula"] = df_disp["matricula"].apply(normalizar_matricula)
    df_disp["muda_turno"] = df_disp["muda_turno"].apply(normalizar_sim_nao)

    df_controle = df_controle.dropna(subset=["matricula"]).copy()
    df_controle["matricula"] = df_controle["matricula"].apply(normalizar_matricula)
    for col in [
        "fim_aquisitivo",
        "limite_gozo",
        "inicio_ferias_1",
        "fim_ferias_1",
        "inicio_ferias_2",
        "fim_ferias_2",
    ]:
        df_controle[col] = pd.to_datetime(df_controle[col], errors="coerce", format="mixed", dayfirst=True)

    df_controle["dias_restantes"] = (
        pd.to_numeric(df_controle["dias_restantes"], errors="coerce")
        .fillna(0)
        .apply(math.ceil)
        .astype(int)
    )

    # Regra de negócio: dentro de uma janela de 365 dias, o saldo considerado
    # pelo modelo não pode ultrapassar 30 dias por colaborador/período.
    # Valores acima disso indicam duplicidade/erro de base para o horizonte tático.
    df_saldos_acima_30 = df_controle[df_controle["dias_restantes"] > 30].copy()
    if not df_saldos_acima_30.empty:
        print(
            f"AVISO: {len(df_saldos_acima_30):,} registro(s) de férias com dias_restantes > 30 foram limitados a 30."
        )
        for _, row in df_saldos_acima_30.head(20).iterrows():
            print(
                f"  matrícula {row['matricula']}: dias_restantes={int(row['dias_restantes'])} -> 30"
            )
        if len(df_saldos_acima_30) > 20:
            print("  ...")
        df_controle.loc[df_controle["dias_restantes"] > 30, "dias_restantes"] = 30
    df_controle = df_controle[
        (df_controle["fim_aquisitivo"].notna())
        & (df_controle["limite_gozo"].notna())
        & (df_controle["fim_aquisitivo"] < data_inicio)
        & (df_controle["limite_gozo"] >= data_inicio)
    ].copy()
    df_controle = (
        df_controle.sort_values("limite_gozo", ascending=True)
        .drop_duplicates(subset=["matricula"], keep="first")
        .copy()
    )

    I_A = df_aloc.loc[df_aloc["tipo"] == 0, "matricula"].tolist()
    I_E = df_aloc.loc[df_aloc["tipo"] == 1, "matricula"].tolist()
    I_S, cargo_suplente, cidade_suplente = _criar_suplentes_potenciais()
    I = I_A + I_E + I_S

    matriculas_validas = set(I_A + I_E)
    df_controle = df_controle[df_controle["matricula"].isin(matriculas_validas)].copy()

    cargo = dict(zip(df_aloc["matricula"], df_aloc["cargo"]))
    cidade = dict(zip(df_aloc["matricula"], df_aloc["cidade"]))
    projeto_original = dict(zip(df_aloc["matricula"], df_aloc["projeto_original"]))
    turno_original = dict(zip(df_aloc["matricula"], df_aloc["turno_original"]))

    cargo.update(cargo_suplente)
    cidade.update(cidade_suplente)
    turno_original.update({i: "diurno" for i in I_S})

    P = sorted(p for p in df_aloc["projeto_original"].dropna().unique().tolist() if str(p).strip())
    R = ["auxiliar", "laboratorista", "tecnico"]
    K = ["diurno", "noturno"]

    cidade_projeto = dict(zip(df_localidade["projeto"], df_localidade["cidade_projeto"]))
    inicio_projeto = dict(zip(df_localidade["projeto"], df_localidade["inicio_projeto"]))
    fim_projeto = dict(zip(df_localidade["projeto"], df_localidade["fim_projeto"]))

    flex = dict(zip(df_disp["matricula"], df_disp["muda_turno"]))
    for i in I:
        flex.setdefault(i, 1)
    for i in I_S:
        flex[i] = 1

    b = dict(zip(df_controle["matricula"], df_controle["dias_restantes"]))
    prazo_data = dict(zip(df_controle["matricula"], df_controle["limite_gozo"]))

    # Nome usado na aba Férias: vem da aba Alocação
    nome_por_matricula = {
        str(row["matricula"]).strip(): str(row["nome_alocacao"]).strip()
        for _, row in df_aloc.iterrows()
        if pd.notna(row["matricula"])
        and pd.notna(row["nome_alocacao"])
        and str(row["nome_alocacao"]).strip()
    }

    # -------------------------------------------------------------------------
    # Mobilizados / mapeados por pessoa-projeto
    # -------------------------------------------------------------------------
    # Fonte local: "Flexibilidade Operacional CTO.xlsx"
    #
    # Aba Mobilizados:
    #   ID_mobilizados | Nome | Cargo | Projeto | Cliente | Status
    #
    # Aba Pessoas:
    #   CPF | Nome | Cargo | Status | Matrícula
    #
    # Regra:
    # - Ignora Status da aba Mobilizados.
    # - Se existe linha Nome + Projeto em Mobilizados, considera que a pessoa está
    #   mapeada/mobilizada naquele projeto.
    # - A matrícula NÃO vem da coluna A de Mobilizados.
    # - A matrícula vem da aba Pessoas, cruzando pelo Nome.
    try:
        df_mob = pd.read_excel(arq_flexibilidade, sheet_name="Mobilizados")
        df_pessoas = pd.read_excel(arq_flexibilidade, sheet_name="Pessoas")
        print("Mobilizados e Pessoas carregados de Flexibilidade Operacional CTO.xlsx")
    except Exception as exc:
        print(
            "AVISO: não foi possível ler Flexibilidade Operacional CTO.xlsx. "
            f"Considerando ninguém mobilizado. Detalhe: {exc}"
        )
        df_mob = pd.DataFrame(columns=["Nome", "Projeto"])
        df_pessoas = pd.DataFrame(columns=["Nome", "Matrícula"])

    df_mob.columns = [str(c).strip() for c in df_mob.columns]
    df_pessoas.columns = [str(c).strip() for c in df_pessoas.columns]

    col_nome_mob = localizar_coluna(df_mob, ["Nome"])
    col_projeto_mob = localizar_coluna(df_mob, ["Projeto"])

    col_nome_pessoas = localizar_coluna(df_pessoas, ["Nome"])
    col_matricula_pessoas = localizar_coluna(df_pessoas, ["Matrícula", "Matricula", "matricula"])

    if col_nome_mob is None or col_projeto_mob is None:
        print(
            "AVISO: aba Mobilizados precisa ter as colunas Nome e Projeto. "
            "Considerando ninguém mobilizado."
        )
        df_mob_tratado = pd.DataFrame(columns=["nome_norm", "projeto", "projeto_norm"])
    else:
        df_mob_tratado = pd.DataFrame({
            "nome_original": df_mob[col_nome_mob],
            "projeto": df_mob[col_projeto_mob],
        })
        df_mob_tratado["nome_norm"] = df_mob_tratado["nome_original"].apply(normalizar_nome)
        df_mob_tratado["projeto"] = df_mob_tratado["projeto"].astype(str).str.strip()
        df_mob_tratado["projeto_norm"] = df_mob_tratado["projeto"].apply(normalizar_projeto)
        df_mob_tratado = df_mob_tratado[
            (df_mob_tratado["nome_norm"] != "")
            & (df_mob_tratado["projeto_norm"] != "")
        ].copy()

    if col_nome_pessoas is None or col_matricula_pessoas is None:
        print(
            "AVISO: aba Pessoas precisa ter as colunas Nome e Matrícula. "
            "Considerando ninguém mobilizado."
        )
        df_pessoas_tratado = pd.DataFrame(columns=["nome_norm", "matricula"])
    else:
        df_pessoas_tratado = pd.DataFrame({
            "nome_original": df_pessoas[col_nome_pessoas],
            "matricula": df_pessoas[col_matricula_pessoas],
        })
        df_pessoas_tratado["nome_norm"] = df_pessoas_tratado["nome_original"].apply(normalizar_nome)
        df_pessoas_tratado["matricula"] = df_pessoas_tratado["matricula"].apply(normalizar_matricula)
        df_pessoas_tratado = df_pessoas_tratado[
            (df_pessoas_tratado["nome_norm"] != "")
            & (df_pessoas_tratado["matricula"] != "")
        ].copy()

    # Mapa Nome normalizado -> matrícula(s), vindo da aba Pessoas.
    nome_para_matriculas_flex = {}
    for _, row in df_pessoas_tratado.iterrows():
        nome_para_matriculas_flex.setdefault(row["nome_norm"], set()).add(row["matricula"])

    def resolver_matricula_por_nome_flex(row):
        nome_norm = row.get("nome_norm", "")
        mats = sorted(nome_para_matriculas_flex.get(nome_norm, set()))

        if len(mats) == 1:
            return mats[0]

        if len(mats) > 1:
            print(
                "AVISO: nome duplicado na aba Pessoas para mobilização: "
                f"{row.get('nome_original', '')}. Linha ignorada."
            )

        return ""

    df_mob_tratado["matricula_modelo"] = df_mob_tratado.apply(
        resolver_matricula_por_nome_flex,
        axis=1,
    )

    # Mapa Projeto normalizado -> Projeto oficial usado no modelo.
    # Aqui a normalização ignora acento, maiúsculas/minúsculas, espaços e hífen.
    projeto_norm_para_modelo = {}
    for p in P:
        p_norm = normalizar_projeto(p)
        if p_norm and p_norm not in projeto_norm_para_modelo:
            projeto_norm_para_modelo[p_norm] = p

    df_mob_tratado["projeto_modelo"] = (
        df_mob_tratado["projeto_norm"]
        .map(projeto_norm_para_modelo)
        .fillna("")
    )

    qtd_original_mob = len(df_mob_tratado)

    # Mantém só matrículas que existem no modelo e projetos que existem no modelo.
    df_mob_tratado = df_mob_tratado[
        df_mob_tratado["matricula_modelo"].isin(matriculas_validas)
        & (df_mob_tratado["projeto_modelo"] != "")
    ].copy()

    mobilizado = {
        (row["matricula_modelo"], row["projeto_modelo"]): 1
        for _, row in df_mob_tratado.iterrows()
    }

    Cmob = 600.0

    print(f"Mobilizações/mapeamentos carregados: {len(mobilizado):,} pares pessoa-projeto")

    linhas_ignoradas_mob = qtd_original_mob - len(df_mob_tratado)
    if linhas_ignoradas_mob > 0:
        print(
            f"AVISO: {linhas_ignoradas_mob:,} linha(s) de Mobilizados foram ignoradas "
            "por nome sem matrícula na aba Pessoas, matrícula fora do modelo ou projeto sem correspondência."
        )

    for i in I_S:
        b[i] = 0
        prazo_data[i] = data_fim
    for i in I:
        b.setdefault(i, 0)
        prazo_data.setdefault(i, data_fim)

    L = {}
    for i in I:
        prazo = prazo_data[i]
        if pd.isna(prazo):
            L[i] = max(T) + 1
            continue
        prazo = prazo.normalize()
        if prazo < data_inicio:
            L[i] = 0
        elif prazo > data_fim:
            L[i] = max(T) + 1
        else:
            L[i] = data_para_t[prazo]
        b[i] = max(int(b[i]), 0)

    ferias_programadas = {(i, t): 0 for i in I for t in T}
    ferias_programadas_datas_total = {i: set() for i in I}
    tem_bloco_aprovado_14 = {i: 0 for i in I}
    consideracoes_solicitacoes = []

    def adicionar_periodo_ferias_programadas(i, inicio, fim):
        inicio = pd.Timestamp(inicio).normalize()
        fim = pd.Timestamp(fim).normalize()
        if fim < inicio:
            raise ErroValidacaoDados(
                f"O período de férias selecionado para a matrícula {i} possui data final anterior à inicial."
            )
        if (fim - inicio).days + 1 >= 14:
            tem_bloco_aprovado_14[i] = 1
        for data in pd.date_range(inicio, fim, freq="D"):
            ferias_programadas_datas_total.setdefault(i, set()).add(data.normalize())
        if fim < data_inicio:
            return
        for data in pd.date_range(max(inicio, data_inicio), fim, freq="D"):
            t = data_para_t.get(data.normalize())
            if t is not None:
                ferias_programadas[(i, t)] = 1

    for _, row in df_controle.iterrows():
        i = row["matricula"]
        if i not in matriculas_validas:
            continue
        for inicio, fim in [
            (row["inicio_ferias_1"], row["fim_ferias_1"]),
            (row["inicio_ferias_2"], row["fim_ferias_2"]),
        ]:
            if pd.isna(inicio) or pd.isna(fim):
                continue
            adicionar_periodo_ferias_programadas(i, inicio, fim)

    solicitacoes_aprovadas_teste = list(solicitacoes_aprovadas_teste or [])
    solicitacoes_aplicadas = 0
    for solicitacao in solicitacoes_aprovadas_teste:
        i = normalizar_matricula(solicitacao.get("matricula"))
        if i not in matriculas_validas:
            consideracoes_solicitacoes.append(
                f"Solicitação da matrícula {i or '(vazia)'} ignorada: o funcionário não "
                "pertence ao conjunto válido do modelo (verifique cargo, desligamento e projeto)."
            )
            continue
        inicio = pd.to_datetime(solicitacao.get("inicio"), errors="coerce")
        fim = pd.to_datetime(solicitacao.get("fim"), errors="coerce")
        if pd.isna(inicio) or pd.isna(fim):
            consideracoes_solicitacoes.append(
                f"Solicitação da matrícula {i} ignorada: período ausente ou inválido."
            )
            continue
        inicio = inicio.normalize()
        fim = fim.normalize()
        if fim < inicio:
            consideracoes_solicitacoes.append(
                f"Solicitação da matrícula {i} ignorada: data final anterior à data inicial."
            )
            continue
        if fim < data_inicio or inicio > data_fim:
            consideracoes_solicitacoes.append(
                f"Solicitação da matrícula {i}, de {inicio:%d/%m/%Y} a {fim:%d/%m/%Y}, "
                "ignorada: período fora do horizonte de 365 dias do modelo."
            )
            continue

        dias_solicitados = set(pd.date_range(inicio, fim, freq="D").normalize())
        dias_ja_programados = ferias_programadas_datas_total.get(i, set())
        dias_sobrepostos = len(dias_solicitados & dias_ja_programados)
        adicionar_periodo_ferias_programadas(i, inicio, fim)
        solicitacoes_aplicadas += 1
        if inicio < data_inicio or fim > data_fim:
            consideracoes_solicitacoes.append(
                f"Solicitação da matrícula {i}, de {inicio:%d/%m/%Y} a {fim:%d/%m/%Y}, "
                "com impacto diário de alocação somente no trecho que intersecta o horizonte; "
                "o período completo foi considerado no saldo de férias."
            )
        if dias_sobrepostos:
            consideracoes_solicitacoes.append(
                f"Solicitação da matrícula {i} considerada, mas {dias_sobrepostos} dia(s) "
                "já constavam como férias programadas e não foram contados em duplicidade."
            )

    if solicitacoes_aprovadas_teste:
        print(
            f"Solicitações pendentes tratadas como aprovadas no cenário: "
            f"{solicitacoes_aplicadas} de {len(solicitacoes_aprovadas_teste)} selecionada(s)"
        )

    dias_ferias_programadas_total = {
        i: len(ferias_programadas_datas_total.get(i, set()))
        for i in I
    }

    a = {(rho, r): 0 for rho in R for r in R}
    for rho in R:
        for r in R:
            if rho == "tecnico" and r in {"tecnico", "laboratorista", "auxiliar"}:
                a[(rho, r)] = 1
            elif rho == "laboratorista" and r in {"laboratorista", "auxiliar"}:
                a[(rho, r)] = 1
            elif rho == "auxiliar" and r == "auxiliar":
                a[(rho, r)] = 1

    matriz_distancias = calcular_matriz_distancias(
        [cidade[i] for i in I if cidade.get(i)],
        [cidade_projeto[p] for p in P if cidade_projeto.get(p)],
    )
    dist = {}
    for i in I:
        for p in P:
            cf = normalizar_cidade(cidade.get(i, ""))
            cp = normalizar_cidade(cidade_projeto.get(p, ""))
            if cf == cp:
                dist[(i, p)] = 0.0
                continue
            chave_distancia = (cf, cp)
            if chave_distancia not in matriz_distancias:
                raise ErroValidacaoDados(
                    "Não foi possível determinar a distância entre as cidades "
                    f"{cidade.get(i, '')} e {cidade_projeto.get(p, '')}. "
                    "Verifique na planilha se elas estão escritas corretamente e tente novamente."
                )
            dist[(i, p)] = matriz_distancias[chave_distancia]

    d = {(p, r, t, k): 0 for p in P for r in R for t in T for k in K}
    for i in I_A:
        p = projeto_original[i]
        r = cargo[i]
        k = turno_original[i]
        ini_p = inicio_projeto.get(p)
        fim_p = fim_projeto.get(p)
        for t in T:
            data = t_para_data[t]
            if pd.notna(ini_p) and data < ini_p.normalize():
                continue
            if pd.notna(fim_p) and data > fim_p.normalize():
                continue
            d[(p, r, t, k)] += 1

    Csal = {"tecnico": 563.13, "laboratorista": 424.69, "auxiliar": 233.91}
    Receita = {"tecnico": 946.06, "laboratorista": 713.48, "auxiliar": 392.98}
    Cdist = 0.8
    Dmax = 300
    gamma = 1

    linhas_dominancia = []
    IS_mantidos = []
    for i in I_S:
        receita_max_i = 0.0
        for t in T:
            melhor_dia = 0.0
            for p in P:
                for r in R:
                    if a[(cargo[i], r)] == 0:
                        continue
                    for k in K:
                        if d[(p, r, t, k)] > 0:
                            melhor_dia = max(melhor_dia, Receita[r])
            receita_max_i += melhor_dia

        custo_i = Csal[cargo[i]] * 30 * 12
        removido = receita_max_i <= custo_i
        if not removido:
            IS_mantidos.append(i)

        linhas_dominancia.append({
            "IS": i,
            "cargo": cargo[i],
            "cidade": cidade[i],
            "receita_maxima_teorica": round(receita_max_i, 2),
            "custo_anual": round(custo_i, 2),
            "decisao": "removido" if removido else "mantido",
        })

    removidos_dominancia = [linha["IS"] for linha in linhas_dominancia if linha["decisao"] == "removido"]
    I_S = IS_mantidos
    I = I_A + I_E + I_S

    df_dominancia = pd.DataFrame(linhas_dominancia)
    try:
        df_dominancia.to_excel(PROJECT_DIR / "Diagnostico_IS_Dominancia.xlsx", index=False)
    except Exception as exc:
        print(f"AVISO: nao foi possivel gerar Diagnostico_IS_Dominancia.xlsx: {exc}")

    print(f"IS removidos por dominancia economica: {len(removidos_dominancia)}")

    pi = {}
    for i in I:
        for p in P:
            excesso = dist[(i, p)] - Dmax
            if i in set(I_S) or excesso <= 0:
                pi[(i, p)] = 0.0
            else:
                pi[(i, p)] = gamma * excesso

    print("\nDados carregados")
    print(f"  |I_A| fixos atuais: {len(I_A)}")
    print(f"  |I_E| suplentes existentes: {len(I_E)}")
    print(f"  |I_S| suplentes potenciais: {len(I_S)}")
    print(f"  |I| total: {len(I)}")
    print(f"  |P| projetos: {len(P)}")
    print(f"  |R| cargos: {len(R)}")
    print(f"  |T| dias: {len(T)}")
    print(f"  Demanda total: {sum(d.values()):,.0f} postos-dia")

    return {
        "I_A": I_A,
        "I_E": I_E,
        "I_S": I_S,
        "I": I,
        "P": P,
        "R": R,
        "K": K,
        "T": T,
        "cargo": cargo,
        "cidade": cidade,
        "projeto_original": projeto_original,
        "nome_por_matricula": nome_por_matricula,
        "turno_original": turno_original,
        "flex": flex,
        "b": b,
        "L": L,
        "ferias_programadas": ferias_programadas,
        "dias_ferias_programadas_total": dias_ferias_programadas_total,
        "tem_bloco_aprovado_14": tem_bloco_aprovado_14,
        "consideracoes_solicitacoes": consideracoes_solicitacoes,
        "a": a,
        "d": d,
        "Csal": Csal,
        "Receita": Receita,
        "Ccontrat": {r: 22 * 12 * Csal[r] for r in R},
        "Cnot": {r: 0.10 * Csal[r] for r in R},
        "Cdist": Cdist,
        "Dmax": Dmax,
        "gamma": gamma,
        "dist": dist,
        "dist_cidade": matriz_distancias,
        "pi": pi,
        "mobilizado": mobilizado,
        "Cmob": Cmob,
        "cidade_projeto": cidade_projeto,
        "inicio_projeto": inicio_projeto,
        "fim_projeto": fim_projeto,
        "t_para_data": t_para_data,
        "data_para_t": data_para_t,
    }
