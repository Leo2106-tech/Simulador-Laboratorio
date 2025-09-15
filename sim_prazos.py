# sim_prazos.py

import streamlit as st
import pandas as pd
from datetime import date, timedelta
import json
import os
import plotly.express as px
from google_drive_loader import carregar_e_filtrar_dados
from param_prazos import simular_prazos_propostas

# =========================================================================
#                   CONSTANTES E CONFIGURAÇÕES
# =========================================================================
# Nome do arquivo que usaremos como nosso "banco de dados" local
LOG_FILE = "propostas_log.json"

DISTRIBUICAO_ENSAIOS = [
    {"Ensaio": "CIUsat", "Tipo Amostra": "Indeformada"}, {"Ensaio": "CIUsat", "Tipo Amostra": "Deformada"},
    {"Ensaio": "CIDsat", "Tipo Amostra": "Indeformada"}, {"Ensaio": "CIDsat", "Tipo Amostra": "Deformada"},
    {"Ensaio": "CADsat", "Tipo Amostra": "Deformada"}, {"Ensaio": "CAUsat", "Tipo Amostra": "Deformada"},
    {"Ensaio": "CAUsat", "Tipo Amostra": "Indeformada"}, {"Ensaio": "CADsat", "Tipo Amostra": "Indeformada"},
    {"Ensaio": "CIU", "Tipo Amostra": "Indeformada"}, {"Ensaio": "BE", "Tipo Amostra": "Indeformada"},
    {"Ensaio": "CID", "Tipo Amostra": "Indeformada"}, {"Ensaio": "CIDsat/GD", "Tipo Amostra": "Deformada"},
    {"Ensaio": "CIUsat/GD", "Tipo Amostra": "Deformada"}, {"Ensaio": "BEP", "Tipo Amostra": "Deformada"},
    {"Ensaio": "UUsat", "Tipo Amostra": "Deformada"}, {"Ensaio": "BE", "Tipo Amostra": "Deformada"},
    {"Ensaio": "PN", "Tipo Amostra": "Indeformada"}, {"Ensaio": "UUsat", "Tipo Amostra": "Indeformada"},
]
OPCOES_ENSAIO = sorted(list(set(item["Ensaio"] for item in DISTRIBUICAO_ENSAIOS)))
OPCOES_TIPO_AMOSTRA = [""] + sorted(list(set(item["Tipo Amostra"] for item in DISTRIBUICAO_ENSAIOS)))
ENSAIO_PADRAO_DEFAULT = "CIUsat"

# =========================================================================
#           FUNÇÕES DE LÓGICA E MANIPULAÇÃO DE DADOS
# =========================================================================

# --- NOVAS FUNÇÕES PARA SALVAR, CARREGAR E LIMPAR ---
def salvar_propostas():
    """Pega os dados do session_state e salva no arquivo LOG_FILE."""
    try:
        # A conversão de data para string é necessária para o formato JSON
        propostas_para_salvar = []
        for p in st.session_state.propostas:
            proposta_copia = p.copy()
            proposta_copia['data_chegada'] = p['data_chegada'].isoformat()
            propostas_para_salvar.append(proposta_copia)

        with open(LOG_FILE, 'w', encoding='utf-8') as f:
            json.dump(propostas_para_salvar, f, indent=4, ensure_ascii=False)
        st.toast("✅ Propostas salvas com sucesso!", icon="✅")
    except Exception as e:
        st.toast(f"🚨 Erro ao salvar propostas: {e}", icon="🚨")

def carregar_propostas():
    """Carrega os dados do LOG_FILE se ele existir."""
    if os.path.exists(LOG_FILE):
        try:
            with open(LOG_FILE, 'r', encoding='utf-8') as f:
                propostas_carregadas = json.load(f)
                # Reconverte a data de string para objeto date
                for p in propostas_carregadas:
                    p['data_chegada'] = date.fromisoformat(p['data_chegada'])
                return propostas_carregadas
        except (json.JSONDecodeError, TypeError):
             # Se o arquivo estiver corrompido ou vazio, retorna uma lista vazia
            return []
    return []

def limpar_tudo():
    """Limpa a tela e o arquivo de log."""
    st.session_state.propostas = []
    if os.path.exists(LOG_FILE):
        os.remove(LOG_FILE)
    st.toast("🗑️ Todas as propostas foram limpas.", icon="🗑️")

def converter_propostas_para_df(propostas_list):
    """Converte a lista de propostas do session_state para um DataFrame padronizado."""
    dados_para_df = []
    for proposta in propostas_list:
        for ensaio in proposta['ensaios']:
            # Para cada ensaio, repetimos a quantidade de vezes especificada
            for _ in range(ensaio['quantidade']):
                dados_para_df.append({
                    'Campanha': proposta['nome_proposta'],
                    'Ensaio': ensaio['ensaio'],
                    'Tipo Amostra': ensaio['tipo_amostra'] if ensaio['tipo_amostra'] else 'Indeformada',
                    # Adicionamos colunas "placeholder" para compatibilidade
                    'ID Ensaio/CP': f"Manual_{proposta['nome_proposta']}",
                    'Origem': 'Manual' # Coluna para identificar a origem dos dados
                })
    if not dados_para_df:
        return pd.DataFrame()
    return pd.DataFrame(dados_para_df)


# Callback para limpar os resultados da simulação anterior se as propostas mudarem
def on_proposta_change():
    st.session_state.prazos_gerais = None
    st.session_state.prazos_detalhados = None

# --- Funções existentes (sem alterações, exceto o callback) ---
def adicionar_proposta(nome_proposta):
    nova_proposta = {"nome_proposta": nome_proposta, "data_chegada": date.today(), "ensaios": [{"ensaio": ENSAIO_PADRAO_DEFAULT, "tipo_amostra": "", "quantidade": 1}]}
    st.session_state.propostas.append(nova_proposta)
    on_proposta_change() # Limpa os resultados ao adicionar proposta

def validar_e_adicionar_proposta_callback():
    novo_nome = st.session_state.novo_nome_proposta
    if not novo_nome:
        st.toast("⚠️ O nome da proposta não pode ser vazio.", icon="⚠️")
        return
    nomes_existentes = [p['nome_proposta'] for p in st.session_state.propostas]
    if novo_nome in nomes_existentes:
        st.toast(f"🚨 A proposta '{novo_nome}' já existe!", icon="🚨")
        return
    adicionar_proposta(novo_nome)
    st.session_state.novo_nome_proposta = ""

def remover_proposta(nome_proposta):
    st.session_state.propostas = [p for p in st.session_state.propostas if p["nome_proposta"] != nome_proposta]
    on_proposta_change() # Limpa os resultados ao remover proposta

def adicionar_ensaio(nome_proposta):
    for p in st.session_state.propostas:
        if p["nome_proposta"] == nome_proposta:
            p["ensaios"].append({"ensaio": ENSAIO_PADRAO_DEFAULT, "tipo_amostra": "", "quantidade": 1})
            on_proposta_change() # Limpa os resultados ao adicionar ensaio
            break

def remover_ensaio(nome_proposta, ensaio_idx):
    for p in st.session_state.propostas:
        if p["nome_proposta"] == nome_proposta:
            p["ensaios"].pop(ensaio_idx)
            on_proposta_change() # Limpa os resultados ao remover ensaio
            break

# =========================================================================
#                   FUNÇÃO PRINCIPAL DE RENDERIZAÇÃO
# =========================================================================

def render():
    st.header("Simulador de Prazos")
    st.markdown("Adicione ou carregue propostas para estimar o prazo de conclusão.")

    # --- NOVO: Lógica de carregamento inicial ---
    # Usamos um 'flag' para garantir que os dados sejam carregados do arquivo apenas uma vez.
    if 'dados_carregados' not in st.session_state:
        st.session_state.propostas = carregar_propostas()
        # Inicializa os placeholders para os resultados da simulação
        st.session_state.prazos_gerais = None
        st.session_state.prazos_detalhados = None
        st.session_state.schedules_por_cenario = None
        st.session_state.propostas_manuais_cache = None

        st.session_state.dados_carregados = True
    
    # --- NOVO: Botões de Ação (Salvar e Limpar) ---
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        st.subheader("Gerenciar Propostas")
    with col2:
        st.button("💾 Salvar Propostas no Log", on_click=salvar_propostas, use_container_width=True, help="Salva o estado atual de todas as propostas para uso futuro.")
    with col3:
        st.button("🗑️ Limpar Tudo", on_click=limpar_tudo, use_container_width=True, help="Remove todas as propostas da tela e do arquivo de log.")


    with st.container(border=True):
        st.markdown("##### Adicionar Nova Proposta")
        c1, c2 = st.columns([3, 1])
        c1.text_input("Nome da Proposta / Cliente", key="novo_nome_proposta", placeholder="Ex: Cliente A - Projeto Y")
        c2.write(" &nbsp; "); c2.button("➕ Adicionar", on_click=validar_e_adicionar_proposta_callback, use_container_width=True)

    st.markdown("---")

    if not st.session_state.propostas:
        st.info("Nenhuma proposta na tela. Adicione uma nova ou recarregue a página se houver um log salvo.")
        return

    # O resto do código para renderizar as propostas permanece o mesmo...
    for i, proposta in enumerate(st.session_state.propostas):
        with st.container(border=True):
            # ... (código para exibir cada proposta, sem alterações) ...
            col1, col2, col3 = st.columns([4, 2, 1])
            with col1:
                st.subheader(f"Proposta: {proposta['nome_proposta']}")
            with col2:
                proposta['data_chegada'] = st.date_input("Data Estimada de Chegada", key=f"data_{proposta['nome_proposta']}", value=proposta['data_chegada'], on_change=on_proposta_change)
            with col3:
                st.write(" &nbsp; ")
                st.button("❌ Remover", key=f"remover_proposta_{proposta['nome_proposta']}", on_click=remover_proposta, args=(proposta['nome_proposta'],), use_container_width=True)
            
            st.markdown("###### Ensaios da Proposta")
            
            c1, c2, c3, c4 = st.columns([3, 2, 1, 1])
            c1.markdown("**Ensaio**"); c2.markdown("**Tipo de Amostra**"); c3.markdown("**Quantidade**")

            for idx, ensaio in enumerate(proposta["ensaios"]):
                col1, col2, col3, col4 = st.columns([3, 2, 1, 1])
                ensaio['ensaio'] = col1.selectbox("Ensaio", OPCOES_ENSAIO, index=OPCOES_ENSAIO.index(ensaio['ensaio']), key=f"ensaio_{proposta['nome_proposta']}_{idx}", label_visibility="collapsed", on_change=on_proposta_change)
                ensaio['tipo_amostra'] = col2.selectbox("Tipo de Amostra", OPCOES_TIPO_AMOSTRA, index=OPCOES_TIPO_AMOSTRA.index(ensaio['tipo_amostra']), key=f"tipo_amostra_{proposta['nome_proposta']}_{idx}", label_visibility="collapsed", on_change=on_proposta_change)
                
                # Substituído st.number_input por st.text_input com validação manual
                # para remover os botões de incremento/decremento.
                quantidade_str = col3.text_input(
                    "Quantidade", 
                    value=str(ensaio['quantidade']), 
                    key=f"quantidade_{proposta['nome_proposta']}_{idx}", 
                    label_visibility="collapsed", 
                    on_change=on_proposta_change
                )
                ensaio['quantidade'] = int(quantidade_str) if quantidade_str.isdigit() and int(quantidade_str) >= 1 else 1

                col4.button("➖", key=f"remover_ensaio_{proposta['nome_proposta']}_{idx}", on_click=remover_ensaio, args=(proposta['nome_proposta'], idx))

            st.button("Adicionar Ensaio", key=f"adicionar_ensaio_{proposta['nome_proposta']}", on_click=adicionar_ensaio, args=(proposta['nome_proposta'],))


    if st.session_state.propostas:
        st.markdown("---")
        if st.button("Executar Simulação de Prazos", use_container_width=True, type="primary"):
            # ETAPA 1: Carregar dados reais do Drive
            with st.spinner("Executando... (1/3) Carregando Dados da Fila Real..."):
                df_real = carregar_e_filtrar_dados()
                if df_real.empty and any(p['data_chegada'] is None for p in st.session_state.propostas):
                     st.error("Falha ao carregar os dados do planejamento. A simulação não pode continuar sem uma base de dados.")
                     st.stop()
            st.success("✅ Fila atual carregada.")

            # ETAPA 2: Processar propostas manuais da tela
            with st.spinner("Executando... (2/3) Processando propostas manuais..."):
                propostas_manuais = st.session_state.get('propostas', [])
                df_manual = converter_propostas_para_df(propostas_manuais)
            st.success("✅ Propostas manuais processadas.")

            # ETAPA 3: Combinar os dois DataFrames
            with st.spinner("Executando... (3/3) Combinando todos os dados..."):
                df_real['Origem'] = 'Planejamento (Drive)'
                df_combinado = pd.concat([df_real, df_manual], ignore_index=True)
            st.success("🚀 Dados combinados!")

            # ETAPA 4: EXECUTAR A SIMULAÇÃO DE CENÁRIOS
            st.markdown("---")
            st.subheader("📊 Resultados da Simulação de Prazos")
            
            # Chamar a nova função orquestradora
            st.session_state.prazos_gerais, st.session_state.prazos_detalhados = simular_prazos_propostas(df_combinado, propostas_manuais)
            # Salva a lista de propostas usada na simulação para referência futura
            st.session_state.propostas_manuais_cache = propostas_manuais

        # ETAPA 5: EXIBIR OS RESULTADOS (agora fora do if do botão)
        resultados_gerais = st.session_state.get('prazos_gerais')
        resultados_detalhados = st.session_state.get('prazos_detalhados')

        if resultados_gerais is not None:
            if not resultados_gerais:
                st.warning("A simulação foi concluída, mas não foi possível calcular os prazos.")
            else:
                st.success("Simulação concluída! Veja abaixo os prazos médios estimados.")
                st.markdown("---")
                
                # --- EXIBIÇÃO DOS PRAZOS GERAIS (CARDS) ---
                st.subheader("Prazo Geral por Proposta")
                for proposta, prazo_em_dias in sorted(resultados_gerais.items()):
                    with st.container(border=True):
                        st.subheader(f"Proposta: {proposta}")
                        
                        data_chegada_proposta = next((p['data_chegada'] for p in st.session_state.propostas_manuais_cache if p['nome_proposta'] == proposta), None)
                        data_chegada_str = f"a partir de {data_chegada_proposta.strftime('%d/%m/%Y')}" if data_chegada_proposta else ""

                        st.metric(
                            label="Prazo Médio Estimado",
                            value=f"{prazo_em_dias:.0f} dias úteis",
                            help=f"Este é o prazo para a conclusão do último ensaio da proposta. O valor é uma média considerando todos os cenários de fechamento das propostas simuladas. {data_chegada_str}"
                        )

                # --- EXIBIÇÃO DOS PRAZOS DETALHADOS (TABELA) ---
                if resultados_detalhados:
                    st.markdown("<br>", unsafe_allow_html=True)
                    st.subheader("Prazos de Entrega Detalhados por Ensaio")
                    
                    # Converte a lista de resultados em um DataFrame
                    df_prazos = pd.DataFrame(resultados_detalhados)
                    
                    # Ordena para melhor visualização
                    df_prazos = df_prazos.sort_values(by=["Proposta", "Ensaio"]).reset_index(drop=True)
                    
                    # Formata a coluna de prazo para exibir apenas o número inteiro
                    df_prazos_formatado = df_prazos.style.format({
                        "Prazo de Entrega (dias úteis)": "{:.0f}"
                    })
                    
                    st.dataframe(df_prazos_formatado, use_container_width=True)
                
                st.info("Nota: Todos os prazos são uma média considerando os diferentes cenários de fechamento das propostas simuladas.")