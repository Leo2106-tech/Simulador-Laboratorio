import pandas as pd
import numpy as np
import re
from collections import defaultdict
from itertools import combinations
import streamlit as st # Para usar st.progress_bar

# =========================================================================
#                   CONFIGURAÇÃO FIXA DE RECURSOS
# =========================================================================
# Capacidade atual do laboratório. Este dicionário é usado em todas as simulações.
CAPACIDADE_RECURSOS_ATUAL = {
    'BANCADA_PREP_ATIVA': 1, 'BANCADA_TARUGO': 3, 'BANCADA_MONTAGEM': 3,
    'LINHA_SAT_CO2': 1, 'BANCADA_DESM': 1,
    'PRENSA_ESPECIAL_ANISO_CICLICO': 1, 'PRENSA_ROMP_ISO': 8,
    'BANCADA_ADEN_CONVENCIONAL': 6, 'PAINEL_SAT_H2O': 60, 'PAINEL_SAT_CP': 12,
    'CELULA_CONVENCIONAL': 55, 'CELULA_CICLICO': 2, 'CELULA_BENDER': 2
}

# =========================================================================
#                   MOTOR DE SIMULAÇÃO (HEURÍSTICA)
# =========================================================================
def _executar_heuristica(ensaios, p, r_j, U, etapas_proc, recursos_proc):
    """
    Função interna que executa a simulação de eventos discretos.
    Recebe todos os parâmetros já calculados.
    """
    capacidade_recurso_cenario = CAPACIDADE_RECURSOS_ATUAL

    p_total = {job: sum(p.get(job, {}).get(stage, 0) for stage in p[job] if stage != 'Prep_Espera') for job in ensaios}
    job_stages = {job: [etapa for etapa in etapas_proc if p.get(job, {}).get(etapa, 0) > 0] for job in ensaios}
    schedule = defaultdict(dict)
    
    resource_instance_available_time = {res: [0] * capacidade_recurso_cenario.get(res, 1) for res in recursos_proc}
    
    job_progress = {job: {'completed_stages': set()} for job in ensaios}
    uncompleted_jobs = set(ensaios)

    while uncompleted_jobs:
        candidate_operations = []
        for job in uncompleted_jobs:
            completed_stages = job_progress[job]['completed_stages']
            next_stage = next((etapa for etapa in job_stages[job] if etapa not in completed_stages), None)
            if not next_stage: continue
            
            job_ready_at = r_j.get(job, 0) if not completed_stages else schedule[job][job_stages[job][len(completed_stages)-1]]['end']
            
            resources_ready_at = 0
            
            required_resource_types = [res for res in recursos_proc if U.get((job, next_stage, res), 0) == 1]
            
            for res_type in required_resource_types:
                current_res_ready_at = min(resource_instance_available_time.get(res_type, [0]))
                if current_res_ready_at > resources_ready_at:
                    resources_ready_at = current_res_ready_at

            earliest_start_time = max(job_ready_at, resources_ready_at)

            candidate_operations.append({'job': job, 'etapa': next_stage, 'start_time': earliest_start_time, 'priority': (r_j.get(job, 0), p_total.get(job, 0))})

        if not candidate_operations:
            break # Todos os jobs foram agendados
            
        best_op = min(candidate_operations, key=lambda op: (op['start_time'], op['priority']))
        job, etapa, start_time = best_op['job'], best_op['etapa'], best_op['start_time']
        duration = p.get(job, {}).get(etapa, 0)
        end_time = start_time + duration

        schedule[job][etapa] = {'start': start_time, 'end': end_time}
        job_progress[job]['completed_stages'].add(etapa)
        
        required_resource_types = [res for res in recursos_proc if U.get((job, etapa, res), 0) == 1]
        for res_type in required_resource_types:
            instance_times = resource_instance_available_time[res_type]
            free_instances = [i for i, t in enumerate(instance_times) if t <= start_time + 1e-5]
            idx_to_update = free_instances[0] if free_instances else instance_times.index(min(instance_times))
            resource_instance_available_time[res_type][idx_to_update] = end_time

        if next((etapa for etapa in job_stages[job] if etapa not in job_progress[job]['completed_stages']), None) is None:
            uncompleted_jobs.remove(job)

    schedule_df = pd.DataFrame([(j, e, s['start'], s['end']) for j, ets in schedule.items() for e, s in ets.items()], columns=['Job', 'Etapa', 'Início', 'Fim'])
    return schedule_df

# =========================================================================
#                   PREPARAÇÃO DOS DADOS E PARÂMETROS
# =========================================================================
def _preparar_parametros_simulacao(df_cenario):
    """
    Função interna que pega um DataFrame de um cenário e calcula todos os
    parâmetros necessários para a heurística.
    """
    df = df_cenario.copy()
    
    # --- Início da lógica de parametrização (idêntica à de param_capacidade) ---
    df['Job'] = df['ID Ensaio/CP'].astype(str) + "_" + df.index.astype(str)

    # A data de chegada já vem do df_manual, precisamos converter para horas
    # Se 'Início Plan Atual' não existir (caso de dados manuais), usamos a data de chegada da proposta
    if 'data_chegada' in df.columns:
        hoje = pd.Timestamp.today().normalize()
        df['Release Date'] = df['data_chegada'].apply(lambda x: max(0, (pd.to_datetime(x) - hoje).days) * 24)
    else: # Para dados do Drive que já têm 'Início Plan Atual'
        df['Release Date'] = df['Início Plan Atual'].apply(lambda x: x * 24 + 7 if pd.notnull(x) else 0)

    def prep_ativa_horas(tipo):
        minutos = 60 if tipo == 'Deformada' else 30
        return minutos / 60.0
    df['Prep_Ativa'] = df['Tipo Amostra'].apply(prep_ativa_horas)
    df['Prep_Espera'] = df['Tipo Amostra'].apply(lambda tipo: 24.0 if tipo == 'Deformada' else 0.0)

    def tempo_formacao_horas(tipo):
        minutos = 30 if tipo == 'Deformada' else 10
        return minutos / 60.0
    df['Tarugo'] = df['Tipo Amostra'].apply(tempo_formacao_horas)
    df['Montagem Célula'] = 10 / 60.0
    ensaios_com_saturacao = ["QCSD", "BE", "BEP"]

    def tem_saturacao(ensaio):
        if isinstance(ensaio, str):
            return 'sat' in ensaio.lower() or ensaio in ensaios_com_saturacao
        return False
    
    def saturacao_agua(ensaio, nome_amostra):
        if not tem_saturacao(ensaio): return 0.0
        if 'aren' in str(nome_amostra).lower(): return 12.0
        else: return 24.0

    def saturacao_co2(ensaio):
        return (20/60.0) if tem_saturacao(ensaio) else 0.0
    
    def contrapressao(ensaio):
        return 4.5 if tem_saturacao(ensaio) else 0.0
    
    df['Sat H2O'] = df.apply(lambda row: saturacao_agua(row['Ensaio'], row.get('Nome Amostra')), axis=1)
    df['Sat CO2'] = df['Ensaio'].apply(saturacao_co2)
    df['Sat Contrapressao'] = df['Ensaio'].apply(contrapressao)

    def tempo_adensamento_horas(nome_amostra):
        nome = str(nome_amostra).lower()
        return (40 / 60.0) if 'aren' in nome else 2.0
    ensaios_sem_adensamento = ['UU', 'UUsat']
    df['Adensamento'] = np.where(df['Ensaio'].isin(ensaios_sem_adensamento), 0.0, df.get('Nome Amostra', 'default').apply(tempo_adensamento_horas))

    def extrair_deformacao(texto):
        if pd.isnull(texto): return 20
        match = re.search(r'Deformação[:\s]*([0-9]+)%', str(texto))
        if match: return int(match.group(1))
        return 20
    df['Deformacao (%)'] = df.get('Especificação Técnica Ensaio', 20).apply(extrair_deformacao)

    def altura_tarugo(nome_ensaio):
        if isinstance(nome_ensaio, str) and 'GD' in nome_ensaio: return 202
        return 102
    df['Altura Tarugo'] = df['Ensaio'].apply(altura_tarugo)
    ensaios_drenados = ["CID", "CIDsat", "CADsat", "CCIDsat", "QCSD", "CIDsat/GD", "CCADsat"]
    ensaios_nao_drenados = ["CIUsat", "CIU", "UU", "UUsat", "CAU", "CAUsat", "EIUsat", "CIUsat/GD", "PN", "CK0", "CCAUsat"]

    def velocidade_carregamento(nome_ensaio):
        if isinstance(nome_ensaio, str):
            for drenado in ensaios_drenados:
                if drenado in nome_ensaio: return 0.045
            for nao_drenado in ensaios_nao_drenados:
                if nao_drenado in nome_ensaio: return 0.09
        return 0
    df['Velocidade'] = df['Ensaio'].apply(velocidade_carregamento)

    def rompido_na_prensa(nome_ensaio):
        return any(tag in nome_ensaio for tag in ensaios_drenados + ensaios_nao_drenados) if isinstance(nome_ensaio, str) else False
    df['Rompido?'] = df['Ensaio'].apply(rompido_na_prensa)

    def calcular_rompimento(row):
        if not row['Rompido?']: return 0
        deform, altura, vel = row['Deformacao (%)'], row['Altura Tarugo'], row['Velocidade']
        if deform is None or vel is None or vel == 0: return 0
        return ((altura * (deform / 100)) / vel) / 60
    df['Rompimento'] = df.apply(calcular_rompimento, axis=1)
    df['Desmontagem'] = 20 / 60.0

    def definir_tipo_ensaio(nome_ensaio):
        nome_ensaio = str(nome_ensaio)
        if nome_ensaio.startswith('CC'): return 'Ciclico'
        elif len(nome_ensaio) > 1 and nome_ensaio[1] == 'I': return 'Iso'
        elif nome_ensaio.startswith('BE'): return 'Bender'
        else: return 'Aniso'
    df['Tipo Ensaio'] = df['Ensaio'].apply(definir_tipo_ensaio)
    
    Parametros = df[['Job', 'Tipo Amostra', 'Ensaio', 'Tipo Ensaio', 'Release Date', 'Prep_Ativa', 'Prep_Espera', 'Tarugo', 'Montagem Célula', 'Sat CO2','Sat H2O','Sat Contrapressao', 'Adensamento', 'Rompimento', 'Desmontagem']].copy()
    ensaios = Parametros['Job'].tolist()
    tipos_ensaio = Parametros.set_index('Job')['Tipo Ensaio'].to_dict()
    etapas_proc = ['Prep_Ativa', 'Prep_Espera', 'Tarugo', 'Montagem Célula', 'Sat_CO2', 'Sat_H2O', 'Sat_Contrapressao', 'Adensamento', 'Rompimento', 'Romp&Adensa', 'Desmontagem', 'Liberacao_Celula']
    r_j = Parametros.set_index('Job')['Release Date'].to_dict()
    p = {job: {} for job in ensaios}

    def to_float(val):
        try: return max(0.0, float(val))
        except (ValueError, TypeError): return 0.0

    for job in ensaios:
        row = Parametros[Parametros['Job'] == job].iloc[0]
        p[job]['Prep_Ativa'] = to_float(row['Prep_Ativa'])
        p[job]['Prep_Espera'] = to_float(row['Prep_Espera'])
        p[job]['Tarugo'] = to_float(row['Tarugo'])
        p[job]['Montagem Célula'] = to_float(row['Montagem Célula'])
        p[job]['Sat_CO2'] = to_float(row['Sat CO2'])
        p[job]['Sat_H2O'] = to_float(row['Sat H2O'])
        p[job]['Sat_Contrapressao'] = to_float(row['Sat Contrapressao'])
        p[job]['Desmontagem'] = to_float(row['Desmontagem'])
        p[job]['Liberacao_Celula'] = 0.0
        tipo_do_ensaio = tipos_ensaio.get(job)
        if tipo_do_ensaio in ['Aniso', 'Ciclico']:
            p[job]['Romp&Adensa'] = to_float(row['Adensamento']) + to_float(row['Rompimento'])
            p[job]['Adensamento'], p[job]['Rompimento'] = 0.0, 0.0
        else:
            p[job]['Romp&Adensa'] = 0.0
            p[job]['Adensamento'], p[job]['Rompimento'] = to_float(row['Adensamento']), to_float(row['Rompimento'])
            
    recursos_proc = list(CAPACIDADE_RECURSOS_ATUAL.keys())
    mapeamento_etapa_recurso = {
        'Prep_Ativa': ['BANCADA_PREP_ATIVA'], 'Tarugo': ['BANCADA_TARUGO'], 'Montagem Célula': ['BANCADA_MONTAGEM', 'CELULA_CONVENCIONAL', 'CELULA_CICLICO', 'CELULA_BENDER'],
        'Sat_CO2': ['LINHA_SAT_CO2', 'CELULA_CONVENCIONAL', 'CELULA_CICLICO', 'CELULA_BENDER'], 'Sat_H2O': ['PAINEL_SAT_H2O', 'CELULA_CONVENCIONAL', 'CELULA_CICLICO', 'CELULA_BENDER'],
        'Sat_Contrapressao': ['PAINEL_SAT_CP', 'CELULA_CONVENCIONAL', 'CELULA_CICLICO', 'CELULA_BENDER'], 'Adensamento': ['BANCADA_ADEN_CONVENCIONAL', 'CELULA_CONVENCIONAL', 'CELULA_CICLICO', 'CELULA_BENDER'],
        'Rompimento': ['PRENSA_ROMP_ISO', 'CELULA_CONVENCIONAL', 'CELULA_CICLICO', 'CELULA_BENDER'], 'Romp&Adensa': ['PRENSA_ESPECIAL_ANISO_CICLICO', 'CELULA_CONVENCIONAL', 'CELULA_CICLICO', 'CELULA_BENDER'],
        'Desmontagem': ['BANCADA_DESM', 'CELULA_CONVENCIONAL', 'CELULA_CICLICO', 'CELULA_BENDER'],
    }
    recursos_celulas = [recurso for recurso in recursos_proc if 'CELULA' in recurso]
    Ajc = pd.DataFrame(0, index=ensaios, columns=recursos_celulas)
    for job in ensaios:
        tipo = tipos_ensaio.get(job)
        if tipo == 'Bender':
            if 'CELULA_BENDER' in Ajc.columns: Ajc.loc[job, 'CELULA_BENDER'] = 1
        elif tipo == 'Ciclico':
            if 'CELULA_CICLICO' in Ajc.columns: Ajc.loc[job, 'CELULA_CICLICO'] = 1
        elif tipo in ['Iso', 'Aniso']:
            if 'CELULA_CONVENCIONAL' in Ajc.columns: Ajc.loc[job, 'CELULA_CONVENCIONAL'] = 1
    U = {}
    for j in ensaios:
        for m in etapas_proc:
            if p.get(j, {}).get(m, 0) > 0:
                recursos_potenciais = mapeamento_etapa_recurso.get(m, [])
                for r in recursos_potenciais:
                    if r in recursos_proc:
                        if r in recursos_celulas:
                            if Ajc.loc[j, r] == 1: U[j, m, r] = 1
                        else: U[j, m, r] = 1
    
    return df, ensaios, p, r_j, U, etapas_proc, recursos_proc

# =========================================================================
#                   FUNÇÃO PRINCIPAL ORQUESTRADORA
# =========================================================================
def simular_prazos_propostas(df_combinado, propostas_manuais):
    """
    Orquestra a simulação de prazos para várias combinações de propostas.

    Args:
        df_combinado (pd.DataFrame): DataFrame com todos os ensaios (fila real + todas as propostas).
        propostas_manuais (list): A lista de dicionários de propostas do st.session_state.

    Returns:
        dict: Um dicionário com o prazo médio em dias para cada proposta.
              Ex: {'Proposta A': 25, 'Proposta B': 30}
    """
    nomes_propostas = [p['nome_proposta'] for p in propostas_manuais]
    df_real = df_combinado[df_combinado['Origem'] == 'Planejamento (Drive)']
    
    # Dicionário para armazenar os prazos de cada proposta em cada cenário
    prazos_detalhados_cenario = defaultdict(list)
    prazos_gerais_cenario = defaultdict(list)

    # Gera todas as combinações não vazias de propostas (cenários)
    cenarios = []
    for i in range(1, len(nomes_propostas) + 1):
        cenarios.extend(combinations(nomes_propostas, i))

    # Barra de progresso para o Streamlit
    barra_progresso = st.progress(0, text="Iniciando simulações de cenários...")

    for i, cenario in enumerate(cenarios):
        nomes_propostas_cenario = list(cenario)
        barra_progresso.progress((i + 1) / len(cenarios), text=f"Simulando cenário: {', '.join(nomes_propostas_cenario)}")

        # 1. Montar o DataFrame para o cenário atual
        df_propostas_cenario = df_combinado[df_combinado['Campanha'].isin(nomes_propostas_cenario)]
        df_cenario_atual = pd.concat([df_real, df_propostas_cenario], ignore_index=True)

        # Adicionar a data de chegada para os cálculos de release date
        map_proposta_data = {p['nome_proposta']: p['data_chegada'] for p in propostas_manuais}
        df_cenario_atual['data_chegada'] = df_cenario_atual['Campanha'].map(map_proposta_data)

        # 2. Preparar os parâmetros para a simulação
        df_cenario_atual, ensaios, p, r_j, U, etapas_proc, recursos_proc = _preparar_parametros_simulacao(df_cenario_atual)

        # 3. Executar a heurística
        schedule_df = _executar_heuristica(ensaios, p, r_j, U, etapas_proc, recursos_proc)

        # 4. Calcular e armazenar o prazo para cada proposta NESTE cenário
        if not schedule_df.empty:
            # Mapear 'Job' de volta para 'Campanha'
            map_job_campanha = df_cenario_atual.set_index('Job')['Campanha'].to_dict()
            schedule_df['Campanha'] = schedule_df['Job'].map(map_job_campanha)
            map_job_ensaio = df_cenario_atual.set_index('Job')['Ensaio'].to_dict()
            schedule_df['Ensaio'] = schedule_df['Job'].map(map_job_ensaio)

            for nome_proposta in nomes_propostas_cenario:
                ensaios_da_proposta = schedule_df[schedule_df['Campanha'] == nome_proposta]
                if not ensaios_da_proposta.empty:
                    # --- 1. CÁLCULO DO PRAZO GERAL DA PROPOSTA (para o st.metric) ---
                    tempo_final_proposta_horas = ensaios_da_proposta['Fim'].max()
                    prazo_dias_uteis_geral = tempo_final_proposta_horas / 17
                    prazos_gerais_cenario[nome_proposta].append(prazo_dias_uteis_geral)

                    # --- 2. CÁLCULO DOS PRAZOS DETALHADOS POR TIPO DE ENSAIO (para a tabela) ---
                    # Itera sobre cada tipo de ensaio único dentro da proposta
                    for tipo_ensaio in ensaios_da_proposta['Ensaio'].unique():
                        # Encontra o tempo de conclusão do último ensaio DESTE TIPO
                        ensaios_do_tipo = ensaios_da_proposta[ensaios_da_proposta['Ensaio'] == tipo_ensaio]
                        tempo_final_ensaio_horas = ensaios_do_tipo['Fim'].max()
                        
                        # Converter para dias úteis
                        prazo_dias_uteis = tempo_final_ensaio_horas / 17 
                        
                        # Armazena o prazo para a combinação (proposta, tipo_ensaio)
                        prazos_detalhados_cenario[(nome_proposta, tipo_ensaio)].append(prazo_dias_uteis)

    # 5. Calcular as médias para ambos os resultados
    prazos_gerais_medios = {}
    for proposta, lista_prazos in prazos_gerais_cenario.items():
        if lista_prazos:
            prazos_gerais_medios[proposta] = np.mean(lista_prazos)

    prazos_detalhados_medios = []
    for (proposta, ensaio), lista_prazos in prazos_detalhados_cenario.items():
        if lista_prazos:
            prazo_medio = np.mean(lista_prazos)
            prazos_detalhados_medios.append({
                "Proposta": proposta,
                "Ensaio": ensaio,
                "Prazo de Entrega (dias úteis)": prazo_medio
            })

    barra_progresso.empty() # Limpa a barra de progresso
    return prazos_gerais_medios, prazos_detalhados_medios
