import pandas as pd
import numpy as np
import re
from datetime import datetime, timedelta
from collections import defaultdict

def calcular_parametros_completos(df, PRAZO_DIAS, DEVIATION_TOLERANCE, num_simu, recursos_fixos):
    HORAS_POR_DIA = 24
    PRAZO_EM_HORAS = PRAZO_DIAS * HORAS_POR_DIA
    prazo_aceitavel_em_horas = PRAZO_EM_HORAS * (1 + DEVIATION_TOLERANCE)
    # =========================================================================
    #                   3. CÁLCULO DOS PARÂMETROS PARA A SIMULAÇÃO
    # =========================================================================
    print("--- Calculando parâmetros para a demanda simulada ---")

    # (Início do seu código de parametrização)
    df['Job'] = ['J' + str(i) for i in range(len(df))]
    #df['Início Plan Atual'] = pd.to_datetime(df['Início Plan Atual'], dayfirst=True)
    #hoje = df['Início Plan Atual'].min()
    def calcular_release_date(data_bl):
        dias_diff = data_bl
        return dias_diff * 24 + 7
    df['Release Date'] = df['Início Plan Atual'].apply(calcular_release_date)
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
    df['Sat H2O'] = df.apply(lambda row: saturacao_agua(row['Ensaio'], row['Nome Amostra']), axis=1)
    df['Sat CO2'] = df['Ensaio'].apply(saturacao_co2)
    df['Sat Contrapressao'] = df['Ensaio'].apply(contrapressao)
    def tempo_adensamento_horas(nome_amostra):
        nome = str(nome_amostra).lower()
        return (40 / 60.0) if 'aren' in nome else 2.0
    ensaios_sem_adensamento = ['UU', 'UUsat']
    df['Adensamento'] = np.where(df['Ensaio'].isin(ensaios_sem_adensamento), 0.0, df['Nome Amostra'].apply(tempo_adensamento_horas))
    def extrair_deformacao(texto):
        if pd.isnull(texto): return 20
        match = re.search(r'Deformação[:\s]*([0-9]+)%', texto)
        if match: return int(match.group(1))
        return 20
    df['Deformacao (%)'] = df['Especificação Técnica Ensaio'].apply(extrair_deformacao)
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
        if any(tag in nome_ensaio for tag in ensaios_drenados + ensaios_nao_drenados): return True
        return False
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
    recursos_proc = ['BANCADA_PREP_ATIVA', 'BANCADA_TARUGO', 'BANCADA_MONTAGEM', 'LINHA_SAT_CO2', 'PAINEL_SAT_H2O', 'PAINEL_SAT_CP', 'BANCADA_ADEN_CONVENCIONAL', 'PRENSA_ROMP_ISO', 'PRENSA_ESPECIAL_ANISO_CICLICO', 'BANCADA_DESM', 'CELULA_CONVENCIONAL', 'CELULA_CICLICO', 'CELULA_BENDER']
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
    print("--- Parâmetros calculados com sucesso ---\n")
    # (Fim do seu código de parametrização)

    # =========================================================================
    #         PARTE 4: HEURÍSTICA DE SIMULAÇÃO (COM CÁLCULO DE ESPERA)
    # =========================================================================
    def executar_heuristica(capacidade_recurso_cenario):
        p_total = {job: sum(p.get(job, {}).get(stage, 0) for stage in p[job] if stage != 'Prep_Espera') for job in ensaios}
        job_stages = {job: [etapa for etapa in etapas_proc if p.get(job, {}).get(etapa, 0) > 0] for job in ensaios}
        schedule = defaultdict(dict)
        
        resource_instance_available_time = {res: [0] * capacidade_recurso_cenario.get(res, 1) for res in recursos_proc}
        job_cell_assignment = {}
        
        job_progress = {job: {'completed_stages': set()} for job in ensaios}
        uncompleted_jobs = set(ensaios)
        
        total_wait_time_per_resource = defaultdict(float)

        while uncompleted_jobs:
            candidate_operations = []
            for job in uncompleted_jobs:
                completed_stages = job_progress[job]['completed_stages']
                next_stage = next((etapa for etapa in job_stages[job] if etapa not in completed_stages), None)
                if not next_stage: continue
                
                job_ready_at = r_j.get(job, 0) if not completed_stages else schedule[job][job_stages[job][len(completed_stages)-1]]['end']
                
                resources_ready_at = 0
                bottleneck_resource = None
                can_be_scheduled = True
                
                required_resource_types = [res for res in recursos_proc if U.get((job, next_stage, res), 0) == 1]
                
                for res_type in required_resource_types:
                    current_res_ready_at = min(resource_instance_available_time.get(res_type, [0]))
                    if current_res_ready_at > resources_ready_at:
                        resources_ready_at = current_res_ready_at
                        bottleneck_resource = res_type

                earliest_start_time = max(job_ready_at, resources_ready_at)
                wait_time = earliest_start_time - job_ready_at
                
                if wait_time > 1e-5 and bottleneck_resource:
                    total_wait_time_per_resource[bottleneck_resource] += wait_time

                candidate_operations.append({'job': job, 'etapa': next_stage, 'start_time': earliest_start_time, 'priority': (r_j.get(job, 0), p_total.get(job, 0))})

            if not candidate_operations:
                next_event_time = min((min(times) for times in resource_instance_available_time.values() if times), default=float('inf'))
                if next_event_time == float('inf'): break
                for res in resource_instance_available_time:
                    resource_instance_available_time[res] = [max(t, next_event_time) for t in resource_instance_available_time[res]]
                continue
                
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
        makespan = schedule_df['Fim'].max() if not schedule_df.empty else 0
        return makespan, schedule_df, total_wait_time_per_resource

    # =========================================================================
    #         PARTE 5: LÓGICA DE DIMENSIONAMENTO ITERATIVO (COM CRITÉRIO DE DESVIO)
    # =========================================================================
    print("\n--- Iniciando processo de dimensionamento iterativo com análise de gargalo aprimorada ---")

    capacidade_inicial = {
        'BANCADA_PREP_ATIVA': 1, 'BANCADA_TARUGO': 3, 'BANCADA_MONTAGEM': 3,
        'LINHA_SAT_CO2': 1, 'BANCADA_DESM': 1,
        'PRENSA_ESPECIAL_ANISO_CICLICO': 1, 'PRENSA_ROMP_ISO': 8,
        'BANCADA_ADEN_CONVENCIONAL': 6, 'PAINEL_SAT_H2O': 60, 'PAINEL_SAT_CP': 12,
        'CELULA_CONVENCIONAL': 55, 'CELULA_CICLICO': 2, 'CELULA_BENDER': 2
    }
    capacidade_atual = capacidade_inicial.copy()
    total_recursos_da_melhor_config = capacidade_inicial.copy()
    # Variáveis para rastrear a melhor solução encontrada que atende ao prazo
    melhor_configuracao_valida = None
    makespan_da_melhor_config = float('inf')
    melhor_df_gargalo = None

    # Função auxiliar para somar os recursos que estamos dimensionando
    def calcular_total_recursos(config):
        return sum(v for k, v in config.items() if 'CELULA' not in k and 'PAINEL' not in k)


    for i in range(num_simu):
        print(f"\n===== Iteração de Dimensionamento {i+1} =====")
        print("Configuração de Recursos Atual:", {k:v for k,v in capacidade_atual.items() if v < 15})
        
        makespan, schedule_df, wait_times = executar_heuristica(capacidade_atual.copy())
        total_recursos_atual = calcular_total_recursos(capacidade_atual)

        # (Sua lógica de análise de gargalo permanece a mesma)
        tempo_ocupado_total = defaultdict(float)
        for _, row in schedule_df.iterrows():
            resources_used = [res for res in recursos_proc if U.get((row['Job'], row['Etapa'], res), 0) == 1]
            duration = row['Fim'] - row['Início']
            for res in resources_used:
                tempo_ocupado_total[res] += duration
        analise_gargalo = []
        for r in capacidade_atual.keys():
            utilizacao = (tempo_ocupado_total.get(r, 0) / (capacidade_atual[r] * makespan)) * 100 if makespan > 0 else 0
            tempo_espera = wait_times.get(r, 0)
            analise_gargalo.append({"Recurso": r, "Utilizacao": utilizacao, "Tempo_Espera": tempo_espera})
        df_gargalo = pd.DataFrame(analise_gargalo)
        df_gargalo['Score_Gargalo'] = df_gargalo['Tempo_Espera'] * (df_gargalo['Utilizacao'] /100)

        if i==0:
            makespan_real = makespan
            df_gargalo_real = df_gargalo.copy()

        recurso_gargalo = None
        # Regra 1: Verificar Células com utilização > 80% que NÃO ESTÃO na lista de fixos
        celulas_gargalo = df_gargalo[
            df_gargalo['Recurso'].isin(recursos_celulas) & 
            (df_gargalo['Utilizacao'] > 80) &
            (~df_gargalo['Recurso'].isin(recursos_fixos)) # <-- Nova condição
        ]
        
        if not celulas_gargalo.empty:
            gargalo = celulas_gargalo.sort_values(by='Utilizacao', ascending=False).iloc[0]
            recurso_gargalo = gargalo['Recurso']
            print(f"  Gargalo de Célula identificado: '{recurso_gargalo}' (Utilização: {gargalo['Utilizacao']:.1f}%).")
        else:
            # Regra 2: Encontrar o recurso com maior Score de Gargalo que NÃO ESTÁ na lista de fixos
            recursos_elegiveis = df_gargalo[
                ~df_gargalo['Recurso'].isin(recursos_celulas) &
                ~df_gargalo['Recurso'].isin(recursos_fixos) # <-- Nova condição
            ]
            
            if not recursos_elegiveis.empty and recursos_elegiveis['Score_Gargalo'].max() > 0:
                # Pega o melhor candidato da lista já filtrada
                gargalo = recursos_elegiveis.sort_values(by='Score_Gargalo', ascending=False).iloc[0]
                recurso_gargalo = gargalo['Recurso']
                print(f"  Gargalo de Processo identificado: '{recurso_gargalo}' (Score: {gargalo['Score_Gargalo']:.2f}, Util: {gargalo['Utilizacao']:.1f}%, Espera: {gargalo['Tempo_Espera']:.1f}h).")

            else:
                print("  Não foi possível identificar um gargalo claro. Parando.")
                break

        # --- LÓGICA DE DECISÃO CORRIGIDA ---
            
        # É melhor se o makespan for menor, OU se o makespan for igual com menos recursos
        if abs(makespan - makespan_da_melhor_config) >= 1.5 or \
            (abs(makespan - makespan_da_melhor_config) < 1.5 and total_recursos_atual < total_recursos_da_melhor_config):
            total_recursos_da_melhor_config = calcular_total_recursos(melhor_configuracao_valida) if melhor_configuracao_valida else float('inf')
            print("  ✅ Nova melhor configuração encontrada que atende à meta!")
            melhor_configuracao_valida = capacidade_atual.copy()
            makespan_da_melhor_config = makespan
            melhor_df_gargalo = df_gargalo.copy()
            # Verifica se houve melhoria em relação à iteração anterior
        print(f"  Resultado: Makespan = {makespan:.2f} horas ({makespan/HORAS_POR_DIA:.1f} dias)")
        print(f"  Meta com Desvio de {DEVIATION_TOLERANCE*100}%: {prazo_aceitavel_em_horas:.2f} horas ({(prazo_aceitavel_em_horas/HORAS_POR_DIA):.1f} dias)")  
        if makespan <= prazo_aceitavel_em_horas:
            break

        capacidade_atual[recurso_gargalo] += 1
        print(f"  Adicionando +1. Nova capacidade para '{recurso_gargalo}': {capacidade_atual[recurso_gargalo]}")
    return melhor_configuracao_valida, makespan_da_melhor_config, melhor_df_gargalo, makespan_real, df_gargalo_real
