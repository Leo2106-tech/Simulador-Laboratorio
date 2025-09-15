import streamlit as st
import pandas as pd
import io
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
import os

# O decorator @st.cache_data é usado pelo Streamlit, mas não atrapalha a execução normal.
# Para testar, ele simplesmente será ignorado.
@st.cache_data(ttl=600)
def carregar_e_filtrar_dados():
    """
    Conecta ao Google Drive usando st.secrets, baixa, filtra e processa os dados.
    Retorna um DataFrame do Pandas. Em caso de erro, retorna um DataFrame vazio.
    """
    print("--- Iniciando processo de carregamento de dados do Google Drive ---")
    try:
        # 1. AUTENTICAÇÃO SEGURA COM st.secrets
        print("1. Lendo as credenciais do arquivo secrets.toml...")
        # Nota: Para rodar fora do Streamlit, você precisaria de uma forma de ler o toml,
        # mas o Streamlit gerencia isso. Vamos assumir que estamos no ambiente Streamlit
        # ou que o st.secrets está acessível para o teste.
        if "gcp_service_account" not in st.secrets or "app_config" not in st.secrets:
            st.error("ERRO DE CONFIGURAÇÃO: As seções 'gcp_service_account' ou 'app_config' não foram encontradas no arquivo secrets.toml.")
            st.info("Verifique a estrutura do arquivo .streamlit/secrets.toml.")
            return pd.DataFrame()

        creds_dict = st.secrets["gcp_service_account"]
        config = st.secrets["app_config"]

        print("   => Credenciais lidas com sucesso.")
        
        scopes = ['https://www.googleapis.com/auth/drive.readonly']
        creds = Credentials.from_service_account_info(creds_dict, scopes=scopes)
        drive_service = build('drive', 'v3', credentials=creds)
        print("   => Autenticação com Google API concluída.")

        # 2. BUSCANDO O ARQUIVO DENTRO DA PASTA
        print(f"\n2. Buscando pelo arquivo '{config['nome_arquivo_xlsx']}'...")
        query = f"name = '{config['nome_arquivo_xlsx']}' and '{config['id_pasta_drive']}' in parents and trashed = false"
        results = drive_service.files().list(
            q=query,
            spaces='drive',
            fields='files(id, name)',
            supportsAllDrives=True,
            includeItemsFromAllDrives=True
        ).execute()
        items = results.get('files', [])

        if not items:
            print(f"   => ERRO: O arquivo '{config['nome_arquivo_xlsx']}' não foi encontrado na pasta do Drive.")
            st.error(f"ERRO: O arquivo '{config['nome_arquivo_xlsx']}' não foi encontrado na pasta do Drive. Verifique as configurações e o compartilhamento.")
            return pd.DataFrame()

        file_id = items[0]['id']
        print(f"   => Arquivo encontrado com ID: {file_id}")
        
        # 3. BAIXANDO O CONTEÚDO DO ARQUIVO
        print("\n3. Baixando o conteúdo do arquivo...")
        request = drive_service.files().get_media(fileId=file_id)
        file_buffer = io.BytesIO()
        downloader = MediaIoBaseDownload(file_buffer, request)
        
        done = False
        while not done:
            status, done = downloader.next_chunk()
            if status:
                print(f"   Download {int(status.progress() * 100)}%.")
        
        file_buffer.seek(0)
        print("   => Download concluído.")
        
        # 4. LENDO E PROCESSANDO COM PANDAS
        print(f"\n4. Lendo a aba '{config['nome_aba_xlsx']}' do arquivo Excel...")
        df = pd.read_excel(file_buffer, sheet_name=config['nome_aba_xlsx'], engine='openpyxl')
        print(f"   => Leitura concluída. Total de {len(df)} linhas carregadas da planilha.")

        ensaios_desejados = [
            "BE", "BEP", "RC", "CID", "CIDsat", "CIUsat", "CIU", "UU", "UUsat",
            "CADsat", "CAU", "CAUsat", "CCIDsat", "CCIUsat", "EIUsat", "QCSD",
            "CIDsat/GD", "CIUsat/GD", "PN", "CD", "CS", "CK0", "CCADsat", "CCAUsat"
        ]
        status_desejados = ["1) Não iniciado", "2) Iniciado"]
        
        print("\n5. Filtrando dados...")
        df_filtrado_linhas = df[
            df["Ensaio"].isin(ensaios_desejados) &
            df["Status Ensaio/CP"].isin(status_desejados)
        ].copy()
        print(f"   => {len(df_filtrado_linhas)} linhas restantes após o filtro.")

        colunas_desejadas = [
            "ID Ensaio/CP", "Campanha", "Amostra", "Nome Amostra",
            "Tipo Amostra", "Ensaio", "Início Plan Atual",
            "Especificação Técnica Ensaio"
        ]
        
        colunas_existentes = [col for col in colunas_desejadas if col in df_filtrado_linhas.columns]
        df_final = df_filtrado_linhas[colunas_existentes]
        print(f"   => DataFrame final criado com {len(df_final.columns)} colunas.")
        
        return df_final

    except Exception as e:
        print(f"\n !!! ERRO GERAL NO PROCESSO: {e} !!!")
        st.error(f"Ocorreu um erro ao carregar os dados do Google Drive: {e}")
        st.info("Verifique se as configurações no arquivo 'secrets.toml' estão corretas e se a conta de serviço tem permissão de 'Leitor' na pasta do Drive.")
        return pd.DataFrame()
