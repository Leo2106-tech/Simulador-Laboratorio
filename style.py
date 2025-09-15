# style.py
import streamlit as st

def aplicar_estilo_personalizado():
    """
    Injeta CSS para estilizar a aplicação com as cores desejadas.
    """
    # Cores extraídas da imagem de referência
    cor_fundo_sidebar = "#2A3946"
    cor_texto_claro = "#FFFFFF"
    cor_botao = "#3E5060"
    cor_botao_hover = "#5A728A"
    cor_rotulo_sidebar = "#CED4DA"
    cor_titulo_app = "#3E5060" # Cor corrigida para os títulos da página principal

    # Código CSS para ser injetado na página
    estilo_css = f"""
        <style>
            /* =================================================================
               ESTILOS DA BARRA LATERAL (SIDEBAR)
               ================================================================= */
            [data-testid="stSidebar"] {{
                background-color: {cor_fundo_sidebar};
                color: {cor_texto_claro};
            }}

            /* Cor dos RÓTULOS (labels) dos widgets na sidebar */
            [data-testid="stSidebar"] label {{
                color: {cor_rotulo_sidebar};
            }}

            /* Garante que o texto das opções do st.radio seja branco */
            [data-testid="stSidebar"] [data-testid="stRadio"] label span {{
                color: {cor_texto_claro} !important;
                font-size: 1.1em;
            }}
            
            /* Regra específica para títulos DENTRO da sidebar */
            [data-testid="stSidebar"] h1,
            [data-testid="stSidebar"] h2,
            [data-testid="stSidebar"] h3,
            [data-testid="stSidebar"] h4 {{
                color: {cor_texto_claro} !important;
            }}

            /* =================================================================
               ESTILOS GERAIS E BOTÕES
               ================================================================= */
            /* Estilo do botão primário (e outros botões) */
            .stButton > button {{
                background-color: {cor_botao};
                color: {cor_texto_claro};
                border: none;
                border-radius: 8px;
                padding: 10px 20px;
            }}
            .stButton > button:hover {{
                background-color: {cor_botao_hover};
                border: none;
            }}
            .stButton > button:active {{
                background-color: {cor_botao};
                border: none;
            }}

            /* Títulos na PÁGINA PRINCIPAL */
            h1, h2, h3 {{
                color: {cor_titulo_app};
            }}

            /* =================================================================
               ESTILOS DE WIDGETS ESPECÍFICOS (RESTAURADOS)
               ================================================================= */
            /* Altera a cor da barra preenchida do slider */
            [data-testid="stSlider"] div[data-baseweb="slider"] div[style*="background: rgb(255, 75, 75);"] {{
                background-color: {cor_botao} !important;
            }}

            /* Altera a cor da bolinha (thumb) do slider */
            [data-testid="stSlider"] div[role="slider"] {{
                background-color: {cor_botao} !important;
            }}
            
            /* Altera a cor do valor (ex: 10%) que aparece acima da bolinha */
            [data-testid="stThumbValue"] {{
                color: {cor_botao} !important;
            }}

            /* Estilo das tags do MultiSelect */
            [data-testid="stMultiSelect"] [data-baseweb="tag"] {{
                background-color: {cor_botao} !important;
                color: {cor_texto_claro} !important;
                border-radius: 6px;
                border: none;
            }}

            /* Altera a cor do "X" de remover do MultiSelect */
            [data-testid="stMultiSelect"] [data-baseweb="tag"] svg {{
                fill: {cor_texto_claro} !important;
            }}
            [data-testid="stMultiSelect"] [data-baseweb="tag"]:hover {{
                background-color: {cor_botao_hover} !important;
            }}

            /* Botões de incremento/decremento do number_input */
            [data-testid="stNumberInput"] button {{
                background-color: {cor_botao} !important;
                color: {cor_texto_claro} !important;
            }}
            [data-testid="stNumberInput"] button:hover {{
                background-color: {cor_botao_hover} !important;
                color: {cor_texto_claro} !important;
            }}
        </style>
    """
    st.markdown(estilo_css, unsafe_allow_html=True)