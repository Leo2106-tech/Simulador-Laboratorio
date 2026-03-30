# app.py
import streamlit as st

# Importe os módulos que você criou
from style import aplicar_estilo_personalizado
import sim_capacidade
import sim_prazos
import sim_precos

# Configuração da página (sem alterações)
st.set_page_config(
    page_title="Simulador do Laboratório",
    layout="wide",
    page_icon="assets/CHMMS_icone-34.png",
    initial_sidebar_state="expanded",
    menu_items={'About': "Simulador de cenários para o Laboratório Central."}
)

# Aplica nosso estilo CSS
aplicar_estilo_personalizado()

# =========================================================================
#             INICIALIZAÇÃO E CONTROLE DE ESTADO (A NOVA LÓGICA)
# =========================================================================

# 1. Inicializamos o 'session_state' para lembrar a escolha do usuário.
#    Se 'tipo_simulacao' não existir na memória, ele é criado com o valor None.
st.session_state.setdefault('tipo_simulacao', None)

# 2. Funções de Callback para os botões.
#    Essas funções alteram o estado da aplicação.
def set_simulacao(tipo):
    """Salva o tipo de simulação escolhido no estado da sessão."""
    st.session_state.tipo_simulacao = tipo

def go_home():
    """Limpa o estado da sessão para voltar ao menu principal."""
    st.session_state.tipo_simulacao = None

# =========================================================================
#                        INTERFACE (UI)
# =========================================================================

# --- Barra Lateral (Sidebar) ---
# A sidebar agora é mais simples. O botão de voltar só aparece se já
# estivermos dentro de uma simulação.
with st.sidebar:
    try:
        st.image("assets/CHMMS_logo_reduzida-16.png", width=180)
    except Exception as e:
        st.warning("Arquivos de imagem não encontrados.")
    
    # Adiciona o botão "Voltar" se uma simulação já foi escolhida
    if st.session_state.tipo_simulacao is not None:
        st.button("↩️ Voltar ao Menu", on_click=go_home, use_container_width=True)


# --- Página Principal ---
# 3. Lógica de renderização: ou mostra o menu, ou a simulação.
if st.session_state.tipo_simulacao is None:
    # --- ESTADO INICIAL: MOSTRA O MENU PRINCIPAL ---
    st.title("Simulador de Cenários do Laboratório Central")
    st.markdown("### Selecione o tipo de simulação que deseja executar:")
    
    st.markdown("<br>", unsafe_allow_html=True) # Espaço
    
    col1, col2, col3 = st.columns([1, 0.1, 1]) # Coluna do meio para espaçamento

    with col1:
        st.button(
            "📈 Simular Capacidade",
            on_click=set_simulacao,
            args=('Capacidade',), # Argumento para a função set_simulacao
            use_container_width=True,
            type="primary"
        )
        st.info("Otimize a quantidade de recursos (prensas, bancadas, etc.) para atender a uma demanda dentro de um prazo específico.")

        st.markdown("<br>", unsafe_allow_html=True)
        st.button(
                "💰 Pricing Propostas",
                on_click=set_simulacao,
                args=('Precos',),
                use_container_width=True,
                type="primary"
            )
        st.info("Adicione propostas para ter uma combinação de preços que maximiza a receita e a chance de fechamento.")

    with col3:
        st.button(
            "🗓️ Simular Prazos",
            on_click=set_simulacao,
            args=('Prazos',),
            use_container_width=True
        )
        st.info("Estime o prazo de conclusão de uma carteira de ensaios com a configuração de recursos atual.")
    
    #st.markdown("<br>", unsafe_allow_html=True)
    #c1, c2, c3 = st.columns([0.55, 1, 0.55])
        

else:
    # --- ESTADO SECUNDÁRIO: MOSTRA A SIMULAÇÃO ESCOLHIDA ---
    if st.session_state.tipo_simulacao == 'Capacidade':
        sim_capacidade.render()
    elif st.session_state.tipo_simulacao == 'Prazos':
        sim_prazos.render()
    elif st.session_state.tipo_simulacao == 'Precos':
        sim_precos.render()