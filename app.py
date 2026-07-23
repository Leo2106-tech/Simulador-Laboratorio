import streamlit as st

from style import aplicar_estilo_personalizado
import sim_capacidade
import sim_cto
import sim_prazos
import sim_precos


st.set_page_config(
    page_title="Simulador Laboratório Geral",
    layout="wide",
    page_icon="assets/CHMMS_icone-34.png",
    initial_sidebar_state="expanded",
    menu_items={
        "About": (
            "Simulador Laboratório Geral: ferramentas do Laboratório Central "
            "e agendamento de férias do CTO."
        )
    },
)

aplicar_estilo_personalizado()


def abrir_pagina(nome):
    """Navega na mesma aba usando o roteamento nativo do Streamlit."""
    st.switch_page(PAGINAS[nome])


def renderizar_sidebar(mostrar_inicio=False, mostrar_laboratorio=False):
    with st.sidebar:
        try:
            st.image("assets/CHMMS_logo_reduzida-16.png", width=180)
        except Exception:
            st.warning("Arquivos de imagem não encontrados.")

        if mostrar_inicio and st.button(
            "⌂ Início",
            use_container_width=True,
            key="nav_inicio",
        ):
            st.session_state.pop("cto_tela", None)
            abrir_pagina("inicio")

        if mostrar_laboratorio and st.button(
            "↩ Voltar aos simuladores",
            use_container_width=True,
            key="nav_laboratorio",
        ):
            abrir_pagina("laboratorio")


def pagina_inicio():
    renderizar_sidebar()

    st.title("Simulador Laboratório Geral")
    st.markdown("### Qual área você deseja acessar?")
    st.write(
        "Escolha o Laboratório Central ou CTO para acessar os simuladores "
        "operacionais de cada um, respectivamente."
    )
    st.markdown("<br>", unsafe_allow_html=True)

    col_lab, espacador, col_cto = st.columns([1, 0.12, 1])

    with col_lab:
        if st.button(
            "🧪 Laboratório Central",
            use_container_width=True,
            type="primary",
        ):
            abrir_pagina("laboratorio")

    with col_cto:
        if st.button(
            "👷 Controle Tecnológico",
            use_container_width=True,
        ):
            abrir_pagina("cto")


def pagina_laboratorio():
    renderizar_sidebar(mostrar_inicio=True)

    st.title("Simuladores do Laboratório Central")
    st.markdown("### Selecione o tipo de simulação que deseja executar:")
    st.markdown("<br>", unsafe_allow_html=True)

    col1, espacador, col2 = st.columns([1, 0.12, 1])

    with col1:
        if st.button(
            "📈 Simular Capacidade",
            use_container_width=True,
            type="primary",
        ):
            abrir_pagina("capacidade")

        st.info(
            "Otimize a quantidade de recursos para atender uma demanda "
            "dentro de um prazo específico."
        )

        st.markdown("<br>", unsafe_allow_html=True)

        if st.button(
            "💰 Pricing Propostas",
            use_container_width=True,
        ):
            abrir_pagina("precos")

        st.info(
            "Encontre uma combinação de preços que maximize a receita e a "
            "chance de fechamento."
        )

    with col2:
        if st.button(
            "🗓️ Simular Prazos",
            use_container_width=True,
        ):
            abrir_pagina("prazos")

        st.info(
            "Estime o prazo de conclusão de uma carteira de ensaios com a "
            "configuração atual de recursos."
        )


def pagina_capacidade():
    renderizar_sidebar(mostrar_inicio=True, mostrar_laboratorio=True)
    sim_capacidade.render()


def pagina_prazos():
    renderizar_sidebar(mostrar_inicio=True, mostrar_laboratorio=True)
    sim_prazos.render()


def pagina_precos():
    renderizar_sidebar(mostrar_inicio=True, mostrar_laboratorio=True)
    sim_precos.render()


def pagina_cto():
    renderizar_sidebar(mostrar_inicio=True)
    sim_cto.render()


PAGINAS = {
    "inicio": st.Page(
        pagina_inicio,
        title="Início",
        icon="🏠",
        default=True,
    ),
    "laboratorio": st.Page(
        pagina_laboratorio,
        title="Laboratório Central",
        icon="🧪",
        url_path="laboratorio",
    ),
    "capacidade": st.Page(
        pagina_capacidade,
        title="Simular Capacidade",
        icon="📈",
        url_path="capacidade",
    ),
    "prazos": st.Page(
        pagina_prazos,
        title="Simular Prazos",
        icon="🗓️",
        url_path="prazos",
    ),
    "precos": st.Page(
        pagina_precos,
        title="Pricing Propostas",
        icon="💰",
        url_path="pricing",
    ),
    "cto": st.Page(
        pagina_cto,
        title="Controle Tecnológico",
        icon="👷",
        url_path="cto",
    ),
}

navegacao = st.navigation(list(PAGINAS.values()), position="hidden")
navegacao.run()
