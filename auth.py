"""Autenticacao Google OIDC para usuarios da Chammas."""

from __future__ import annotations

from typing import Any

import streamlit as st


ALLOWED_DOMAIN = "chammas.com.br"
_PLACEHOLDER_PREFIX = "SUBSTITUA_"


def _user_value(user: Any, key: str, default: str = "") -> str:
    """Le um atributo de st.user de forma compativel com diferentes versoes."""
    if hasattr(user, "get"):
        return str(user.get(key, default) or default)
    return str(getattr(user, key, default) or default)


def user_email(user: Any) -> str:
    """Retorna o e-mail normalizado do usuario autenticado."""
    return _user_value(user, "email").strip().casefold()


def is_allowed_email(email: str) -> bool:
    """Aceita somente enderecos cujo dominio exato seja chammas.com.br."""
    local_part, separator, domain = email.strip().casefold().rpartition("@")
    return bool(separator and local_part and domain == ALLOWED_DOMAIN)


def _oauth_is_configured() -> bool:
    """Evita iniciar o OAuth enquanto as credenciais ainda forem placeholders."""
    try:
        auth_config = st.secrets["auth"]
        required_values = (
            str(auth_config.get("client_id", "")),
            str(auth_config.get("client_secret", "")),
            str(auth_config.get("cookie_secret", "")),
        )
    except (KeyError, TypeError):
        return False

    return all(
        value and not value.startswith(_PLACEHOLDER_PREFIX)
        for value in required_values
    )


def _render_login() -> None:
    """Exibe a pagina de entrada sem renderizar o restante do aplicativo."""
    col_left, col_center, col_right = st.columns([1, 1.2, 1])

    with col_center:
        try:
            st.image("assets/CHMMS_logo_reduzida-16.png", width=220)
        except Exception:
            pass

        st.title("Simulador Laboratorio Geral")
        st.write(
            "Entre com sua conta corporativa Google para acessar os simuladores."
        )

        if not _oauth_is_configured():
            st.warning(
                "A autenticacao Google ainda nao foi configurada. Preencha a secao "
                "[auth] nos segredos do aplicativo e reinicie o Streamlit."
            )
            return

        if st.button(
            "Continuar com Google",
            type="primary",
            use_container_width=True,
        ):
            try:
                st.login()
            except Exception as exc:
                st.error(
                    "Nao foi possivel iniciar o login. Confira as credenciais OAuth "
                    "e o redirect URI configurado no Google Cloud."
                )
                with st.expander("Detalhes tecnicos"):
                    st.code(str(exc))

        st.caption(f"Acesso exclusivo para contas @{ALLOWED_DOMAIN}.")


def require_chammas_user() -> Any:
    """Interrompe a aplicacao ate existir uma sessao corporativa valida."""
    if not getattr(st.user, "is_logged_in", False):
        _render_login()
        st.stop()

    email = user_email(st.user)
    if not is_allowed_email(email):
        st.error(
            "Acesso nao autorizado. Entre com uma conta corporativa "
            f"@{ALLOWED_DOMAIN}."
        )
        st.caption(
            f"Conta autenticada: {email or 'e-mail nao informado pelo Google'}"
        )
        if st.button("Sair e trocar de conta", type="primary"):
            st.logout()
        st.stop()

    return st.user
