import streamlit as st


def render_citation_card(source: dict, index: int) -> None:
    """
    Renders a single citation card in the UI.
    """
    st.markdown(f"""
    <div class="citation-card">
        <span class="citation-num">[{source['number']}]</span>
        <span class="citation-text">{source['citation']}</span>
    </div>
    """, unsafe_allow_html=True)


def render_message(role: str, content: str,
                   sources: list = None) -> None:
    """
    Renders a chat message bubble.
    role: 'user' or 'assistant'
    """
    if role == "user":
        st.markdown(f"""
        <div class="message user-message">
            <div class="message-content">{content}</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="message assistant-message">
            <div class="message-content">{content}</div>
        </div>
        """, unsafe_allow_html=True)

        # Render citations below the answer
        if sources:
            st.markdown('<div class="sources-block">', unsafe_allow_html=True)
            st.markdown('<p class="sources-label">Sources</p>',
                        unsafe_allow_html=True)
            for source in sources:
                render_citation_card(source, source["number"])
            st.markdown('</div>', unsafe_allow_html=True)


def render_status_badge(label: str, value: str,
                        color: str = "blue") -> str:
    """
    Returns HTML for a status badge.
    """
    colors = {
        "green":  ("#0f6e56", "#e1f5ee"),
        "blue":   ("#185fa5", "#e6f1fb"),
        "amber":  ("#854f0b", "#faeeda"),
        "gray":   ("#5f5e5a", "#f1efe8"),
    }
    text_color, bg_color = colors.get(color, colors["blue"])
    return (f'<span style="background:{bg_color};color:{text_color};'
            f'padding:3px 10px;border-radius:20px;font-size:12px;'
            f'font-weight:500">{label}: {value}</span>')