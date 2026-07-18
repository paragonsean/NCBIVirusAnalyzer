"""Class-based Streamlit app routing."""

from __future__ import annotations

from collections.abc import Callable

import streamlit as st


class StreamlitWorkflowApp:
    """Render a dropdown-selected set of Streamlit workflows."""

    def __init__(
        self,
        title: str,
        caption: str,
        workflows: dict[str, Callable[[], None]],
        page_title: str | None = None,
    ):
        self.title = title
        self.caption = caption
        self.workflows = workflows
        self.page_title = page_title or title

    def run(self) -> None:
        st.set_page_config(page_title=self.page_title, layout="wide")
        st.title(self.title)
        st.caption(self.caption)

        product = st.selectbox(
            "Data product / workflow",
            list(self.workflows.keys()),
            index=0,
        )
        st.divider()
        self.workflows[product]()
