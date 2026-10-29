"""Display primitives: rich output in Jupyter, plain text everywhere else.

Every reporter goes through these four functions so a pipeline run reads the same
whether it is executed from ``REPORT.ipynb`` (coloured section headers, styled tables,
inline figures) or from ``make train`` (markdown-ish text).
"""

from __future__ import annotations

import pandas as pd

from ..evaluation import md_table, style_table

# header level -> colour, in the spirit of ml-pipe's display_title
_LEVELS = {"#": "#1f3b73", "##": "#2f5597", "###": "#5b6b7a", "####": "#8b1a1a", "#####": "inherit"}
_BOXES = {
    "info": ("#e8f0fb", "#2f5597"),
    "success": ("#e6f4ea", "#1e7b34"),
    "warning": ("#fff4e5", "#b35c00"),
    "error": ("#fdecea", "#a61b1b"),
}


def in_notebook() -> bool:
    """True when running under an IPython kernel (JupyterLab, nbconvert, VS Code)."""
    try:
        from IPython import get_ipython
    except ImportError:  # pragma: no cover
        return False
    ip = get_ipython()
    return ip is not None and ip.__class__.__name__ == "ZMQInteractiveShell"


def display_title(prefix: str, title: str) -> None:
    """Section header. ``prefix`` is one of ``#`` .. ``#####`` (phase, analysis, table, ...)."""
    if prefix not in _LEVELS:
        raise ValueError(f"prefix must be one of {list(_LEVELS)}, got {prefix!r}")
    if in_notebook():
        from IPython.display import HTML, display

        level = len(prefix)
        top = {1: "1.2em", 2: "1.0em"}.get(level, "0.6em")
        display(HTML(f'<h{level} style="color:{_LEVELS[prefix]}; margin:{top} 0 0.3em 0">{title}</h{level}>'))
    else:
        print(f"\n{prefix} {title}")


def display_markdown(text: str) -> None:
    if in_notebook():
        from IPython.display import Markdown, display

        display(Markdown(text))
    else:
        print(text)


def display_info_box(text: str, kind: str = "info") -> None:
    """Coloured call-out (``info`` / ``success`` / ``warning`` / ``error``)."""
    bg, fg = _BOXES.get(kind, _BOXES["info"])
    if in_notebook():
        from IPython.display import HTML, display

        display(HTML(f'<div style="background:{bg}; border-left:4px solid {fg}; color:{fg}; '
                     f'padding:0.5em 0.8em; margin:0.4em 0; border-radius:3px">{text}</div>'))
    else:
        print(f"[{kind}] {text}")


def display_dataframe(df, title: str | None = None, *, bars: bool = True, digits: int = 4,
                      caption: str | None = None, highlight_best: bool = True) -> None:
    """Show a DataFrame (styled with bars in notebooks) or a ready ``Styler``.

    Text fallback prints a markdown table of the underlying frame.
    """
    if title:
        display_title("####", title)
    if in_notebook():
        from IPython.display import display

        if isinstance(df, pd.DataFrame):
            obj = style_table(df, digits=digits, caption=caption, highlight_best=highlight_best) if bars else df.round(digits)
        else:
            obj = df
        display(obj)
    else:
        frame = df if isinstance(df, pd.DataFrame) else df.data
        if caption:
            print(f"({caption})")
        print(md_table(frame, digits))


def display_figure(fig, dpi: int = 110) -> None:
    """Show a matplotlib figure inline as PNG (backend-independent) and close it."""
    import io

    import matplotlib.pyplot as plt

    if in_notebook():
        from IPython.display import Image, display

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
        display(Image(data=buf.getvalue()))
    plt.close(fig)
