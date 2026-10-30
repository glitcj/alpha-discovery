"""Display primitives: rich output in Jupyter, plain text everywhere else.

Every reporter goes through these functions so a pipeline run reads the same whether it is
executed from ``REPORT.ipynb`` (banner + numbered section headers, styled tables, inline figures
with captions) or from ``make train`` (markdown-ish text). ``display_figure(fig, name=...)`` also
writes ``FIGURE_DIR / <name>.png`` when the pipeline has set :data:`FIGURE_DIR`.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..evaluation import md_table, style_table
from ..style import INK, SERIES, STATUS

#: set by the pipeline (``figures=True``) while the reporters run; ``None`` = do not save
FIGURE_DIR: Path | None = None

# header level -> colour, in the spirit of ml-pipe's display_title
_LEVELS = {"#": "#1f3b73", "##": "#2f5597", "###": "#5b6b7a", "####": "#8b1a1a", "#####": "inherit"}
_BOXES = {
    "info": ("#e8f0fb", "#2f5597"),
    "success": ("#e6f4ea", "#1e7b34"),
    "warning": ("#fff4e5", "#b35c00"),
    "error": ("#fdecea", "#a61b1b"),
}
_FONT = 'font-family: system-ui, -apple-system, "Segoe UI", Helvetica, Arial, sans-serif;'


def in_notebook() -> bool:
    """True when running under an IPython kernel (JupyterLab, nbconvert, VS Code)."""
    try:
        from IPython import get_ipython
    except ImportError:  # pragma: no cover
        return False
    ip = get_ipython()
    return ip is not None and ip.__class__.__name__ == "ZMQInteractiveShell"


def _html(s: str) -> None:
    from IPython.display import HTML, display

    display(HTML(s))


def display_banner(title: str, chips: dict | None = None, subtitle: str | None = None) -> None:
    """Level-1 header: a coloured band with the title, a muted subtitle and key: value chips."""
    chips = chips or {}
    if in_notebook():
        chip_html = "".join(
            f'<span style="display:inline-block; background:rgba(255,255,255,0.16); border-radius:999px; '
            f'padding:0.15em 0.7em; margin:0.35em 0.4em 0 0; font-size:0.78em; color:#ffffff">'
            f'<span style="opacity:0.75">{k}</span>&nbsp; <b>{v}</b></span>' for k, v in chips.items())
        sub = f'<div style="opacity:0.85; font-size:0.9em; margin-top:0.2em">{subtitle}</div>' if subtitle else ""
        _html(f'<div style="{_FONT} background:linear-gradient(90deg, #1f3b73, #2f5597); color:#ffffff; '
              f'border-radius:8px; padding:0.9em 1.2em; margin:1.2em 0 0.8em 0">'
              f'<div style="font-size:1.45em; font-weight:600; letter-spacing:0.01em">{title}</div>{sub}'
              f'<div>{chip_html}</div></div>')
    else:
        print(f"\n# {title}" + (f" — {subtitle}" if subtitle else ""))
        if chips:
            print("  " + " · ".join(f"{k}: {v}" for k, v in chips.items()))


def display_title(prefix: str, title: str, subtitle: str | None = None, number: str | None = None) -> None:
    """Section header. ``prefix`` is one of ``#`` .. ``#####``; ``number`` prefixes the title ("2.1")."""
    if prefix not in _LEVELS:
        raise ValueError(f"prefix must be one of {list(_LEVELS)}, got {prefix!r}")
    label = f"{number}&nbsp; {title}" if number else title
    plain = f"{number} {title}" if number else title
    if in_notebook():
        level = len(prefix)
        colour = _LEVELS[prefix]
        if level == 1:
            display_banner(title, subtitle=subtitle)
            return
        if level == 2:
            sub = (f'<div style="color:{INK["muted"]}; font-style:italic; font-size:0.86em; margin:0.15em 0 0 0">'
                   f'{subtitle}</div>') if subtitle else ""
            _html(f'<div style="{_FONT} margin:1.4em 0 0.6em 0; padding-bottom:0.3em; border-bottom:2px solid '
                  f'{colour}"><h2 style="color:{colour}; margin:0; font-size:1.25em; font-weight:600">{label}</h2>'
                  f'{sub}</div>')
            return
        top = "0.9em" if level == 3 else "0.6em"
        sub = (f' <span style="color:{INK["muted"]}; font-weight:normal; font-style:italic; font-size:0.85em">'
               f'— {subtitle}</span>') if subtitle else ""
        _html(f'<h{level} style="{_FONT} color:{colour}; margin:{top} 0 0.3em 0; font-weight:600">'
              f'{label}{sub}</h{level}>')
    else:
        print(f"\n{prefix} {plain}" + (f" — {subtitle}" if subtitle else ""))


def display_markdown(text: str) -> None:
    if in_notebook():
        from IPython.display import Markdown, display

        display(Markdown(text))
    else:
        print(text)


def display_caption(text: str) -> None:
    """Muted italic line under a figure: what to look for."""
    if in_notebook():
        _html(f'<div style="{_FONT} color:{INK["muted"]}; font-style:italic; font-size:0.85em; '
              f'margin:-0.2em 0 1.0em 0.2em">{text}</div>')
    else:
        print(f"_{text}_")


def display_info_box(text: str, kind: str = "info") -> None:
    """Coloured call-out (``info`` / ``success`` / ``warning`` / ``error``)."""
    bg, fg = _BOXES.get(kind, _BOXES["info"])
    if in_notebook():
        _html(f'<div style="{_FONT} background:{bg}; border-left:4px solid {fg}; color:{fg}; '
              f'padding:0.55em 0.9em; margin:0.5em 0; border-radius:4px; line-height:1.45">{text}</div>')
    else:
        print(f"[{kind}] {text}")


def display_status(text: str, status: str = "good", title: str | None = None) -> None:
    """Verdict box in the reserved status palette (``good`` / ``warning`` / ``serious`` / ``critical``)."""
    colour = STATUS.get(status, STATUS["warning"])
    icon = {"good": "✔", "warning": "▲", "serious": "■", "critical": "✖"}.get(status, "•")
    if in_notebook():
        head = f'<div style="font-weight:600; margin-bottom:0.25em">{icon} {title}</div>' if title else f"{icon} "
        _html(f'<div style="{_FONT} border:1px solid {colour}; border-left:6px solid {colour}; '
              f'background:{INK["surface"]}; color:{INK["primary"]}; padding:0.7em 1em; margin:0.6em 0; '
              f'border-radius:6px; line-height:1.5">{head}{text}</div>')
    else:
        print(f"[{status}] {icon} {title + ': ' if title else ''}{text}")


def display_info_grid(items: dict, title: str | None = None) -> None:
    """ml-pipe style "Information" block: key / value pairs in a compact grid."""
    if in_notebook():
        cells = "".join(
            f'<div style="padding:0.35em 0.8em; border-bottom:1px solid {INK["grid"]}">'
            f'<div style="color:{INK["muted"]}; font-size:0.75em; text-transform:uppercase; '
            f'letter-spacing:0.04em">{k}</div><div style="color:{INK["primary"]}; font-size:0.95em">{v}</div></div>'
            for k, v in items.items())
        head = f'<div style="font-weight:600; color:{INK["secondary"]}; margin-bottom:0.3em">{title}</div>' if title else ""
        _html(f'<div style="{_FONT} margin:0.4em 0 0.8em 0">{head}<div style="display:grid; '
              f'grid-template-columns:repeat(auto-fill, minmax(180px, 1fr)); gap:0 0.6em; background:{INK["page"]}; '
              f'border-radius:6px; padding:0.3em 0.2em">{cells}</div></div>')
    else:
        if title:
            print(f"#### {title}")
        for k, v in items.items():
            print(f"- {k}: {v}")


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


def display_figure(fig, dpi: int = 110, *, name: str | None = None, caption: str | None = None) -> Path | None:
    """Show a matplotlib figure inline as PNG (backend-independent), save it when ``name`` and
    :data:`FIGURE_DIR` are set, print the caption, and close it. Returns the saved path."""
    import io

    import matplotlib.pyplot as plt

    path = None
    if name and FIGURE_DIR is not None:
        FIGURE_DIR.mkdir(parents=True, exist_ok=True)
        path = Path(FIGURE_DIR) / f"{name}.png"
        fig.savefig(path, format="png", dpi=dpi, bbox_inches="tight")
    if in_notebook():
        from IPython.display import Image, display

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
        display(Image(data=buf.getvalue()))
    elif path is not None:
        print(f"[figure] {path.name}")
    plt.close(fig)
    if caption:
        display_caption(caption)
    return path


__all__ = ["FIGURE_DIR", "in_notebook", "display_banner", "display_title", "display_markdown", "display_caption",
           "display_info_box", "display_status", "display_info_grid", "display_dataframe", "display_figure",
           "SERIES"]
