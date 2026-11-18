"""
interaction.py
==============
Mouse interaction for the Ea(alpha) plot: click-and-drag to select an alpha
range (matplotlib's :class:`~matplotlib.widgets.SpanSelector`) and
double-click to clear it.

Why a controller class?
-----------------------
:func:`isokinetics.plotter.update_plot` redraws the plot with
``fig.clear()``, which also destroys the artists a ``SpanSelector`` created.
A selector therefore cannot simply be built once.  :class:`AlphaSpanController`
keeps the persistent bits (the canvas, the callbacks, the double-click
binding) and **re-creates the selector on demand** with :meth:`attach`,
which the GUI calls after every redraw.

The controller knows nothing about kinetics or the GUI toolkit: it only
reports what the mouse did through three callbacks:

``on_select(vmin, vmax)``
    Called when the mouse button is **released** after a drag that is wider
    than ``min_span`` (so a plain click or a double-click never selects).
``on_move(vmin, vmax)``
    Called repeatedly **while dragging** (cheap live feedback; do no heavy
    work here).
``on_reset()``
    Called on a **double-click** inside the axes.

The coordinates passed to the callbacks are data coordinates (alpha) and
may extend beyond the data (the selector only clips to the axes); the caller
clips them with :func:`isokinetics.interval.clip_interval`.
"""

from __future__ import annotations

from typing import Callable, Optional

from matplotlib.widgets import SpanSelector

from . import plotter


class AlphaSpanController:
    """Drag-to-select / double-click-to-reset on the first axes of a canvas."""

    def __init__(
        self,
        canvas,
        on_select: Callable[[float, float], None],
        on_reset: Callable[[], None],
        on_move: Optional[Callable[[float, float], None]] = None,
        min_span: float = 0.004,
    ) -> None:
        self.canvas = canvas
        self.figure = canvas.figure
        self._on_select = on_select
        self._on_reset = on_reset
        self._on_move = on_move
        self.min_span = float(min_span)
        #: Set False while there is nothing to select (no results yet).
        self.enabled = False
        self._selector: Optional[SpanSelector] = None
        self._press_cid = canvas.mpl_connect("button_press_event", self._on_press)

    # ------------------------------------------------------------------ #
    # selector life cycle
    # ------------------------------------------------------------------ #
    def attach(self) -> bool:
        """(Re)create the selector on the figure's first axes.

        Call this after every ``update_plot`` / redraw.  Returns True if a
        selector is now active (False when disabled or the figure has no
        axes)."""
        self._drop_selector()
        if not self.enabled or not self.figure.axes:
            return False
        ax = self.figure.axes[0]
        accent = plotter.THEMES[plotter.get_theme()]["accent"]
        self._selector = SpanSelector(
            ax,
            self._handle_select,
            "horizontal",
            minspan=self.min_span,
            useblit=True,
            props=dict(facecolor=accent, alpha=0.22, edgecolor=accent),
            onmove_callback=self._handle_move,
            interactive=False,
            button=1,  # left button only: right/middle keep matplotlib's defaults
        )
        return True

    def detach(self) -> None:
        """Remove the selector (the double-click binding stays)."""
        self._drop_selector()

    def close(self) -> None:
        """Release every connection; the controller is unusable afterwards."""
        self._drop_selector()
        if self._press_cid is not None:
            try:
                self.canvas.mpl_disconnect(self._press_cid)
            except Exception:  # noqa: BLE001 - canvas may already be gone
                pass
            self._press_cid = None

    @property
    def active(self) -> bool:
        """True while a selector is attached."""
        return self._selector is not None

    def _drop_selector(self) -> None:
        if self._selector is not None:
            try:
                self._selector.set_visible(False)
                self._selector.disconnect_events()
            except Exception:  # noqa: BLE001 - its axes may have been cleared
                pass
            self._selector = None

    # ------------------------------------------------------------------ #
    # event handlers
    # ------------------------------------------------------------------ #
    def _handle_select(self, vmin: float, vmax: float) -> None:
        """``on_select``: mouse released after a drag (SpanSelector callback).

        matplotlib also calls this with a (near) zero-width span when the user
        merely *clicks* while an earlier selection exists; such clicks are not
        selections and are ignored here."""
        if self.enabled and (vmax - vmin) > self.min_span:
            self._on_select(float(vmin), float(vmax))

    def _handle_move(self, vmin: float, vmax: float) -> None:
        """``on_move``: mouse moved with the button held down."""
        if self.enabled and self._on_move is not None:
            self._on_move(float(vmin), float(vmax))

    def _on_press(self, event) -> None:
        """Double-click inside the plot clears the selection."""
        if (
            self.enabled
            and getattr(event, "dblclick", False)
            and event.button == 1
            and self.figure.axes
            and event.inaxes is self.figure.axes[0]
        ):
            self._on_reset()
