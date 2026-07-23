# Required Notice: Copyright (c) Shahzeb Imran and Egg contributors
#
# PolyForm Noncommercial License 2.0.0-pre.2
# https://github.com/bezmi/egg/blob/main/LICENSE.md
# Free to use and redistribute for personal and noncommercial purposes.
# See the license for details.
# For commercial licensing, contact s.imran@tuta.io

"""editable(show_if=...) hides run-parameters by another parameter's value.

The panel is rebuilt from the source on every change, so a param whose
show_if condition is not met by the current values simply drops out — and
reappears when the referenced param changes back.
"""

from egg.webui.scene import guard_params, set_guard_param, visible_params

CODE = """
import egg.webui as egg_webui

a = egg_webui.params(
    smoother=egg_webui.editable("jacobi", choices=["jacobi", "fas", "control_point"]),
    c2_weight=egg_webui.editable(
        0.0, label="interface C2 weight", show_if={"smoother": ["jacobi", "fas"]}
    ),
    ratio=egg_webui.editable(
        2, label="control ratio", show_if={"smoother": "control_point"}
    ),
)
"""


def _visible(code):
    return {p.name for p in visible_params(guard_params(code))}


def test_show_if_is_parsed():
    by_name = {p.name: p.show_if for p in guard_params(CODE)}
    assert by_name["interface C2 weight"] == {"smoother": ["jacobi", "fas"]}
    assert by_name["control ratio"] == {"smoother": "control_point"}
    # A plain param has no condition.
    assert by_name["smoother"] is None


def test_visibility_tracks_the_referenced_value():
    # jacobi: the node-mode knob shows, the control knob hides.
    vis = _visible(CODE)
    assert "interface C2 weight" in vis
    assert "control ratio" not in vis

    # switch to control_point: they swap.
    code2 = set_guard_param(CODE, "smoother", "control_point")
    vis2 = _visible(code2)
    assert "interface C2 weight" not in vis2
    assert "control ratio" in vis2

    # fas is in the C2 knob's allowed list.
    code3 = set_guard_param(CODE, "smoother", "fas")
    assert "interface C2 weight" in _visible(code3)


def test_param_without_show_if_is_always_visible():
    assert "smoother" in _visible(CODE)
    assert "smoother" in _visible(set_guard_param(CODE, "smoother", "control_point"))
