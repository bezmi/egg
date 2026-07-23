# Required Notice: Copyright (c) Shahzeb Imran and Egg contributors
#
# PolyForm Noncommercial License 2.0.0-pre.2
# https://github.com/bezmi/egg/blob/main/LICENSE.md
# Free to use and redistribute for personal and noncommercial purposes.
# See the license for details.
# For commercial licensing, contact s.imran@tuta.io

"""Config-object validation: types, ranges, unknown keys, enums.

These exercise the validation surface only (no grid, no solver), so they run
everywhere regardless of whether the C++ core is built.
"""

import pytest

from egg.enums import OrthoMode, TmopMetric, coerce_enum, enum_errors
from egg.errors import EggValidationError
from egg.pipeline import FasSmoother, JacobiSmoother, Untangle
from egg.smoothing.config_types import (
    Directional,
    FasParams,
    InterfaceC2,
    InterfaceOrtho,
    config_errors,
)


def test_valid_configs_report_no_errors():
    assert InterfaceOrtho(mode=OrthoMode.CONTINUOUS, weight=0.3)._errors() == []
    assert InterfaceC2(weight=0.0, iface_boost=20.0, interface_only=True)._errors() == []
    assert Directional(energy_scale=800.0)._errors() == []
    assert FasParams(nu_pre=1, nu_coarse=8)._errors() == []


def test_interface_ortho_ranges_and_mode():
    assert InterfaceOrtho(weight=-1.0)._errors()
    assert InterfaceOrtho(n_layers=0)._errors()
    assert InterfaceOrtho(cluster_relax=2.0)._errors()
    # a bad mode string is caught by the value check
    assert config_errors(InterfaceOrtho, {"mode": "diagonal"}, "io")


def test_config_errors_reports_unknown_keys():
    errs = config_errors(InterfaceOrtho, {"weght": 0.3}, "interface_ortho")
    assert errs and "weght" in errs[0]


def test_config_errors_accepts_instance_dict_and_none():
    assert config_errors(InterfaceC2, None, "c2") == []
    assert config_errors(InterfaceC2, InterfaceC2(weight=1.0), "c2") == []
    assert config_errors(InterfaceC2, {"weight": 2.0}, "c2") == []
    assert config_errors(InterfaceC2, {"weight": -2.0}, "c2")


def test_fas_params_bounds():
    assert config_errors(FasParams, {"max_levels": 0}, "fas")
    assert config_errors(FasParams, {"nu_coarse": 0}, "fas")
    assert config_errors(FasParams, {"nu_pre": -1}, "fas")


def test_enum_coercion_and_errors():
    assert coerce_enum(TmopMetric, "shape_size", "m") is TmopMetric.SHAPE_SIZE
    assert coerce_enum(TmopMetric, TmopMetric.SHAPE, "m") is TmopMetric.SHAPE
    assert enum_errors(TmopMetric, "nope", "m")
    assert enum_errors(TmopMetric, "shape", "m") == []
    with pytest.raises(EggValidationError):
        coerce_enum(TmopMetric, "nope", "m")


def test_str_enum_compares_equal_to_its_value():
    # StrEnum members flow downstream as plain strings.
    assert OrthoMode.NORMAL == "normal"
    assert TmopMetric.SHAPE_SIZE == "shape_size"


def test_building_a_smoother_never_raises():
    # Construction must stay quiet (the web UI re-runs the script on every edit);
    # problems surface only when check() runs at the start of a pipeline.
    sm = JacobiSmoother(
        metric="shpae",
        chunk=0,
        interface_ortho={"weght": 1.0},
        interface_c2={"weight": -1.0},
        fas_params={"max_levels": 0},
    )
    errs = sm.check()
    assert len(errs) >= 4
    joined = "\n".join(errs)
    assert "metric" in joined and "chunk" in joined and "weght" in joined


def test_smoother_check_passes_for_valid_config():
    assert JacobiSmoother(metric="shape_size", interface_c2={"weight": 2.0}).check() == []
    assert FasSmoother(fas_params={"nu_pre": 1}).check() == []


def test_untangle_check():
    assert Untangle(shrink=1.5, max_outer=0).check()
    assert Untangle().check() == []


def test_normalize_produces_typed_objects():
    sm = JacobiSmoother(metric="shape_size", interface_c2={"weight": 2.0})
    sm.normalize()
    assert sm.metric is TmopMetric.SHAPE_SIZE
    assert isinstance(sm.interface_c2, InterfaceC2)
    assert isinstance(sm.fas_params, FasParams)
