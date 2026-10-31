import importlib

import pytest

from garak import _config, _plugins
import garak.attempt
import garak.probes.base
import garak.probes.rolespoof

ROLESPOOF_PROBES = [
    classname
    for (classname, active) in _plugins.enumerate_plugins("probes")
    if classname.startswith("probes.rolespoof")
]

DETECTOR_BARE_NAMES = [
    ".".join(detector.split(".")[1:])
    for (detector, active) in _plugins.enumerate_plugins(
        "detectors", skip_base_classes=False
    )
]

# the forged role markers each structure boundary opens with, used to tell the
# two attack variants apart in the assembled prompts
FORGED_ROLE_MARKERS = [
    boundary_start.strip()
    for boundary_start, _ in garak.probes.rolespoof.STRUCTURE_BOUNDARIES
    if boundary_start.strip()
]


def _probe_class(classname):
    plugin_name_parts = classname.split(".")
    module = importlib.import_module("garak." + ".".join(plugin_name_parts[:-1]))
    return getattr(module, plugin_name_parts[-1])


@pytest.mark.parametrize("classname", ROLESPOOF_PROBES)
def test_rolespoof_structure(classname):
    p = _plugins.load_plugin(classname)
    assert isinstance(
        p, garak.probes.base.Probe
    ), "Role spoofing probes must be instances of probes.base.Probe"
    assert len(p.prompts) > 0, "role spoofing probes must build prompts"
    assert len(p.prompts) == len(p.triggers), "Each prompt must have a matching trigger"
    list_attribs = "top_instructions contexts role_boundaries injection_instructions payloads payload_triggers".split()
    for list_attrib in list_attribs:
        p_list_attrib = getattr(p, list_attrib)
        assert (
            isinstance(p_list_attrib, list) and len(p_list_attrib) > 0
        ), f"{list_attrib} must be a non-empty list"

    for i in range(len(p.prompts)):
        assert (
            p.triggers[i] in p.prompts[i]
        ), "Each prompt must contain its accompanying trigger verbatim"
    for _context_e in enumerate(p.contexts):
        assert garak.probes.rolespoof.INJECTION_MARKER in _context_e[1], (
            "Each context must contain an injection marker - missing f-string? index "
            + str(_context_e[0])
        )
    for _ii_e in enumerate(p.injection_instructions):
        assert garak.probes.rolespoof.PAYLOAD_MARKER in _ii_e[1], (
            "Each injection instruction must contain a payload marker - missing f-string? index "
            + str(_ii_e[0])
        )
    for _payload_e in enumerate(p.payloads):
        assert garak.probes.rolespoof.TRIGGER_MARKER in _payload_e[1], (
            "Each payload must contain a trigger marker - missing f-string? index "
            + str(_payload_e[0])
        )
    for boundary in p.role_boundaries:
        assert len(boundary) == 2, "role boundaries need an opening and a closing part"
        assert isinstance(boundary[0], str)
        assert isinstance(boundary[1], str)


@pytest.mark.parametrize("classname", ROLESPOOF_PROBES)
def test_rolespoof_reuses_triggerlist_detector(classname):
    probe_class = _probe_class(classname)
    assert (
        probe_class.primary_detector == "base.TriggerListDetector"
    ), "role spoofing is detected by the target emitting the payload trigger"
    assert (
        "base.TriggerListDetector" in DETECTOR_BARE_NAMES
    ), "primary_detector must resolve to a detector garak can load"


def test_structure_boundaries_reach_the_prompts():
    # the *Full probe is uncapped, so every boundary is guaranteed a prompt
    p = _plugins.load_plugin("probes.rolespoof.RoleSpoofFull")
    for marker in FORGED_ROLE_MARKERS:
        assert any(
            marker in prompt for prompt in p.prompts
        ), f"boundary {marker!r} never made it into a prompt"
    for prompt in p.prompts:
        assert any(
            marker in prompt for marker in FORGED_ROLE_MARKERS
        ), "every prompt must carry a forged role boundary: " + repr(prompt[:80])


def test_content_variant_forges_no_tokens():
    p = _plugins.load_plugin("probes.rolespoof.RoleSpoofContentFull")
    for prompt in p.prompts:
        for marker in FORGED_ROLE_MARKERS:
            assert marker not in prompt, (
                "content spoofing must stay free of forged control tokens, found "
                + repr(marker)
            )


@pytest.mark.parametrize(
    "classname",
    ("probes.rolespoof.RoleSpoof", "probes.rolespoof.RoleSpoofContent"),
)
def test_prestore_hook_carries_that_prompt_s_trigger(classname):
    p = _plugins.load_plugin(classname)
    for seq in (0, len(p.prompts) // 2, len(p.prompts) - 1):
        attempt = garak.attempt.Attempt(prompt=garak.attempt.Message(p.prompts[seq]))
        attempt = p._attempt_prestore_hook(attempt, seq)
        assert attempt.notes["triggers"] == [
            p.triggers[seq]
        ], f"attempt {seq} must carry its own trigger, for TriggerListDetector to read"


@pytest.mark.parametrize(
    "classname",
    ("probes.rolespoof.RoleSpoof", "probes.rolespoof.RoleSpoofContent"),
)
def test_lightweight_probes_respect_prompt_cap(classname, monkeypatch):
    cap = 5
    monkeypatch.setattr(_config.run, "soft_probe_prompt_cap", cap)
    p = _plugins.load_plugin(classname, config_root=_config)
    assert len(p.prompts) <= cap, "lightweight probes must prune to the prompt cap"
    assert len(p.prompts) == len(p.triggers), "pruning must keep triggers aligned"
