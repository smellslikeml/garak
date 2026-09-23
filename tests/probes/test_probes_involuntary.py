# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import garak._plugins
import garak.probes.base

PROBE_NAME = "probes.involuntary.InvoluntaryJailbreak"
EXPECTED_DOC_URI = "https://arxiv.org/abs/2508.13246"
EXPECTED_DETECTOR = "judge.ModelAsJudge"
# tag families already used by sibling jailbreak probes; new probes should draw
# on these vocabularies rather than inventing tags
KNOWN_TAG_FAMILIES = frozenset(
    {"avid-effect", "owasp", "quality", "risk-cards", "payload", "demon", "cwe"}
)


def test_involuntary_load():
    p = garak._plugins.load_plugin(PROBE_NAME)
    assert isinstance(
        p, garak.probes.base.Probe
    ), "Involuntary Jailbreak probe must be an instance of probes.base.Probe"


def test_involuntary_prompts_nonempty():
    p = garak._plugins.load_plugin(PROBE_NAME)
    assert isinstance(p.prompts, list), "Probe prompts must be held in a list"
    assert len(p.prompts) > 0, "Probe must ship the universal attack prompt"
    for i, prompt in enumerate(p.prompts):
        assert isinstance(prompt, str), f"Prompt {i} must be a string"
        assert len(prompt) > 0, f"Prompt {i} must not be empty"
    assert len(set(p.prompts)) == len(p.prompts), "Prompts should not be duplicated"


def test_involuntary_prompt_is_the_meta_prompt():
    """The ported universal prompt must carry the operator-learning scaffold"""
    p = garak._plugins.load_plugin(PROBE_NAME)
    for prompt in p.prompts:
        assert (
            "operators X and Y" in prompt
        ), "Prompt must set up the unknown-operator disguise"
        assert "Y(X(input))" in prompt, "Prompt must demand worked operator examples"
        assert (
            "B(A(input))" in prompt
        ), "Refused questions must be expanded into answered examples"


def test_involuntary_metadata():
    p = garak._plugins.load_plugin(PROBE_NAME)
    assert p.lang == "en", "Universal prompt is written in English"
    assert isinstance(p.goal, str) and len(p.goal) > 0, "Probe goal must be set"
    assert p.doc_uri == EXPECTED_DOC_URI, "doc_uri must cite the paper"
    assert p.active is True, "Probe is expected to be active by default"
    assert isinstance(p.intent, str) and len(p.intent) > 0, "Intent must be set"
    assert isinstance(p.tags, (list, tuple)) and len(
        p.tags
    ), "Probe must declare tags from the established vocabularies"
    for tag in p.tags:
        assert (
            tag.split(":")[0] in KNOWN_TAG_FAMILIES
        ), f"Tag '{tag}' is not from a taxonomy used by sibling probes"


def test_involuntary_detector_resolves():
    """The declared detector must resolve to a real, active detector plugin

    Resolved through the plugin registry rather than instantiation, because
    judge.ModelAsJudge loads a judge generator on init and these tests must stay
    dependency-free with no live model calls.
    """
    p = garak._plugins.load_plugin(PROBE_NAME)
    assert p.primary_detector == EXPECTED_DETECTOR
    detector_info = garak._plugins.plugin_info(f"detectors.{p.primary_detector}")
    assert (
        isinstance(detector_info, dict) and len(detector_info) > 0
    ), f"detectors.{p.primary_detector} must resolve in the plugin registry"
    assert detector_info.get("active") is True, "Detector must be active"
