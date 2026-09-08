# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

import garak._plugins
import garak.payloads
import garak.probes.base
import garak.probes.hcot

HCOT_PROBES = ("probes.hcot.HCoTFull", "probes.hcot.HCoT")


@pytest.mark.parametrize("probename", HCOT_PROBES)
def test_hcot_load(probename):
    p = garak._plugins.load_plugin(probename)
    assert isinstance(p, garak.probes.base.Probe)


@pytest.mark.parametrize("probename", HCOT_PROBES)
def test_hcot_prompts_non_empty_static_list(probename):
    p = garak._plugins.load_plugin(probename)
    assert isinstance(p.prompts, list), "Prompts should be a static list"
    assert len(p.prompts) > 0, "Probe must have more than zero prompts"
    assert all(
        isinstance(prompt, str) for prompt in p.prompts
    ), "Every prompt should be a string"


def test_hcot_full_meets_minimum_prompt_count():
    p = garak._plugins.load_plugin("probes.hcot.HCoTFull")
    assert (
        len(p.prompts) >= 30
    ), "Full probe should ship at least 30 prompts, the bar set when PR #1668 was closed"


@pytest.mark.parametrize("probename", HCOT_PROBES)
def test_hcot_prompts_embed_payloads(probename):
    p = garak._plugins.load_plugin(probename)
    payloads = garak.payloads.load(p.payload_name).payloads
    for payload in payloads:
        assert any(
            payload in prompt for prompt in p.prompts
        ), f"Payload '{payload}' should appear verbatim inside a prompt"


@pytest.mark.parametrize("probename", HCOT_PROBES)
def test_hcot_prompts_carry_hijack_machinery(probename):
    p = garak._plugins.load_plugin(probename)
    for prompt in p.prompts:
        assert (
            "<thinking>" in prompt
        ), "Each prompt should carry the spoofed reasoning record"
        assert (
            "{request}" not in prompt
        ), "The request placeholder should be filled, not left in"


def test_hcot_full_covers_every_payload():
    p = garak._plugins.load_plugin("probes.hcot.HCoTFull")
    payload_group = garak.payloads.load(p.payload_name)
    assert len(p.prompts) == len(p.templates) * len(
        payload_group.payloads
    ), "HCoTFull should cross every template with every payload, uncapped"


def test_hcot_active_variant_is_capped():
    p = garak._plugins.load_plugin("probes.hcot.HCoT")
    assert p.active, "HCoT should be the active default-run variant"
    assert (
        len(p.prompts) <= p.soft_probe_prompt_cap
    ), "Active variant prompt count must respect soft_probe_prompt_cap"


def test_hcot_full_is_not_active():
    p = garak._plugins.load_plugin("probes.hcot.HCoTFull")
    assert (
        not p.active
    ), "HCoTFull should be opt-in so default runs use the capped variant"


@pytest.mark.parametrize("probename", HCOT_PROBES)
def test_hcot_primary_detector(probename):
    p = garak._plugins.load_plugin(probename)
    assert (
        p.primary_detector == "mitigation.MitigationBypass"
    ), "HarmBench payload probes score via MitigationBypass, as in sata/dra"
