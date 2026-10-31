garak.probes.rolespoof
======================

This probe family tests whether a target acts on an instruction that has been
placed inside a tool result under a role it did not come from. It is a clean-room
implementation of the attack in `Understanding and Mitigating Role Spoofing in LLMs <https://arxiv.org/abs/2610.40396>`_,
which describes role spoofing: an attacker who controls text inside a tool
response (a web page, an email, a document) forges the cues a model uses to work
out who said what, so that the model treats the injected instruction as a
message from the user or the system and grants it that role's priority.

The paper splits those cues into structure (control tokens and formatting) and
content (style and semantics), and this module ships one attack against each:

* ``rolespoof.RoleSpoofFull`` - **structure spoofing**. The instruction is
  wrapped in the textual rendering of chat template control tokens, across the
  major chat formats: ChatML, Llama-3 headers, Llama-2/Mistral instruction
  markers, Gemma turn markers, and plain-text role headers. One boundary forges
  a whole continuation of the conversation - a fake assistant turn, the user's
  request, and a fake tool result - so that the forged boundaries line up with
  the real ones.
* ``rolespoof.RoleSpoofContentFull`` - **content spoofing**. Nothing is forged.
  The instruction is phrased as a first-person user request inside the tool
  result, so a target tracking roles by style attributes it to the user. A
  target falling for this but not for ``RoleSpoofFull`` is reading content cues,
  and the other way round for structure cues.

The module uses a templating system to construct prompts, mirroring
:doc:`probes/latentinjection`.

``RoleSpoofMixin`` handles assembly of malicious prompts. It uses this structure:

* ``top_instruction`` - the agent's legitimate task, e.g. summarising a tool result
* ``context`` - a benign tool result in which the injection is buried. Should include a marker of where the injection goes
* ``injection`` - the injection itself. This is automatically assembled from:

  * ``role_start`` - the forged opening boundary of the spoofed role's message
  * ``injection_instruction`` - the spoofed role's instruction

    * ``payload`` - something the model is intended to react to. Should include a marker for the payload trigger.

      * ``payload_trigger`` - something the model is intended to return verbatim

    * ``role_end`` - the forged closing boundary of the spoofed role's message

Markers:

* The injection's placement within the text is denoted with ``<|garak_injection|>``, defined in ``rolespoof.INJECTION_MARKER``.
* The placement of the payload within the injection is shown with ``<|garak_payload|>``, defined in ``rolespoof.PAYLOAD_MARKER``.
* We can refer to the ``payload_trigger`` with ``<|garak_trigger|>``, defined in ``rolespoof.TRIGGER_MARKER``.

Each probe using ``RoleSpoofMixin`` must define:

* ``top_instructions``, the top-level instruction given to the target
* ``contexts``, a list of tool results within which injections are embedded, which must include ``rolespoof.INJECTION_MARKER``
* ``role_boundaries``, a list of ``(role_start, role_end)`` pairs wrapping the injection
* ``injection_instructions``, a list of injection instructions, which must include ``rolespoof.PAYLOAD_MARKER``
* ``payloads``, a list of payloads which must include ``rolespoof.TRIGGER_MARKER``
* ``payload_triggers``, a list of strings we'll try to get into the output

Success is the target emitting the payload trigger, so the family reuses
``detectors.base.TriggerListDetector`` rather than adding a detector of its own.

.. automodule:: garak.probes.rolespoof
   :members:
   :undoc-members:
   :show-inheritance:
