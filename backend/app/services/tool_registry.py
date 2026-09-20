"""
Central registry of specialist tools the orchestrator can select from.
Keeping this explicit (rather than hard-coding dispatch logic inline)
is what makes the "auditable execution summary" honest: every response
can point at exactly which registry entry was used.
"""
from dataclasses import dataclass


@dataclass
class ToolSpec:
    name: str
    task: str
    required_mode: str          # single_image | cross_modal_pair | bi_temporal_pair
    engine: str                 # human-readable model/engine description
    status: str = "active"      # active | planned


TOOL_REGISTRY: dict[str, ToolSpec] = {
    "vqa": ToolSpec(
        name="RS-VQA",
        task="vqa",
        required_mode="single_image",
        engine="SmolVLM-500M-Instruct + LoRA (BigEarthNet-adapted)",
    ),
    "captioning": ToolSpec(
        name="RS-Captioning",
        task="captioning",
        required_mode="single_image",
        engine="SmolVLM-500M-Instruct + LoRA (BigEarthNet-adapted)",
    ),
    "grounding": ToolSpec(
        name="Grounding-DINO",
        task="grounding",
        required_mode="single_image",
        engine="Grounding DINO (zero-shot open-vocabulary detection)",
    ),
    "change_vqa": ToolSpec(
        name="Change-VQA",
        task="change_vqa",
        required_mode="bi_temporal_pair",
        engine="Siamese U-Net (LEVIR-CD) + SmolVLM temporal reasoning",
    ),
    "change_description": ToolSpec(
        name="Change-Description",
        task="change_description",
        required_mode="bi_temporal_pair",
        engine="Siamese U-Net (LEVIR-CD) + SmolVLM temporal reasoning",
    ),
    "optical_sar_fusion": ToolSpec(
        name="Optical+SAR Fusion",
        task="optical_sar_fusion",
        required_mode="cross_modal_pair",
        engine="Dual-branch gated fusion net + SAR backscatter physics",
    ),
}


def list_tools() -> list[ToolSpec]:
    return list(TOOL_REGISTRY.values())


def get_tool(task: str) -> ToolSpec:
    if task not in TOOL_REGISTRY:
        raise KeyError(f"No registered tool for task '{task}'")
    return TOOL_REGISTRY[task]
