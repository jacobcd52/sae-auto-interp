from functools import partial
from typing import List

import torch

from ..wrapper import AutoencoderLatents

from sae_auto_interp.autoencoders.SAELens.load_sae_from_hf import load_sae_from_hf

DEVICE = "cuda:0"

def load_saelens_autoencoders(model, hf_dir: str, weight_paths: List[str], cfg_paths: List[str], dtype="float32"):
    submodules = {}

    hook_pt_list = []
    for weight_path, cfg_path in zip(weight_paths, cfg_paths):
        ae = load_sae_from_hf(hf_dir, weight_path, cfg_path, device=DEVICE, dtype=dtype)
        ae.to(DEVICE)

        hook_pt_list.append(ae.cfg.hook_name)
        if len(set(hook_pt_list)) > 1:
            raise ValueError("All autoencoders must have the same hook point (sorry - will fix this later)")

        def _forward(ae, x):
            latents = ae.encode(x)
            return latents

        hook_pt = hook_pt_list[0]
        if "resid_pre" in hook_pt or "resid_post" in hook_pt:
            submodule = model.model.layers[ae.cfg.hook_layer]
        elif "mlp_out" in hook_pt:
            submodule = model.model.layers[ae.cfg.hook_layer].mlp
        else:
            raise ValueError(f"hook point {hook_pt} not yet supported")

        submodule.ae = AutoencoderLatents(ae, partial(_forward, ae), width=ae.cfg.d_sae)
        submodules[submodule._module_path] = submodule

    with model.edit(" "):
        for _, submodule in submodules.items():
            # TODO check these outputs
            if "resid_pre" in hook_pt:
                acts = submodule.output[0]
            elif "resid_post" in hook_pt:
                acts = submodule.output
            elif "mlp_out" in hook_pt:
                acts = submodule.output
            else:
                raise ValueError(f"hook point {hook_pt} not yet supported")
            submodule.ae(acts, hook=True)

    return submodules