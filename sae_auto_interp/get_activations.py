# import sys
# sys.path.append("/root/sae-auto-interp/")

from nnsight import LanguageModel
from simple_parsing import ArgumentParser
import torch

from sae_auto_interp.autoencoders import load_saelens_autoencoders
from sae_auto_interp.config import CacheConfig
from sae_auto_interp.features import FeatureCache
from sae_auto_interp.utils import load_filter, load_tokenized_data


def get_activations(sae_repo : str,
                    sae_weights_file : str,
                    sae_cfg_file : str,
                    feature_idx_list : list[int],
                    dataset_name : str = "stas/openwebtext-10k",
                    dataset_split : str = "train",
                    model_name : str = "google/gemma-2b-it",
                    batch_size : int = 32,
                    ctx_len : int = 128,
                    n_tokens : int = 1_000_000,
                    ):
    
    cfg = CacheConfig(batch_size=batch_size, ctx_len=ctx_len, n_tokens=n_tokens, n_splits=5)
    model = LanguageModel(model_name, device_map="auto", dispatch=True)

    submodule_dict = load_saelens_autoencoders(
        model,
        sae_repo,
        [sae_weights_file],
        [sae_cfg_file],
    )

    module_filter = None #{list(submodule_dict.values())[0]._module_path : torch.tensor(feature_idx_list, device=model.device)}

    tokens = load_tokenized_data(
        cfg.ctx_len,
        model.tokenizer,
        dataset_name,
        dataset_split,
    )

    cache = FeatureCache(
        model, submodule_dict, batch_size=cfg.batch_size, filters=module_filter
    )

    cache.run(cfg.n_tokens, tokens)

    cache.save_splits(
        n_splits=cfg.n_splits,
        save_dir="/root/sae-auto-interp/splits",
    )

    # for convenience, we return (model, SAE width)
    return model, list(submodule_dict.values())[0].ae.ae.cfg.d_sae
