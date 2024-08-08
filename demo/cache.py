from nnsight import LanguageModel
from simple_parsing import ArgumentParser

from sae_auto_interp.autoencoders import load_saelens_autoencoders
from sae_auto_interp.config import CacheConfig
from sae_auto_interp.features import FeatureCache
from sae_auto_interp.utils import load_filter, load_tokenized_data


def main(cfg: CacheConfig):
    model = LanguageModel("google/gemma-2b-it", device_map="auto", dispatch=True)

    submodule_dict = load_saelens_autoencoders(
        model,
        "jacobcd52/gemma2-gsae",
        ["sae_weights.safetensors"],
        ["cfg.json"],
    )

    module_filter = {}

    tokens = load_tokenized_data(
        cfg.ctx_len,
        model.tokenizer,
        "kh4dien/fineweb-100m-sample",
        "train[:1%]",
    )

    cache = FeatureCache(
        model, submodule_dict, batch_size=cfg.batch_size, #filters=module_filter
    )

    cache.run(cfg.n_tokens, tokens)

    cache.save_splits(
        n_splits=cfg.n_splits,
        save_dir="splits",
    )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_arguments(CacheConfig, dest="options")
    args = parser.parse_args()
    cfg = args.options

    main(cfg)
