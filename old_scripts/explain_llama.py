import os
os.environ["CONFIG_PATH"] = "/mnt/ssd-1/gpaulo/SAE-Zoology/sae_auto_interp/configs/gpaulo_llama_256.yaml"

import asyncio
from nnsight import LanguageModel 
from tqdm import tqdm
import torch
from sae_auto_interp.explainers import SimpleExplainer, ExplainerInput
from sae_auto_interp.clients import get_client
from sae_auto_interp.utils import execute_model, load_tokenized_data
from sae_auto_interp.features import FeatureRecord
import random
import argparse
from sae_auto_interp import cache_config as CONFIG

argparser = argparse.ArgumentParser()
argparser.add_argument("--layers", type=str, default="12,14")
args = argparser.parse_args()
layers = [int(layer) for layer in args.layers.split(",") if layer.isdigit()]

# Load model 
model = LanguageModel("meta-llama/Meta-Llama-3-8B", device_map="cpu", dispatch=True,torch_dtype =torch.bfloat16)
print("Model loaded")
# Load tokenized data
n_tokens = 10_400_000
dataset_repo =  "kh4dien/fineweb-100m-sample"
batch_len = 256
tokens  = load_tokenized_data(model.tokenizer,n_tokens=n_tokens, dataset_repo=dataset_repo, batch_len=batch_len,dataset_split="train")

print("Tokenized data loaded")
# Load features to explain

# Raw features contains locations
raw_features_path = "raw_features_llama"

explainer_inputs_1=[]
explainer_inputs_2=[]
explainer_inputs_3=[]
seed = 22
random.seed(seed)
        
for layer in layers:
    records = FeatureRecord.from_tensor(
        tokens,
        layer_index=layer,
        selected_features=torch.arange(1000,2000),
        raw_dir= raw_features_path,
        max_examples=10000
    )

    for record in records:
        all_examples = record.examples
        if len(all_examples) < 500:
            continue
        top500 = all_examples[:500]

        examples = random.sample(top500, 20)+random.sample(all_examples[:-5], 15)+all_examples[-5:]
        random.shuffle(examples)
        
        for example in examples:
            example.decode(model.tokenizer)
            
        explainer_inputs_1.append(
            ExplainerInput(
                train_examples=examples,
                record=record
            )
        )
        


client = get_client("outlines", "meta-llama/Meta-Llama-3-8B-Instruct", base_url="http://127.0.0.1:8000")

explainer = SimpleExplainer(client)
print("Running 1")
explainer_out_dir = "saved_explanations/llama_1/"
asyncio.run(
    execute_model(
        explainer, 
        explainer_inputs_1,
        output_dir=explainer_out_dir,
    )
)
