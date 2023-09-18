"""
Copyright 2023 Google LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import os
import torch
from diffusers import StableDiffusionPipeline
import numpy as np
import matplotlib.pyplot as plt
from transformers import CLIPImageProcessor, CLIPModel
import glob


def load_tokens(pipe, data, device):
    """
    Adds the new learned tokens into the predefined dictionary of pipe.
    """
    added_tokens = []
    for t_ in data.keys():
        added_tokens.append(t_)
    num_added_tokens = pipe.tokenizer.add_tokens(added_tokens)
    pipe.text_encoder.resize_token_embeddings(len(pipe.tokenizer))
    for token_ in data.keys():
        ref_token = pipe.tokenizer.tokenize(token_)
        ref_indx = pipe.tokenizer.convert_tokens_to_ids(ref_token)[0]
        embd_cur = data[token_].to(device).to(dtype=torch.float16)
        pipe.text_encoder.text_model.embeddings.token_embedding.weight[ref_indx] = embd_cur


def save_rev_samples(output_path, path_to_embed, model_id, device):
    if not os.path.exists(f"{output_path}/final_samples"):
        os.mkdir(f"{output_path}/final_samples")
    prompts_title = ["Vl", "Vr", "Vl Vr"]
    prompt_to_vec = {}
    assert os.path.exists(path_to_embed)
    data = torch.load(path_to_embed)
    combined = []
    prompts = []
    for w_ in data.keys():
        prompt_to_vec[w_] = data[w_]
        combined.append(w_)
        prompts.append(w_)
    prompts.append(" ".join(combined))

    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16, safety_checker=None, requires_safety_checker=False).to(device)
    load_tokens(pipe, prompt_to_vec, device)

    print("Prompts loaded to pipe ...")
    print(prompt_to_vec.keys())

    gen_seeds = [4321, 95, 11, 87654]
    num_images_per_seed = 10

    plt.figure(figsize=(20,20))
    for i in range(len(prompts)):
        images_per_seed = []
        for gen_seed in gen_seeds:
            with torch.no_grad():
                torch.manual_seed(gen_seed)
                images = pipe(prompt=[prompts[i]] * num_images_per_seed, num_inference_steps=25, guidance_scale=7.5).images
            images_per_seed.extend(images)
        
        # plot results
        plot_stacked = []
        for j in range(int(len(images_per_seed) / 4)):
            images_staked_h = np.hstack([np.asarray(img) for img in images_per_seed[j * 4:j * 4 + 4]])
            plot_stacked.append(images_staked_h)
        im_stack = np.vstack(plot_stacked)
        plt.subplot(1,len(prompts), i + 1)
        plt.imshow(im_stack)
        plt.axis("off")
        plt.title(prompts_title[i], size=24)
    plt.savefig(f"{output_path}/final_samples.jpg")


def generate_training_data(code_path, node, output_path, device, model_id, model_id_clip):
    node_code = torch.load(code_path)[node]
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16, safety_checker=None, requires_safety_checker=False).to(device)
    load_tokens(pipe, {"<*>": node_code}, device)
    print("Prompts loaded to pipe ...")
    gen_seeds = [4321, 95, 11, 87654]
    num_images_per_seed = 10

    clip_model = CLIPModel.from_pretrained(model_id_clip)
    preprocess = CLIPImageProcessor.from_pretrained(model_id_clip)

    plt.figure(figsize=(20,20))
    images_per_seed = []
    for gen_seed in gen_seeds:
        with torch.no_grad():
            torch.manual_seed(gen_seed)
            images = pipe(prompt=["<*>"] * num_images_per_seed, num_inference_steps=25, guidance_scale=7.5).images
        images_per_seed.extend(images)

    # saves the clip embeddings for all images
    with torch.no_grad():
        images_preprocess = [preprocess(image, return_tensors="pt")["pixel_values"] for image in images_per_seed]
        stacked_images = torch.cat(images_preprocess)
        embedding_a = clip_model.get_image_features(stacked_images)
        emb_norm = torch.norm(embedding_a, dim=1)
        clip_embed_all = embedding_a / emb_norm.unsqueeze(1)
    
    sim_mat = (clip_embed_all @ clip_embed_all.T)
    mean_over_rows = sim_mat.mean(dim=0)
    sorted_inds = np.array(mean_over_rows).argsort()[-10:][::-1]
    
    best_images = []
    for j in sorted_inds:
        best_images.append(images_per_seed[j])
        images_per_seed[j].save(f"{output_path}/{j}.jpg")

    # plot results
    plot_stacked = []
    for j in range(int(len(images_per_seed) / 4)):
        images_staked_h = np.hstack([np.asarray(img) for img in images_per_seed[j * 4:j * 4 + 4]])
        plot_stacked.append(images_staked_h)
    im_stack = np.vstack(plot_stacked)
    plt.subplot(1, 2, 1)
    plt.imshow(im_stack)
    plt.axis("off")
    plt.title(node, size=24)
    
    plot_stacked = []
    for j in range(int(len(best_images) / 5)):
        images_staked_h = np.hstack([np.asarray(img) for img in best_images[j * 5:j * 5 + 5]])
        plot_stacked.append(images_staked_h)
    im_stack = np.vstack(plot_stacked)
    plt.subplot(1, 2, 2)
    plt.imshow(im_stack)
    plt.axis("off")
    plt.title("Chosen", size=24)
    
    if not os.path.exists(f"{output_path}/generated_images_summary"):
        os.mkdir(f"{output_path}/generated_images_summary")
    plt.savefig(f"{output_path}/generated_images_summary/final_samples.jpg")

    del clip_model
    del pipe
    torch.cuda.empty_cache()
    


def save_children_nodes(parent_node, children_node_path, concept_output_path, device, MODEL_ID, MODEL_ID_CLIP):
    node_number = int(parent_node[1:])
    left_child_number = node_number * 2 + 1
    right_child_number = left_child_number + 1
    data = torch.load(children_node_path)
    left_child_code = {f"v{left_child_number}" :data["<*>"]}
    right_child_code = {f"v{right_child_number}": data["<&>"]}
    if not os.path.exists(f"{concept_output_path}/v{left_child_number}"):
        os.mkdir(f"{concept_output_path}/v{left_child_number}")
    if not os.path.exists(f"{concept_output_path}/v{right_child_number}"):
        os.mkdir(f"{concept_output_path}/v{right_child_number}")
    torch.save(left_child_code, f"{concept_output_path}/v{left_child_number}/embeds.bin")
    torch.save(right_child_code, f"{concept_output_path}/v{right_child_number}/embeds.bin")
    print(f"Results saved to:\n[{concept_output_path}/v{left_child_number}/embeds.bin]\n[{concept_output_path}/v{right_child_number}/embeds.bin]")

    files_l = glob.glob(f"{concept_output_path}/v{left_child_number}/*.png") + glob.glob(f"{concept_output_path}/v{left_child_number}/*.jpg") + glob.glob(f"{concept_output_path}/v{left_child_number}/*.jpeg")
    files_r = glob.glob(f"{concept_output_path}/v{right_child_number}/*.png") + glob.glob(f"{concept_output_path}/v{right_child_number}/*.jpg") + glob.glob(f"{concept_output_path}/v{right_child_number}/*.jpeg")
    if not len(files_l):
        generate_training_data(f"{concept_output_path}/v{left_child_number}/embeds.bin", f"v{left_child_number}", f"{concept_output_path}/v{left_child_number}", device, MODEL_ID, MODEL_ID_CLIP)
    if not len(files_r):
        generate_training_data(f"{concept_output_path}/v{right_child_number}/embeds.bin", f"v{right_child_number}", f"{concept_output_path}/v{right_child_number}", device, MODEL_ID, MODEL_ID_CLIP)




