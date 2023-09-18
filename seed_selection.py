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

import matplotlib.pyplot as plt
from PIL import Image
import torch
import numpy as np
import os
import math
import torch
from diffusers import StableDiffusionPipeline
from IPython.display import clear_output
from transformers import CLIPImageProcessor, CLIPModel
from PIL import Image
import re
import argparse
import utils

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_to_new_tokens", type=str, help="Path to directory with the new embeddings")
    parser.add_argument("--node", type=str, help="Path to directory with the new embeddings")
    parser.add_argument("--model_id", type=str, default="runwayml/stable-diffusion-v1-5")
    parser.add_argument("--model_id_clip", type=str, default="openai/clip-vit-base-patch32")
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.device = device
    return args


def get_tree_tokens(args, seeds):
    """
    Load the learned tokens into "prompt_to_vec" dict, there should be two new tokens per node per seed.
    """
    prompt_to_vec = {}
    prompts_per_seed = {}
    for seed in seeds:
        path_to_embed = f"{args.path_to_new_tokens}/{args.node}/{args.node}_seed{seed}/learned_embeds-steps-200.bin"
        assert os.path.exists(path_to_embed)
        data = torch.load(path_to_embed)
        prompts_per_seed[seed] = []
        combined = []
        for w_ in data.keys():
            key_ = w_.replace("<", "").replace(">","")
            new_key = f"<{key_}_{seed}>" # <*_seed> / <&_seed>
            prompt_to_vec[new_key] = data[w_]
            combined.append(new_key)
            prompts_per_seed[seed].append(new_key)
        prompts_per_seed[seed].append(" ".join(combined))
    return prompt_to_vec, prompts_per_seed


if __name__ == "__main__":
    args = parse_args()
    if not os.path.exists(f"{args.path_to_new_tokens}/{args.node}/consistency_test"):
        os.mkdir(f"{args.path_to_new_tokens}/{args.node}/consistency_test")
    seeds = [0, 1000, 1234, 111]
    prompts_title = ["Vl", "Vr", "Vl Vr"]
    prompt_to_vec, prompts_per_seed = get_tree_tokens(args, seeds)
    # prompts_per_seed is {seed: ["<*_seed>", "<&_seed>", "<*_seed> <&_seed>"]}
    print(prompts_per_seed)
    
    pipe = StableDiffusionPipeline.from_pretrained(args.model_id, torch_dtype=torch.float16, safety_checker=None, requires_safety_checker=False).to(args.device)
    utils.load_tokens(pipe, prompt_to_vec, args.device)

    print("Prompts loaded to pipe ...")
    print(prompt_to_vec.keys())

    clip_model = CLIPModel.from_pretrained(args.model_id_clip)
    preprocess = CLIPImageProcessor.from_pretrained(args.model_id_clip)

    prompts_to_images = {}
    prompts_to_clip_embeds = {}
    final_sim_score = {}
    gen_seeds = [4321, 95, 11, 87654]
    num_images_per_seed = 10
    for seed in seeds:
        plt.figure(figsize=(20,15))
        prompts_to_images[seed] = {}
        prompts_to_clip_embeds[seed] = {}
        cur_prompts = prompts_per_seed[seed]
        for i in range(len(cur_prompts)):
            images_per_seed = []
            for gen_seed in gen_seeds:
                with torch.no_grad():
                    torch.manual_seed(gen_seed)
                    images = pipe(prompt=[cur_prompts[i]] * num_images_per_seed, num_inference_steps=25, guidance_scale=7.5).images
                images_per_seed.extend(images)
            
            # plot results
            plot_stacked = []
            for j in range(int(len(images_per_seed) / 4)):
                images_staked_h = np.hstack([np.asarray(img) for img in images_per_seed[j * 4:j * 4 + 4]])
                plot_stacked.append(images_staked_h)
            im_stack = np.vstack(plot_stacked)
            plt.subplot(1,len(cur_prompts) + 1, i + 1)
            plt.imshow(im_stack)
            plt.axis("off")
            plt.title(prompts_title[i], size=24)

            # saves the clip embeddings for all images
            images_preprocess = [preprocess(image, return_tensors="pt")["pixel_values"] for image in images_per_seed]
            stacked_images = torch.cat(images_preprocess)
            embedding_a = clip_model.get_image_features(stacked_images)
            emb_norm = torch.norm(embedding_a, dim=1)
            embedding_a = embedding_a / emb_norm.unsqueeze(1)

            prompts_to_images[seed][cur_prompts[i]] = images_per_seed
            prompts_to_clip_embeds[seed][cur_prompts[i]] = embedding_a

        # sim matrix per seed
        cur_prompts = prompts_per_seed[seed]
        num_prompts = len(cur_prompts)
        sim_matrix = np.zeros((num_prompts, num_prompts))
        for i, k1 in enumerate(cur_prompts):
            for j, k2 in enumerate(cur_prompts):
                sim_mat = (prompts_to_clip_embeds[seed][k1] @ prompts_to_clip_embeds[seed][k2].T)
                if k1 == k2: 
                    sim_ = torch.triu(sim_mat, diagonal=1).sum() / torch.triu(torch.ones(sim_mat.shape), diagonal=1).sum()
                else:
                    sim_ = sim_mat.mean()
                sim_matrix[i, j] = sim_
        plt.subplot(1, len(cur_prompts) + 1, len(cur_prompts) + 1)
        plt.imshow(sim_matrix, vmin=0.4, vmax=0.9)
        for i, k1 in enumerate(prompts_title):
            plt.text(i, -0.9, f"{k1}", ha="center", va="center", size=16)
        for i, k2 in enumerate(prompts_title):
            plt.text(-1, i, f"{k2}", ha="center", va="center", size=16)
        for x in range(sim_matrix.shape[1]):
            for y in range(sim_matrix.shape[0]):
                plt.text(x, y, f"{sim_matrix[y, x]:.2f}", ha="center", va="center", size=18)
        plt.xlim([-1.5, len(cur_prompts) - 0.5])
        plt.ylim([len(cur_prompts)- 0.5, -1.5])
        plt.axis("off")
        
        
        s_l, s_r, s_lr = sim_matrix[0, 0], sim_matrix[1, 1], sim_matrix[0, 1]
        final_sim_score[seed] = (s_l + s_r) + (min(s_l, s_r) - s_lr) 
        plt.suptitle(f"Seed Score [{final_sim_score[seed]:.2f}]", size=28)
        plt.savefig(f"{args.path_to_new_tokens}/{args.node}/consistency_test/seed{seed}.jpg")
    torch.save(final_sim_score, f"{args.path_to_new_tokens}/{args.node}/consistency_test/seed_scores.bin")
    print(final_sim_score)
    
