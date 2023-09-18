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

import argparse
import multiprocessing as mp
import os
import subprocess as sp
import sys
import torch
from shutil import copyfile
import utils
import glob

MODEL_ID = "runwayml/stable-diffusion-v1-5"
MODEL_ID_CLIP = "openai/clip-vit-base-patch32"
device = "cuda" if torch.cuda.is_available() else "cpu"

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent_data_dir", type=str, help="Path to directory with the training samples")
    parser.add_argument("--node", type=str, help="which node to split (v0, v1..) the corresponding images should be under 'parent_data_dir/vi'")
    parser.add_argument("--test_name", type=str, default="test", help="your GPU id")
    parser.add_argument("--max_train_steps", type=int, default=201, help="your GPU id")
    parser.add_argument("--GPU_ID", type=int, default=0, help="your GPU id")
    parser.add_argument("--multiprocess", type=int, default=0)
    
    args = parser.parse_args()
    return args


def run_seed(args, seed):
    print("seed", seed)
    exit_code = sp.run(["accelerate", "launch", "--gpu_ids", f"{args.GPU_ID}", "textual_inversion_decomposed.py",
                        "--train_data_dir", f"input_concepts/{args.parent_data_dir}/{args.node}",
                        "--placeholder_token", "<*> <&>",
                        "--validation_prompt", "<*>,<&>,<*> <&>",
                        "--output_dir", f"outputs/{args.parent_data_dir}/{args.node}/{args.test_name}_seed{seed}/",
                        "--seed", f"{seed}",
                        "--max_train_steps", f"{args.max_train_steps}",
                        "--validation_steps", "100"
                        ])
    if exit_code.returncode:
        sys.exit(1)


if __name__ == "__main__":
    args = parse_args()

    training_data_dir = f"input_concepts/{args.parent_data_dir}/{args.node}"
    if not os.path.exists(training_data_dir):
        raise AssertionError("There is no data in " + training_data_dir)
    files = glob.glob(f"{training_data_dir}/*.png") + glob.glob(f"{training_data_dir}/*.jpg") + glob.glob(f"{training_data_dir}/*.jpeg")

    if not len(files) > 1:
        if not os.path.exists(f"{training_data_dir}/embeds.bin"):
            raise AssertionError("There is no child code in [" + training_data_dir + "/embeds.bin] to generate the data. Please run with parent node first.")
        print("Generating dataset...")
        utils.generate_training_data(f"{training_data_dir}/embeds.bin", args.node, training_data_dir, device, MODEL_ID, MODEL_ID_CLIP)

    # run textual inversion for 200 steps
    if args.multiprocess:
        ncpus = 10
        P = mp.Pool(ncpus)  # Generate pool of workers

    seeds = [0, 1000, 1234, 111]
    for seed in seeds:
        if args.multiprocess:
            P.apply_async(run_seed, (args, seed))
        else:
            run_seed(args, seed)
    
    if args.multiprocess:
        P.close()
        P.join()  # start processes
    

    # Run seed selection
    sp.run(["python", "seed_selection.py",
            "--path_to_new_tokens", f"outputs/{args.parent_data_dir}", 
            "--node", f"{args.node}"])
    seeds_scores = torch.load(f"outputs/{args.parent_data_dir}/{args.node}/consistency_test/seed_scores.bin")
    best_seed = max(seeds_scores, key=lambda k: seeds_scores[k])
    print(f"Best seed [{best_seed}]")

    # Continue textual inversion
    print(f"Resume running with seed [{best_seed}]...")
    exit_code = sp.run(["accelerate", "launch", "--gpu_ids", f"{args.GPU_ID}", "textual_inversion_decomposed.py",
                        "--train_data_dir", f"input_concepts/{args.parent_data_dir}/{args.node}",
                        "--placeholder_token", "<*> <&>",
                        "--validation_prompt", "<*>,<&>,<*> <&>",
                        "--output_dir", f"outputs/{args.parent_data_dir}/{args.node}/{args.test_name}_seed{best_seed}/",
                        "--seed", f"{best_seed}",
                        "--max_train_steps", f"{1000}",
                        "--validation_steps", "100",
                        "--resume_from_checkpoint", f"outputs/{args.parent_data_dir}/{args.node}/{args.node}_seed{best_seed}/checkpoint-200",
                        "--checkpointing_steps", "2000"
                        ])

    copyfile(f"outputs/{args.parent_data_dir}/{args.node}/{args.node}_seed{best_seed}/learned_embeds.bin",
         f"outputs/{args.parent_data_dir}/{args.node}/learned_embeds.bin")
    copyfile(f"outputs/{args.parent_data_dir}/{args.node}/{args.node}_seed{best_seed}/learned_embeds-steps-1000.bin",
         f"outputs/{args.parent_data_dir}/{args.node}/learned_embeds-steps-1000.bin")
    
    # Saves some samples of the final node 
    utils.save_children_nodes(args.node, f"outputs/{args.parent_data_dir}/{args.node}/learned_embeds-steps-1000.bin", f"input_concepts/{args.parent_data_dir}", device, MODEL_ID, MODEL_ID_CLIP)
    utils.save_rev_samples(f"outputs/{args.parent_data_dir}/{args.node}", f"outputs/{args.parent_data_dir}/{args.node}/learned_embeds-steps-1000.bin", MODEL_ID, device)
