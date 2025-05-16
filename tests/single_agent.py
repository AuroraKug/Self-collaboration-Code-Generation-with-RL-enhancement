import os
import sys
import json
import argparse
import tqdm
import time

# Add project root to sys.path to allow imports from core, utils, etc.
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from datasets import load_dataset
from utils import prompt_split_humaneval, code_truncate
from core.interface import ProgramInterface

def generate_code_simple(problem_prompt_str: str, 
                         model_name: str, 
                         max_tokens: int, 
                         temperature: float, 
                         top_p: float, 
                         itf_instance: ProgramInterface) -> str:
    """
    Calls the language model with a simple prompt and extracts the code.
    problem_prompt_str is typically task['prompt'] from HumanEval.
    """
    messages = [
        {
            "role": "system", 
            "content": "You are a code generation AI. Given a problem description (which might include a function signature and docstring), provide the complete Python function code that solves the problem. Output only the raw Python code block. Do not include any explanations or surrounding text."
        },
        {"role": "user", "content": problem_prompt_str}
    ]

    try:
        responses = itf_instance.run(
            prompt=messages,
            majority_at=1,  # Single generation
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p
        )
        if responses and responses[0]:
            raw_model_output = responses[0]
            extracted_code = code_truncate(raw_model_output)
            return extracted_code if extracted_code.strip() else "error"
        else:
            print("Model returned no response or empty response.")
            return "error"
    except Exception as e:
        print(f"Error calling model or processing response: {e}")
        return "error"

def main():
    parser = argparse.ArgumentParser(description="Single agent baseline for code generation.")
    parser.add_argument('--dataset', type=str, default='humaneval', help='Dataset to use (default: humaneval)')
    parser.add_argument('--lang', type=str, default='python', help='Language (default: python)')
    parser.add_argument('--output_path', type=str, default='hunmaneval_output_gpt-3.5-turbo_baseline.jsonl', help='Path to save the output JSONL file (default: hunmaneval_output_gpt-3.5-turbo_baseline.jsonl)')
    parser.add_argument('--model', type=str, default='gpt-3.5-turbo', help='Model name to use')
    parser.add_argument('--max_tokens', type=int, default=1024, help='Max tokens for model generation')
    parser.add_argument('--temperature', type=float, default=0.0, help='Temperature for model generation')
    parser.add_argument('--top_p', type=float, default=1.0, help='Top_p for model generation')
    
    args = parser.parse_args()

    # Initialize ProgramInterface for model calls
    # Ensure OPENAI_API_KEY environment variable is set for this to work.
    try:
        itf = ProgramInterface(model=args.model, stop='') 
    except Exception as e:
        print(f"Failed to initialize ProgramInterface: {e}. Ensure OPENAI_API_KEY is set.")
        sys.exit(1)

    # Load dataset
    if args.dataset == 'humaneval':
        if args.lang == 'python':
            try:
                dataset = load_dataset("openai_humaneval")
                dataset_key = ["test"]
            except Exception as e:
                print(f"Failed to load humaneval dataset: {e}")
                sys.exit(1)
        else:
            print(f"Language {args.lang} not supported for humaneval dataset in this script.")
            sys.exit(1)
    else:
        print(f"Dataset {args.dataset} not supported in this script.")
        sys.exit(1)

    # Ensure output directory exists
    output_dir = os.path.dirname(args.output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    print(f"Starting generation for {args.dataset} with model {args.model}. Output will be saved to {args.output_path}")

    with open(args.output_path, 'w') as f:
        for key in dataset_key:
            pbar = tqdm.tqdm(dataset[key], total=len(dataset[key]), desc=f"Processing {key}")
            for task in pbar:
                if args.dataset == 'humaneval':
                    task_id = task['task_id']
                    raw_task_prompt = task['prompt'] # This is the full prompt for HumanEval (signature + docstring)
                    entry_point = task['entry_point']
                    test_script = task['test']

                    # Split the raw prompt to get the part before the function definition (e.g., imports)
                    before_func, _, _, _ = prompt_split_humaneval(raw_task_prompt, entry_point)
                    
                    # The problem description passed to the simple model will be the raw_task_prompt
                    extracted_code = generate_code_simple(
                        raw_task_prompt,
                        args.model,
                        args.max_tokens,
                        args.temperature,
                        args.top_p,
                        itf
                    )

                    completion_to_write = ""
                    if extracted_code == "error" or not extracted_code.strip():
                        print(f"Warning: Failed to generate or extract valid code for {task_id}. Completion will be empty.")
                        completion_to_write = "" 
                    else:
                        completion_to_write = extracted_code
                    
                    solution_entry = {
                        'task_id': task_id,
                        'prompt': before_func + "\n",  # Stuff before the function def (e.g., imports)
                        'test': test_script,
                        'entry_point': entry_point,
                        'completion': completion_to_write, # The actual function code generated
                    }
                    f.write(json.dumps(solution_entry) + '\n')
                    f.flush()
    
    print(f"Processing complete. Output written to {args.output_path}")

if __name__ == '__main__':
    main()
