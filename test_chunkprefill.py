import os
import time
import argparse
import random
import string
from pathlib import Path
import torch
 
from vllm import LLM, SamplingParams
from datasets import load_dataset, Features, Value, Sequence
from transformers import AutoTokenizer


# os.environ["ASCEND_RT_VISIBLE_DEVICES"]="0,1,2,3,4,5,6,7"
os.environ["ASCEND_RT_VISIBLE_DEVICES"]="8,9,10,11,12,13,14,15"
# os.environ["ASCEND_RT_VISIBLE_DEVICES"]="12,13,14,15"
os.environ["VLLM_ASCEND_ENABLE_MLP_OPTIMIZE"] = "1"
# os.environ["VLLM_ASCEND_ENABLE_CP"]="1"
os.environ["VLLM_ASCEND_ENABLE_CONTEXT_PARALLEL"] = "1"
os.environ["VLLM_VERSION"] = "0.11.0"
os.environ["ASCEND_LAUNCH_BLOCKING"]="1"
# os.environ["ASCEND_LAUNCH_BLOCKING"]="0"
os.environ["VLLM_LOGGING_LEVEL"]="DEBUG"
os.environ["ASDOPS_LOG_TO_FILE"]="1"
os.environ["ASDOPS_LOG_LEVEL"]="INFO"

def generate_prompts(prompt, length):
    model_path = "/mnt/share/weights/DeepSeek-V2-Lite/"
    # model_path = "/mnt/share/t00608739/Qwen3-30B-A3B-Instruct-2507"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    encoded = tokenizer(
        prompt[0],
        truncation=False,  
        return_tensors="pt"
    )
    sample_length = encoded["input_ids"].shape[1]
    print(f"生成前tokens: {sample_length} tokens")
    decoded_str = tokenizer.decode(encoded["input_ids"][0,:length], skip_special_tokens=True)
    encoded = tokenizer(
        decoded_str,
        truncation=False,  
        return_tensors="pt"
    )
    second_ = encoded["input_ids"].shape[1]
    print(f"生成后tokens: {second_} tokens")
    return decoded_str

def check_token_len(model_path, prompts):
    for i, prompt in enumerate(prompts):
        print(f">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>yyyyyyy")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        encoded = tokenizer(
            prompt,
            truncation=False,  # 不截断，我们自己控制长度
            return_tensors="pt"
        )
            
        # 获取当前样本的token长度
        sample_length = encoded["input_ids"].shape[1]
        print(f"提示词{i}的长度: {sample_length} tokens")

if __name__ == "__main__":
    torch.cuda.empty_cache()
    parser = argparse.ArgumentParser()
 
    parser.add_argument('--input_len', type=int, default=1024)
    parser.add_argument('--output_len', type=int, default=50)
    parser.add_argument('--bs', type=int, default=1)
    # parser.add_argument('--model_path', type=str, default="/mnt/weight/deepseekv3-lite-base-latest")
    parser.add_argument('--model_path', type=str, default="/mnt/share/weights/DeepSeek-V2-Lite/")
    # parser.add_argument('--model_path', type=str, default="/mnt/share/t00608739/Qwen3-30B-A3B-Instruct-2507")
    parser.add_argument('--tp', type=int, default=2)   # 4 to 8
    parser.add_argument('--cp', type=int, default=2)   # 4 to 8
    parser.add_argument('--dcp', type=int, default=2)   # 4 to 8
    parser.add_argument('--profiler_dir', type=str, default=None)
    parser.add_argument('-p', '--profiling', action="store_true")
    parser.add_argument('--iter_times', type=int, default=1)
    parser.add_argument('-c', '--enable_chunked_prefill', default=True)
 
    args = parser.parse_args()

    def generate_odd_queue_string(length):
        return ' '.join(str(2*i + 1) for i in range(length))
    
 
    sampling_params = SamplingParams(temperature = 0, max_tokens=args.output_len)
    # sampling_params = SamplingParams(temperature = 0.8, top_p = 0.95, max_tokens=args.output_len)
    # sampling_params = SamplingParams(temperature = 0.6, top_k = 40, top_p = 0.95, repetition_penalty = 1.03, ignore_eos=True,  max_tokens=args.output_len)
    llm = LLM(model=args.model_path,
          trust_remote_code=True,
          enforce_eager=True,
          tensor_parallel_size=args.tp,  # tp=8
          # context_parallel_size=args.cp,  # tp=8
          prefill_context_parallel_size=args.cp,  # tp=8
          decode_context_parallel_size=args.dcp,
          enable_prefix_caching=False,
          enable_expert_parallel=False,
          enable_chunked_prefill=True,
          max_num_batched_tokens=64, #1024, #16384  1024  74000  131072
          max_model_len=4096,   # 128K  131072
          additional_config={"ascend_scheduler_config": {"enabled": False}},
          compilation_config={"cudagraph_mode":"FULL_DECODE_ONLY"},
          # max_num_seqs=1,
          block_size=128,
          cp_kv_cache_interleave_size=32,
          gpu_memory_utilization=0.9  # Ä¬ÈÏÖµ0.9
          )
 
    base = 400
    for i in range(1):
        # prompts = [
        #     generate_odd_queue_string(base)+" " 
        # ]
        # prompt1 = ["Scarlett O'Hara was not beautiful, but men seldom realized it when caught by her charm as the Tarleton twins were.  In her face were too sharply blended the delicate features of her mother, a Coast aristocrat of French descent, and the heavy ones of her florid Irish father.  But it was an arresting face, pointed of chin, square of jaw.  Her eyes were pale green without a touch of hazel, starred with bristly black lashes and slightly tilted at the ends. Above them, her thick black brows slanted upward, cutting a startling oblique line in her magnolia-white skin--that skin so prized by Southern women and so carefully guarded with bonnets, veils and mittens against hot Georgia suns. Seated with Stuart and Brent Tarleton in the cool shade of the porch of Tara, her father's plantation, that bright April afternoon of 1861, she made a pretty picture.  Her new green flowered-muslin dress spread its twelve yards of billowing material over her hoops and exactly matched the flat-heeled green morocco slippers her father had recently brought her from Atlanta. The dress set off to perfection the seventeen-inch waist, the smallest in three counties, and the tightly fitting basque showed breasts well matured for her sixteen years.  But for all the modesty of her spreading skirts, the demureness of hair netted smoothly into a chignon and the quietness of small white hands folded in her lap, her true self was poorly concealed. The green eyes in the carefully sweet face were turbulent, willful, lusty with life, distinctly at variance with her decorous demeanor. Her"]
        # prompt1 = generate_prompts(prompt1,257)
        # prompt2 = ["Scarlett O'Hara was not beautiful, but men seldom realized it when caught by her charm as the Tarleton twins were.  In her face were too sharply blended the delicate features of her mother, a Coast aristocrat of French descent, and the heavy ones of her florid Irish father.  But it was an arresting face, pointed of chin, square of jaw.  Her eyes were pale green without a touch of hazel, starred with bristly black lashes and slightly tilted at the ends. Above them, her thick black brows slanted upward, cutting a startling oblique line in her magnolia-white skin--that skin so prized by Southern women and so carefully guarded with bonnets, veils and mittens against hot Georgia suns. Seated with Stuart and Brent Tarleton in the cool shade of the porch of Tara, her father's plantation, that bright April afternoon of 1861, she made a pretty picture.  Her new green flowered-muslin dress spread its twelve yards of billowing material over her hoops and exactly matched the flat-heeled green morocco slippers her father had recently brought her from Atlanta. The dress set off to perfection the seventeen-inch waist, the smallest in three counties, and the tightly fitting basque showed breasts well matured for her sixteen years.  But for all the modesty of her spreading skirts, the demureness of hair netted smoothly into a chignon and the quietness of small white hands folded in her lap, her true self was poorly concealed. The green eyes in the carefully sweet face were turbulent, willful, lusty with life, distinctly at variance with her decorous demeanor. Her"]
        # prompt2 = generate_prompts(prompt2,257)
        # prompts = [prompt1, prompt2]
        # prompts = [
        #     # "The capital of France is", # (6 + 2)  / 2 = 4
        #     # "Hello, my name is Tom, I am", # (9 + 3) / 2
        #     # "The president of United States is", # (7 + 1) / 2 = 4
        #     # "AI future is", # 4 / 2
        #     # "Janet\u2019s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?",
        #     # "Hello, my name is Tom, I am", # (9 + 3) / 2
        #     # generate_prompts(["Hello, my name is Tom, I am"], 256),
        #     # "The president of United States is", # (7 + 1) / 2 = 4
        #     # "While Joanne is gathering apples from her family\u2019s orchard, her sister comes outside to help her. Joanne gathers 30 apples from the tallest trees, half this amount from the shortest trees, and more apples from the average trees. Compared with Joanne, her sister gathers twice as many apples from the tallest trees and 3 times as many apples from the shortest trees. She doesn't take any from the average trees. If the sisters have gathered a combined total of 500 apples, how many apples did Joanne gather from the average trees? Please give me the answer.", # 4 / 2
        #     # "While Joanne is gathering apples from her family\u2019s orchard, her sister comes outside to help her. Joanne gathers 30 apples from the tallest trees, half this amount from the shortest trees, and more apples from the average trees. Compared with Joanne, her sister gathers twice as many apples from the tallest trees and 3 times as many apples from the shortest trees. She doesn't take any from the average trees. If the sisters have gathered a combined total of 500 apples, how many apples did Joanne gather from the average trees?", # 4 / 2
        #     # "Kim has started his own housekeeping business and is calculating how much profit he will make from his clients. He already has 3 clients, but is talking to another 5 potential clients and feels confident enough to include them in his calculations. Each client\u2019s home will need 2 bottles of bleach and a pack of cloths to clean. Bottles of bleach will cost $2 each and packs of cloths will cost $5 each. These are his only expenses. He calculates that his total income each week will be $92. Profit is the difference between total income and total expenses, so how much profit, in dollars, will Lucas make each week? Please give me the answer.",
        #     generate_prompts(["Kim has started his own housekeeping business and is calculating how much profit he will make from his clients. He already has 3 clients, but is talking to another 5 potential clients and feels confident enough to include them in his calculations. Each client\u2019s home will need 2 bottles of bleach and a pack of cloths to clean. Bottles of bleach will cost $2 each and packs of cloths will cost $5 each. These are his only expenses. He calculates that his total income each week will be $92. Profit is the difference between total income and total expenses, so how much profit, in dollars, will Lucas make each week? Please give me the answer."], 256),
        #     "The capital of France is", # (6 + 2)  / 2 = 4
        #     "AI future is", # 4 / 2
        #     "The president of United States is", # (7 + 1) / 2 = 4
        #     "Hello, my name is Tom, I am", # (9 + 3) / 2
        # ]
        # prompts = [
        #     "The capital of France is"*128
        # ]
        prompts = [ 
        "Answer the following question.The last line of the response should follow this format: \"answer:$ANSWER\" (without quotes), where ANSWER is a number. Let's think step by step.\n\nQuestion: Janet’s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?",
        "Answer the following question.The last line of the response should follow this format: \"answer:$ANSWER\" (without quotes), where ANSWER is a number. Let's think step by step.\n\nQuestion: Every day, Wendi feeds each of her chickens three cups of mixed chicken feed, containing seeds, mealworms and vegetables to help keep them healthy.  She gives the chickens their feed in three separate meals. In the morning, she gives her flock of chickens 15 cups of feed.  In the afternoon, she gives her chickens another 25 cups of feed.  How many cups of feed does she need to give her chickens in the final meal of the day if the size of Wendi's flock is 20 chickens?",
        "Answer the following question.The last line of the response should follow this format: \"answer:$ANSWER\" (without quotes), where ANSWER is a number. Let's think step by step.\n\nQuestion: Carla is downloading a 200 GB file. Normally she can download 2 GB/minute, but 40% of the way through the download, Windows forces a restart to install updates, which takes 20 minutes. Then Carla has to restart the download from the beginning. How load does it take to download the file?",
        "Answer the following question.The last line of the response should follow this format: \"answer:$ANSWER\" (without quotes), where ANSWER is a number. Let's think step by step.\n\nQuestion: John drives for 3 hours at a speed of 60 mph and then turns around because he realizes he forgot something very important at home.  He tries to get home in 4 hours but spends the first 2 hours in standstill traffic.  He spends the next half-hour driving at a speed of 30mph, before being able to drive the remaining time of the 4 hours going at 80 mph.  How far is he from home at the end of those 4 hours?"
        ]
        # prompts = [
        #     "Kim has started his own housekeeping business and is calculating how much profit he will make from his clients. He already has 3 clients, but is talking to another 5 potential clients and feels confident enough to include them in his calculations. Each client\u2019s home will need 2 bottles of bleach and a pack of cloths to clean. Bottles of bleach will cost $2 each and packs of cloths will cost $5 each. These are his only expenses. He calculates that his total income each week will be $92. Profit is the difference between total income and total expenses, so how much profit, in dollars, will Lucas make each week?",
	    #     "The capital of France is",
        #     "Hello, my name is Tom, I am",
        #     "The president of United States is"
        # ]
        check_token_len("/mnt/share/weights/DeepSeek-V2-Lite/", prompts)
        # check_token_len("/mnt/share/t00608739/Qwen3-30B-A3B-Instruct-2507", prompts)
        t0 = time.time()
        for _ in range(args.iter_times):
            outputs = llm.generate(prompts, sampling_params)
        t1 = time.time()
        print(f"TTFT: {(t1 - t0) * 1000 / (args.iter_times * args.bs)} ms")
     
        for i, output in enumerate(outputs):
            prompt = output.prompt
            generated_text = output.outputs[0].text
            # print(
            #     f"req_num: {i}\nGenerated text: {generated_text!r}"
            # )
            prompt = prompt.split(" ")
            print(
                #f"prompt:{prompt}\n"
                #f"req_num: {i}\n[{prompt}] -> Generated text: {generated_text!r}\n"
                f"req_num: {i}\n[{prompt[-5:]}] -> Generated text: {generated_text!r}\n"
                f"Token ids: {output.outputs[0].token_ids}\n"
            )
     
    print("end.")

