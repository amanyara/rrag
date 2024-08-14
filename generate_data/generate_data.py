"""
run:
python tt.py \
  --raw_file_path="./seed_data_6/wikiQA_gpt/data/train-00000-of-00001.parquet"
  --model_name="Mistral-7B-Instruct-v0_3" \
  --output_dir="./rewrite_data" \
"""

import fire
from tqdm import tqdm
import pandas as pd
import utils
import time
import json
import multiprocessing


def run_in_parallel(rag, model_name, prompt_t, seed_data):
    generate_data_dict = utils.encode_prompt_and_generate_data(rag_r=rag,
                                                               prompt_path=prompt_t,
                                                               seed_path=seed_data,
                                                               model_name_id=model_name)

    return generate_data_dict


def main(
        raw_file_path="./seed_data_9/wikiQA_gpt/data/train-00000-of-00001.parquet",
        model_name="Qwen1.5-32B-Chat-GPTQ-Int4",
        output_dir="./rewrite_data"
        ):

    prompt_templates_list = ["./prompt_templates/butter_fingers_perturbation.txt",
                             "./prompt_templates/duplicate_punctuation.txt",
                             "./prompt_templates/insert_abbreviation.txt",
                             "./prompt_templates/leet_letters.txt",
                             "./prompt_templates/random_upper_transformation.txt",
                             "./prompt_templates/shuffle_word.txt",
                             "./prompt_templates/visual_attack_letters.txt",
                             "./prompt_templates/whitespace_perturbation.txt",
                             "./prompt_templates/all_type.txt"
                             ]

    seed_data_list = ["./seed_data_9/butter_fingers_perturbation.jsonl",
                      "./seed_data_9/duplicate_punctuation.jsonl",
                      "./seed_data_9/insert_abbreviation.jsonl",
                      "./seed_data_9/leet_letters.jsonl",
                      "./seed_data_9/random_upper_transformation.jsonl",
                      "./seed_data_9/shuffle__word.jsonl",
                      "./seed_data_9/visual_attack.jsonl",
                      "./seed_data_9/whitespace_perturbation.jsonl",
                      "./seed_data_9/all_type.jsonl"
                      ]

    start_time = time.time()
    df = pd.read_parquet(raw_file_path)

    all_processed_data_list = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing Data"):
        rag = row["context"]
        input_ = row["question"]
        output = row["reworded_answer"]
        all_generate_data_dict = {"rag": rag, "input": input_, "output": output}
        # 创建进程池
        with multiprocessing.Pool(processes=len(prompt_templates_list)) as pool:
            # 将参数传递到并行进程中
            results = pool.starmap(run_in_parallel, zip([rag]*len(prompt_templates_list),
                                                        [model_name]*len(prompt_templates_list),
                                                        prompt_templates_list,
                                                        seed_data_list))

        for r in results:
            all_generate_data_dict.update(r)

        # 关闭进程
        pool.close()
        pool.join()

        all_processed_data_list.append(all_generate_data_dict)

        if len(all_processed_data_list) < 2:
            continue
        else:

            with open(f'{output_dir}/test008.jsonl', 'w', encoding='utf-8') as f:
                for item in all_processed_data_list:
                    json_f = json.dumps(item, ensure_ascii=False)
                    f.write(json_f + '\n')

            print(f"Final total generated instructions: {len(all_processed_data_list)}")
            end_time = time.time()
            print("程序运行时间：", end_time - start_time, "秒")
            break


if __name__ == "__main__":
    fire.Fire(main())



