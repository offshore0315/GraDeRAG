import json
import pandas as pd
import os
import random

# --- 配置区 ---
WEBQSP_PATH = 'G:\GraDeRAG\Data\WebQSP.train.json'
GRAILQA_PATH = 'G:\GraDeRAG\Data\grailqa_v1.0_train.json'
OUTPUT_CSV_PATH = 'data/annotation_sheet_source.csv'

# --- 修改后的配置 ---
SAMPLES_PER_DATASET = 600
RANDOM_SEED = 42
FINAL_COLUMNS = [
    'task_id', 'original_question', 'source_dataset', 'correct_answer',
    'correct_path_cypher', 'malicious_answer', 'malicious_path_cypher',
    'annotator', 'status'
]


# --- 配置区结束 ---

def process_webqsp(file_path):
    """
    处理WebQSP数据集。
    新逻辑：只提取同时包含 'EntityName' 和 'AnswerArgument' 的答案。
    """
    print(f"Processing WebQSP from: {file_path} with strict filtering...")
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    valid_records = []
    for q_data in data['Questions']:
        # --- 核心修改点：严格过滤和格式化 ---
        valid_answers = []
        for parse in q_data['Parses']:
            for ans in parse['Answers']:
                name = ans.get('EntityName')
                entity_id = ans.get('AnswerArgument')

                # 必须同时存在name和id，且不为空字符串
                if name and entity_id:
                    formatted_answer = f"{name} ({entity_id})"
                    valid_answers.append(formatted_answer)

        # 如果这个问题下没有任何一个答案满足条件，则跳过此问题
        if not valid_answers:
            continue

        # 去重后记录
        unique_answers = sorted(list(set(valid_answers)))
        record = {
            'task_id': f"webqsp_{q_data['QuestionId']}",
            'original_question': q_data['ProcessedQuestion'],
            'source_dataset': 'webqsp',
            'correct_answer': ' | '.join(unique_answers)
        }
        valid_records.append(record)

    print(f"Found {len(valid_records)} valid records in WebQSP.")
    return valid_records


def process_grailqa(file_path):
    """
    处理GrailQA数据集。
    新逻辑：只提取同时包含 'friendly_name' 和 'answer_argument' 的答案。
    """
    print(f"Processing GrailQA from: {file_path} with strict filtering...")
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    valid_records = []
    for q_data in data:
        # --- 核心修改点：严格过滤和格式化 ---
        valid_answers = []
        for ans in q_data['answer']:
            name = ans.get('friendly_name')
            entity_id = ans.get('answer_argument')

            # 必须同时存在name和id，且不为空字符串
            if name and entity_id:
                formatted_answer = f"{name} ({entity_id})"
                valid_answers.append(formatted_answer)

        # 如果这个问题下没有任何一个答案满足条件，则跳过此问题
        if not valid_answers:
            continue

        unique_answers = sorted(list(set(valid_answers)))
        record = {
            'task_id': f"grailqa_{q_data['qid']}",
            'original_question': q_data['question'],
            'source_dataset': 'grailqa',
            'correct_answer': ' | '.join(unique_answers)
        }
        valid_records.append(record)

    print(f"Found {len(valid_records)} valid records in GrailQA.")
    return valid_records


def main():
    """主执行函数，包含新的采样逻辑"""
    os.makedirs('data', exist_ok=True)
    all_records = []

    random.seed(RANDOM_SEED)

    # --- 新的采样逻辑 ---
    # 1. 先处理整个文件，得到所有符合高质量标准的记录
    # 2. 然后从这些高质量记录中进行采样
    if os.path.exists(WEBQSP_PATH):
        webqsp_valid_records = process_webqsp(WEBQSP_PATH)
        num_to_sample = min(SAMPLES_PER_DATASET, len(webqsp_valid_records))
        print(f"Sampling {num_to_sample} records from WebQSP valid pool...")
        all_records.extend(random.sample(webqsp_valid_records, k=num_to_sample))
    else:
        print(f"Warning: WebQSP file not found at {WEBQSP_PATH}")

    if os.path.exists(GRAILQA_PATH):
        grailqa_valid_records = process_grailqa(GRAILQA_PATH)
        num_to_sample = min(SAMPLES_PER_DATASET, len(grailqa_valid_records))
        print(f"Sampling {num_to_sample} records from GrailQA valid pool...")
        all_records.extend(random.sample(grailqa_valid_records, k=num_to_sample))
    else:
        print(f"Warning: GrailQA file not found at {GRAILQA_PATH}")

    if not all_records:
        print("No data processed that meets the strict criteria. Exiting.")
        return

    df = pd.DataFrame(all_records)

    df['correct_path_cypher'] = ''
    df['malicious_answer'] = ''
    df['malicious_path_cypher'] = ''
    df['annotator'] = ''
    df['status'] = 'pending'
    df = df[FINAL_COLUMNS]
    df = df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

    df.to_csv(OUTPUT_CSV_PATH, index=False, encoding='utf-8-sig')

    print("-" * 50)
    print(f"Successfully created the high-quality annotation sheet!")
    print(f"File saved at: {OUTPUT_CSV_PATH}")
    print(f"Total tasks prepared: {len(df)}")
    print("Each 'correct_answer' now follows the 'Content (ID)' format.")
    print("-" * 50)


if __name__ == '__main__':
    main()