import json
import csv
import time
import spacy
import gensim.downloader
from graderag_attacker import GraDeRAttackerV2  # 导入新版攻击器
from typing import List, Dict, Tuple, Set

from tqdm import tqdm

from data.build_dynamic_vocab import build_vocab_dynamic

# (这里的 build_dynamic_topic_vocabulary 函数保持不变，可以从之前复制过来)


def parse_path_info_final(path_data: Dict) -> Dict:
    """
    从数据条目中解析出路径的详细信息。
    此版本能正确处理单跳和“摘要式”多跳路径。
    """
    start_node_name = path_data["start_node"]["name"].replace("_", " ").lower()
    end_node_name = path_data["end_node"]["name"].replace("_", " ").lower()

    # 提取所有关系类型，并从Freebase路径中取最后一部分作为关键词
    all_rels = [r.split('/')[-1].replace("_", " ").lower()
                for r in path_data["relationship_types"]]

    # --- 为适应度函数准备数据 ---
    # 1. 节点别名 (NodeCov): 只包含起点和终点，因为中间节点未知
    node_aliases = {
        start_node_name: {start_node_name},
        end_node_name: {end_node_name}
    }

    # 2. 关系关键词 (RelCov): 包含路径上的所有关系
    rel_keywords = {rel: {rel} for rel in all_rels}

    # 3. 目标别名 (Target): 总是路径的终点
    target_aliases = {end_node_name}

    # 4. 路径三元组 (Adj): 只有在路径是单跳时才能构建
    path_triples = set()
    if len(all_rels) == 1:
        path_triples.add((start_node_name, all_rels[0], end_node_name))

    # 5. 用于构建V_topic的文本
    path_texts = [start_node_name, end_node_name] + all_rels

    return {
        "path_triples": path_triples,
        "node_aliases": node_aliases,
        "rel_keywords": rel_keywords,
        "target_aliases": target_aliases,
        "path_texts": path_texts
    }


def run_all_experiments_v2(
    dataset_path: str, v_base_path: str, output_csv_path: str, attacker_config: Dict
):
    print("--- Initializing Models and Data for V2 Experiment ---")
    with open(dataset_path, 'r') as f:
        dataset = json.load(f)
    with open(v_base_path, 'r') as f:
        v_base = {line.strip() for line in f}

    glove_vectors = gensim.downloader.load('glove-wiki-gigaword-100')
    print("All models loaded.")

    attacker = GraDeRAttackerV2(
        v_base=v_base, embedding_model=glove_vectors, **attacker_config)

    csv_headers = [
        "idx", "question", "status", "final_suffix", "QC", "SL", "PPL_Delta",
        "original_PAS", "attacked_PAS", "Delta_PAS",
        "ASR_flag", "NodeCov", "RelCov", "Adj", "Target"
    ]

    results_log = []

    for i, data_entry in enumerate(tqdm(dataset, desc="Overall Experiment Progress")):
        question = data_entry['question']

        malicious_path_info = parse_path_info_final(
            data_entry['malicious_path'])
        correct_path_info = parse_path_info_final(data_entry['correct_path'])

        # 将当前问题信息设置到tqdm的描述中
        tqdm.write(f"\n{'='*30} Attacking Question #{i + 1}/{len(dataset)} {'='*30}")
        tqdm.write(f"Question: {question}")

        response_orig = attacker._get_rag_response(question)
        pas_orig_details = attacker._calculate_score_tilde(
            response_orig, malicious_path_info) if response_orig else {"Score_tilde": 0}
        original_pas = pas_orig_details.get('Score_tilde', 0)

        tqdm.write("Building dynamic vocabulary...")
        v_topic = build_vocab_dynamic(
            v_base=v_base_path, malicious_path=data_entry)

        result = attacker.attack(
            original_query=question,
            malicious_path_info=malicious_path_info,
            correct_path_info=correct_path_info,
            v_topic=v_topic,
            max_suffix_len=5
        )

        details = result.get('details', {})
        suffix_list = result.get("suffix", [])
        attacked_pas = details.get("Score_tilde", 0)
        is_successful = 1 if attacker._is_attack_successful(details) else 0

        log_entry = {
            "idx": i + 1, "question": question, "status": result.get("status"),
            "final_suffix": " ".join(suffix_list),
            "QC": result.get("total_queries"),
            "SL": len(suffix_list),
            "PPL_Delta": details.get("PPL_Delta"),
            "original_PAS": original_pas,
            "attacked_PAS": attacked_pas,
            "Delta_PAS": attacked_pas - original_pas,
            "ASR_flag": is_successful,
            "NodeCov": details.get("NodeCov"), "RelCov": details.get("RelCov"),
            "Adj": details.get("Adj"), "Target": details.get("Target")
        }
        results_log.append(log_entry)

        with open(output_csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=csv_headers)
            writer.writeheader()
            writer.writerows(results_log)

    print(f"\n\n{'='*30} All Final Experiments Finished! {'='*30}")
    print(f"Results saved to '{output_csv_path}'")


if __name__ == '__main__':
    # 确保文件路径正确
    DATASET_FILE = "./data/annotated_dataset_v5_controlled_mix.json"
    V_BASE_FILE = "../../Adversarial suffix/v_combined.txt"
    RESULTS_FILE = "experiment_results_v2.csv"

    # §4.1.5 中的默认超参数
    config = {
        "rag_api_url": "http://127.0.0.1:8899/query",
        "population_size": 64, "f_scale": 0.5, "cr_rate": 0.7,
        "max_generations": 100, "plateau_patience": 10,
        "weights": {'alpha': 0.25, 'beta': 0.20, 'rho': 0.25, 'omega': 0.30},
    }

    run_all_experiments_v2(
        dataset_path=DATASET_FILE,
        v_base_path=V_BASE_FILE,
        output_csv_path=RESULTS_FILE,
        attacker_config=config
    )
