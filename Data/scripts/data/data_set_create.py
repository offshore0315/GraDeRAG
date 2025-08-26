import os
import json

from neo4j import GraphDatabase, exceptions
from openai import OpenAI  # <--- 修改点 1
from tqdm import tqdm
import logging
import random

# --- 1. 配置与初始化 ---

# 设置日志记录
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')



# 配置 Neo4j 数据库连接
NEO4J_URI = "neo4j://10.26.48.154:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "zgl123456"

# --- 修改点 2: 配置 OpenAI API ---
OPENAI_API_KEY = "sk-9a5bb786d2b94e28a148de3574e01602"
OPENAI_BASE_URL = "https://api.deepseek.com/v1" # 如果.env里没有，则为None
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found in .env file.")

# 初始化OpenAI客户端
client = OpenAI(
    api_key=OPENAI_API_KEY,
    base_url=OPENAI_BASE_URL # 如果为None，会自动使用官方地址
)
# 选择一个支持JSON模式的强大模型
LLM_MODEL_NAME = "deepseek-chat" # 或者 "gpt-4o", "gpt-3.5-turbo"

PROMPT_QUESTION_GENERATION = """
角色: "你是一位语言专家和问题构造师，精通如何根据事实提出自然的人类问题。"
上下文 (Context): "我将为你提供一条来自知识图谱的真实路径，它代表了一个事实。路径格式为 `(起始实体) -[关系类型]-> (结束实体)`。"
提供的路径: "{path_string}"
任务 (Task): "请根据以上提供的路径，生成一个包含 2到3个 不同提问方式的列表。这些问题都应指向同一个答案，但措辞、风格或侧重点应有所不同。请以JSON格式返回，包含一个键 'generated_questions'，值为一个问题字符串列表。"
"""
PROMPT_MALICIOUS_CHOICE = """
角色: "你是一位精通知识图谱和逻辑推理的专家数据标注员。"
上下文 (Context):
生成的问题 (Generated Question): "{question}"
已知的正确答案 (Known Correct Answer): "{correct_answer}"
候选的周边真实路径列表 (从1到{path_count}编号):
---
{path_list}
---
任务 (Task): "请仔细阅读以上的“候选路径列表”。你的任务是从中挑选一个路径的编号 (number)，该路径的终点应该是一个对于问题来说“看似合理但错误”的答案。挑选时必须遵守以下规则：1. 路径终点（答案）绝对不能是 '{correct_answer}'。2. 挑选出的答案应该具有一定的迷惑性。请以JSON格式返回你挑选出的那个路径的编号，JSON中只包含一个键 'chosen_path_number'，值为整数。"
"""

# --- 3. 辅助函数 (保持不变) ---
def call_llm(prompt: str):
    try:
        response = client.chat.completions.create(model=LLM_MODEL_NAME, messages=[{"role": "system", "content": "You are a helpful assistant designed to output JSON."}, {"role": "user", "content": prompt}], response_format={"type": "json_object"}, temperature=0.2, timeout=120)
        response_text = response.choices[0].message.content
        return json.loads(response_text)
    except Exception as e:
        logging.error(f"LLM API or JSON parsing Error: {e}")
        return None

def format_path_for_prompt(path) -> str:
    start_node = path.start_node
    end_node = path.end_node
    rel_types = " -> ".join([rel["type"] for rel in path.relationships])
    return f"({start_node['name']}) -[{rel_types}]-> ({end_node['name']})"

def format_paths_for_selection(records: list) -> (str, list):
    path_strings, structured_paths = [], []
    for i, record in enumerate(records):
        path = record["path"]
        start_node, end_node = path.start_node, path.end_node
        rel_types = [rel["type"] for rel in path.relationships]
        path_str_for_prompt = f"({start_node['name']}) -[{' -> '.join(rel_types)}]-> ({end_node['name']})"
        path_strings.append(f"Path {i+1}: {path_str_for_prompt}")
        
        start_name_escaped = start_node['name'].replace('"', '""')
        end_name_escaped = end_node['name'].replace('"', '""')
        cypher_for_path = f'MATCH (s:Entity {{name: "{start_name_escaped}"}}), (e:Entity {{name: "{end_name_escaped}"}}) MATCH p=shortestPath((s)-[*]->(e)) RETURN e.name LIMIT 1'
        
        structured_paths.append({
            "answer": end_node['name'],
            "start_node": {"id": start_node['id'], "name": start_node['name']},
            "end_node": {"id": end_node['id'], "name": end_node['name']},
            "relationship_types": rel_types
        })
    return "\n".join(path_strings), structured_paths

# --- 4. 主流程函数 (保持不变) ---
def process_path(session, seed_path):
    correct_answer = seed_path.end_node["name"]
    correct_path_str = format_path_for_prompt(seed_path)
    start_node_name = seed_path.start_node['name']

    # Step 2: LLM generates question
    prompt_gen_q = PROMPT_QUESTION_GENERATION.format(path_string=correct_path_str)
    response_gen_q = call_llm(prompt_gen_q)
    if not response_gen_q or "generated_questions" not in response_gen_q or not response_gen_q["generated_questions"]:
        logging.warning(f"Skipping path '{correct_path_str}': LLM failed to generate a question list.")
        return None
    question = random.choice(response_gen_q["generated_questions"])
    logging.info(f"Generated Question for '{correct_path_str}': '{question}'")

    # Step 3: Explore and select malicious path
    explore_query = "MATCH (startNode:Entity {name: $start_node_name}) CALL apoc.path.expandConfig(startNode, {minLevel: 1, maxLevel: 2, uniqueness: 'NODE_GLOBAL', maxDegree: 25}) YIELD path RETURN path LIMIT 50"
    try:
        result = session.run(explore_query, start_node_name=start_node_name)
        candidate_records = list(result)
    except Exception as e:
        logging.warning(f"Skipping path '{correct_path_str}': Failed to explore neighborhood. Error: {e}")
        return None
    if not candidate_records:
        logging.warning(f"Skipping path '{correct_path_str}': No explorable paths found from start node.")
        return None

    path_list_str, structured_paths = format_paths_for_selection(candidate_records)
    prompt_select_mal = PROMPT_MALICIOUS_CHOICE.format(question=question, correct_answer=correct_answer, path_count=len(structured_paths), path_list=path_list_str)
    response_select_mal = call_llm(prompt_select_mal)
    if not response_select_mal or "chosen_path_number" not in response_select_mal:
        logging.warning(f"Skipping path '{correct_path_str}': LLM failed to select a malicious path.")
        return None

    try:
        chosen_index = int(response_select_mal["chosen_path_number"]) - 1
        chosen_path_info = structured_paths[chosen_index]
        if chosen_path_info["answer"] == correct_answer:
            logging.warning(f"Skipping path '{correct_path_str}': LLM chose the correct answer as malicious.")
            return None
    except (ValueError, IndexError, TypeError) as e:
        logging.warning(f"Skipping path '{correct_path_str}': LLM returned an invalid choice. Error: {e}")
        return None
    
    # Step 4: Consolidate output
    correct_path_data = {"start_node": {"id": seed_path.start_node['id'], "name": seed_path.start_node['name']}, "end_node": {"id": seed_path.end_node['id'], "name": seed_path.end_node['name']}, "relationship_types": [rel['type'] for rel in seed_path.relationships]}
    malicious_path_data = {"start_node": chosen_path_info["start_node"], "end_node": chosen_path_info["end_node"], "relationship_types": chosen_path_info["relationship_types"]}
    final_data_point = {"question": question, "correct_path": correct_path_data, "malicious_path": malicious_path_data}
    logging.info(f"Successfully created data point for '{correct_path_str}'.")
    return final_data_point

def sample_paths(session, path_query, target_count, all_rel_types, samples_per_type=3):
    """辅助函数，执行分层采样直到达到目标数量。"""
    paths = []
    path_ids = set()
    for rel_type in tqdm(all_rel_types, desc=f"Sampling {target_count} paths ({path_query[:15]}...)"):
        if len(paths) >= target_count:
            break
        try:
            results = session.run(path_query, rel_type=rel_type, limit=samples_per_type)
            for record in results:
                path = record['p']
                path_unique_id = tuple(sorted([node.element_id for node in path.nodes] + [rel.element_id for rel in path.relationships]))
                if path_unique_id not in path_ids:
                    paths.append(path)
                    path_ids.add(path_unique_id)
        except Exception:
            continue
    if len(paths) > target_count:
        paths = random.sample(paths, target_count)
    logging.info(f"Sampled {len(paths)} paths (target was {target_count}).")
    return paths

if __name__ == "__main__":
    # --- 关键修改：定义新的目标数量 ---
    TARGET_1_HOP = 200
    TARGET_2_HOP = 200
    TARGET_3_HOP = 100

    # --- 关键修改：定义三种采样查询 ---
    query_1hop = """
        MATCH p=(n:Entity)-[r:RELATION]->(m:Entity)
        WHERE r.type = $rel_type AND n.name IS NOT NULL AND m.name IS NOT NULL AND n <> m
        RETURN p LIMIT $limit
    """
    query_2hop = """
        MATCH p=(n:Entity)-[r1:RELATION]->(m:Entity)-[r2:RELATION]->(o:Entity)
        WHERE r1.type = $rel_type AND n.name IS NOT NULL AND o.name IS NOT NULL AND n <> o
        RETURN p LIMIT $limit
    """
    query_3hop = """
        MATCH p=(n:Entity)-[r1:RELATION]->(m:Entity)-[r2:RELATION]->(o:Entity)-[r3:RELATION]->(q:Entity)
        WHERE r1.type = $rel_type AND n.name IS NOT NULL AND q.name IS NOT NULL AND n <> o
        RETURN p LIMIT $limit
    """

    db_driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    
    with db_driver.session() as session:
        logging.info("Fetching all distinct relationship types...")
        rel_types_result = session.run("MATCH ()-[r:RELATION]->() RETURN DISTINCT r.type AS type")
        all_rel_types = [record['type'] for record in rel_types_result]
        random.shuffle(all_rel_types)
        logging.info(f"Found {len(all_rel_types)} distinct relationship types.")

        # --- 关键修改：分三步进行采样 ---
        logging.info(f"--- Starting to sample {TARGET_1_HOP} single-hop paths ---")
        single_hop_paths = sample_paths(session, query_1hop, TARGET_1_HOP, all_rel_types)

        logging.info(f"--- Starting to sample {TARGET_2_HOP} 2-hop paths ---")
        multi_hop_paths_2 = sample_paths(session, query_2hop, TARGET_2_HOP, all_rel_types)

        logging.info(f"--- Starting to sample {TARGET_3_HOP} 3-hop paths ---")
        multi_hop_paths_3 = sample_paths(session, query_3hop, TARGET_3_HOP, all_rel_types)

    # --- 关键修改：合并三组路径 ---
    seed_paths = single_hop_paths + multi_hop_paths_2 + multi_hop_paths_3
    random.shuffle(seed_paths)
    logging.info(f"Total seed paths to process: {len(seed_paths)} ({len(single_hop_paths)} 1-hop, {len(multi_hop_paths_2)} 2-hop, {len(multi_hop_paths_3)} 3-hop)")

    # --- 执行数据生成流程 (保持不变) ---
    final_dataset = []
    with db_driver.session() as db_session:
        for path in tqdm(seed_paths, desc="Generating Final Dataset"):
            data_point = process_path(db_session, path)
            if data_point:
                final_dataset.append(data_point)
            
    # --- 保存结果 ---
    output_filename = "annotated_dataset_v5_controlled_mix.json"
    with open(output_filename, "w", encoding="utf-8") as f:
        json.dump(final_dataset, f, indent=2, ensure_ascii=False)

    logging.info(f"\nAnnotation finished. {len(final_dataset)} data points saved to {output_filename}")
    db_driver.close()