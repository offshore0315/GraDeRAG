import json
from neo4j import GraphDatabase
import re

# ==============================================================================
# 步骤 0: 配置 (Configuration)
# ==============================================================================
NEO4J_URI = "neo4j://10.26.48.154:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "zgl123456" 
WEBQSP_FILE_PATH = "../../WebQSP.train.covered.json" 
NEO4J_NODE_MID_PROPERTY = "id" 
# ==============================================================================
# 步骤 1: 从WebQSP文件加载任务并动态构建元数据
# ==============================================================================
def load_tasks_and_metadata_from_webqsp(file_path):
    """
    从WebQSP JSON文件中加载任务，并同时动态构建实体和关系的映射字典。
    """
    print(f"🔄 Loading tasks and building metadata from: {file_path}...")
    tasks, entity_map, relation_map = [], {}, {}
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f).get("Questions", [])
    except FileNotFoundError:
        print(f"❌ Error: WebQSP file not found at '{file_path}'.")
        return [], {}, {}
    except (json.JSONDecodeError, AttributeError):
        print(f"❌ Error: Could not decode JSON or find 'Questions' key in '{file_path}'.")
        return [], {}, {}

    for i, question_data in enumerate(data):
        if not question_data.get("Parses"): continue
        parse = question_data["Parses"][0]
        if not parse.get("Answers") or not parse.get("TopicEntityMid"): continue

        topic_entity_mid = parse["TopicEntityMid"]
        answer_entity_mid = parse["Answers"][0].get("AnswerArgument")

        if not answer_entity_mid: continue

        # 将MID转换为Neo4j兼容的格式 (e.g., m.0123 -> /m/0123)
        topic_entity_neo4j_id = "/" + topic_entity_mid.replace(".", "/")
        answer_entity_neo4j_id = "/" + answer_entity_mid.replace(".", "/")

        entity_map[topic_entity_mid] = parse["TopicEntityName"]
        entity_map[answer_entity_mid] = parse["Answers"][0]["EntityName"]
        
        inferential_chain = parse.get("InferentialChain") or []
        for rel_dot in inferential_chain:
            rel_slash = "/" + rel_dot.replace(".", "/")
            relation_map[rel_dot] = rel_slash

        # 使用ID属性和转换后的Neo4j ID来构建Cypher路径
        correct_path_cypher = f"MATCH path = (h:Entity {{{NEO4J_NODE_MID_PROPERTY}: '{topic_entity_neo4j_id}'}})-"
        path_segments = [f"[:`{relation_map[rel_dot]}`]->" for rel_dot in inferential_chain]
        
        if path_segments:
            correct_path_cypher += "".join(path_segments)[:-2] + f"(t:Entity {{{NEO4J_NODE_MID_PROPERTY}: '{answer_entity_neo4j_id}'}}) RETURN path"
        else:
            correct_path_cypher = "N/A"

        tasks.append({
            "task_id": question_data["QuestionId"], "original_question": question_data["RawQuestion"],
            "core_entity": parse["TopicEntityName"],
            "core_entity_mid": topic_entity_mid,
            "core_entity_neo4j_id": topic_entity_neo4j_id, # 存储转换后的ID以供查询
            "source_dataset": "WebQSP",
            "correct_answer": parse["Answers"][0]["EntityName"], "correct_path_cypher": correct_path_cypher
        })
        # if i < 10:
        #     print(tasks[-1])  # 打印前10个任务以供检查
        
        
    print(f"✅ Loaded {len(tasks)} tasks and discovered {len(entity_map)} entities and {len(relation_map)} relations.")
    return tasks, entity_map, relation_map

# ==============================================================================
# 步骤 2: 从Neo4j动态提取【分层】的局部上下文
# ==============================================================================
def extract_multi_hop_contexts(driver, core_entity_neo4j_id: str, entity_map, relation_map, max_hops: int = 3) -> dict:
    """
    一次性查询多跳邻居，并按跳数分层组织上下文。
    """
    print(f"\n🔍 Querying Neo4j for up to {max_hops}-hop contexts around ID: '{core_entity_neo4j_id}'...")
    
    all_hops_contexts = {f"{i}-hop": {"entities": {}, "relations": {}, "triples": set()} for i in range(1, max_hops + 1)}

    # 【最终修正】使用配置的ID属性和Neo4j格式的ID进行查询
    query = f"MATCH p = (h {{{NEO4J_NODE_MID_PROPERTY}: $entity_id}})-[*1..{max_hops}]-(t) RETURN p LIMIT 300"
    
    with driver.session() as session:
        results = session.run(query, entity_id=core_entity_neo4j_id)
        print("query: ", query, "entity_id", core_entity_neo4j_id)
        slash_to_dot_map = {v: k for k, v in relation_map.items()}

        for record in results:
            path = record["p"]
            path_length = len(path)

            for rel in path.relationships:
                h_node, t_node, r_type = rel.start_node, rel.end_node, rel.type
                
                # 从节点中获取Neo4j格式的ID
                h_mid_slash = h_node.get(NEO4J_NODE_MID_PROPERTY)
                t_mid_slash = t_node.get(NEO4J_NODE_MID_PROPERTY)
                
                # 将其转换回原始的点格式MID，以便在我们的映射表中查找名称
                h_mid_dot = h_mid_slash[1:].replace("/",".") if h_mid_slash else None
                t_mid_dot = t_mid_slash[1:].replace("/",".") if t_mid_slash else None

                h_name = entity_map.get(h_mid_dot)
                t_name = entity_map.get(t_mid_dot)

                if not all([h_mid_dot, t_mid_dot, h_name, t_name, r_type]): continue
                
                r_id = slash_to_dot_map.get(r_type)
                
                if not r_id: continue
                
                for i in range(path_length, max_hops + 1):
                    context = all_hops_contexts[f"{i}-hop"]
                    context["entities"][h_mid_dot], context["entities"][t_mid_dot] = h_name, t_name
                    context["relations"][r_id] = r_type
                    context["triples"].add((h_mid_dot, r_id, t_mid_dot))

    for hop_key, context in all_hops_contexts.items():
        context["triples"] = list(context["triples"])
        print(f"  - {hop_key} context size: {len(context['triples'])} triples.")

    return all_hops_contexts

# ==============================================================================
# 步骤 3: 带决策能力的Gemini API调用函数
# ==============================================================================
def generate_malicious_sample_with_decision(task, all_hops_contexts):
    """
    模拟一个两阶段的Gemini API调用：
    1. 决策阶段：选择最佳的上下文跳数。
    2. 生成阶段：基于选择的上下文创建攻击样本。
    """
    context_summary = {hop: len(data["triples"]) for hop, data in all_hops_contexts.items() if data["triples"]}
    if not context_summary:
        print("🚫 Cannot generate sample: All contexts are empty.")
        return None

    # 此处省略了决策prompt，因为它只影响内部逻辑
    
    # 模拟Gemini的决策
    chosen_hop = 2 
    if context_summary.get("2-hop", 0) > 200 and context_summary.get("1-hop", 0) > 0:
        chosen_hop = 1
    elif context_summary.get("2-hop", 0) == 0 and context_summary.get("3-hop", 0) > 0:
        chosen_hop = 3
    
    print(f"🧠 Gemini decided to use {chosen_hop}-hop context.")
    
    # 【修正】返回的字典只包含最终需要的字段
    simulated_api_response = {
      "malicious_answer": f"Plausible answer from {chosen_hop}-hop context",
      "malicious_path_cypher": f"A misleading Cypher path found within {chosen_hop}-hop context"
    }
    return simulated_api_response

# ==============================================================================
# 步骤 4: 主执行流程 (Main Execution Flow)
# ==============================================================================
if __name__ == "__main__":
    all_tasks, entity_map, relation_map = load_tasks_and_metadata_from_webqsp(WEBQSP_FILE_PATH)
    
    if all_tasks:
        driver = None
        try:
            driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
            driver.verify_connectivity()
            print("\n🚀 Neo4j connection successful!")
            
            final_results_for_csv = []

            for task in all_tasks:
                print(f"\n{'='*20} Processing Task: {task['task_id']} {'='*20}")
                print(f"Question: {task['original_question']}")

                # 【最终修正】传入转换后的Neo4j ID进行查询
                multi_hop_contexts = extract_multi_hop_contexts(
                    driver, task["core_entity_neo4j_id"], entity_map, relation_map
                )

                malicious_data = generate_malicious_sample_with_decision(task, multi_hop_contexts)

                if malicious_data:
                    final_task_for_csv = {
                        "task_id": task["task_id"],
                        "original_question": task["original_question"],
                        "correct_answer": task["correct_answer"],
                        "correct_path_cypher": task["correct_path_cypher"],
                        "malicious_answer": malicious_data["malicious_answer"],
                        "malicious_path_cypher": malicious_data["malicious_path_cypher"],
                        "status": "Completed"
                    }
                    
                    final_results_for_csv.append(final_task_for_csv)

                    print("\n--- Final Task for CSV ---")
                    print(json.dumps(final_task_for_csv, indent=2, ensure_ascii=False))
                    print("--------------------------")
            
            with open("attack_tasks.json", "w", encoding="utf-8") as f:
                json.dump(final_results_for_csv, f, indent=2, ensure_ascii=False)
            print("\n\n✅ 所有任务处理完毕，并已保存至 attack_tasks.json")


        except Exception as e:
            print(f"\n🚨 发生了一个错误: {e}")
        finally:
            if driver:
                driver.close()
                print("\n🔌 Neo4j connection closed.")