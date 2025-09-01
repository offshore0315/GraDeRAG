# scripts/check_entity_coverage.py (版本 3 - 增加保存功能)
import os
import json
from neo4j import GraphDatabase
from tqdm import tqdm


def get_webqsp_entities_with_normalization(file_path: str) -> set:
    """从WebQSP JSON文件中提取所有唯一的主题实体ID，并将其标准化。"""
    print(f"Reading and standardizing WebQSP entities from {file_path}...")
    entities = set()
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            questions = data.get("Questions", [])
            for q in tqdm(questions, desc="Extracting WebQSP entities"):
                parses = q.get("Parses")
                if parses and len(parses) > 0:
                    topic_entity_mid = parses[0].get("TopicEntityMid")
                    if topic_entity_mid:
                        normalized_mid = topic_entity_mid
                        if not normalized_mid.startswith('/'):
                            normalized_mid = '/' + normalized_mid.replace('.', '/', 1)
                        entities.add(normalized_mid)
    except FileNotFoundError:
        print(f"Error: WebQSP file not found at {file_path}")
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {file_path}")
    return entities

def get_neo4j_entities(driver) -> set:
    """从Neo4j数据库中获取所有唯一的实体ID。"""
    print("Fetching all entity IDs from Neo4j...")
    entities = set()
    query = "MATCH (n:Entity) RETURN n.id AS entityId"
    with driver.session(database="neo4j") as session:
        results = session.read_transaction(lambda tx: list(tx.run(query)))
        for record in tqdm(results, desc="Fetching Neo4j entities"):
            entities.add(record["entityId"])
    return entities

# --- 新增功能：筛选并保存已覆盖的问答对 ---
def filter_and_save_covered_questions(original_file_path: str, output_file_path: str, existing_entities: set):
    """
    读取原始WebQSP文件，筛选出主题实体存在于 aistrong_existing_entities 集合中的问答对，
    并将其以原格式保存到新的JSON文件中。
    """
    print(f"\nFiltering questions and saving to {output_file_path}...")
    covered_questions = []
    
    try:
        with open(original_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            questions = data.get("Questions", [])
            for q in tqdm(questions, desc="Filtering questions"):
                parses = q.get("Parses")
                if parses and len(parses) > 0:
                    topic_entity_mid = parses[0].get("TopicEntityMid")
                    if topic_entity_mid:
                        normalized_mid = topic_entity_mid
                        if not normalized_mid.startswith('/'):
                            normalized_mid = '/' + normalized_mid.replace('.', '/', 1)
                        
                        # 检查标准化后的ID是否存在于我们的Neo4j实体集合中
                        if normalized_mid in existing_entities:
                            covered_questions.append(q) # 保留整个原始问题对象

        # 以与原始文件相同的结构创建新数据
        new_data = {"Questions": covered_questions}

        # 写入新的JSON文件
        with open(output_file_path, 'w', encoding='utf-8') as f:
            # indent=2 使JSON文件格式化，易于阅读
            # ensure_ascii=False 确保中文字符等正确写入
            json.dump(new_data, f, indent=2, ensure_ascii=False)
            
        print(f"Successfully saved {len(covered_questions)} covered questions to {output_file_path}")

    except Exception as e:
        print(f"An error occurred during filtering and saving: {e}")


def main():
    # --- 配置 ---
    webqsp_file_path = '../WebQSP.train.json'
    # 新增：定义过滤后文件的输出路径
    output_file_path = '../WebQSP.train.covered.json' 
    
    uri = "neo4j://10.26.48.154:7687"
    username = "neo4j"
    password = "zgl123456"
    
    if not password:
        print("Error: NEO4J_PASSWORD not found.")
        return

    driver = GraphDatabase.driver(uri, auth=(username, password))
    
    try:
        # --- 步骤 1: 获取实体集合 ---
        webqsp_entities = get_webqsp_entities_with_normalization(webqsp_file_path)
        if not webqsp_entities:
            print("No entities extracted from WebQSP. Exiting.")
            return
            
        neo4j_entities = get_neo4j_entities(driver)
        if not neo4j_entities:
            print("No entities found in Neo4j. Exiting.")
            return

        # --- 步骤 2: 分析与报告覆盖率 ---
        common_entities = webqsp_entities.intersection(neo4j_entities)
        
        print("\n--- Entity Coverage Analysis (ID Standardized) ---")
        print(f"Total unique entities in FB15k-237 (Neo4j): {len(neo4j_entities)}")
        print(f"Total unique topic entities in WebQSP:       {len(webqsp_entities)}")
        print("-" * 50)
        print(f"Entities found in both (Intersection):       {len(common_entities)}")
        print(f"Coverage Rate: {(len(common_entities) / len(webqsp_entities) * 100):.2f}%")
        print("--------------------------------------------------")
        
        # --- 步骤 3: [新功能] 筛选并保存已覆盖的问答对 ---
        # 我们使用从Neo4j获取的完整实体集(neo4j_entities)作为判断标准
        filter_and_save_covered_questions(webqsp_file_path, output_file_path, neo4j_entities)

    finally:
        driver.close()
        print("\nProcess complete. Neo4j driver closed.")

if __name__ == "__main__":
    main()