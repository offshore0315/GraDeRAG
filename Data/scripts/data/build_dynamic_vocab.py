import collections
import os
import json
import re
import spacy  # 引入 spaCy


def load_vocabulary_from_file(filename="v_base.txt"):
    """
    从文件中加载词汇表。文件应每行包含一个单词。
    """
    print(f"--- 开始从文件 '{filename}' 中加载基础词汇表 ---")
    if not os.path.exists(filename):
        print(f"错误: 基础词汇表文件 '{filename}' 不存在！")
        print("请先运行 base 脚本生成该文件。")
        return set()  # 返回一个空集合

    with open(filename, 'r', encoding='utf-8') as f:
        # 使用集合推导式高效读取，并去除每行末尾的换行符
        vocabulary = {line.strip() for line in f}

    print(f"加载成功！从 '{filename}' 中读取了 {len(vocabulary)} 个词。")
    return vocabulary


def save_vocabulary_to_file(vocabulary, filename):
    """
    将词汇表集合保存到文件，每行一个单词，按字母排序。
    """
    print(f"\n--- 开始将合并后的词汇表保存到 '{filename}' ---")
    sorted_vocab = sorted(list(vocabulary))
    with open(filename, 'w', encoding='utf-8') as f:
        for word in sorted_vocab:
            f.write(word + "\n")
    print(f"保存成功！共 {len(sorted_vocab)} 个词汇已写入 '{filename}'。")


def extract_keywords_from_relation(relation_string: str):
    """
    正则提取关键词
    """

    cleand_string = re.sub(r'[_./]', ' ', relation_string)

    keywords = {word for word in cleand_string.split() if len(word)
                > 2 and word.isalpha()}

    common_fb_words = {
        'base',
        'schemastaging',
        'measurement_unit',
        'dated_money_value',
        'organization_extra',
        'phone_sandbox',
        'people'
    }

    return keywords - common_fb_words


def generate_v_topic(malicious_path, nlp, top_k_similar=5):
    """
    针对恶意路径动态生成主题相关词汇表。
    """

    aggregated_text = []

    start_node_name = malicious_path["start_node"]["name"].replace("_", " ")
    end_node_name = malicious_path["end_node"]["name"].replace("_", " ")
    aggregated_text.append(start_node_name)
    aggregated_text.append(end_node_name)

    relation_keywords = set()
    for rel in malicious_path["relationship_types"]:
        relation_keywords.update(extract_keywords_from_relation(rel))
    aggregated_text.extend(list(relation_keywords))

    topic_document = " ".join(aggregated_text)
    doc = nlp(topic_document.lower())
    core_keywords = set()
    for token in doc:
        if not token.is_stop and not token.is_punct and token.pos_ in {"NOUN", "PROPN", "VERB", "ADJ"}:
            core_keywords.add(token.lemma_.lower())
    print(f"提取到的核心关键词: {core_keywords}")

    v_topic = set(core_keywords)

    for keyword in core_keywords:
        if keyword in nlp.vocab and nlp.vocab[keyword].has_vector:
            queries = [
                w for w in nlp.vocab if w.has_vector and w.is_lower and w.is_alpha and w != nlp.vocab[keyword]]

            by_similarity = sorted(
                queries, key=lambda w: nlp.vocab[keyword].similarity(w), reverse=True)

            for similar_word in by_similarity[:top_k_similar]:
                print("test: ", similar_word.text)
                v_topic.add(similar_word.text.lower())

    return core_keywords, v_topic


def build_vocab_dynamic(v_base, malicious_path):

    vocabulary = load_vocabulary_from_file(v_base)
    try:
        nlp = spacy.load("en_core_web_md")
        print("成功加载 spaCy 语言模型 'en_core_web_md'。")
    except OSError:
        print("错误: 未找到 spaCy 语言模型 'en_core_web_md'。")
        print("请运行以下命令安装该模型：")
        print("python -m spacy download en_core_web_md")
        exit(1)

    malicious_data = malicious_path["malicious_path"]

    print("\n--- 动态生成主题相关词汇表 ---")

    core_keywords, v_topic = generate_v_topic(
        malicious_data, nlp, top_k_similar=10)

    print(f"核心关键词 ({len(core_keywords)}): {core_keywords}")
    print(f"主题相关词汇表 ({len(v_topic)}): {v_topic}")

    v_allow = vocabulary.union(v_topic)
    print("最终候选词汇表 (V_allow) 已生成。", v_allow)
    print("\n--- 统计信息 ---")
    print(f"基础词汇表 (V_base) 大小: {len(vocabulary)}")
    print(f"动态主题词汇表 (V_topic) 大小: {len(v_topic)}")
    print(f"最终候选词汇表 (V_allow) 大小: {len(v_allow)}")
    return v_allow


if __name__ == "__main__":

    base_vocabulary_filename = "../../../Adversarial suffix/v_combined.txt"
    combined_vocab_filename = "v_combined.txt"

    try:
        nlp = spacy.load("en_core_web_md")
        print("成功加载 spaCy 语言模型 'en_core_web_md'。")
    except OSError:
        print("错误: 未找到 spaCy 语言模型 'en_core_web_md'。")
        print("请运行以下命令安装该模型：")
        print("python -m spacy download en_core_web_md")
        exit(1)

    vocabulary = load_vocabulary_from_file(base_vocabulary_filename)

    malicious_path_file = "./annotated_dataset_v5_controlled_mix.json"

    malicious_data = json.load(
        open(malicious_path_file, 'r', encoding='utf-8'))

    for idx, item in enumerate(malicious_data):
        print(f"\n=================================================")
        print(f"处理数据条目 #{idx+1}")
        print(f"问题: {item['question']}")
        print(f"=================================================")

        malicious_path = item["malicious_path"]

        print("\n--- 动态生成主题相关词汇表 ---")

        core_keywords, v_topic = generate_v_topic(
            malicious_path, nlp, top_k_similar=10)

        print(f"核心关键词 ({len(core_keywords)}): {core_keywords}")
        print(f"主题相关词汇表 ({len(v_topic)}): {v_topic}")

        v_allow = vocabulary.union(v_topic)
        print("最终候选词汇表 (V_allow) 已生成。", v_allow)
        print("\n--- 统计信息 ---")
        print(f"基础词汇表 (V_base) 大小: {len(vocabulary)}")
        print(f"动态主题词汇表 (V_topic) 大小: {len(v_topic)}")
        print(f"最终候选词汇表 (V_allow) 大小: {len(v_allow)}")

        output_filename = f"v_topic_{idx+1}.txt"
        save_vocabulary_to_file(v_topic, output_filename)
        break
    print("\n所有数据条目处理完毕。")
