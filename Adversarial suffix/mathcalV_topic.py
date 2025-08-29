import spacy
import gensim.downloader
from spacy.lang.en.stop_words import STOP_WORDS


def build_dynamic_topic_vocabulary(
        path_texts: list[str],
        spacy_model,
        gensim_model,
        top_k_expansion: int = 5
) -> set[str]:
    """
    为给定的恶意路径文本，构建一个动态主题词汇表(V_topic)。

    Args:
        path_texts (list[str]): 包含路径中所有节点和关系文本的列表。
        spacy_model: 已加载的spaCy语言模型。
        gensim_model: 已加载的gensim词向量模型。
        top_k_expansion (int): 为每个核心关键词查找的近邻词数量。

    Returns:
        set[str]: 包含所有主题词的集合。
    """

    # 步骤一：文本聚合
    aggregated_text = " ".join(path_texts)
    print(f"1. 聚合文本: '{aggregated_text}'")

    # 步骤二：核心词提取
    doc = spacy_model(aggregated_text)
    core_keywords = set()
    for token in doc:
        # 筛选名词、动词、形容词，且非停用词和标点
        if token.pos_ in ["NOUN", "PROPN", "VERB", "ADJ"] and not token.is_stop and not token.is_punct:
            core_keywords.add(token.lemma_.lower())

    print(f"\n2. 提取出的核心关键词: {core_keywords}")

    # 步骤三：语义扩展
    topic_vocabulary = set()
    for keyword in core_keywords:
        # 将原始关键词加入词汇表
        topic_vocabulary.add(keyword)
        try:
            # 查找近邻词
            neighbors = gensim_model.most_similar(keyword, topn=top_k_expansion)
            for neighbor, _ in neighbors:
                topic_vocabulary.add(neighbor.lower())
        except KeyError:
            # 如果词向量模型中不存在该词，则跳过
            # print(f"  - 词 '{keyword}' 不在GloVe模型中，跳过扩展。")
            pass

    print(f"\n3. 经过语义扩展后的最终主题词汇表 (V_topic): \n{sorted(list(topic_vocabulary))}")

    return topic_vocabulary


# ==================
#  运行示例
# ==================
if __name__ == "__main__":
    print("--- GraDeRAG 动态主题词汇表构建脚本 ---")

    # 加载所需模型
    # 如果是首次运行，gensim会自动下载glove-wiki-gigaword-100模型
    print("\n正在加载Gensim词向量模型 (首次运行可能需要几分钟下载)...")
    glove_vectors = gensim.downloader.load('glove-wiki-gigaword-100')
    print("Gensim模型加载完毕。")

    print("\n正在加载spaCy语言模型...")
    nlp = spacy.load("en_core_web_sm")
    print("spaCy模型加载完毕。")

    # 定义我们的攻击场景
    malicious_path_texts = [
        "Titanic",
        "Leonardo DiCaprio",
        "starred in",  # 关系词
        "The Aviator",
        "directed by",  # 关系词
        "Martin Scorsese"
    ]

    print("\n" + "=" * 50)
    print("开始为以下恶意路径构建动态主题词汇表:")
    print(malicious_path_texts)
    print("=" * 50 + "\n")

    # 构建 V_topic
    v_topic = build_dynamic_topic_vocabulary(
        path_texts=malicious_path_texts,
        spacy_model=nlp,
        gensim_model=glove_vectors,
        top_k_expansion=5
    )

    print(f"\n构建完成！共生成 {len(v_topic)} 个独特的词语。")