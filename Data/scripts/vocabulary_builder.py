# vocabulary_builder.py
import spacy
import gensim.downloader
import warnings
from functools import lru_cache

# 忽略gensim的FutureWarning
warnings.filterwarnings(action='ignore', category=FutureWarning, module='gensim')

@lru_cache(maxsize=1)
def load_models():
    """加载并缓存spacy和gensim模型以避免重复加载。"""
    print("正在加载Gensim词向量模型 (首次运行可能需要几分钟下载)...")
    glove_vectors = gensim.downloader.load('glove-wiki-gigaword-100')
    print("Gensim模型加载完毕。")
    
    print("正在加载spaCy语言模型...")
    nlp = spacy.load("en_core_web_sm")
    print("spaCy模型加载完毕。")
    return nlp, glove_vectors

def build_dynamic_topic_vocabulary(
    path_texts: list[str],
    top_k_expansion: int = 5
) -> set[str]:
    """
    为给定的恶意路径文本，构建一个动态主题词汇表(V_topic)。
    """
    spacy_model, gensim_model = load_models()
    
    aggregated_text = " ".join(path_texts)
    doc = spacy_model(aggregated_text)
    core_keywords = set()
    
    for token in doc:
        if token.pos_ in ["NOUN", "PROPN", "VERB", "ADJ"] and not token.is_stop and not token.is_punct:
            core_keywords.add(token.lemma_.lower())
    
    topic_vocabulary = set(core_keywords)
    for keyword in core_keywords:
        try:
            neighbors = gensim_model.most_similar(keyword, topn=top_k_expansion)
            for neighbor, _ in neighbors:
                topic_vocabulary.add(neighbor.lower())
        except KeyError:
            pass
            
    return topic_vocabulary

def get_v_allow(v_base_path: str, path_texts: list[str]) -> list[str]:
    """
    构建最终的 V_allow 词汇表。
    """
    # 加载 V_base
    with open(v_base_path, 'r', encoding='utf-8') as f:
        v_base = set(line.strip().lower() for line in f)
        
    # 构建 V_topic
    v_topic = build_dynamic_topic_vocabulary(path_texts)
    
    # 合并
    v_allow_set = v_base.union(v_topic)
    
    # 返回列表形式，便于索引
    return sorted(list(v_allow_set))

if __name__ == '__main__':
    # 示例
    malicious_path_texts = [
        "Titanic", "Leonardo DiCaprio", "starred in", 
        "The Aviator", "directed by", "Martin Scorsese"
    ]
    
    # 确保你有一个 data/v_base.txt 文件
    try:
        v_allow = get_v_allow('data/v_base.txt', malicious_path_texts)
        print(f"\n为示例路径构建的 V_allow 词汇表示例 (前20个): {v_allow[:20]}")
        print(f"V_allow 总大小: {len(v_allow)}")
    except FileNotFoundError:
        print("错误: 请先创建 data/v_base.txt 文件。")