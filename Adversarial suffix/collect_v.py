import collections
from datasets import load_dataset
from spacy.lang.en.stop_words import STOP_WORDS
import os # 引入 os 模块来检查文件是否存在

# ==================
#  辅助函数
# ==================

def load_vocabulary_from_file(filename="v_base.txt"):
    """
    从文件中加载词汇表。文件应每行包含一个单词。
    """
    print(f"--- 开始从文件 '{filename}' 中加载基础词汇表 ---")
    if not os.path.exists(filename):
        print(f"错误: 基础词汇表文件 '{filename}' 不存在！")
        print("请先运行 base 脚本生成该文件。")
        return set() # 返回一个空集合

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


# 这是最终修正版的函数
def mine_adversarial_words_from_dataset(dataset_name="llm-attacks/AdvBench", top_n=100):
    """
    从一个对抗性提示数据集中，挖掘高频出现的、非停用词的指令词。
    现在从本地文件加载。
    """
    print(f"\n--- 开始从本地文件自动挖掘指令词 ---")

    # 1. 加载本地数据集
    local_file_path = "advbench_train.parquet" # 确保这个文件和你的脚本在同一个目录下
    try:
        dataset = load_dataset("parquet", data_files={'train': local_file_path}, split='train')
        print(f"本地数据集 '{local_file_path}' 加载成功。")
    except FileNotFoundError:
        print(f"错误: 本地数据集文件 '{local_file_path}' 未找到！")
        print("请确保你已经下载了该文件并放置在正确的目录下。")
        return set()
    except Exception as e:
        print(f"加载本地数据集失败: {e}")
        return set()

    # 2. 聚合所有提示文本
    # *** 唯一的修改在这里：将 'goal' 改为 'prompt' ***
    all_prompts_text = " ".join([item['prompt'] for item in dataset])

    # 3. 词频统计
    words = all_prompts_text.lower().split()
    word_counts = collections.Counter(words)

    # 4. 过滤与排序
    mined_words = set()
    count = 0
    # 从最常见的词开始，过滤掉停用词和非字母词
    for word, freq in word_counts.most_common():
        if count >= top_n:
            break
        # 筛选条件：纯字母、长度大于1、非停用词
        if word.isalpha() and len(word) > 1 and word not in STOP_WORDS:
            mined_words.add(word)
            count += 1

    print(f"挖掘完成！从本地数据集中提取了 Top-{len(mined_words)} 个高频指令词。")
    return mined_words


# ==================
#  主程序
# ==================
if __name__ == "__main__":
    # 定义输入和输出文件名
    base_vocab_filename = "v_base.txt"
    combined_vocab_filename = "v_combined.txt"

    # 步骤一：加载由 base 脚本生成的静态基础词汇表
    base_vocabulary = load_vocabulary_from_file(base_vocab_filename)

    # 如果基础词汇表加载失败（比如文件不存在），可以选择中止程序
    if not base_vocabulary:
        print("\n由于基础词汇表加载失败，程序已中止。")
    else:
        # 步骤二：从数据集中挖掘新的对抗性词汇
        mined_adversarial_words = mine_adversarial_words_from_dataset(top_n=100)

        # 步骤三：合并两个词汇集合
        print("\n--- 开始合并词汇表 ---")
        combined_vocabulary = base_vocabulary.union(mined_adversarial_words)

        # 步骤四：打印统计信息
        print(f"基础词汇表大小: {len(base_vocabulary)}")
        print(f"新挖掘的词汇大小: {len(mined_adversarial_words)}")
        print(f"合并后总词汇表大小: {len(combined_vocabulary)}")

        # 步骤五：将最终的词汇表保存到新文件
        save_vocabulary_to_file(combined_vocabulary, combined_vocab_filename)

        print("\n--- 所有操作完成 ---")