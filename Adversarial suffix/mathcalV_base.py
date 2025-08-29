import wordfreq
import re

# (将上面定义的两个函数 get_high_frequency_words 和 get_adversarial_instruction_words 粘贴在这里)
def get_high_frequency_words(top_n=5000, lang='en'):
    """
    获取指定语言中最高频的前 N 个单词。
    """
    # wordfreq.top_n_list 直接返回一个单词列表，例如 ['the', 'of', 'a', ...]
    top_words_list = wordfreq.top_n_list(lang, top_n)
    # 直接将这个列表转换为集合即可，无需解包
    top_words = set(top_words_list)
    print(f"步骤一：成功获取 {len(top_words)} 个英语高频词。")
    return top_words


def get_adversarial_instruction_words_enhanced():
    """
    返回一个经过扩展和精炼的、用于对抗性攻击的指令词集合。
    """
    # 1. 引导与转换 (Guiding & Transition) - 改变话题或焦点
    guiding_words = {
        'instead', 'however', 'focusing', 'regarding', 'alternatively', 'moreover',
        'additionally', 'rather', 'specifically', 'particularly', 'consider',
        'otherwise', 'meantime', 'henceforth', 'furthermore'
    }

    # 2. 逻辑与关系 (Logical & Relational) - 构建关系或提问
    logical_words = {
        'related', 'associated', 'connection', 'between', 'among', 'about',
        'concerning', 'like', 'such', 'as', 'who', 'what', 'when', 'where', 'why', 'how',
        'which', 'whose', 'whom'
    }

    # 3. 自然化填充 (Naturalizing Fillers) - 使后缀更像人类语言
    filler_words = {
        'actually', 'interestingly', 'basically', 'really', 'simply', 'fact',
        'course', 'maybe', 'perhaps', 'especially', 'absolutely', 'definitely',
        'positively', 'literally'
    }

    # 4. 比较与最高级 (Comparison & Superlatives) - 引导进行价值判断
    comparison_words = {
        'more', 'less', 'better', 'worse', 'best', 'worst', 'most', 'least',
        'another', 'different', 'similar', 'other'
    }

    # 5. 排除与否定 (Exclusion & Negation) - 强制模型忽略某些信息
    exclusion_words = {
        'without', 'except', 'not', 'never', 'no', 'neither', 'nor',
        'disregard', 'forget', 'ignore'
    }

    # 6. 因果与解释 (Causation & Explanation) - 诱导模型进行解释
    causal_words = {
        'because', 'since', 'therefore', 'explain', 'consequently', 'thus',
        'reason', 'result', 'cause'
    }

    # 7. 假设与推测 (Hypotheticals & Speculation) - 将模型置于非事实性语境
    hypothetical_words = {
        'imagine', 'suppose', 'if', 'could', 'would', 'might', 'possibly'
    }

    # 组合所有集合
    instruction_words = guiding_words.union(logical_words).union(filler_words) \
        .union(comparison_words).union(exclusion_words) \
        .union(causal_words).union(hypothetical_words)

    print(f"步骤二（增强版）：成功定义 {len(instruction_words)} 个攻击性指令词。")
    return instruction_words


def build_static_base_vocabulary(output_filename="v_base.txt"):
    """
    执行所有步骤，构建、合并、清洗并保存最终的静态基础词汇表。
    """
    print("--- 开始构建静态基础词汇表 (V_base) ---")

    # 步骤一和步骤二
    high_freq_words = get_high_frequency_words(top_n=5000)
    # *** 修正：调用你已经定义的增强版函数 ***
    instruction_words = get_adversarial_instruction_words_enhanced()

    # 步骤三：合并
    v_base = high_freq_words.union(instruction_words)
    print(f"\n步骤三：合并后，词汇表初始大小为 {len(v_base)}。")

    # 步骤四：清洗
    # - 只保留纯英文字母的单词
    # - 单词长度在2到15之间
    cleaned_v_base = set()
    for word in v_base:
        if word and word.isalpha() and 2 <= len(word) <= 15:
            cleaned_v_base.add(word.lower())  # 确保所有词都是小写

    print(f"步骤四：清洗后，最终词汇表大小为 {len(cleaned_v_base)}。")

    # 步骤五：保存到文件
    # 将集合转为列表并排序，以便于查看和版本控制
    sorted_v_base = sorted(list(cleaned_v_base))
    with open(output_filename, 'w', encoding='utf-8') as f:
        for word in sorted_v_base:
            f.write(word + "\n")

    print(f"步骤五：静态基础词汇表已成功保存到文件 '{output_filename}'。")
    print("\n--- 构建完成 ---")


# ==================
#  运行构建脚本
# ==================
if __name__ == "__main__":
    build_static_base_vocabulary()