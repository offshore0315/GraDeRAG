import numpy as np
import requests
import re
import math
import time
import torch
from typing import List, Dict, Tuple, Set, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM  # 不再使用 pipeline
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm, trange


# ===============================================================
# 全新、正确的 PerplexityCalculatorV2
# ===============================================================


class PerplexityCalculatorV2:
    """
    使用基础模型正确计算自回归模型 (如 GPT-2) 的困惑度。
    """

    def __init__(self, model_name: str = 'gpt2'):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        #fmt: off
        print(f"Initializing PerplexityCalculator with '{model_name}' on device '{self.device}'...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name).to(self.device)
        self.model.eval()  # 设置为评估模式
        print("PerplexityCalculatorV2 ready.")

    def get_perplexity(self, text: str) -> float:
        """
        计算给定文本的困惑度 PPL = exp(cross_entropy_loss)。
        """
        if not text:
            return 0.0
        try:
            # 1. 对文本进行分词
            inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
            input_ids = inputs["input_ids"]

            # 2. 在不计算梯度的情况下，获取模型的输出
            with torch.no_grad():
                # 将 labels 设置为 input_ids，模型会自动计算损失
                outputs = self.model(input_ids=input_ids, labels=input_ids)
                loss = outputs.loss

            # 3. 计算 PPL
            ppl = torch.exp(loss).item()
            return ppl
        except Exception:
            return float('inf')


# ===============================================================
# GraDeRAttackerV2 (已更新为使用新的 PerplexityCalculatorV2)
# ===============================================================
class GraDeRAttackerV2:
    """
    GraDeRAG 攻击算法的重构和优化版本。
    """

    def __init__(
        self,
        rag_api_url: str,
        v_base: Set[str],
        embedding_model,
        population_size: int = 64, f_scale: float = 0.5, cr_rate: float = 0.7,
        max_generations: int = 100, plateau_patience: int = 10,
        lambda_ppl: float = 0.05, lambda_len: float = 0.01,
        weights: Dict[str, float] = {'alpha': 0.25,
                                     'beta': 0.20, 'rho': 0.25, 'omega': 0.30},
        lambda_ctr: float = 0.5,
        softmatch_theta: float = 0.7, softmatch_beta: float = 0.1,
        adj_sigma_w: int = 15,
        success_criteria: Dict[str, float] = {
            'Target': 0.9, 'Adj': 0.6, 'NodeCov': 0.5, 'RelCov': 0.5}
    ):
        self.rag_api_url = rag_api_url
        self.v_base = v_base
        self.embedding_model = embedding_model
        self.embedding_dim = embedding_model.vector_size
        self.NP, self.F, self.CR = population_size, f_scale, cr_rate
        self.G_max, self.T_patience = max_generations, plateau_patience
        self.lambda_ppl, self.lambda_len = lambda_ppl, lambda_len
        self.weights = weights
        self.lambda_ctr = lambda_ctr
        self.softmatch_theta, self.softmatch_beta = softmatch_theta, softmatch_beta
        self.adj_sigma_w = adj_sigma_w
        self.success_criteria = success_criteria

        print("Initializing helper models for Attacker V2...")
        self.sbert_model = SentenceTransformer(
            'all-MiniLM-L6-v2', device='cuda' if torch.cuda.is_available() else 'cpu')

        # !! 核心修正：使用新的、正确的 PerplexityCalculatorV2 !!
        self.ppl_calculator = PerplexityCalculatorV2()

        print("Attacker V2 helper models ready.")
        self.query_count = 0
        self.v_allow_embeddings = None
        self.v_allow_list = None

    def _soft_match(self, text: str, keywords: Set[str]) -> float:
        if not keywords or not text:
            return 0.0
        text_ngrams = []
        words = text.lower().split()
        for n in range(1, 4):
            for i in range(len(words) - n + 1):
                text_ngrams.append(" ".join(words[i:i+n]))
        if not text_ngrams:
            return 0.0
        ngram_embeds = self.sbert_model.encode(
            text_ngrams, convert_to_tensor=True)
        keyword_embeds = self.sbert_model.encode(
            list(keywords), convert_to_tensor=True)
        cos_sims = util.pytorch_cos_sim(ngram_embeds, keyword_embeds)
        max_sim = torch.max(cos_sims).item()
        score = 1 / \
            (1 + math.exp(-(max_sim - self.softmatch_theta) / self.softmatch_beta))
        return score

    def _get_rag_response(self, query: str) -> Optional[Dict]:
        try:
            response = requests.post(self.rag_api_url, json={
                                     "query": query}, timeout=60)
            response.raise_for_status()
            self.query_count += 1
            return response.json()
        except requests.RequestException:
            return None

    def _calculate_score_tilde(self, response: Dict, path_info: Dict) -> Dict:
        text = response.get("answer", "")
        node_aliases = path_info['node_aliases']
        rel_keywords = path_info['rel_keywords']
        path_triples = path_info['path_triples']
        target_aliases = path_info['target_aliases']

        node_cov_sum = sum(self._soft_match(text, aliases)
                           for aliases in node_aliases.values())
        node_cov = node_cov_sum / len(node_aliases) if node_aliases else 0

        rel_cov_sum = sum(self._soft_match(text, keywords)
                          for keywords in rel_keywords.values())
        rel_cov = rel_cov_sum / len(rel_keywords) if rel_keywords else 0

        adj_sum = 0
        text_lower = text.lower()
        for head, rel, tail in path_triples:
            try:
                pos_h = min([text_lower.find(
                    a) for a in node_aliases[head] if a in text_lower] or [math.inf])
                pos_r = min([text_lower.find(k)
                            for k in rel_keywords[rel] if k in text_lower] or [math.inf])
                pos_t = min([text_lower.find(
                    a) for a in node_aliases[tail] if a in text_lower] or [math.inf])
                if pos_h < pos_r < pos_t:
                    dist_hr = pos_r - pos_h
                    dist_rt = pos_t - pos_r
                    k_hr = math.exp(-dist_hr**2 / (2 * self.adj_sigma_w**2))
                    k_rt = math.exp(-dist_rt**2 / (2 * self.adj_sigma_w**2))
                    adj_sum += k_hr * k_rt
            except KeyError:
                continue
        adj = adj_sum / len(path_triples) if path_triples else 0

        target_hit = self._soft_match(text, target_aliases)

        score = (self.weights['alpha'] * node_cov + self.weights['beta'] * rel_cov +
                 self.weights['rho'] * adj + self.weights['omega'] * target_hit)

        return {"Score_tilde": score, "NodeCov": node_cov, "RelCov": rel_cov, "Adj": adj, "Target": target_hit}

    def _calculate_loss(self, suffix: List[str], original_query: str, malicious_path_info: Dict, correct_path_info: Dict) -> Tuple[float, Dict]:
        attack_query = f"{original_query} {' '.join(suffix)}"
        response = self._get_rag_response(attack_query)
        if not response:
            return float('inf'), {}

        malicious_scores = self._calculate_score_tilde(
            response, malicious_path_info)
        correct_scores = self._calculate_score_tilde(
            response, correct_path_info)

        ctr_pen = self.lambda_ctr * (
            self.weights['alpha'] * correct_scores['NodeCov'] +
            self.weights['beta'] * correct_scores['RelCov'] +
            self.weights['rho'] * correct_scores['Adj']
        )

        final_score = malicious_scores['Score_tilde'] - ctr_pen

        # 使用正确的PPL计算
        ppl_q = self.ppl_calculator.get_perplexity(original_query)
        ppl_q_s = self.ppl_calculator.get_perplexity(attack_query)
        ppl_delta = max(0, math.log(ppl_q_s) - math.log(ppl_q)
                        ) if ppl_q > 0 and ppl_q_s > 0 else 0

        loss = (1 - final_score) + self.lambda_ppl * \
            ppl_delta + self.lambda_len * len(suffix)

        all_details = {**malicious_scores, "CtrPen": ctr_pen,
                       "PPL_Delta": ppl_delta, "loss": loss}
        return loss, all_details

    def _encode(self, words: List[str]) -> np.ndarray:
        return np.concatenate([self.embedding_model[w] for w in words if w in self.embedding_model])

    def _decode(self, vector: np.ndarray, L: int) -> List[str]:
        sub_vectors = np.array_split(vector, L)
        words = []
        for sub_vec in sub_vectors:
            sims = np.dot(self.v_allow_embeddings, sub_vec) / (np.linalg.norm(
                self.v_allow_embeddings, axis=1) * np.linalg.norm(sub_vec))
            words.append(self.v_allow_list[np.argmax(sims)])
        return words

    def _is_attack_successful(self, details: Dict) -> bool:
        if not details:
            return False
        return all(details.get(k, 0) >= v for k, v in self.success_criteria.items())

    def attack(self, original_query: str, malicious_path_info: Dict, correct_path_info: Dict, v_topic: Set[str], max_suffix_len: int = 5) -> Dict:
        start_time = time.time()
        self.query_count = 0

        # --- 词汇表构建 ---
        print("Building V_allow and embedding matrix...")
        v_allow_set = self.v_base.union(v_topic)
        self.v_allow_list = [w for w in v_allow_set if w in self.embedding_model]
        self.v_allow_embeddings = np.array([self.embedding_model[w] for w in self.v_allow_list])
        print(f"  - V_base size: {len(self.v_base)}")
        print(f"  - V_topic size: {len(v_topic)}")
        print(f"  - V_allow size (in model): {len(self.v_allow_list)}")

        best_suffix_so_far, best_loss_so_far, best_details = [], float('inf'), {}

        # --- 顺序长度课程学习 ---
        for L in range(1, max_suffix_len + 1):
            print(f"\n{'='*20} Attacking with Suffix Length L = {L} {'='*20}")

            # --- 初始化种群 ---
            print("  Initializing and encoding population...")
            population = np.zeros((self.NP, L * self.embedding_dim))
            for i in range(self.NP):
                random_words = np.random.choice(self.v_allow_list, L, replace=True)
                population[i] = self._encode(random_words)

            # --- 评估初始种群 (这里最耗时，使用 tqdm) ---
            print("  Evaluating initial population...")
            losses = np.full(self.NP, float('inf'))
        
            for i in trange(self.NP, desc=f"  Initial Eval (L={L})"):
                suffix = self._decode(population[i], L)
                loss, details = self._calculate_loss(suffix, original_query, malicious_path_info, correct_path_info)
                losses[i] = loss
                if loss < best_loss_so_far:
                    best_loss_so_far, best_suffix_so_far, best_details = loss, suffix, details
            print(f"  Initial population evaluated. Best loss so far: {best_loss_so_far:.4f}")

            plateau_counter = 0
            last_best_gen_loss = np.min(losses)

            # --- 差分进化迭代 ---
            
            gen_iterator = trange(self.G_max, desc=f"  DE Evolution (L={L})")
            for g in gen_iterator:
                for i in range(self.NP):
                    idxs = [idx for idx in range(self.NP) if idx != i]
                    a, b, c = population[np.random.choice(idxs, 3, replace=False)]
                    mutant = a + self.F * (b - c)
                    cross_points = np.random.rand(L * self.embedding_dim) < self.CR
                    trial = np.where(cross_points, mutant, population[i])

                    trial_suffix = self._decode(trial, L)
                    trial_loss, details = self._calculate_loss(trial_suffix, original_query, malicious_path_info, correct_path_info)

                    if trial_loss < losses[i]:
                        losses[i], population[i] = trial_loss, trial
                        if trial_loss < best_loss_so_far:
                            best_loss_so_far, best_suffix_so_far, best_details = trial_loss, trial_suffix, details
                            # 更新tqdm的附加信息
                            gen_iterator.set_postfix(best_loss=f"{best_loss_so_far:.4f}")

                if self._is_attack_successful(best_details):
                    print("\nAttack successful!")
                    # 关闭tqdm进度条
                    gen_iterator.close()
                    return {"status": "Success", "suffix": best_suffix_so_far, "details": best_details, "total_queries": self.query_count, "time_taken": time.time() - start_time}

                current_best_gen_loss = np.min(losses)
                if current_best_gen_loss < last_best_gen_loss:
                    last_best_gen_loss, plateau_counter = current_best_gen_loss, 0
                else:
                    plateau_counter += 1

                if plateau_counter >= self.T_patience:
                    print(f"\n  [L={L}] Plateau reached after {g+1} generations. Moving to next length.")
                    # 关闭tqdm进度条
                    gen_iterator.close()
                    break

            # 确保在循环正常结束后也关闭进度条
            if gen_iterator.n < gen_iterator.total -1: # 如果不是正常完成
                gen_iterator.close()


        print("\nAttack finished. Max length/generations reached.")
        return {"status": "Failed", "suffix": best_suffix_so_far, "details": best_details, "total_queries": self.query_count, "time_taken": time.time() - start_time}
