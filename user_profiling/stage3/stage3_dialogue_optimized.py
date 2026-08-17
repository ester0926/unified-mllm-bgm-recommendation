"""
Stage 3: Dialogue Generation (Optimized Version)
優化項目：
1. Persona Snippet 預先生成並快取（避免重複呼叫 LLM）
2. 批次處理對話生成（減少 I/O 開銷）
3. 可選的品質評估（加快實驗速度）
4. 改善的斷點續傳機制
5. 平行處理選項
"""

import json
import logging
import jsonlines
import re
import os
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import pickle

# 導入模組
from config import PathConfig, ModelConfig, AblationConfig
from llm_client import OllamaClient
from prompt_builder import build_prompt
from qa_validator import validate_dialogue
from quality_evaluator import QualityEvaluatorLLM

log_filename = "logs/stage3_generation_optimized.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename, encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class TwoPhaseGeneratorOptimized:
    def __init__(self, enable_quality_eval=False, use_cache=True, num_workers=1):
        """
        Args:
            enable_quality_eval: 是否啟用品質評估（關閉可大幅加速）
            use_cache: 是否使用 Persona Snippet 快取
            num_workers: 平行處理的線程數（建議 2-4）
        """
        self.client = OllamaClient(
            model_name=ModelConfig.LLM_MODEL_NAME,
            base_url=ModelConfig.OLLAMA_BASE_URL
        )
        self.ablation_mode = AblationConfig.SYNTHESIS_MODE
        self.metadata = self._load_metadata()
        self.enable_quality_eval = enable_quality_eval
        self.use_cache = use_cache
        self.num_workers = num_workers
        
        # Persona Snippet 快取
        self.persona_cache_file = PathConfig.STAGE3_OUTPUT_DIR / "persona_cache.pkl"
        self.persona_cache = self._load_persona_cache() if use_cache else {}
        
        # 只在需要時初始化評估器
        if enable_quality_eval:
            logger.info("Initializing Quality Evaluator (Judge)...")
            self.evaluator = QualityEvaluatorLLM() 
            logger.info("Quality Evaluator Ready.")
        else:
            self.evaluator = None
            logger.info("Quality Evaluation DISABLED for faster processing.")

    def _load_metadata(self):
        """載入 Stage 1 的 Enriched Metadata"""
        meta_path = PathConfig.MUSIC_METADATA_ENRICHED
        if meta_path.exists():
            with open(meta_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}

    def _load_persona_cache(self):
        """載入 Persona Snippet 快取"""
        if self.persona_cache_file.exists():
            try:
                with open(self.persona_cache_file, 'rb') as f:
                    cache = pickle.load(f)
                logger.info(f"Loaded {len(cache)} cached persona snippets.")
                return cache
            except Exception as e:
                logger.warning(f"Failed to load persona cache: {e}")
        return {}

    def _save_persona_cache(self):
        """儲存 Persona Snippet 快取"""
        if self.use_cache:
            self.persona_cache_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.persona_cache_file, 'wb') as f:
                pickle.dump(self.persona_cache, f)

    def _get_cache_key(self, core_sbs, negative_sbs):
        """生成快取鍵（基於音樂 ID 組合）"""
        core_ids = tuple(sorted([item.get('music_id', '') for item in core_sbs]))
        neg_ids = tuple(sorted([item.get('music_id', '') for item in negative_sbs]))
        return (core_ids, neg_ids)

    def generate_persona_snippets(self, core_sbs, negative_sbs):
        """Phase 1: 生成喜歡與不喜歡的畫像（支援快取）"""
        
        # 檢查快取
        if self.use_cache:
            cache_key = self._get_cache_key(core_sbs, negative_sbs)
            if cache_key in self.persona_cache:
                return self.persona_cache[cache_key]
        
        # 1. Generate Likes Snippet
        seeds_pos = [item.get('semantic_seed', '') for item in core_sbs if item.get('semantic_seed')]
        if not seeds_pos: 
            seeds_pos = [f"Likes {item.get('genre', 'unknown')}" for item in core_sbs]
        
        prompt_pos = f"""
        TASK: Summarize user preferences.
        INPUT:
        {chr(10).join(seeds_pos[:5])}
        
        INSTRUCTIONS:
        1. Summarize the user's music TASTE in 1 sentence.
        2. Focus on Genre, Mood, and Instruments.
        3. OUTPUT ONLY THE SENTENCE. NO CHAT.
        """
        try:
            res_pos = self.client.generate(prompt_pos, temperature=0.3).strip().strip('"')
        except: 
            res_pos = "The user enjoys pop and electronic music."

        # 2. Generate Dislikes Snippet
        seeds_neg = [item.get('semantic_seed', '') for item in negative_sbs if item.get('semantic_seed')]
        if not seeds_neg:
            res_neg = "No specific dislikes."
        else:
            prompt_neg = f"""
            TASK: Extraction of Dislikes.
            INPUT (Rejected Songs):
            {chr(10).join(seeds_neg[:5])}
            
            INSTRUCTIONS:
            1. Identify what the user HATES (e.g., loud drums, screaming vocals).
            2. Write ONE sentence starting with "The user dislikes..." or "The user avoids...".
            3. CRITICAL: DO NOT ASK QUESTIONS. DO NOT ASK FOR DATA. OUTPUT ONLY THE SUMMARY.
            """
            try:
                res_neg = self.client.generate(prompt_neg, temperature=0.3).strip().strip('"')
                if "please provide" in res_neg.lower() or "?" in res_neg:
                    res_neg = "The user dislikes the specific style of the rejected tracks."
            except: 
                res_neg = "The user avoids discordant sounds."
        
        # 儲存到快取
        if self.use_cache:
            cache_key = self._get_cache_key(core_sbs, negative_sbs)
            self.persona_cache[cache_key] = (res_pos, res_neg)
            
        return res_pos, res_neg

    def _parse_response_flexible(self, response: str):
        """彈性解析器（與原版相同）"""
        plan = ""
        dialogue_text = response
        
        if "[DIALOGUE]" in response:
            parts = response.split("[DIALOGUE]")
            dialogue_text = parts[1].strip()
            if "[PLAN]" in parts[0]:
                plan = parts[0].split("[PLAN]")[1].strip()
        
        turns = []
        current_turn = {}
        lines = dialogue_text.split('\n')
        
        for line in lines:
            line = line.strip()
            if not line: continue
            if "omitted" in line or ("..." in line and len(line) < 20): continue
            if line.lower().startswith("turn ") or line == "---": continue
            
            user_match = re.match(r'^(User|USER):\s*"?([^"]*)"?$', line)
            if not user_match:
                user_match = re.match(r'^(User|USER):\s*(.*)', line)

            bot_match = re.match(r'^(Recommender|RECOMMENDER|Assistant):\s*(.*)', line)
            
            if user_match:
                if "role" in current_turn: turns.append(current_turn)
                content = user_match.group(2).strip()
                if content.startswith('"') and content.endswith('"'): 
                    content = content[1:-1]
                current_turn = {"role": "User", "content": content}
            elif bot_match:
                if "role" in current_turn: turns.append(current_turn)
                current_turn = {"role": "Recommender", "content": bot_match.group(2).strip()}
            elif current_turn:
                current_turn["content"] += " " + line
        
        if current_turn: turns.append(current_turn)
        for i, t in enumerate(turns): 
            t['turn'] = (i // 2) + 1
            
        return plan, turns

    def generate_dialogue(self, target_meta, pos_snippet, neg_snippet, search_history, dialogue_type):
        """Phase 2: 對話合成（支援快速失敗）"""
        max_retries = 3
        last_failure_reason = None
        
        for attempt in range(max_retries):
            prompt = build_prompt(
                target_music=target_meta,
                ltp_text=pos_snippet,
                search_history=search_history,
                dialogue_type=dialogue_type,
                turn_number=10,
                retry_attempt=attempt,
                last_failure_reason=last_failure_reason,
                user_dislikes=neg_snippet,
                use_cot=(self.ablation_mode != 'free_form')
            )

            try:
                # 第一次嘗試用較高溫度，重試時降溫
                temp = 0.7 if attempt == 0 else 0.4
                raw_res = self.client.generate(prompt, temperature=temp)
                plan, turns = self._parse_response_flexible(raw_res)
                
                if not turns:
                    last_failure_reason = "Format Error: No dialogue turns parsed."
                    continue

                # 簡化驗證（只做基本檢查，跳過複雜驗證）
                full_persona = f"{pos_snippet} {neg_snippet}"
                is_valid, f_type, f_reason = validate_dialogue(
                    turns, dialogue_type, full_persona, retry_attempt=attempt
                )
                
                if is_valid:
                    # 只在啟用時才評估品質
                    scores = {}
                    if self.enable_quality_eval:
                        scores = self.evaluator.evaluate(turns, full_persona, dialogue_type)
                    return raw_res, turns, scores, attempt
                
                last_failure_reason = f"{f_type}: {f_reason}"
                
            except Exception as e:
                logger.error(f"Gen error: {e}")
                last_failure_reason = "System Error"
                
        return "", [], {}, max_retries

    def _get_existing_ids(self, output_file):
        """讀取已存在的 music_id_type（優化版）"""
        existing = set()
        if output_file.exists():
            try:
                with jsonlines.open(output_file, 'r') as reader:
                    for obj in reader:
                        mid = obj.get('music_id')
                        dtype = obj.get('dialogue_type')
                        if mid and dtype:
                            existing.add(f"{mid}_{dtype}")
                logger.info(f"Resuming... Found {len(existing)} existing dialogues.")
            except Exception as e:
                logger.warning(f"Error reading existing file: {e}")
        return existing

    def _process_single_file(self, f_path, existing_keys, types, out_file):
        """處理單個檔案（用於平行處理）"""
        results = []
        
        try:
            with open(f_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            music_id = data['target_music']
            
            # 預檢查：如果所有類型都完成了就跳過
            if all(f"{music_id}_{t}" in existing_keys for t in types):
                return results

            balanced_hist = data.get('balanced_history', {})
            core_sbs = balanced_hist.get('core_sbs', [])
            explor_sbs = balanced_hist.get('exploratory_sbs', [])
            negative_sbs = balanced_hist.get('negative_sbs', [])
            
            if not core_sbs:
                return results
            
            # 1. Generate Snippets (只生成一次)
            pos_snippet, neg_snippet = self.generate_persona_snippets(core_sbs, negative_sbs)
            main_target_meta = self.metadata.get(
                music_id, 
                {"title": "Unknown Track", "genre": "Unknown", "tags": []}
            )
            
            for dtype in types:
                # 逐類型檢查 Resume
                if f"{music_id}_{dtype}" in existing_keys:
                    continue

                target_meta = main_target_meta
                hist_context = core_sbs[:3]
                
                if dtype == "Exploratory" and explor_sbs:
                    hist_context = explor_sbs[:3]
                elif dtype == "Negative" and negative_sbs:
                    neg_item = negative_sbs[0]
                    target_meta = {
                        "title": neg_item.get('title', 'Unknown'),
                        "artist": neg_item.get('artist', 'Unknown'),
                        "genre": neg_item.get('genre', 'Unknown'),
                        "tags": neg_item.get('tags', [])
                    }
                    hist_context = core_sbs[:3]

                raw, turns, scores, retries = self.generate_dialogue(
                    target_meta, pos_snippet, neg_snippet, hist_context, dtype
                )
                
                if turns:
                    result = {
                        "music_id": music_id,
                        "dialogue_target_id": target_meta.get('title'),
                        "dialogue_type": dtype,
                        "persona_snippet": pos_snippet,
                        "negative_snippet": neg_snippet,
                        "dialogue_raw": raw,
                        "dialogue_turns": turns,
                        "quality_scores": scores,
                        "retry_count": retries
                    }
                    results.append(result)
                else:
                    logger.warning(f"FAILED: {dtype} for {music_id}")
                    
        except Exception as e:
            logger.error(f"Error {f_path.name}: {e}")
        
        return results

    def run(self):
        """主流程（優化版）"""
        suffix = AblationConfig.get_experiment_suffix()
        input_dir = PathConfig.STAGE2_OUTPUT_DIR / AblationConfig.SAMPLING_STRATEGY
        if not input_dir.exists(): 
            input_dir = PathConfig.STAGE2_OUTPUT_DIR
        
        history_files = list(input_dir.glob(f"*{suffix}__history.json"))
        if not history_files: 
            history_files = list(input_dir.glob("*__history.json"))

        out_dir = PathConfig.STAGE3_OUTPUT_DIR / self.ablation_mode
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / "dialogues.jsonl"
        
        logger.info(f"Input: {input_dir}")
        logger.info(f"Output: {out_file}")
        logger.info(f"Quality Evaluation: {'ENABLED' if self.enable_quality_eval else 'DISABLED'}")
        logger.info(f"Persona Cache: {'ENABLED' if self.use_cache else 'DISABLED'}")
        logger.info(f"Workers: {self.num_workers}")
        
        existing_keys = self._get_existing_ids(out_file)
        
        types = ["Positive", "Exploratory", "Negative"]
        if self.ablation_mode == 'single_template': 
            types = ["Positive"]
        
        # 批次寫入緩衝
        write_buffer = []
        buffer_size = 10  # 每 10 個結果寫入一次
        
        with jsonlines.open(out_file, 'a') as writer:
            if self.num_workers > 1:
                # 平行處理
                with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                    futures = {
                        executor.submit(
                            self._process_single_file, 
                            f_path, existing_keys, types, out_file
                        ): f_path 
                        for f_path in history_files
                    }
                    
                    for future in tqdm(
                        as_completed(futures), 
                        total=len(futures), 
                        desc="Synthesizing (Parallel)"
                    ):
                        results = future.result()
                        write_buffer.extend(results)
                        
                        # 批次寫入
                        if len(write_buffer) >= buffer_size:
                            for result in write_buffer:
                                writer.write(result)
                                existing_keys.add(
                                    f"{result['music_id']}_{result['dialogue_type']}"
                                )
                            write_buffer = []
            else:
                # 序列處理（原版）
                for f_path in tqdm(history_files, desc="Synthesizing"):
                    results = self._process_single_file(f_path, existing_keys, types, out_file)
                    write_buffer.extend(results)
                    
                    # 批次寫入
                    if len(write_buffer) >= buffer_size:
                        for result in write_buffer:
                            writer.write(result)
                            existing_keys.add(
                                f"{result['music_id']}_{result['dialogue_type']}"
                            )
                        write_buffer = []
            
            # 寫入剩餘結果
            for result in write_buffer:
                writer.write(result)

        # 儲存 Persona Cache
        self._save_persona_cache()
        
        logger.info(f"Stage 3 Complete.")

if __name__ == "__main__":
    # 快速模式：關閉品質評估、啟用快取、使用 2 個線程
    generator = TwoPhaseGeneratorOptimized(
        enable_quality_eval=False,  # 關閉可加速 3-5 倍
        use_cache=True,             # 避免重複生成 persona
        num_workers=4               # 平行處理（建議 2-4）
    )
    generator.run()