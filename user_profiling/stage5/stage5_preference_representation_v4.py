"""
Stage 5: Core Set Preference Representation (Enhanced Version)
功能：
1. 從 Stage 4 提取顯性偏好（summary_text + salient_facts）
2. 從 Stage 2 提取隱性偏好（PersonaX 採樣音樂）
3. 使用 CLIP-T 編碼顯性偏好，AST 編碼隱性偏好
4. 語義相似度加權融合
5. 支援消融實驗（Explicit Only / Implicit Only / Hybrid）
6. 輸出 HDF5 格式供 Stage 6 使用

更新日誌：
- 整合舊版本 (stage5_fusion.py) 的消融實驗支援
- 保留新版本的 CLIP-T 編碼器與語義加權
- 改進 Stage 4 輸出相容性
- 完整的 salient_facts 處理邏輯
"""

import json
import logging
import h5py
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List, Tuple
import jsonlines
from transformers import CLIPTextModel, CLIPTokenizer
import warnings
warnings.filterwarnings('ignore')

# ==================== 配置區 ====================

class PathConfig:
    """路徑配置 - 請根據實際情況修改"""
    # 輸入路徑
    STAGE4_PROFILES = Path(r"data/user_profiling/long_term_preference\stage4_recLLM\profiles.jsonl")
    STAGE2_HISTORY_DIR = Path(r"data/user_profiling/long_term_preference\stage2_history\personax")
    STAGE1_METADATA = Path(r"data/user_profiling/music_metadata_simple\music_metadata_enriched.json")
    HDF5_DIR = Path(r"data/optimized_musechat_features_float16_v3")  # v3 新結構
    
    # 輸出路徑
    STAGE5_OUTPUT_DIR = Path(r"data/user_profiling/stage5_output")
    PREFERENCE_VECTORS = STAGE5_OUTPUT_DIR / "preference_vectors.h5"
    GENERATION_LOG = STAGE5_OUTPUT_DIR / "generation_log.jsonl"

class ModelConfig:
    """模型配置"""
    # CLIP Text Encoder
    CLIP_MODEL = "openai/clip-vit-base-patch32"
    CLIP_DIM = 512
    EXPLICIT_PROJ_DIM = 256

    # AST
    AST_DIM = 768
    IMPLICIT_PROJ_DIM = 256

    # ── 輸出維度（MVT-Fusion 的統一接口）───────────────────────
    # 三種消融模式輸出維度一致，MVT-Fusion 無需修改架構
    OUTPUT_DIM = 256

    # 加權參數
    BETA = 2.0

    # ==================== 消融實驗配置 ====================
    # 'hybrid'        → concat(explicit_256, implicit_256) → Linear(512→256) → P_ltp [256D]
    # 'explicit_only' → explicit_256                       → Linear(256→256) → P_ltp [256D]
    # 'implicit_only' → implicit_256                       → Linear(256→256) → P_ltp [256D]
    # 三種模式輸出維度相同，MVT-Fusion 接口統一，消融結果可直接比較
    REPRESENTATION_MODE = "implicit_only"

    # ==================== 預對齊訓練配置 ====================
    PRETRAIN_ALIGNMENT   = True
    PRETRAIN_EPOCHS      = 200
    PRETRAIN_BATCH_SIZE  = 256
    PRETRAIN_LR          = 5e-4
    PRETRAIN_TEMPERATURE = 0.07
    PRETRAIN_SAVE_PATH   = PathConfig.STAGE5_OUTPUT_DIR / "projection_weights.pt"

# ==================== 日誌設定 ====================
PathConfig.STAGE5_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(PathConfig.STAGE5_OUTPUT_DIR / "stage5.log", encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# ==================== 跨模態預對齊訓練 ====================

class CrossModalAligner:
    """
    Stage 5 跨模態對比預對齊
    ─────────────────────────────────────────────────────────
    目標：讓 W_explicit（512→256）和 W_implicit（768→256）
          在投影後的 256D 空間中，相同 pair 的文字向量和音樂向量
          語義相近，不同 pair 的語義遠離。

    監督信號：MuseChat HDF5 的天然配對
      text_features [77,512] (masked mean pool) ↔ target_music_all_cls [12,768] (mean pool)

    Loss：InfoNCE（雙向對稱版，等同 CLIP 訓練目標）
      L = 0.5 * (L_text2audio + L_audio2text)

    訓練完成後 W_explicit / W_implicit 具備跨模態語義對齊能力，
    Stage 6 端對端訓練再以任務梯度做任務導向微調。
    """

    def __init__(self, W_explicit: torch.nn.Linear,
                       W_implicit: torch.nn.Linear,
                       device: torch.device):
        self.W_explicit = W_explicit
        self.W_implicit = W_implicit
        self.device     = device
        self.temp       = ModelConfig.PRETRAIN_TEMPERATURE

    # ── 資料收集 ──────────────────────────────────────────────
    def collect_pairs(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        從所有 HDF5 收集 (text_vec [512], audio_vec [768]) 配對。
        text：masked mean pool；audio：mean pool over 12 segments。
        """
        text_list, audio_list = [], []
        h5_dir   = PathConfig.HDF5_DIR
        h5_files = sorted(h5_dir.glob("musechat_features_*.h5"))

        logger.info(f"收集預對齊訓練資料（{len(h5_files)} 個 HDF5）...")
        for fpath in tqdm(h5_files, desc="收集配對"):
            try:
                with h5py.File(fpath, 'r') as f:
                    pairs_grp = f.get('pairs', f)
                    for key in pairs_grp.keys():
                        try:
                            grp = pairs_grp[key]
                            if 'text_features'        not in grp: continue
                            if 'target_music_all_cls' not in grp: continue

                            # text: masked mean pool [77,512] → [512]
                            txt = grp['text_features'][()].astype(np.float32)
                            norms = np.linalg.norm(txt, axis=1)
                            mask  = (norms > 0.01).astype(np.float32)
                            if mask.sum() == 0: mask = np.ones(len(txt), np.float32)
                            txt_vec = (txt * mask[:, None]).sum(0) / mask.sum()

                            # audio: mean pool [12,768] → [768]
                            aud_vec = grp['target_music_all_cls'][()].astype(np.float32).mean(0)

                            text_list.append(txt_vec)
                            audio_list.append(aud_vec)
                        except Exception:
                            continue
            except Exception:
                continue

        logger.info(f"收集到 {len(text_list)} 個訓練配對")
        return np.array(text_list, np.float32), np.array(audio_list, np.float32)

    # ── InfoNCE Loss ─────────────────────────────────────────
    def info_nce_loss(self, text_proj: torch.Tensor,
                            audio_proj: torch.Tensor) -> torch.Tensor:
        """
        雙向 InfoNCE（對稱版 CLIP loss）

        text_proj  : [B, 256]，L2 normalized
        audio_proj : [B, 256]，L2 normalized
        """
        B = text_proj.shape[0]
        labels = torch.arange(B, device=self.device)   # 對角線是正例

        # 相似度矩陣 [B, B]
        sim = torch.matmul(text_proj, audio_proj.T) / self.temp

        # 兩個方向的交叉熵
        loss_t2a = torch.nn.functional.cross_entropy(sim,   labels)
        loss_a2t = torch.nn.functional.cross_entropy(sim.T, labels)
        return (loss_t2a + loss_a2t) / 2

    # ── 訓練主流程 ────────────────────────────────────────────
    def train(self) -> Dict[str, float]:
        """
        執行預對齊訓練，回傳最終 loss 統計。
        只訓練 W_explicit 和 W_implicit，CLIP / AST 凍結。
        """
        text_all, audio_all = self.collect_pairs()
        if len(text_all) == 0:
            logger.warning("⚠️ 無訓練資料，跳過預對齊")
            return {}

        # Dataset → DataLoader
        dataset = torch.utils.data.TensorDataset(
            torch.from_numpy(text_all),
            torch.from_numpy(audio_all)
        )
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=ModelConfig.PRETRAIN_BATCH_SIZE,
            shuffle=True, drop_last=True
        )

        optimizer = torch.optim.AdamW(
            list(self.W_explicit.parameters()) +
            list(self.W_implicit.parameters()),
            lr=ModelConfig.PRETRAIN_LR, weight_decay=1e-4
        )
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=ModelConfig.PRETRAIN_LR,
            steps_per_epoch=len(loader),
            epochs=ModelConfig.PRETRAIN_EPOCHS,
            pct_start=0.1    # 前 10% warmup，後 90% cosine decay
        )

        logger.info(f"開始跨模態預對齊訓練（{ModelConfig.PRETRAIN_EPOCHS} epochs，"
                    f"batch={ModelConfig.PRETRAIN_BATCH_SIZE}，"
                    f"T={self.temp}）")

        history = []
        for epoch in range(1, ModelConfig.PRETRAIN_EPOCHS + 1):
            self.W_explicit.train()
            self.W_implicit.train()
            epoch_loss = 0.0

            for txt_batch, aud_batch in loader:
                txt_batch = txt_batch.to(self.device)
                aud_batch = aud_batch.to(self.device)

                # 投影 + L2 正規化
                txt_proj = self.W_explicit(txt_batch)
                aud_proj = self.W_implicit(aud_batch)
                txt_proj = torch.nn.functional.normalize(txt_proj, dim=1)
                aud_proj = torch.nn.functional.normalize(aud_proj, dim=1)

                loss = self.info_nce_loss(txt_proj, aud_proj)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(self.W_explicit.parameters()) +
                    list(self.W_implicit.parameters()), 1.0
                )
                optimizer.step()
                scheduler.step()
                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(loader)
            history.append(avg_loss)
            logger.info(f"  Epoch {epoch:2d}/{ModelConfig.PRETRAIN_EPOCHS}  "
                        f"loss={avg_loss:.4f}  lr={scheduler.get_last_lr()[0]:.2e}")

        # 儲存訓練好的投影層權重
        save_path = ModelConfig.PRETRAIN_SAVE_PATH
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'W_explicit': self.W_explicit.state_dict(),
            'W_implicit': self.W_implicit.state_dict(),
            'final_loss': history[-1],
            'loss_history': history,
        }, save_path)
        logger.info(f"✅ 投影層權重已儲存：{save_path}")

        self.W_explicit.eval()
        self.W_implicit.eval()
        return {'final_loss': history[-1], 'loss_history': history}


class PreferenceVectorGenerator:
    """混合偏好向量生成器（增強版）"""
    
    def __init__(self):
        """初始化編碼器與投影矩陣"""
        logger.info("=" * 60)
        logger.info("🚀 Stage 5: Enhanced Preference Vector Generator")
        logger.info(f"Mode: {ModelConfig.REPRESENTATION_MODE}")
        logger.info("=" * 60)
        
        # 檢查 GPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Device: {self.device}")
        
        # 載入 CLIP Text Encoder
        logger.info("載入 CLIP Text Encoder...")
        self.clip_tokenizer = CLIPTokenizer.from_pretrained(ModelConfig.CLIP_MODEL)
        self.clip_model = CLIPTextModel.from_pretrained(ModelConfig.CLIP_MODEL).to(self.device)
        self.clip_model.eval()
        
        # 投影矩陣
        self.W_explicit = torch.nn.Linear(
            ModelConfig.CLIP_DIM, 
            ModelConfig.EXPLICIT_PROJ_DIM
        ).to(self.device)
        self.W_implicit = torch.nn.Linear(
            ModelConfig.AST_DIM, 
            ModelConfig.IMPLICIT_PROJ_DIM
        ).to(self.device)
        
        # ── 輸出投影頭 W_out（統一輸出維度，消融實驗共用接口）──
        # hybrid        : W_out_hybrid  Linear(512→256)
        # explicit_only : W_out_single  Linear(256→256)
        # implicit_only : W_out_single  Linear(256→256)
        # 三種模式輸出皆為 256D，MVT-Fusion 接口無需修改
        self.W_out_hybrid = torch.nn.Linear(
            ModelConfig.EXPLICIT_PROJ_DIM + ModelConfig.IMPLICIT_PROJ_DIM,
            ModelConfig.OUTPUT_DIM
        ).to(self.device)
        self.W_out_single = torch.nn.Linear(
            ModelConfig.EXPLICIT_PROJ_DIM,   # 256→256，也適用 implicit_only
            ModelConfig.OUTPUT_DIM
        ).to(self.device)
        torch.nn.init.xavier_uniform_(self.W_out_hybrid.weight)
        torch.nn.init.xavier_uniform_(self.W_out_single.weight)

        # ── 跨模態預對齊訓練（可選）────────────────────────────
        if ModelConfig.PRETRAIN_ALIGNMENT:
            save_path = ModelConfig.PRETRAIN_SAVE_PATH
            if save_path.exists():
                # 已有訓練好的權重，直接載入
                logger.info(f"載入預對齊投影層權重：{save_path}")
                ckpt = torch.load(save_path, map_location=self.device)
                self.W_explicit.load_state_dict(ckpt['W_explicit'])
                self.W_implicit.load_state_dict(ckpt['W_implicit'])
                logger.info(f"  (訓練時最終 loss = {ckpt.get('final_loss', 'N/A'):.4f})")
            else:
                # 執行預對齊訓練
                aligner = CrossModalAligner(
                    self.W_explicit, self.W_implicit, self.device
                )
                stats = aligner.train()
                logger.info(f"預對齊完成，最終 loss = {stats.get('final_loss', 'N/A'):.4f}")
        else:
            logger.info("⚠️ 跳過預對齊（PRETRAIN_ALIGNMENT=False），"
                        "W_explicit/W_implicit 保持 Xavier 隨機初始化")
        
        # 載入元數據
        logger.info("載入 Stage 1 元數據...")
        with open(PathConfig.STAGE1_METADATA, 'r', encoding='utf-8') as f:
            self.metadata = json.load(f)
        
        # 載入音樂嵌入
        logger.info("載入音樂嵌入索引...")
        self.music_embeddings = self._load_music_embeddings()
        
        logger.info("✅ 初始化完成！")
    
    def _load_music_embeddings(self) -> Dict[str, np.ndarray]:
        """
        載入預計算的音樂 AST 嵌入（v3 新結構版）

        新 HDF5 結構：
          pairs/
            {video_id}_{candidate_music_id}/
              target_music_all_cls   [12, 768] float16  ← 目標音樂 CLS token
              candidate_music_all_seq [12, 1214, 768]   ← 候選音樂完整序列
              text_features          [77, 512]
              video_features_all     [12, 768]

        索引策略：
          - 從 pair key 用 rsplit('_', 1) 分離 video_id 和 candidate_music_id
          - 對 target_music_all_cls [12, 768] 做 mean pool → [768]
          - 以 video_id 和 candidate_music_id 雙索引儲存，供 Stage 2 core_sbs 查詢
        """
        embeddings: Dict[str, np.ndarray] = {}
        hdf5_files = sorted(PathConfig.HDF5_DIR.glob("musechat_features_*.h5"))

        if not hdf5_files:
            logger.warning(f"⚠️ 未找到音樂嵌入文件：{PathConfig.HDF5_DIR}")
            return embeddings

        logger.info(f"從 {len(hdf5_files)} 個 HDF5 文件載入音樂嵌入（v3 結構）...")
        for hdf5_path in tqdm(hdf5_files, desc="Loading embeddings"):
            try:
                with h5py.File(hdf5_path, 'r') as f:
                    pairs_grp = f.get('pairs', f)
                    for pair_key in pairs_grp.keys():
                        try:
                            grp = pairs_grp[pair_key]

                            # ── 讀取 target_music_all_cls [12, 768] ──────────
                            if 'target_music_all_cls' not in grp:
                                continue
                            raw = grp['target_music_all_cls'][()].astype(np.float32)
                            # [12, 768] → mean pool → [768]
                            vec = raw.mean(axis=0)

                            # ── 從 pair_key 分離兩個 music_id ────────────────
                            # pair_key 格式：{video_id}_{candidate_music_id}
                            # video_id 通常為 YouTube ID（11 碼），但不固定，用 rsplit 安全分割
                            parts = pair_key.rsplit('_', 1)
                            if len(parts) == 2:
                                video_id, candidate_id = parts
                                if video_id    not in embeddings:
                                    embeddings[video_id]    = vec
                                if candidate_id not in embeddings:
                                    embeddings[candidate_id] = vec
                            # 完整 pair_key 也存一份，作為保底查詢
                            if pair_key not in embeddings:
                                embeddings[pair_key] = vec

                        except Exception as e:
                            logger.debug(f"讀取 pair {pair_key} 失敗: {e}")
                            continue

            except Exception as e:
                logger.warning(f"讀取 {hdf5_path.name} 失敗: {e}")
                continue

        logger.info(f"✅ 載入 {len(embeddings)} 個音樂嵌入")

        # 品質檢查
        if embeddings:
            sample = list(embeddings.values())[0]
            logger.info(f"嵌入品質：shape={sample.shape}, "
                        f"mean={np.mean(sample):.4f}, std={np.std(sample):.4f}")

        return embeddings
    
    def encode_explicit_preference(
        self, profile: Dict
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        編碼顯性偏好（整合 summary_text 與 salient_facts）

        回傳 (projected_vec [256], clip_emb [512])：
          - projected_vec：經 W_explicit 投影後的 256D 向量（供 P_ltp 拼接）
          - clip_emb：原始 CLIP 512D 向量（供 _compute_semantic_weights 複用，避免重複 forward）

        NOTE：W_explicit 若 PRETRAIN_ALIGNMENT=True，已透過 InfoNCE 完成跨模態預對齊；
              Stage 6 端對端訓練將進一步以推薦任務梯度做任務導向微調。

        優先級：salient_facts > summary_text > 零向量
        """
        facts = profile.get('salient_facts', [])
        if facts:
            if isinstance(facts[0], dict):
                fact_texts = [f.get('fact', '') for f in facts]
            else:
                fact_texts = list(facts)
            text_content = '. '.join(t for t in fact_texts if t)
        else:
            text_content = profile.get('summary_text', '')

        if not text_content or not text_content.strip():
            zero_proj = np.zeros(ModelConfig.EXPLICIT_PROJ_DIM, dtype=np.float32)
            zero_clip = np.zeros(ModelConfig.CLIP_DIM,           dtype=np.float32)
            return zero_proj, zero_clip

        with torch.no_grad():
            inputs = self.clip_tokenizer(
                text_content, return_tensors='pt',
                padding=True, truncation=True, max_length=77
            ).to(self.device)
            outputs   = self.clip_model(**inputs)
            clip_emb  = outputs.pooler_output          # [1, 512]
            proj_vec  = self.W_explicit(clip_emb)      # [1, 256]

        return (proj_vec.cpu().numpy().flatten(),
                clip_emb.cpu().numpy().flatten())
    
    def encode_implicit_preference(
        self,
        core_sbs: List[Dict],
        explicit_clip_emb: np.ndarray   # 複用已算好的 CLIP 512D，避免重複 forward
    ) -> np.ndarray:
        """
        編碼隱性偏好（語義相似度加權）

        NOTE：W_implicit 若 PRETRAIN_ALIGNMENT=True，已透過 InfoNCE 完成跨模態預對齊；
              Stage 6 端對端訓練將進一步以推薦任務梯度做任務導向微調。

        缺失嵌入的 music_id 直接跳過（不補隨機值，避免污染加權結果）。
        """
        if not core_sbs:
            return np.zeros(ModelConfig.IMPLICIT_PROJ_DIM, dtype=np.float32)

        music_embs:     List[np.ndarray] = []
        semantic_seeds: List[str]        = []
        skipped = 0

        for item in core_sbs:
            music_id      = item.get('music_id', '')
            semantic_seed = item.get('semantic_seed', '')

            if music_id in self.music_embeddings:
                music_embs.append(self.music_embeddings[music_id])
                semantic_seeds.append(semantic_seed)
            else:
                skipped += 1   # 跳過缺失嵌入，不補隨機值

        if skipped:
            logger.debug(f"跳過 {skipped} 個缺失嵌入的 music_id（不補隨機值）")

        if not music_embs:
            return np.zeros(ModelConfig.IMPLICIT_PROJ_DIM, dtype=np.float32)

        music_embs_arr = np.array(music_embs)   # [K, 768]

        # 語義相似度加權（複用傳入的 CLIP 向量，避免重複 forward）
        weights = self._compute_semantic_weights(
            semantic_seeds, explicit_clip_emb
        )

        # 加權平均
        weighted_emb = np.average(music_embs_arr, axis=0, weights=weights)  # [768]

        # 投影到 256 維
        with torch.no_grad():
            t = torch.tensor(weighted_emb, dtype=torch.float32).unsqueeze(0).to(self.device)
            implicit_vec = self.W_implicit(t)   # [1, 256]

        return implicit_vec.cpu().numpy().flatten()
    
    def _compute_semantic_weights(
        self,
        semantic_seeds:    List[str],
        explicit_clip_emb: np.ndarray   # 已算好的 CLIP 512D，直接使用
    ) -> np.ndarray:
        """
        計算語義相似度 Softmax 權重。
        接受已計算好的 CLIP 向量，避免對 explicit_text 重複 forward。
        """
        if not semantic_seeds:
            return np.ones(len(semantic_seeds)) / max(len(semantic_seeds), 1)

        with torch.no_grad():
            # 顯性偏好向量（直接用傳入的 numpy 向量）
            explicit_t = torch.tensor(
                explicit_clip_emb, dtype=torch.float32
            ).unsqueeze(0).to(self.device)   # [1, 512]

            # 編碼語義種子
            seeds_inputs = self.clip_tokenizer(
                semantic_seeds, return_tensors='pt',
                padding=True, truncation=True, max_length=77
            ).to(self.device)
            seeds_emb = self.clip_model(**seeds_inputs).pooler_output  # [K, 512]

            # 餘弦相似度 → softmax 權重
            explicit_norm = explicit_t  / explicit_t.norm(dim=1, keepdim=True).clamp(min=1e-8)
            seeds_norm    = seeds_emb   / seeds_emb.norm(dim=1, keepdim=True).clamp(min=1e-8)
            similarities  = torch.matmul(seeds_norm, explicit_norm.T).squeeze(-1)  # [K]

            weights = torch.softmax(ModelConfig.BETA * similarities, dim=0)
            return weights.cpu().numpy()
    
    def generate_mixed_vector(
        self,
        profile: Dict,
        history: Dict
    ) -> Tuple[np.ndarray, Dict]:
        """
        生成 P_ltp（三種消融模式輸出維度統一為 256D）

        hybrid        : W_out_hybrid( concat(explicit_256, implicit_256) ) → [256D]
        explicit_only : W_out_single( explicit_256 )                       → [256D]
        implicit_only : W_out_single( implicit_256 )                       → [256D]

        設計原則：
        - 不使用零填充（補零非語義中立，會污染推薦模型的梯度）
        - 三種模式輸出維度相同，MVT-Fusion 接口統一，消融結果可直接比較
        - W_out_hybrid / W_out_single 僅作離線特徵生成階段的固定維度對齊映射
          （Xavier 隨機初始化後不再更新），不參與下游 Unified MLLM 的梯度更新；
          後續任務梯度僅更新下游可訓練的 ltp_proj（見論文第 3.2.6 節之描述）
        """
        core_sbs = history.get('balanced_history', {}).get('core_sbs', [])

        # ── 顯性偏好（256D + 原始 CLIP 512D 供加權複用）────────
        explicit_proj_vec, explicit_clip_emb = self.encode_explicit_preference(profile)

        # ── 隱性偏好（256D，複用 clip_emb）────────────────────
        implicit_vec = self.encode_implicit_preference(core_sbs, explicit_clip_emb)

        # ── 消融模式：各自用真實向量，統一經 W_out 輸出 256D ───
        mode = ModelConfig.REPRESENTATION_MODE

        with torch.no_grad():
            if mode == 'hybrid':
                # concat 512D → W_out_hybrid → 256D
                combined = np.concatenate([explicit_proj_vec, implicit_vec])
                t = torch.tensor(combined, dtype=torch.float32).unsqueeze(0).to(self.device)
                p_ltp = self.W_out_hybrid(t).cpu().numpy().flatten()

            elif mode == 'explicit_only':
                # explicit 256D → W_out_single → 256D
                t = torch.tensor(explicit_proj_vec, dtype=torch.float32).unsqueeze(0).to(self.device)
                p_ltp = self.W_out_single(t).cpu().numpy().flatten()

            else:  # implicit_only
                # implicit 256D → W_out_single → 256D
                t = torch.tensor(implicit_vec, dtype=torch.float32).unsqueeze(0).to(self.device)
                p_ltp = self.W_out_single(t).cpu().numpy().flatten()

        facts = profile.get('salient_facts', [])
        metadata = {
            'music_id':          profile.get('music_id') or profile.get('target_music'),
            'mode':              mode,
            'output_dim':        int(p_ltp.shape[0]),          # 永遠是 256
            'explicit_dim':      int(explicit_proj_vec.shape[0]),
            'implicit_dim':      int(implicit_vec.shape[0]),
            'core_sbs_count':    len(core_sbs),
            'core_sbs_matched':  sum(
                1 for item in core_sbs
                if item.get('music_id', '') in self.music_embeddings
            ),
            'has_facts':         bool(facts),
            'has_summary':       bool(profile.get('summary_text')),
        }

        return p_ltp, metadata


# ==================== 主流程 ====================

class Stage5Pipeline:
    """Stage 5 完整流程（增強版）"""
    
    def __init__(self):
        self.generator = PreferenceVectorGenerator()
        self.stats = {
            'total': 0,
            'success': 0,
            'failed': 0,
            'missing_history': 0,
            'missing_embedding': 0,
            'mode_counts': {
                'used_facts': 0,
                'used_summary': 0,
                'no_explicit': 0
            }
        }
    
    def load_profiles(self) -> List[Dict]:
        """載入 Stage 4 profiles（相容性改進）"""
        logger.info(f"載入 Stage 4 profiles: {PathConfig.STAGE4_PROFILES}")
        profiles = []
        
        with jsonlines.open(PathConfig.STAGE4_PROFILES, 'r') as reader:
            for obj in reader:
                # 相容兩種命名方式
                if 'music_id' not in obj and 'target_music' in obj:
                    obj['music_id'] = obj['target_music']
                elif 'target_music' not in obj and 'music_id' in obj:
                    obj['target_music'] = obj['music_id']
                
                profiles.append(obj)
        
        logger.info(f"✅ 載入 {len(profiles)} 個 profiles")
        return profiles
    
    def load_history(self, music_id: str) -> Dict:
        """載入 Stage 2 PersonaX 歷史"""
        history_path = PathConfig.STAGE2_HISTORY_DIR / f"{music_id}__history.json"
        
        if not history_path.exists():
            logger.warning(f"⚠️ 找不到歷史文件: {history_path}")
            self.stats['missing_history'] += 1
            return {}
        
        with open(history_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def run(self):
        """執行完整流程"""
        logger.info("=" * 60)
        logger.info("🚀 Stage 5: Core Set Preference Representation")
        logger.info(f"Mode: {ModelConfig.REPRESENTATION_MODE}")
        logger.info("=" * 60)
        
        # 載入 profiles
        profiles = self.load_profiles()
        self.stats['total'] = len(profiles)
        
        # 準備 HDF5 輸出
        mode_suffix = f"_{ModelConfig.REPRESENTATION_MODE}" if ModelConfig.REPRESENTATION_MODE != "hybrid" else ""
        output_path = Path(str(PathConfig.PREFERENCE_VECTORS).replace(".h5", f"{mode_suffix}.h5"))
        output_h5 = h5py.File(output_path, 'w')
        vectors_group = output_h5.create_group('preference_vectors')
        
        # 準備 log
        log_path = Path(str(PathConfig.GENERATION_LOG).replace(".jsonl", f"{mode_suffix}.jsonl"))
        log_writer = jsonlines.open(log_path, 'w')
        
        # 逐個處理
        logger.info(f"開始生成 {len(profiles)} 個偏好向量...")
        
        for profile in tqdm(profiles, desc="Generating vectors"):
            music_id = profile.get('music_id') or profile.get('target_music')
            
            try:
                # 載入歷史
                history = self.load_history(music_id)
                
                if not history:
                    self.stats['failed'] += 1
                    continue
                
                # 生成混合向量
                mixed_vec, metadata = self.generator.generate_mixed_vector(profile, history)
                
                # 統計使用的偏好來源
                if metadata.get('has_facts'):
                    self.stats['mode_counts']['used_facts'] += 1
                elif metadata.get('has_summary'):
                    self.stats['mode_counts']['used_summary'] += 1
                else:
                    self.stats['mode_counts']['no_explicit'] += 1
                
                # 存儲到 HDF5
                vectors_group.create_dataset(
                    music_id,
                    data=mixed_vec,
                    dtype='float32'
                )
                
                # 寫入日誌
                log_entry = {
                    'music_id': music_id,
                    'status': 'success',
                    **metadata
                }
                log_writer.write(log_entry)
                
                self.stats['success'] += 1
                
            except Exception as e:
                logger.error(f"❌ {music_id} 失敗: {e}")
                log_writer.write({
                    'music_id': music_id,
                    'status': 'failed',
                    'error': str(e)
                })
                self.stats['failed'] += 1
        
        # 關閉文件
        output_h5.close()
        log_writer.close()
        
        # 輸出統計
        self._print_stats(output_path, log_path)
    
    def _print_stats(self, output_path, log_path):
        """輸出統計資訊"""
        logger.info("\n" + "=" * 60)
        logger.info("📊 Stage 5 統計")
        logger.info("=" * 60)
        logger.info(f"輸出維度: {ModelConfig.OUTPUT_DIM}D（三種模式統一）")
        logger.info(f"總數: {self.stats['total']}")
        logger.info(f"成功: {self.stats['success']} ({self.stats['success']/self.stats['total']*100:.1f}%)")
        logger.info(f"失敗: {self.stats['failed']}")
        logger.info(f"  - 缺少歷史: {self.stats['missing_history']}")
        logger.info(f"  - 缺少嵌入: {self.stats['missing_embedding']}")
        
        logger.info(f"\n偏好來源統計:")
        logger.info(f"  - 使用 salient_facts: {self.stats['mode_counts']['used_facts']}")
        logger.info(f"  - 使用 summary_text: {self.stats['mode_counts']['used_summary']}")
        logger.info(f"  - 無顯性偏好: {self.stats['mode_counts']['no_explicit']}")
        
        logger.info("=" * 60)
        logger.info(f"✅ 輸出文件: {output_path}")
        logger.info(f"✅ 日誌文件: {log_path}")


# ==================== 主程式 ====================

def main():
    """主程式入口"""
    try:
        pipeline = Stage5Pipeline()
        pipeline.run()
        logger.info("\n🎉 Stage 5 完成！")
        logger.info(f"Mode: {ModelConfig.REPRESENTATION_MODE}")
    except Exception as e:
        logger.error(f"❌ Stage 5 失敗: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()