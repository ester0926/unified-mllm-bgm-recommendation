"""
config.py — Unified MLLM 超參數與架構配置
Pointwise 版本（v2）Plan B：新增 [RANK] token 作為 ranking readout

Plan B 主要變更（相比原 Pointwise v2）：
  - prefix 順序：[VIDEO, MUSIC, LTP, TEXT_CLIP] → [VIDEO, LTP, TEXT_CLIP, MUSIC]
  - 新增 [RANK] token：放在 prompt 末尾，作為 ranking readout token
  - special_tokens：5 個（加入 [RANK]），vocab = 32000 + 5 = 32005

消融實驗擴充（v3）：
  - 新增 active_modalities：控制哪些模態注入 prefix
  - multimodal_prefix_len 由 __post_init__ 自動從 active_modalities 推導
    完整實驗 = 4；w/o 任一模態 = 3
  - exp_01 hybrid（全模態）：active_modalities=["video","ltp","text","music"]
  - exp_04 w/o P_ltp：      active_modalities=["video","text","music"]
  - exp_05 w/o Video：      active_modalities=["ltp","text","music"]
  - exp_06 w/o Text：       active_modalities=["video","ltp","music"]
  - exp_07 w/o Music：      active_modalities=["video","ltp","text"]
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ModelConfig:

    # ── 凍結編碼器輸出維度 ────────────────────────────────────────────────────
    video_dim: int = 768
    music_dim: int = 768
    text_dim: int = 512
    ltp_dim: int = 256

    # ── 投影層設計 ─────────────────────────────────────────────────────────────
    llama_hidden_dim: int = 4096
    projector_hidden_dim: int = 2048
    projector_dropout: float = 0.05

    # ── 候選音樂設定（Pointwise：1 首）────────────────────────────────────────
    num_candidates: int = 1

    # ── Modality 嵌入 ──────────────────────────────────────────────────────────
    num_modality_types: int = 4   # 0=VIDEO, 1=MUSIC, 2=LTP, 3=TEXT_CLIP

    # ── LLaMA-2-7B 設定 ───────────────────────────────────────────────────────
    llama_model_name: str = "meta-llama/Llama-2-7b-hf"
    max_seq_len: int = 512
    max_new_tokens: int = 128

    # ── LoRA 設定 ─────────────────────────────────────────────────────────────
    lora_rank: int = 32
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "v_proj"]
    )
    lora_bias: str = "none"

    # ── 多任務損失權重 ─────────────────────────────────────────────────────────
    lambda_rank: float = 0.5
    lambda_gen: float = 0.5

    # ── ★ 消融實驗：模態選擇 ───────────────────────────────────────────────────
    # 控制哪些模態注入 prefix，決定 multimodal_prefix_len。
    # __post_init__ 會將此列表正規化（保持 video→ltp→text→music 順序），
    # 並自動設定 multimodal_prefix_len = len(active_modalities)。
    #
    # 設定範例：
    #   exp_01 全模態：        ["video", "ltp", "text", "music"]  → prefix_len=4
    #   exp_04 w/o P_ltp：     ["video", "text", "music"]         → prefix_len=3
    #   exp_05 w/o Video：     ["ltp", "text", "music"]           → prefix_len=3
    #   exp_06 w/o Text：      ["video", "ltp", "music"]          → prefix_len=3
    #   exp_07 w/o Music：     ["video", "ltp", "text"]           → prefix_len=3
    #
    # ⚠️ 勿直接設定 multimodal_prefix_len；它由 __post_init__ 自動推導
    active_modalities: List[str] = field(
        default_factory=lambda: ["video", "ltp", "text", "music"]
    )

    # ── 序列佈局（由 __post_init__ 自動推導）──────────────────────────────────
    # 此欄位宣告保留以供型別提示，實際值由 __post_init__ 覆蓋
    multimodal_prefix_len: int = 4

    # music_token_offset：MUSIC 在完整 prefix 中的位置（pos3，供 XAI 使用）
    music_token_offset: int = 3

    # ── [RANK] Token（Plan B 核心）────────────────────────────────────────────
    rank_special_token: str = "[RANK]"

    def __post_init__(self):
        """
        ★ 自動從 active_modalities 推導 multimodal_prefix_len。

        同時正規化 active_modalities：
          1. 過濾非法模態名稱
          2. 強制維持固定順序 video → ltp → text → music
             （確保 projectors.py 的 prefix 拼接順序一致）
        """
        _canonical_order = ["video", "ltp", "text", "music"]
        self.active_modalities = [
            m for m in _canonical_order if m in self.active_modalities
        ]
        if not self.active_modalities:
            raise ValueError(
                "active_modalities 不能為空！"
                "至少需要包含一種模態：video / ltp / text / music"
            )
        self.multimodal_prefix_len = len(self.active_modalities)


@dataclass
class TrainConfig:

    # ── 資料路徑 ──────────────────────────────────────────────────────────────
    data_dir: str = "./data/optimized_musechat_features_float16_v3"
    json_dir: str = "./data/musechat_json"
    pair_index_cache: str = "./cache/pair_index.json"
    song_bank_cache:  str = "./cache/song_bank"

    # ── 資料集分割比例 ─────────────────────────────────────────────────────────
    train_ratio: float = 0.90
    val_ratio: float = 0.05
    test_ratio: float = 0.05
    split_seed: int = 42

    # ── Batch 設定 ────────────────────────────────────────────────────────────
    micro_batch_size: int = 4
    accumulation_steps: int = 16
    eval_batch_size: int = 8

    # ── 訓練週期 ──────────────────────────────────────────────────────────────
    num_epochs: int = 10

    # ── 優化器 ────────────────────────────────────────────────────────────────
    learning_rate: float = 2e-4
    lora_learning_rate: float = 2e-4
    weight_decay: float = 5e-4
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "cosine"
    max_grad_norm: float = 1.0

    # ── 混合精度 ──────────────────────────────────────────────────────────────
    use_bf16: bool = True
    use_fp16: bool = False

    # ── 記憶體優化 ────────────────────────────────────────────────────────────
    use_gradient_checkpointing: bool = True
    use_flash_attention: bool = False

    # ── 評估設定 ──────────────────────────────────────────────────────────────
    music_pool_size: int = 500
    pointwise_eval_batch_size: int = 32

    # ── 輸出路徑 ──────────────────────────────────────────────────────────────
    output_dir: str = "./checkpoints"
    log_dir: str = "./logs"
    best_model_metric: str = "recall@1"

    # ── Ranking Loss 類型 ─────────────────────────────────────────────────────
    ranking_loss_type: str = "bpr"

    # ── 續訓設定 ──────────────────────────────────────────────────────────────
    resume_checkpoint: Optional[str] = None

    # ── Tokenizer 特殊 Token ──────────────────────────────────────────────────
    # Plan B：5 個 token（加入 [RANK]），vocab = 32000 + 5 = 32005
    # ⚠️ 消融實驗（exp_04~07）的 special_tokens 與主實驗完全相同：
    #   移除模態 ≠ 移除 special token（避免 vocab 大小不一致）
    special_tokens: List[str] = field(default_factory=lambda: [
        "[VIDEO]", "[MUSIC]", "[LTP]", "[TEXT_CLIP]", "[RANK]"
    ])

    # ── Prompt 模板 ───────────────────────────────────────────────────────────
    system_prompt: str = (
        "You are an expert music recommendation assistant for short videos. "
        "You analyze video content, user's long-term music preferences, and "
        "a candidate track to determine if it is a suitable recommendation."
    )
