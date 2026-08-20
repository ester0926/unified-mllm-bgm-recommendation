"""
用途：定義多模態推薦模型，整合影片、音訊、文字與 LTP 偏好表示。
輸入：訓練或推論程式傳入的多模態特徵與 LTP 表示。
輸出：推薦分數、投影後特徵或生成模型需要的表示。
"""

import logging
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from transformers import LlamaForCausalLM
from peft import LoraConfig, get_peft_model, TaskType, PeftModel

from .projectors import MultimodalProjectors

logger = logging.getLogger(__name__)


class UnifiedMLLM(nn.Module):
    """
    Pointwise Unified MLLM（Plan B）。

    每次 forward 接受一首候選音樂，
    ranking score 由 [RANK] token 的最終隱藏狀態輸出。
    """

    def __init__(self, model_config, tokenizer):
        super().__init__()
        self.config = model_config
        self.tokenizer = tokenizer
        self.num_candidates = 1
        # 消融實驗：prefix 長度由 active_modalities 動態決定
        # 完整實驗 = 4；w/o 任一模態 = 3
        self.active_modalities = getattr(
            model_config, "active_modalities",
            ["video", "ltp", "text", "music"]
        )
        self.multimodal_prefix_len = len(self.active_modalities)

        # ── [RANK] token id（Plan B 核心）─────────────────────────────────────
        rank_token_str = getattr(model_config, "rank_special_token", "[RANK]")
        self.rank_token_id = tokenizer.convert_tokens_to_ids(rank_token_str)
        if self.rank_token_id == tokenizer.unk_token_id:
            raise ValueError(
                f"[RANK] token ('{rank_token_str}') not found in tokenizer vocab. "
                f"Please call tokenizer.add_special_tokens({{'additional_special_tokens': ['{rank_token_str}']}}) "
                f"before constructing UnifiedMLLM."
            )
        logger.info(f"[RANK] token id = {self.rank_token_id}")

        # ── 1. 多模態投影層 ────────────────────────────────────────────────────
        self.projectors = MultimodalProjectors(
            video_dim=model_config.video_dim,
            music_dim=model_config.music_dim,
            text_dim=model_config.text_dim,
            ltp_dim=model_config.ltp_dim,
            llama_hidden_dim=model_config.llama_hidden_dim,
            projector_hidden_dim=model_config.projector_hidden_dim,
            dropout=model_config.projector_dropout,
        )

        # ── 2. 載入 LLaMA-2-7B + LoRA ────────────────────────────────────────
        logger.info(f"載入 LLaMA-2: {model_config.llama_model_name}")
        self.llama = LlamaForCausalLM.from_pretrained(
            model_config.llama_model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )

        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=model_config.lora_rank,
            lora_alpha=model_config.lora_alpha,
            lora_dropout=model_config.lora_dropout,
            target_modules=model_config.lora_target_modules,
            bias=model_config.lora_bias,
            modules_to_save=None,
        )
        self.llama = get_peft_model(self.llama, lora_config)
        self.llama.enable_input_require_grads()
        self.llama.gradient_checkpointing_enable()
        logger.info("Gradient Checkpointing 已啟用")

        for name, param in self.llama.named_parameters():
            if "lora_" not in name:
                param.requires_grad = False

        # ── 3. Ranking Head（Plan B：輸入為 [RANK] token 隱藏狀態）────────────
        # [RANK] token 可 attend 所有 prefix 模態 + tokenized t3 prompt
        # 輸入：[RANK] token 最終隱藏狀態 (B, 4096)
        # 輸出：relevance score (B,)
        self.ranking_head = nn.Sequential(
            nn.LayerNorm(model_config.llama_hidden_dim),
            nn.Linear(model_config.llama_hidden_dim, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1),
        )
        self._init_ranking_head()
        self._print_trainable_parameters()

    def _init_ranking_head(self):
        for module in self.ranking_head.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                nn.init.zeros_(module.bias)

    def _print_trainable_parameters(self):
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total     = sum(p.numel() for p in self.parameters())
        logger.info(f"可訓練參數: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    # ──────────────────────────────────────────────────────────────────────────
    # 前向傳播
    # ──────────────────────────────────────────────────────────────────────────

    def forward(
        self,
        video_feat: torch.Tensor,           # (B, video_dim)
        music_candidates: torch.Tensor,     # (B, 1, music_dim)
        ltp_feat: torch.Tensor,             # (B, ltp_dim)
        text_feat: torch.Tensor,            # (B, text_dim)
        input_ids: torch.Tensor,            # (B, seq_len)，含 [RANK] token
        attention_mask: torch.Tensor,       # (B, seq_len)
        labels: Optional[torch.Tensor] = None,
        compute_gen_loss: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Plan B forward：ranking score 來自 [RANK] token 隱藏狀態。

        Returns:
            {
              "ranking_score" : (B,)         — [RANK] token 的 relevance score
              "loss_gen"      : scalar | None — generation loss（labels 存在時）
            }
        """
        B = video_feat.size(0)
        device = video_feat.device

        # ── Step 1：投影多模態特徵 → prefix (B, n_active, 4096) ─────────────
        # 消融實驗：只注入 active_modalities 中的模態
        # 完整實驗 prefix 長度 = 4；w/o 任一模態 = 3
        multimodal_prefix = self.projectors(
            video_feat=video_feat,
            music_candidates=music_candidates,
            ltp_feat=ltp_feat,
            text_feat=text_feat,
            active_modalities=self.active_modalities,
        )
        actual_prefix_len = multimodal_prefix.shape[1]   # 動態長度（3 或 4）

        # ── Step 2：文字 token 嵌入 ──────────────────────────────────────────
        token_embeddings = self.llama.get_base_model().model.embed_tokens(input_ids)
        # (B, seq_len, 4096)

        # ── Step 3：拼接 prefix + text embeddings ────────────────────────────
        inputs_embeds = torch.cat([multimodal_prefix, token_embeddings], dim=1)
        # (B, 4 + seq_len, 4096)

        # ── Step 4：Attention Mask ────────────────────────────────────────────
        # ★ 用實際 prefix 長度（消融時為 3，全模態為 4）
        prefix_attention = torch.ones(B, actual_prefix_len,
                                      dtype=torch.long, device=device)
        full_attention_mask = torch.cat([prefix_attention, attention_mask], dim=1)

        # ── Step 5：Labels（generation supervision）──────────────────────────
        full_labels = None
        if labels is not None and compute_gen_loss:
            prefix_labels = torch.full(
                (B, actual_prefix_len), -100,
                dtype=torch.long, device=device
            )
            full_labels = torch.cat([prefix_labels, labels], dim=1)

        # ── Step 6：LLaMA 前向傳播 ──────────────────────────────────────────
        llama_outputs = self.llama(
            inputs_embeds=inputs_embeds,
            attention_mask=full_attention_mask,
            labels=full_labels,
            output_hidden_states=True,
            return_dict=True,
        )

        # ── Step 7：找 [RANK] token 位置，提取其隱藏狀態 ─────────────────────
        # input_ids 是文字序列（不含 prefix），[RANK] 在 prompt 末尾
        # rank_pos_text：[RANK] 在文字序列中的位置
        # rank_pos_full：在完整序列（prefix + text）中的位置
        rank_mask = (input_ids == self.rank_token_id)   # (B, seq_len)

        # 安全檢查：若某筆找不到 [RANK]，記錄警告並 fallback 到 MUSIC token 位置
        rank_found = rank_mask.any(dim=1)               # (B,)
        if not rank_found.all():
            missing = (~rank_found).sum().item()
            logger.warning(
                "[RANK] token not found in %d/%d sample(s). "
                "Check tokenizer.add_special_tokens() and build_prompt(). "
                "Falling back to music_token_offset=%d.",
                missing, B, self.config.music_token_offset,
            )

        # argmax 找第一個 [RANK] 位置（每筆應只有一個）
        rank_pos_text = rank_mask.long().argmax(dim=1)                   # (B,)
        rank_pos_full = actual_prefix_len + rank_pos_text                # (B,) ★ 動態

        # 備用處理：找不到 [RANK] 的樣本改用 MUSIC token 位置
        fallback_pos = torch.full_like(
            rank_pos_full, self.config.music_token_offset
        )
        rank_pos_full = torch.where(rank_found, rank_pos_full, fallback_pos)

        # 取 [RANK] hidden state
        last_hidden = llama_outputs.hidden_states[-1]   # (B, 4+seq_len, 4096)
        batch_idx   = torch.arange(B, device=device)
        rank_hidden = last_hidden[batch_idx, rank_pos_full, :]   # (B, 4096)

        # ── Step 8：Ranking Head → scalar relevance score ────────────────────
        ranking_score = self.ranking_head(rank_hidden).squeeze(-1)   # (B,)

        return {
            "ranking_score": ranking_score,
            "loss_gen":      llama_outputs.loss,
        }

    # ──────────────────────────────────────────────────────────────────────────
    # Generate（生成推薦解釋，只在 positive music 上呼叫）
    # ──────────────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def generate(
        self,
        video_feat: torch.Tensor,
        music_candidates: torch.Tensor,
        ltp_feat: torch.Tensor,
        text_feat: torch.Tensor,
        input_ids: torch.Tensor,        # prompt-only，末尾含 [RANK]
        attention_mask: torch.Tensor,
        max_new_tokens: int = 128,
        do_sample: bool = False,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        生成推薦解釋文字。
        input_ids 只含 prompt 部分（末尾有 [RANK]），生成從 [RANK] 後開始。
        """
        embed_fn = self.llama.get_input_embeddings()
        target_device = embed_fn.weight.device
        target_dtype  = embed_fn.weight.dtype

        video_feat       = video_feat.to(target_device)
        music_candidates = music_candidates.to(target_device)
        ltp_feat         = ltp_feat.to(target_device)
        text_feat        = text_feat.to(target_device)
        input_ids        = input_ids.to(target_device)
        attention_mask   = attention_mask.to(target_device)

        B = video_feat.size(0)

        multimodal_prefix = self.projectors(
            video_feat=video_feat,
            music_candidates=music_candidates,
            ltp_feat=ltp_feat,
            text_feat=text_feat,
            active_modalities=self.active_modalities,   # ★ 消融時只注入 active 模態
        ).to(dtype=target_dtype)

        token_embs = embed_fn(input_ids).to(dtype=target_dtype)
        inputs_embeds = torch.cat([multimodal_prefix, token_embs], dim=1)

        # ★ prefix_mask 長度也要動態
        actual_prefix_len = multimodal_prefix.shape[1]
        prefix_mask = torch.ones(B, actual_prefix_len,
                                 dtype=attention_mask.dtype, device=target_device)
        full_attention_mask = torch.cat([prefix_mask, attention_mask], dim=1)

        generated = self.llama.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=full_attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature if do_sample else None,
            top_p=top_p if do_sample else None,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        return generated, None

    # ──────────────────────────────────────────────────────────────────────────
    # 參數分組、儲存
    # ──────────────────────────────────────────────────────────────────────────

    def get_parameter_groups(self, lr_projectors: float, lr_lora: float):
        projector_params = list(self.projectors.parameters()) \
                         + list(self.ranking_head.parameters())
        lora_params = [p for n, p in self.llama.named_parameters()
                       if "lora_" in n and p.requires_grad]
        return [
            {"params": projector_params, "lr": lr_projectors, "name": "projectors"},
            {"params": lora_params,      "lr": lr_lora,       "name": "lora"},
        ]

    def save_model(self, save_dir: str):
        import os
        os.makedirs(save_dir, exist_ok=True)
        torch.save(self.projectors.state_dict(),   os.path.join(save_dir, "projectors.pt"))
        torch.save(self.ranking_head.state_dict(), os.path.join(save_dir, "ranking_head.pt"))
        self.llama.save_pretrained(save_dir)
        logger.info(f"模型已儲存至: {save_dir}")