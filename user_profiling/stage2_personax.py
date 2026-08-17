"""
Stage 2: PersonaX Balanced History Selection
功能：根據目標音樂，從資料庫中選出具代表性的歷史序列 (Core/Exploratory/Negative)。
特點：
1. 支援 PersonaX 平衡採樣 (K-Means + Scoring)
2. 支援消融實驗策略切換 (Top-N, Random, Full)
3. 自動讀取 Stage 1 產出的語義 metadata

修正項目：
1. 輸出檔名改為使用 Target Music ID (避免 top1_music 覆蓋問題)
2. 根據 Sampling Strategy 自動建立子資料夾
3. 輸出結果增加 similarity 與 personax_score 欄位

1. 修正負向樣本 (Negative SBS) 選到 Target 本身的問題。
2. 修正 top_n 與 random 策略中的負樣本排除邏輯。
"""

import json
import numpy as np
import h5py
import logging
from pathlib import Path
from glob import glob
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances

# 導入全局配置
from stage3.config import PathConfig, AlgoConfig, AblationConfig

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PersonaXSelector:
    def __init__(self, embedding_matrix, music_ids, id_to_idx, metadata):
        self.embedding_matrix = embedding_matrix
        self.music_ids = music_ids
        self.id_to_idx = id_to_idx
        self.metadata = metadata
        
        self.pool_size = AlgoConfig.POOL_SIZE
        self.n_clusters = AlgoConfig.N_CLUSTERS
        self.alpha = AlgoConfig.ALPHA

    def select_history(self, target_music_id):
        if target_music_id not in self.id_to_idx:
            return None

        # 1. 準備目標向量
        target_idx = self.id_to_idx[target_music_id]
        target_emb = self.embedding_matrix[target_idx].reshape(1, -1)
        
        # 2. 計算全域相似度
        sims = cosine_similarity(target_emb, self.embedding_matrix)[0]
        
        # 標記 Target 自身，避免它被選入 Positive (設為 -999)
        # 但要注意，這個 -999 會影響 Negative 排序，後面要過濾掉
        sims[target_idx] = -999.0

        # --- 消融實驗分流 ---
        strategy = AblationConfig.SAMPLING_STRATEGY
        
        if strategy == 'full':
            # Ablation-1a: 完整歷史 (取 Top 100)
            indices = np.argsort(sims)[::-1][:100]
            # 過濾掉 Target (雖然已設 -999 排在最後，但保險起見)
            indices = [i for i in indices if i != target_idx]
            return self._wrap_result(target_music_id, indices, [], [], sims[indices], [], [])

        elif strategy == 'top_n':
            # Ablation-1b: Top-N
            # 正向排序 (由大到小)
            sorted_indices_desc = np.argsort(sims)[::-1]
            # 負向排序 (由小到大)
            sorted_indices_asc = np.argsort(sims)

            # 選 Positive (排除 -999 的 Target，雖然它在最後)
            pos_candidates = [i for i in sorted_indices_desc if i != target_idx]
            
            core_indices = pos_candidates[:AlgoConfig.CORE_SIZE]
            explor_indices = pos_candidates[AlgoConfig.CORE_SIZE : AlgoConfig.CORE_SIZE + AlgoConfig.EXPLOR_SIZE]
            
            # 選 Negative (排除 -999 的 Target，它在最前面)
            neg_indices = [i for i in sorted_indices_asc if i != target_idx][:AlgoConfig.NEG_SIZE]
            
            return self._wrap_result(target_music_id, core_indices, explor_indices, neg_indices, 
                                     sims[core_indices], sims[explor_indices], sims[neg_indices])

        elif strategy == 'random':
            # Ablation-1b variant: Random
            all_indices = np.arange(len(self.music_ids))
            all_indices = all_indices[all_indices != target_idx] # 排除自己
            
            selected = np.random.choice(all_indices, size=AlgoConfig.CORE_SIZE + AlgoConfig.EXPLOR_SIZE + AlgoConfig.NEG_SIZE, replace=False)
            
            core_indices = selected[:AlgoConfig.CORE_SIZE]
            explor_indices = selected[AlgoConfig.CORE_SIZE : AlgoConfig.CORE_SIZE + AlgoConfig.EXPLOR_SIZE]
            neg_indices = selected[AlgoConfig.CORE_SIZE + AlgoConfig.EXPLOR_SIZE:]
            
            return self._wrap_result(target_music_id, core_indices, explor_indices, neg_indices,
                                     sims[core_indices], sims[explor_indices], sims[neg_indices])

        else: # Default: 'personax'
            return self._run_personax_logic(target_music_id, target_idx, sims)

    def _run_personax_logic(self, target_music_id, target_idx, sims):
        """Ablation-1c: PersonaX 平衡採樣"""
        
        # 1. 建構候選池 (排除 Target)
        # 因為 target 設為 -999，argsort 降序排時它會在最後，所以取前 POOL_SIZE 安全
        candidate_indices = np.argsort(sims)[::-1][:self.pool_size]
        candidate_embs = self.embedding_matrix[candidate_indices]
        
        # 2. 行為聚類
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(candidate_embs)
        centers = kmeans.cluster_centers_

        core_sbs_data = [] 
        selected_embs_s = [] 

        # 3. 平衡評分
        for k in range(self.n_clusters):
            cluster_mask = (labels == k)
            member_local_indices = np.where(cluster_mask)[0]
            if len(member_local_indices) == 0: continue
                
            member_embs = candidate_embs[member_local_indices]
            center = centers[k].reshape(1, -1)
            
            # Proto
            dists_center = euclidean_distances(member_embs, center).flatten()
            max_dist_c = np.max(dists_center) + 1e-9
            proto_scores = 1.0 - (dists_center / max_dist_c)
            
            # Div
            if not selected_embs_s:
                div_scores = np.ones(len(member_local_indices))
            else:
                s_matrix = np.array(selected_embs_s)
                dists_s = euclidean_distances(member_embs, s_matrix)
                avg_dists = np.mean(dists_s, axis=1)
                max_dist_s = np.max(avg_dists) + 1e-9
                div_scores = avg_dists / max_dist_s

            scores = self.alpha * proto_scores + (1 - self.alpha) * div_scores
            
            best_local_idx = np.argmax(scores)
            best_score = scores[best_local_idx]
            best_global_idx = candidate_indices[member_local_indices[best_local_idx]]
            
            core_sbs_data.append((best_global_idx, best_score))
            selected_embs_s.append(self.embedding_matrix[best_global_idx])

        # 4. 補足數量
        existing_indices = set(x[0] for x in core_sbs_data)
        for idx in candidate_indices:
            if len(core_sbs_data) >= AlgoConfig.CORE_SIZE: break
            if idx not in existing_indices:
                core_sbs_data.append((idx, 0.0))
                existing_indices.add(idx)
        
        core_indices = [x[0] for x in core_sbs_data]
        core_px_scores = [x[1] for x in core_sbs_data]

        # 5. Exploratory SBS
        remaining_pool = [i for i in candidate_indices if i not in existing_indices]
        if remaining_pool:
            explor_indices = np.random.choice(
                remaining_pool, 
                size=min(len(remaining_pool), AlgoConfig.EXPLOR_SIZE), 
                replace=False
            )
        else:
            explor_indices = []

        # 6. Negative SBS (修正重點)
        # argsort 升序排列，Target (-999) 會在第 0 位
        # 我們要排除 Target (target_idx)
        sorted_asc_indices = np.argsort(sims)
        neg_indices = []
        for idx in sorted_asc_indices:
            if idx != target_idx:
                neg_indices.append(idx)
            if len(neg_indices) >= AlgoConfig.NEG_SIZE:
                break

        return self._wrap_result(
            target_music_id, 
            core_indices, explor_indices, neg_indices,
            sims[core_indices], sims[explor_indices], sims[neg_indices],
            personax_scores=core_px_scores
        )

    def _wrap_result(self, target_mid, core_idxs, explor_idxs, neg_idxs, 
                     core_sims, explor_sims, neg_sims, personax_scores=None):
        return {
            "target_music": target_mid,
            "balanced_history": {
                "core_sbs": self._format_meta(core_idxs, core_sims, personax_scores),
                "exploratory_sbs": self._format_meta(explor_idxs, explor_sims),
                "negative_sbs": self._format_meta(neg_idxs, neg_sims)
            }
        }

    def _format_meta(self, indices, similarities, px_scores=None):
        results = []
        for i, idx in enumerate(indices):
            mid = self.music_ids[idx]
            info = self.metadata.get(mid, {})
            
            entry = {
                "music_id": mid,
                "title": info.get("title", "unknown"),
                "artist": info.get("artist", "unknown"),
                "genre": info.get("genre", "unknown"),
                "tags": info.get("tags", [])[:5],
                "semantic_seed": info.get("semantic_seed", ""),
                "similarity": float(similarities[i])
            }
            if px_scores and i < len(px_scores):
                entry["personax_score"] = float(px_scores[i])
            results.append(entry)
        return results

# ================================
# 主流程 (與前版相同，僅保留作為參考)
# ================================
def load_data():
    logger.info(f"Loading metadata from {PathConfig.MUSIC_METADATA_ENRICHED}")
    with open(PathConfig.MUSIC_METADATA_ENRICHED, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
        
    music_embeddings = []
    music_ids = []
    id_to_idx = {}
    
    hdf5_files = glob(str(PathConfig.HDF5_DIR / "musechat_features_*.h5"))
    logger.info(f"Found {len(hdf5_files)} HDF5 files. Loading embeddings...")
    
    for path in tqdm(hdf5_files, desc="Loading HDF5"):
        try:
            with h5py.File(path, 'r') as f:
                if 'pairs' not in f: continue
                pairs = f['pairs']
                for pid in pairs.keys():
                    if len(pid) < 11: continue
                    t_id = pid[:11]
                    c_id = pid[-11:]
                    
                    if t_id not in id_to_idx and t_id in metadata:
                        emb = np.array(pairs[pid]['target_music_cls']).flatten()
                        id_to_idx[t_id] = len(music_ids)
                        music_ids.append(t_id)
                        music_embeddings.append(emb)
                    
                    if c_id not in id_to_idx and c_id in metadata:
                        emb = np.array(pairs[pid]['candidate_music_cls']).flatten()
                        id_to_idx[c_id] = len(music_ids)
                        music_ids.append(c_id)
                        music_embeddings.append(emb)
        except Exception:
            pass
            
    logger.info(f"Loaded {len(music_ids)} embeddings.")
    return np.array(music_embeddings), music_ids, id_to_idx, metadata

def process_one_sample(args):
    json_path, selector, output_dir = args
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        target_id = data.get("target_music")
        if not target_id: return None
        
        result = selector.select_history(target_id)
        if result:
            out_name = f"{target_id}__history.json"
            out_path = output_dir / out_name
            
            with open(out_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            return target_id
    except Exception as e:
        logger.error(f"Error processing {json_path}: {e}")
        return None

if __name__ == "__main__":
    emb_matrix, mids, mid_map, meta = load_data()
    selector = PersonaXSelector(emb_matrix, mids, mid_map, meta)
    
    json_files = glob(str(PathConfig.JSON_DIR / "**" / "top1_music.json"), recursive=True)
    logger.info(f"Found {len(json_files)} input samples.")
    
    strategy_name = AblationConfig.SAMPLING_STRATEGY
    out_dir = PathConfig.STAGE2_OUTPUT_DIR / strategy_name
    out_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Starting Selection Strategy: {strategy_name}")
    logger.info(f"Output Directory: {out_dir}")
    
    with ProcessPoolExecutor(max_workers=4) as executor:
        args_list = [(p, selector, out_dir) for p in json_files]
        futures = [executor.submit(process_one_sample, arg) for arg in args_list]
        
        count = 0
        for f in tqdm(as_completed(futures), total=len(futures)):
            if f.result(): count += 1
            
    logger.info(f"Stage 2 Complete. Generated {count} history files.")