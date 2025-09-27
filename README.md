# news-multichannel-recommendation-system

# news-recommendation-system-project [WIP]

- [Introduction](#introduction)
  - [Problem Statement & Approach](#problem-statement--approach)
  - [Evaluation Metric](#evaluation-metric)
- [Data](#data)
  - [Reading Modes](#reading-modes)
  - [Overview](#overview)
  - [Utility Functions](#utility-functions)
- [Model Development](#model-development)
  - [Multi-Channel Recall Dictionary](#multi-channel-recall-dictionary)
  - [Similarity Matrices](#similarity-matrices)
  - [ItemCF and UserCF Recall](#itemcf-and-usercf-recall)
  - [Embedding Recall with Faiss](#embedding-recall-with-faiss)
  - [Cold Start Problem](#cold-start-problem)
  - [Recall Fusion and Evaluation](#recall-fusion-and-evaluation)

---

flowchart TD
    A[User Click Logs] --> B[Data Preprocessing]

    B --> C1[ItemCF Recall]
    B --> C2[UserCF Recall]
    B --> C3[Embedding Recall (Faiss)]
    B --> C4[Cold Start Module]

    C1 --> D[Multi-Channel Merge]
    C2 --> D
    C3 --> D
    C4 --> D

    D --> E[Candidate Articles]
    E --> F[Ranking / Submission]


# Introduction  

## Problem Statement & Approach  
**Objective:** Recommend news articles to users based on historical browsing/click logs collected from a real product.  

**Approach:** Reframe the task as **CTR prediction**:  
**(user, article) → click probability**, then select top-K per user. To scale candidate generation, use a **multi-channel recall** strategy (simple/fast heuristics from different perspectives) followed by a ranking/fusion step. This balances **latency** and **recall**.  

## Evaluation Metric  
Each submission includes **5** recommended articles per user (ranked by predicted click probability). We score whether the ground-truth last click is within the top-5, rewarding higher ranks:  
score(user) = Σ[k=1→5] s(user, k) / k  
If the true article is at rank 1, the contribution is 1; if at rank 2, it is 1/2; otherwise 0, etc.  

---

# Data  

## Reading Modes  
1. **Debug:** Build a baseline quickly on a sampled training subset (`train_click_log_sample`) to validate code paths.
2. **Offline validation:** Use full training logs (`train_click_log`), split into **train/val** for model and hyper-parameter selection.
3. **Online:** Train on all available logs and predict on test logs (`train_click_log + test_click_log`).

## Overview
- **Scale:** ~300k users, ~3M clicks, ~360k articles with precomputed embeddings.  
- **Splits:** 200k users for training; 50k users for test-A; 50k users for test-B.
- **Files:**
  - `train_click_log.csv` — user click logs (train)
  - `testA_click_log.csv` — user click logs (test A)
  - `articles.csv` — article metadata (category, words, timestamps)
  - `articles_emb.csv` — article embedding vectors

| **Field** | **Description** |
|:--:|:--|
| user_id | user id |
| click_article_id | clicked article id |
| click_timestamp | click timestamp |
| click_environment | click environment |
| click_deviceGroup | device group |
| click_os | operating system |
| click_country | country |
| click_region | region |
| click_referrer_type | referrer type |
| article_id | article id (links to clicks) |
| category_id | article category id |
| created_at_ts | article creation timestamp |
| words_count | article word count |
| emb_1..emb_249 | article embedding dimensions |

## Utility Functions

**Get User-Article-Time**
```python
def get_user_item_time(click_df):
    click_df = click_df.sort_values('click_timestamp')

    def make_item_time_pair(df):
        return list(zip(df['click_article_id'], df['click_timestamp']))

    uit = (click_df.groupby('user_id')[['click_article_id','click_timestamp']]
           .apply(lambda x: make_item_time_pair(x))
           .reset_index().rename(columns={0:'item_time_list'}))
    return dict(zip(uit['user_id'], uit['item_time_list']))
```

**Get Article-User-Time Function**
```python
def get_item_user_time_dict(click_df):
    """
    Map each item to [(user_id, click_timestamp), ...]
    """
    def make_user_time_pair(df):
        return list(zip(df['user_id'], df['click_timestamp']))

    click_df = click_df.sort_values('click_timestamp')
    iut = (click_df.groupby('click_article_id')[['user_id','click_timestamp']]
           .apply(lambda x: make_user_time_pair(x))
           .reset_index().rename(columns={0:'user_time_list'}))
    return dict(zip(iut['click_article_id'], iut['user_time_list']))
```

**Getting Historical v.s. Last Click (per user)**
```python
def get_hist_and_last_click(all_click):
    all_click = all_click.sort_values(['user_id','click_timestamp'])
    click_last_df = all_click.groupby('user_id').tail(1)

    def hist_func(df):  # all but last if available
        return df if len(df) == 1 else df[:-1]

    click_hist_df = all_click.groupby('user_id').apply(hist_func).reset_index(drop=True)
    return click_hist_df, click_last_df
```  
**Getting Article Attribute (for rules/cold start**
```python
def get_item_info_dict(item_info_df):
    max_min_scaler = lambda x : (x-np.min(x))/(np.max(x)-np.min(x))
    item_info_df['created_at_ts'] = item_info_df[['created_at_ts']].apply(max_min_scaler)

    item_type_dict    = dict(zip(item_info_df['click_article_id'], item_info_df['category_id']))
    item_words_dict   = dict(zip(item_info_df['click_article_id'], item_info_df['words_count']))
    item_created_dict = dict(zip(item_info_df['click_article_id'], item_info_df['created_at_ts']))
    return item_type_dict, item_words_dict, item_created_dict
```
**Getting User Historical Click Information**  
We derive user-level features to guide recall/ranking and cold-start rules:
* Topic set (category_set) — captures stable interests; used to filter/boost candidates by category.
* Item history (both ids_set and ids_seq) — ids_set for fast “already seen” checks; ids_seq (timestamp-sorted) for sequence models and recency features.
Avoid sets only if you plan sequence modeling—order matters.
* Average article length (avg_words) — behavioral proxy for long/short-form preference; helps rule filters.
* Last clicked article’s creation time (last_created_ts_norm) — recency anchor for “freshness” rules.

Implementation notes/pitfalls:
* Leakage: Fit any normalization (min/max) on train only; reuse params for val/test.
* Missing data: Default safe values (e.g., empty set, np.nan→median) to keep pipelines robust.
* Time zones: Ensure click_timestamp and created_at_ts share the same timezone/units.
* Performance: Prefer vectorized groupby ops over per-row apply; avoid heavy Python lambdas when possible.
* Memory: If data are large, return one dict keyed by user_id with a small payload, instead of four parallel dicts.

```python
from typing import Dict, Any, Tuple, Iterable
import numpy as np
import pandas as pd

def get_user_hist_item_info_dict(
    all_click: pd.DataFrame,
    *,
    created_min: float = None,
    created_max: float = None,
    return_seq: bool = True
) -> Tuple[Dict[int, Any], Dict[int, Any], Dict[int, float], Dict[int, float], Dict[int, Iterable[int]]]:
    """
    Build user-level history features for recall/ranking/cold-start rules.

    Returns five dictionaries keyed by user_id:
      1) category_set: set of categories historically clicked
      2) ids_set:     set of clicked article_ids (fast membership)
      3) avg_words:   average words_count across clicked articles
      4) last_created_ts_norm: normalized creation time of the last clicked article
      5) ids_seq (optional): timestamp-ordered sequence of clicked article_ids (for sequence models)

    Normalization of created_at_ts uses provided (created_min, created_max) if given;
    otherwise computes min/max from the provided DataFrame (be careful about leakage—fit on train).
    """

    # Ensure needed columns exist
    required = {'user_id', 'click_article_id', 'category_id', 'words_count', 'created_at_ts', 'click_timestamp'}
    missing = required - set(all_click.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    # Pre-sort once for consistent "last" and sequences
    df = all_click.sort_values(['user_id', 'click_timestamp'])

    # 1) category_set
    cats = df.groupby('user_id')['category_id'].agg(lambda s: set(s.dropna())).to_dict()

    # 2) ids_set
    ids_set = df.groupby('user_id')['click_article_id'].agg(lambda s: set(s.dropna())).to_dict()

    # 5) ids_seq (ordered)
    if return_seq:
        ids_seq = df.groupby('user_id')['click_article_id'].apply(list).to_dict()
    else:
        ids_seq = {u: [] for u in df['user_id'].unique()}

    # 3) avg_words
    avg_words = df.groupby('user_id')['words_count'].mean().to_dict()

    # 4) last_created_ts (then normalize)
    last_created = (
        df.groupby('user_id')['created_at_ts']
          .agg(lambda s: s.iloc[-1])
          .astype(float)
    )

    # Fit/accept normalization bounds
    cmin = float(last_created.min()) if created_min is None else float(created_min)
    cmax = float(last_created.max()) if created_max is None else float(created_max)
    if cmax == cmin:  # guard against divide-by-zero
        last_created_norm = {u: 0.0 for u in last_created.index}
    else:
        last_created_norm = {int(u): (float(v) - cmin) / (cmax - cmin) for u, v in last_created.items()}

    # Cast keys to python ints for stable JSON/CSV serialization
    category_set = {int(u): cats.get(u, set()) for u in df['user_id'].unique()}
    ids_set      = {int(u): ids_set.get(u, set()) for u in df['user_id'].unique()}
    avg_words    = {int(u): float(avg_words.get(u, np.nan)) for u in df['user_id'].unique()}
    ids_seq      = {int(u): ids_seq.get(u, []) for u in df['user_id'].unique()}

    return category_set, ids_set, avg_words, last_created_norm, ids_seq
```  
**Getting the Top-K Most Clicked Articles**
```python
def get_item_topk_click(click_df, k):
    topk_click = click_df['click_article_id'].value_counts().index[:k]
    return topk_click
```

# Model Development   
## Multi-Channel Recall Dictionary
```python
# Multi-channel recall containers
user_multi_recall_dict = {
    'itemcf_sim_itemcf_recall': {},
    'embedding_sim_item_recall': {},
    'cold_start_recall': {}
}
```
* 'itemcf_sim_itemcf_recall': {}: Recall results based on item-based collaborative filtering. This key stores recall results obtained from item-based CF algorithms.
* 'embedding_sim_item_recall': {}: Recall results based on embedding similarity. This key stores recall results obtained from embedding similarity algorithms.
* 'cold_start_recall': {}: Recall results for cold start scenarios. This key stores recall results obtained from cold start strategies.

## Recall Evaluation Function  
After performing recall, it’s often necessary to adjust methods or parameters to achieve better recall results. Since recall quality determines the upper bound of ranking performance, a recall evaluation function is provided.

## Similarity Matrices
This section mainly generates similarity matrices using collaborative filtering and vector retrieval. The similarity matrices are divided into user2user and item2item. Below we first obtain the item2item similarity matrix based on itemCF.

**itemcf i2i_sim**  
ItemCF (i2i) with association weights. We account for click order, click time gap, creation time gap, and content length gap.

1. **Similarity accumulation process**:  
sim(i, j) = Σ_{u ∈ U_ij} 1 / log(1 + |N(u)|)  
   Where:  
   - *sim(i, j)* is the similarity between items *i* and *j*.  
   - *U<sub>ij</sub>* is the set of users who clicked both items *i* and *j*.  
   - *N(u)* is the number of items clicked by user *u*.  

2. **Similarity normalization process**:   
sim_normalized(i, j) = sim(i, j) / sqrt( c(i) × c(j) )  
   Where:  
   - *sim_normalized(i, j)* is the normalized similarity between items *i* and *j*.  
   - *c(i)* and *c(j)* are the total click counts of items *i* and *j* respectively.    

```python
def itemcf_sim(all_click_df):
    user_item_time = get_user_item_time(all_click_df)
    i2i_sim, item_cnt = defaultdict(dict), defaultdict(int)

    for _, item_time_list in user_item_time.items():
        for loc1, (i, ti) in enumerate(item_time_list):
            item_cnt[i] += 1
            for loc2, (j, tj) in enumerate(item_time_list):
                if i == j: 
                    continue
                loc_alpha  = 1.0 if loc2 > loc1 else 0.7
                loc_weight = loc_alpha * (0.9 ** (abs(loc2 - loc1) - 1))
                time_w     = np.exp(0.7 ** abs(ti - tj))
                create_w   = np.exp(0.8 ** abs(item_created_time_dict[i] - item_created_time_dict[j]))
                words_w    = np.exp(0.7 ** abs(item_words_dict[i] - item_words_dict[j]))
                i2i_sim[i][j] = i2i_sim[i].get(j, 0.0) + (loc_weight * time_w * create_w * words_w) / math.log(len(item_time_list) + 1)

    i2i_norm = defaultdict(dict)
    for i, nbrs in i2i_sim.items():
        for j, w in nbrs.items():
            i2i_norm[i][j] = w / math.sqrt(item_cnt[i] * item_cnt[j])
    return i2i_norm
```

Example user click data:  
- **User A**: clicked items 1, 2, 3  
- **User B**: clicked items 1, 2  
- **User C**: clicked items 2, 3  

Steps:  
1. **Items 1 and 2**: Both User A and User B clicked them → increase `i2i_sim[1][2]` by `1 / log(3)` (User A’s list length = 3).  
2. **Items 1 and 3**: User A clicked both; User C clicked item 3 → increase similarity by `1 / log(3)`.  
3. **Items 2 and 3**: User A and User C clicked both → increase similarity by `1 / log(3)`.  

**userCF u2u_sim with activity prior**  
When computing user-user similarity, simple association rules can also be used. For example, user activeness weight, where user click count is taken as the activity indicator.
```python
def get_user_activate_degree_dict(all_click_df):
    cnt = all_click_df.groupby('user_id')['click_article_id'].count()
    mm = MinMaxScaler()
    return dict(zip(cnt.index, mm.fit_transform(cnt.to_frame())[:, 0]))

def usercf_sim(all_click_df, activate):
    item_users = get_item_user_time_dict(all_click_df)
    u2u, u_cnt = defaultdict(dict), defaultdict(int)

    for _, user_time_list in item_users.items():
        for u, _ in user_time_list:
            u_cnt[u] += 1
        for u, _ in user_time_list:
            for v, _ in user_time_list:
                if u == v: 
                    continue
                weight = 50 * (activate.get(u, 0) + activate.get(v, 0))
                u2u[u][v] = u2u[u].get(v, 0.0) + weight / math.log(len(user_time_list) + 1)

    for u, nbrs in u2u.items():
        for v, w in nbrs.items():
            u2u[u][v] = w / math.sqrt(u_cnt[u] * u_cnt[v])
    return u2u
```
## ItemCF and UserCF Recall
**itemcf recall**   
Using collaborative filtering and embeddings, we have the article similarity matrix. Now apply itemCF recall: recommend items similar to user’s history. Association rules are used:
Consider weight of order of historical vs similar articles
Consider article creation time difference
Consider content similarity weight (embedding). Note: embedding similarity does not cover all pairs, so special handling is needed.  
**userCF Recall**  
Based on user-based collaborative filtering: recommend items clicked by similar users.
Association rules are added to weight recommended items based on relations between:
Target user’s historical clicked articles
Similar users’ historical clicked articles
Weights are computed as the sum of: similarity, creation time difference, and relative position between the items.
## Embedding Recall with Faiss
**Embedding Recall with Faiss**   
Used to retrieve similar items efficiently (u2i/i2i) at scale. Faiss supports exact/approximate search and compression (e.g., PQ).
```python
import faiss

vecs = np.asarray(item_vectors, dtype=np.float32)   # [N, d]
q    = np.asarray([query_vector], dtype=np.float32) # [1, d]

index = faiss.IndexFlatL2(vecs.shape[1])
index.add(vecs)
D, I = index.search(q, k=5)  
```  
## Cold Start Problem
For users/items with sparse history, apply simple rules on top of embedding recall to keep candidates aligned with recent interests.

### Analysis in the current scenario  
We find that only ~30,000 articles appear in the logs, but the total article library has over 300,000. If a test user’s last click is an article not seen in the logs, then we face an article cold start scenario. Furthermore, ~20% of test users have only one recorded click, which makes recommendations difficult and raises potential user cold start issues. In this section, we focus on solutions for article cold start, but also mention feasible approaches for user cold start.

### Article Cold Start  
Here the goal is not to solve article cold start for its own sake, but because users may click articles not seen in the log. We need to select from ~270,000 unseen articles to generate potential recommendations. This can be treated as a recall strategy.  

Approaches:  
- Use embedding-based recall to select articles similar to a user’s historical clicks in vector space.  
- Apply rule-based filtering to embedding recall results. For example:  
  - Keep articles with the same topic as historical clicks.  
  - Keep articles with similar word counts.  
  - Keep articles with creation times closer to the user’s last click or on the same day.  
This increases the likelihood that the recalled articles match user interests.

### User Cold Start  
Analysis shows ~20% of users in the test set have only one click. For these sparse users, we might apply additional strategies:  
- Add supplemental recall strategies tailored to sparse users.  
- After ranking, apply rule-based augmentation to include some articles.  

The main goal under cold start conditions is to recall potentially interesting articles based on limited historical behavior and article attributes.  

Logic:   
- Iterate through embedding-based recalled articles. For each article, gather user history information (topics, average word count, last click time) and current article information (topic, word count, creation time).  
- Apply rules:  
  - The article must not appear in the user’s history (novelty).  
  - The article’s topic must be in the user’s topic set (similarity).  
  - The word count must not differ from the average by more than 200 (content length similarity).  
  - The creation time must be within 90 days of the last click (recency).  
- If all conditions are met, add the article to the user’s cold start recall list.  
- Finally, sort the cold start recall list by score and select the top `recall_item_num` articles. Save the results as a pickle file.  

```python
def cold_start_items(user_recall, user_topic_set, user_mean_words, user_last_created_ts,
                     item_topic, item_words, item_created_ts, seen_item_ids, recall_item_num):
    out = {}
    for u, cand in user_recall.items():
        last_ts = datetime.fromtimestamp(user_last_created_ts[u])
        keep = []
        for item, score in cand:
            if item in seen_item_ids: 
                continue
            if item_topic[item] not in user_topic_set[u]:
                continue
            if abs(item_words[item] - user_mean_words[u]) > 200:
                continue
            if abs((datetime.fromtimestamp(item_created_ts[item]) - last_ts).days) > 90:
                continue
            keep.append((item, score))
        out[u] = sorted(keep, key=lambda x: x[1], reverse=True)[:recall_item_num]
    return out
```

## Recall Fusion and Evaluation 
Multi-channel recall merge combines all recall results into one unified candidate list:  
* Recall based on item similarity from itemCF
* Recall based on embedding similarity
* Recall based on cold start strategies
```python
def combine_recall_results(user_multi_recall_dict, weight=None, topk=25):
    weight = weight or {k: 1.0 for k in user_multi_recall_dict}
    final = defaultdict(dict)

    def _norm(items):
        if len(items) < 2: 
            return items
        mx, mn = items[0][1], items[-1][1]
        return [(it, 1.0 if mx == mn else (sc - mn) / (mx - mn)) for it, sc in items]

    for method, per_user in user_multi_recall_dict.items():
        w = weight.get(method, 1.0)
        for u, sorted_items in per_user.items():
            for it, sc in _norm(sorted_items):
                final[u][it] = final[u].get(it, 0.0) + w * sc

    return {u: sorted(d.items(), key=lambda x: x[1], reverse=True)[:topk] for u, d in final.items()}

def metrics_recall(user_recall, last_click_df, topk=50):
    last = dict(zip(last_click_df['user_id'], last_click_df['click_article_id']))
    users = len(user_recall)
    for k in range(10, topk + 1, 10):
        hits = sum(last[u] in {it for it, _ in items[:k]} for u, items in user_recall.items())
        print(f"topk: {k} | hit_num: {hits} | hit_rate: {hits/users:.5f} | user_num: {users}")
```
