import pandas as pd
import numpy as np
import json

df = pd.read_csv('data/D2.csv')
feat_cols = [c for c in df.columns if c.startswith('feature_')]
labeled = df[df['is_test'] == 1].copy()
unlabeled = df[df['is_test'] == 0].copy()

out = {}

# --- overview stats ---
out['overview'] = {
    'rows': int(len(df)),
    'materials': int(df['MaterialID'].nunique()),
    'features': len(feat_cols),
    'missing': int(df.isnull().sum().sum()),
    'duplicates': int(df.duplicated().sum()),
    'labeled_materials': int(labeled['MaterialID'].nunique()),
    'unlabeled_materials': int(unlabeled['MaterialID'].nunique()),
    'rows_per_material_median': float(df.groupby('MaterialID').size().median()),
    'rows_per_material_min': int(df.groupby('MaterialID').size().min()),
    'rows_per_material_max': int(df.groupby('MaterialID').size().max()),
}

# --- target balance (labeled only, per material) ---
tgt_per_mat = labeled.groupby('MaterialID')['target'].first()
vc = tgt_per_mat.value_counts().sort_index()
out['target_balance'] = {'fail_0': int(vc.get(0,0)), 'pass_1': int(vc.get(1,0))}

# --- is_test x target cross tab (per material) ---
mat_level = df.groupby('MaterialID').agg(is_test=('is_test','first'), target=('target','first'))
out['is_test_target'] = {
    'is_test0_target0': int(((mat_level.is_test==0)&(mat_level.target==0)).sum()),
    'is_test0_target1': int(((mat_level.is_test==0)&(mat_level.target==1)).sum()),
    'is_test1_target0': int(((mat_level.is_test==1)&(mat_level.target==0)).sum()),
    'is_test1_target1': int(((mat_level.is_test==1)&(mat_level.target==1)).sum()),
}

# --- correlation of per-material aggregated features with target ---
agg = labeled.groupby('MaterialID')[feat_cols].agg(['mean'])
agg.columns = [f'{c}_mean' for c,_ in agg.columns]
agg['target'] = tgt_per_mat
corr = agg.corr()['target'].drop('target')
corr = corr.reindex(corr.abs().sort_values(ascending=False).index)
out['feature_corr'] = [{'feature': k.replace('_mean',''), 'corr': round(float(v),4)} for k,v in corr.items()]

# --- feature_8 distribution by target (histogram) ---
def hist_by_target(col, bins=40, clip_q=(0.005,0.995)):
    lo, hi = labeled[col].quantile(clip_q[0]), labeled[col].quantile(clip_q[1])
    edges = np.linspace(lo, hi, bins+1)
    result = {}
    for t in [0,1]:
        vals = labeled.loc[labeled.target==t, col]
        vals = vals.clip(lo, hi)
        counts, _ = np.histogram(vals, bins=edges)
        result[str(t)] = counts.tolist()
    return {'edges': edges.tolist(), 'counts': result}

out['hist_feature_8'] = hist_by_target('feature_8')
out['hist_feature_9'] = hist_by_target('feature_9')
out['hist_feature_10'] = hist_by_target('feature_10')

# --- sample time series: pick 3 materials target=0, 3 target=1 ---
sample_ids = []
for t in [0,1]:
    ids = tgt_per_mat[tgt_per_mat==t].index[:4].tolist()
    sample_ids.extend(ids)

series = []
for mid in sample_ids:
    sub = labeled[labeled.MaterialID==mid].sort_values('duration_ms')
    series.append({
        'material_id': int(mid),
        'target': int(tgt_per_mat[mid]),
        'duration': sub['duration_ms'].round(4).tolist(),
        'feature_8': sub['feature_8'].round(4).tolist(),
        'feature_10': sub['feature_10'].round(4).tolist(),
        'step': sub['StepID'].tolist(),
    })
out['sample_series'] = series

# --- feature-feature correlation heatmap (row-level, subsample for speed) ---
sub = labeled[feat_cols].sample(n=20000, random_state=42)
fc = sub.corr().round(2)
out['feature_heatmap'] = {
    'labels': feat_cols,
    'matrix': fc.values.tolist()
}

with open('eda_outputs/report_data.json','w') as f:
    json.dump(out, f)

print("Saved report_data.json")
print("keys:", list(out.keys()))
print("json size (KB):", len(json.dumps(out))/1024)
