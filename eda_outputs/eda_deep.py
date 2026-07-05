import pandas as pd
import numpy as np

pd.set_option('display.max_columns', 50)
pd.set_option('display.width', 200)

df = pd.read_csv('data/D2.csv')
feat_cols = [c for c in df.columns if c.startswith('feature_')]

# Only is_test==1 has real labels (is_test==0 always target==0 -> placeholder)
labeled = df[df['is_test'] == 1].copy()
print("Labeled (is_test==1) materials:", labeled['MaterialID'].nunique())
print("Unlabeled (is_test==0) materials:", df[df['is_test']==0]['MaterialID'].nunique())

# aggregate per material: mean, std, min, max of each feature + duration
agg = labeled.groupby('MaterialID')[feat_cols + ['duration_ms']].agg(['mean','std','min','max'])
agg.columns = ['_'.join(c) for c in agg.columns]
target_per_mat = labeled.groupby('MaterialID')['target'].first()
agg['target'] = target_per_mat

# correlation of aggregated features with target
corr = agg.corr()['target'].drop('target').sort_values(key=lambda s: -s.abs())
print("\nTop 20 |correlation| of per-material aggregated features with target:")
print(corr.head(20))

# rows per step distribution among labeled
print("\nStep row counts (labeled only):")
print(labeled.groupby(['MaterialID'])['StepID'].apply(lambda s: s.value_counts().to_dict()).head())

# target balance
print("\nTarget balance among labeled materials:")
print(target_per_mat.value_counts(normalize=True))

# save aggregated for plotting
agg.to_csv('eda_outputs/agg_per_material.csv')

# raw feature stats split by target (row level, labeled only) for a few features
print("\nRow-level feature means split by target (labeled only):")
print(labeled.groupby('target')[feat_cols].mean().T)
