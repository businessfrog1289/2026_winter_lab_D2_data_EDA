# -*- coding: utf-8 -*-
"""
D2.csv EDA 스크립트
- 구조: MaterialID(자재) x StepID(공정 2단계) 시계열, duration_ms(0~1 정규화 시간축)
- feature_1~20: 공정 센서값(표준화됨), is_test/target: train은 target=0(정상)만, test는 0/1 혼재
- 실행: python eda_D2.py  (같은 폴더에 D2.csv 있어야 함)
- 결과: ./eda_outputs/ 폴더에 png 저장
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------
# 0. 설정
# ---------------------------------------------------------
import platform
import matplotlib.font_manager as fm

def set_korean_font():
    system = platform.system()
    candidates = []
    if system == 'Windows':
        candidates = ['Malgun Gothic', 'NanumGothic']
    elif system == 'Darwin':
        candidates = ['AppleGothic', 'NanumGothic']
    else:
        candidates = ['NanumGothic', 'Noto Sans CJK KR', 'Noto Sans KR']

    available = {f.name for f in fm.fontManager.ttflist}
    for name in candidates:
        if name in available:
            plt.rcParams['font.family'] = 'sans-serif'
            plt.rcParams['font.sans-serif'] = [name] + plt.rcParams.get('font.sans-serif', [])
            print(f'[font] 사용 폰트: {name}')
            return
    print('[font] 한글 폰트를 찾지 못했습니다. 그림의 한글이 깨질 수 있습니다. '
          '(Windows는 보통 Malgun Gothic이 기본 설치되어 있어야 합니다)')

# 순서 중요: sns.set_style이 font.family를 sans-serif로 되돌리므로 스타일 적용 "이후"에 폰트를 지정해야 함
sns.set_style('whitegrid')
set_korean_font()
plt.rcParams['axes.unicode_minus'] = False

OUT_DIR = 'eda_outputs'
os.makedirs(OUT_DIR, exist_ok=True)

def save(fig, name):
    fig.savefig(os.path.join(OUT_DIR, name), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'[saved] {name}')

df = pd.read_csv('data/D2.csv')
feat_cols = [c for c in df.columns if c.startswith('feature_')]

# ---------------------------------------------------------
# 1. 기본 구조 요약 (텍스트)
# ---------------------------------------------------------
print('=== 기본 정보 ===')
print('shape:', df.shape)
print('MaterialID 개수:', df['MaterialID'].nunique())
print('결측치 합계:', df.isna().sum().sum())
print(pd.crosstab(df['is_test'], df['target'], margins=True))

# 자재 단위 라벨 테이블 (분석 편의를 위해 미리 생성)
mat_label = df.groupby('MaterialID').agg(
    is_test=('is_test', 'first'),
    target=('target', 'first'),
    n_rows=('MaterialID', 'size')
).reset_index()

# ---------------------------------------------------------
# 2. train/test/target 분포
# ---------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

ct = pd.crosstab(mat_label['is_test'], mat_label['target'])
ct.plot(kind='bar', stacked=True, ax=axes[0], color=['#4C72B0', '#DD8452'])
axes[0].set_title('자재 단위 is_test x target 분포')
axes[0].set_xlabel('is_test (0=train, 1=test)')
axes[0].set_xticklabels(['train(0)', 'test(1)'], rotation=0)
axes[0].legend(title='target')

sns.countplot(data=mat_label[mat_label.is_test == 1], x='target', ax=axes[1], palette='Set2')
axes[1].set_title('test 자재 내 target(정상/이상) 비율')
save(fig, '01_train_test_target_dist.png')

# ---------------------------------------------------------
# 3. 자재별 시퀀스 길이(공정 시간 길이) 분포
# ---------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
sns.histplot(mat_label['n_rows'], bins=20, ax=axes[0], kde=True)
axes[0].set_title('자재별 전체 행 수(시퀀스 길이) 분포')

step_len = df.groupby(['MaterialID', 'StepID']).size().reset_index(name='n')
sns.boxplot(data=step_len, x='StepID', y='n', ax=axes[1])
axes[1].set_title('StepID별 시퀀스 길이 분포')
save(fig, '02_sequence_length_dist.png')

# ---------------------------------------------------------
# 4. feature 전체 분포 (정상 vs 이상, test 데이터만 - 라벨 다양성 있음)
# ---------------------------------------------------------
test_df = df[df['is_test'] == 1].copy()

n_cols = 4
n_rows = int(np.ceil(len(feat_cols) / n_cols))
fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 4 * n_rows))
axes = axes.flatten()
for i, col in enumerate(feat_cols):
    sns.kdeplot(data=test_df, x=col, hue='target', ax=axes[i],
                common_norm=False, fill=True, alpha=0.3, palette=['#4C72B0', '#C44E52'])
    axes[i].set_title(col)
for j in range(len(feat_cols), len(axes)):
    axes[j].axis('off')
fig.suptitle('feature별 분포 (정상=0 vs 이상=1, test set, row-level)', y=1.01, fontsize=16)
save(fig, '03_feature_distributions_by_target.png')

# ---------------------------------------------------------
# 5. 상관관계 히트맵 (row-level raw feature 간)
# ---------------------------------------------------------
fig, ax = plt.subplots(figsize=(12, 10))
corr = df[feat_cols + ['duration_ms']].corr()
sns.heatmap(corr, cmap='coolwarm', center=0, ax=ax, annot=False, square=True)
ax.set_title('feature 간 상관관계 (row-level)')
save(fig, '04_feature_correlation_heatmap.png')

# ---------------------------------------------------------
# 6. 자재 단위 집계(mean/std) 후 target과의 상관관계
# ---------------------------------------------------------
agg = test_df.groupby('MaterialID')[feat_cols].agg(['mean', 'std'])
agg.columns = ['_'.join(c) for c in agg.columns]
agg['target'] = test_df.groupby('MaterialID')['target'].first()

target_corr = agg.corr()['target'].drop('target').sort_values(key=np.abs, ascending=False)

fig, ax = plt.subplots(figsize=(8, 10))
top_corr = target_corr.head(20)
colors = ['#C44E52' if v > 0 else '#4C72B0' for v in top_corr.values]
ax.barh(top_corr.index[::-1], top_corr.values[::-1], color=colors[::-1])
ax.set_title('자재 단위 feature 통계량과 target의 상관계수 (상위 20개)')
ax.axvline(0, color='k', linewidth=0.8)
save(fig, '05_target_correlation_ranking.png')

print('\n=== target과 상관관계 높은 상위 10개 ===')
print(target_corr.head(10))

# ---------------------------------------------------------
# 7. 핵심 feature (상관 top 3) 박스플롯: target별 비교
# ---------------------------------------------------------
top3_base = sorted(set([c.rsplit('_', 1)[0] for c in target_corr.head(6).index]), key=lambda x: x)[:3]
fig, axes = plt.subplots(1, len(top3_base), figsize=(6 * len(top3_base), 5))
if len(top3_base) == 1:
    axes = [axes]
for ax, base_col in zip(axes, top3_base):
    plot_df = test_df[[base_col, 'target']]
    sns.boxplot(data=plot_df, x='target', y=base_col, ax=ax, palette=['#4C72B0', '#C44E52'])
    ax.set_title(f'{base_col} vs target')
save(fig, '06_top_features_boxplot.png')

# ---------------------------------------------------------
# 8. 개별 자재 시계열 트레이스 (정상 vs 이상, 핵심 feature)
# ---------------------------------------------------------
normal_ids = mat_label[(mat_label.is_test == 1) & (mat_label.target == 0)]['MaterialID'].sample(3, random_state=42).tolist()
abnormal_ids = mat_label[(mat_label.is_test == 1) & (mat_label.target == 1)]['MaterialID'].sample(3, random_state=42).tolist()

trace_feats = top3_base if len(top3_base) > 0 else feat_cols[:3]

fig, axes = plt.subplots(len(trace_feats), 1, figsize=(12, 4 * len(trace_feats)), sharex=True)
if len(trace_feats) == 1:
    axes = [axes]
for ax, feat in zip(axes, trace_feats):
    for mid in normal_ids:
        sub = df[df.MaterialID == mid].sort_values('duration_ms')
        ax.plot(sub['duration_ms'], sub[feat], color='#4C72B0', alpha=0.6, label='정상' if mid == normal_ids[0] else None)
    for mid in abnormal_ids:
        sub = df[df.MaterialID == mid].sort_values('duration_ms')
        ax.plot(sub['duration_ms'], sub[feat], color='#C44E52', alpha=0.6, label='이상' if mid == abnormal_ids[0] else None)
    ax.axvline(0.6, color='gray', linestyle='--', linewidth=1, label='Step 경계(대략)')
    ax.set_title(f'{feat} 시계열 트레이스 (정상 3개 vs 이상 3개 자재)')
    ax.set_ylabel(feat)
    ax.legend()
axes[-1].set_xlabel('duration_ms (정규화 시간)')
save(fig, '07_timeseries_traces_normal_vs_abnormal.png')

# ---------------------------------------------------------
# 9. PCA 2D 임베딩 (자재 단위 집계 feature, target 색상)
# ---------------------------------------------------------
X = agg.drop(columns='target').fillna(0)
y = agg['target']
X_scaled = StandardScaler().fit_transform(X)

pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)

fig, ax = plt.subplots(figsize=(8, 7))
for label, color, name in [(0, '#4C72B0', '정상'), (1, '#C44E52', '이상')]:
    mask = y == label
    ax.scatter(X_pca[mask, 0], X_pca[mask, 1], c=color, label=name, alpha=0.6, s=25)
ax.set_title(f'PCA 2D 임베딩 (자재 단위 feature 통계량, 설명분산 {pca.explained_variance_ratio_.sum():.1%})')
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.legend()
save(fig, '08_pca_embedding.png')

# ---------------------------------------------------------
# 10. t-SNE 2D 임베딩 (참고용, 시간 오래 걸릴 수 있음)
# ---------------------------------------------------------
tsne = TSNE(n_components=2, random_state=42, perplexity=30, init='pca')
X_tsne = tsne.fit_transform(X_scaled)

fig, ax = plt.subplots(figsize=(8, 7))
for label, color, name in [(0, '#4C72B0', '정상'), (1, '#C44E52', '이상')]:
    mask = y == label
    ax.scatter(X_tsne[mask, 0], X_tsne[mask, 1], c=color, label=name, alpha=0.6, s=25)
ax.set_title('t-SNE 2D 임베딩 (자재 단위 feature 통계량)')
ax.legend()
save(fig, '09_tsne_embedding.png')

# ---------------------------------------------------------
# 11. StepID별 duration_ms 분포 & 공정 경계 확인
# ---------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 5))
sns.boxplot(data=df, x='StepID', y='duration_ms', ax=ax)
ax.set_title('StepID별 duration_ms 분포 (공정 단계 경계 확인)')
save(fig, '10_step_duration_boundary.png')

print('\n모든 그림 저장 완료 ->', os.path.abspath(OUT_DIR))
