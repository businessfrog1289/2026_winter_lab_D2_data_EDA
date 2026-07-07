# -*- coding: utf-8 -*-
"""
D2.csv 후속 분석 5 - PCA를 이용한 차원 축소(20개 -> 2개)와 PC1/PC2 시계열 트레이스
목적: feature 20개 + 심한 다중공선성 문제를, row-level PCA로 압축한 PC1/PC2 두 개만으로도
      정상/이상을 여전히 잘 구분해낼 수 있는지 확인한다.
실행: python eda_D2_followup5.py (data/D2.csv 필요)
결과: eda_outputs/ 에 23~25 이미지 저장
"""
import os, platform
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import accuracy_score, f1_score

OUT_DIR = 'eda_outputs'
os.makedirs(OUT_DIR, exist_ok=True)

def set_korean_font():
    system = platform.system()
    candidates = ['Malgun Gothic', 'NanumGothic'] if system == 'Windows' else \
                 ['AppleGothic', 'NanumGothic'] if system == 'Darwin' else \
                 ['NanumGothic', 'Noto Sans CJK KR', 'Noto Sans KR']
    available = {f.name for f in fm.fontManager.ttflist}
    for name in candidates:
        if name in available:
            plt.rcParams['font.family'] = 'sans-serif'
            plt.rcParams['font.sans-serif'] = [name] + plt.rcParams.get('font.sans-serif', [])
            print(f'[font] 사용 폰트: {name}')
            return
    print('[font] 한글 폰트를 찾지 못했습니다.')

sns.set_style('whitegrid')
set_korean_font()
plt.rcParams['axes.unicode_minus'] = False

def save(fig, name):
    fig.savefig(os.path.join(OUT_DIR, name), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'[saved] {name}')

df = pd.read_csv('data/D2.csv')
feat_cols = [c for c in df.columns if c.startswith('feature_')]

# ---------------------------------------------------------
# 1) row-level(시점 단위)로 20개 feature 원본 그대로 표준화 후 PCA
#    - row-level을 쓰는 이유: PC1(t), PC2(t) "시계열 트레이스"를 그리려면
#      각 시점(row)마다 주성분 점수가 있어야 하므로, 자재 단위로 미리 요약하면 안 됨
# ---------------------------------------------------------
scaler = StandardScaler()
X = scaler.fit_transform(df[feat_cols])
pca_full = PCA(n_components=20, random_state=42)
pca_full.fit(X)

# ---------------------------------------------------------
# 2) 몇 개의 주성분을 쓸지 결정 (scree plot)
# ---------------------------------------------------------
ratio = pca_full.explained_variance_ratio_
cum = np.cumsum(ratio)
fig, ax = plt.subplots(figsize=(9, 5.5))
x = np.arange(1, 21)
ax.bar(x, ratio, color='#4C72B0', label='개별 설명분산')
ax.plot(x, cum, color='#C44E52', marker='o', markersize=4, label='누적 설명분산')
ax.axhline(0.8, color='gray', linestyle='--', linewidth=1)
ax.axvline(2.5, color='black', linestyle=':', linewidth=1)
ax.set_xlabel('주성분 번호')
ax.set_ylabel('설명분산 비율')
ax.set_title('20개 feature 전체 PCA — 주성분별/누적 설명분산 (row-level)')
ax.set_xticks(x)
ax.legend()
save(fig, '23_pca_scree_20features.png')
print('PC1:', ratio[0], 'PC1+PC2:', cum[1], 'PC1~5 누적:', cum[4])

# ---------------------------------------------------------
# 3) PC1, PC2 두 개만 선택해서 재적합 + 로딩(기여도) 확인
# ---------------------------------------------------------
pca2 = PCA(n_components=2, random_state=42)
pcs = pca2.fit_transform(X)
df['PC1'] = pcs[:, 0]
df['PC2'] = pcs[:, 1]

loadings = pd.DataFrame(pca2.components_.T, index=feat_cols, columns=['PC1', 'PC2'])
print('\n=== PC1 기여도 상위 5개 ===')
print(loadings['PC1'].abs().sort_values(ascending=False).head(5))
print('\n=== PC2 기여도 상위 5개 ===')
print(loadings['PC2'].abs().sort_values(ascending=False).head(5))

fig, axes = plt.subplots(1, 2, figsize=(13, 6))
for ax, pc in zip(axes, ['PC1', 'PC2']):
    top = loadings[pc].sort_values(key=np.abs, ascending=True).tail(10)
    colors = ['#C44E52' if v > 0 else '#4C72B0' for v in top.values]
    ax.barh(top.index, top.values, color=colors)
    ax.set_title(f'{pc} 로딩 (기여도) 상위 10개 feature')
    ax.axvline(0, color='k', linewidth=0.8)
save(fig, '24_pc_loadings.png')

# ---------------------------------------------------------
# 4) PC1(t), PC2(t) 시계열 트레이스 (정상 3개 vs 이상 3개 자재)
# ---------------------------------------------------------
test_df = df[df.is_test == 1]
mat_label = test_df.groupby('MaterialID')['target'].first()
normal_ids = mat_label[mat_label == 0].sample(3, random_state=42).index.tolist()
abn_ids = mat_label[mat_label == 1].sample(3, random_state=42).index.tolist()

fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
for ax, pc in zip(axes, ['PC1', 'PC2']):
    for mid in normal_ids:
        sub = df[df.MaterialID == mid].sort_values('duration_ms')
        ax.plot(sub['duration_ms'], sub[pc], color='#4C72B0', alpha=0.7, label='정상' if mid == normal_ids[0] else None)
    for mid in abn_ids:
        sub = df[df.MaterialID == mid].sort_values('duration_ms')
        ax.plot(sub['duration_ms'], sub[pc], color='#C44E52', alpha=0.7, label='이상' if mid == abn_ids[0] else None)
    ax.axvline(0.6, color='gray', linestyle='--', linewidth=1, label='Step 경계(대략)' if pc == 'PC1' else None)
    ax.set_ylabel(pc)
    ax.set_title(f'{pc} 시계열 트레이스 (정상 3개 vs 이상 3개 자재)')
    ax.legend()
axes[-1].set_xlabel('duration_ms (정규화 시간)')
save(fig, '25_pc_timeseries_trace.png')

# ---------------------------------------------------------
# 5) 자재 단위 PC1/PC2 평균으로 target 분리력 및 분류 성능 확인
# ---------------------------------------------------------
agg = test_df.groupby('MaterialID')[['PC1', 'PC2']].mean()
y = test_df.groupby('MaterialID')['target'].first()
print('\nPC1_mean vs target 상관:', agg['PC1'].corr(y))
print('PC2_mean vs target 상관:', agg['PC2'].corr(y))
print('정상 PC2 범위:', agg['PC2'][y == 0].min(), '~', agg['PC2'][y == 0].max())
print('이상 PC2 범위:', agg['PC2'][y == 1].min(), '~', agg['PC2'][y == 1].max())

clf = LogisticRegression(max_iter=2000)
pred = cross_val_predict(clf, agg[['PC1', 'PC2']], y, cv=StratifiedKFold(5, shuffle=True, random_state=42))
acc, f1 = accuracy_score(y, pred), f1_score(y, pred)
print(f'\nPC1+PC2 로지스틱회귀 (5-fold CV): accuracy={acc:.4f}, f1={f1:.4f}')

print('\n모든 후속 분석5 완료 ->', os.path.abspath(OUT_DIR))