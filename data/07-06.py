# -*- coding: utf-8 -*-
"""
D2.csv - 다중공선성/PCA 보고서용 전체 이미지 통합 생성 스크립트
목적: "다중공선성 진단과 PCA를 통한 차원 축소" 보고서에 필요한 이미지 8개를
      한 번에 일관되게(같은 실행 환경, 같은 random_state) 생성한다.
실행: python eda_D2_pca_final.py (data/D2.csv 필요)
      pip install statsmodels --break-system-packages  (VIF 계산용, 없으면 먼저 설치)
결과: eda_outputs/ 에 아래 8개 이미지 저장
  1) vif_initial.png                  - VIF 진단 (20개 전체)
  2) pca_scree_all.png                - scree plot (20개 전체, feature_8/9 포함)
  3) pc_loadings_all.png              - PC1/PC2 로딩 (20개 전체)
  4) pc_timeseries_all.png            - PC1(t)/PC2(t) 시계열 (20개 전체)
  5) pca_scree_no89.png               - scree plot (feature_8/9 제외 18개)
  6) pc_loadings_no89.png             - PC1/PC2 로딩 (feature_8/9 제외)
  7) pc_timeseries_no89.png           - PC1(t)/PC2(t) 시계열 (feature_8/9 제외)
  8) decision_boundary_compare.png    - 로지스틱 vs RandomForest 결정 경계 비교
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import accuracy_score, f1_score
from statsmodels.stats.outliers_influence import variance_inflation_factor

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
feat_cols_no89 = [c for c in feat_cols if c not in ('feature_8', 'feature_9')]

test_df = df[df.is_test == 1]
mat_label = test_df.groupby('MaterialID')['target'].first()
normal_ids = mat_label[mat_label == 0].sample(3, random_state=42).index.tolist()
abn_ids = mat_label[mat_label == 1].sample(3, random_state=42).index.tolist()


def pca_pipeline(cols, tag, n_scree=None):
    """row-level PCA 파이프라인: scree, loadings, PC1/PC2 시계열, 분류성능까지 한 번에"""
    n_scree = n_scree or len(cols)
    X = StandardScaler().fit_transform(df[cols])

    # (a) scree plot
    pca_full = PCA(n_components=n_scree, random_state=42)
    pca_full.fit(X)
    ratio = pca_full.explained_variance_ratio_
    cum = np.cumsum(ratio)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    x = np.arange(1, n_scree + 1)
    ax.bar(x, ratio, color='#4C72B0', label='개별 설명분산')
    ax.plot(x, cum, color='#C44E52', marker='o', markersize=4, label='누적 설명분산')
    ax.axhline(0.8, color='gray', linestyle='--', linewidth=1)
    ax.axvline(2.5, color='black', linestyle=':', linewidth=1)
    ax.set_xlabel('주성분 번호'); ax.set_ylabel('설명분산 비율')
    ax.set_title(f'{tag} PCA — 주성분별/누적 설명분산 (row-level)')
    ax.set_xticks(x); ax.legend()
    save(fig, f'pca_scree_{"all" if tag=="전체 20개" else "no89"}.png')
    print(f'[{tag}] PC1={ratio[0]:.3f}, PC1+PC2={cum[1]:.3f}')

    # (b) PC1, PC2 재적합 + 로딩
    pca2 = PCA(n_components=2, random_state=42)
    pcs = pca2.fit_transform(X)
    df[f'PC1_{tag}'] = pcs[:, 0]
    df[f'PC2_{tag}'] = pcs[:, 1]
    loadings = pd.DataFrame(pca2.components_.T, index=cols, columns=['PC1', 'PC2'])

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    for ax, pc in zip(axes, ['PC1', 'PC2']):
        top = loadings[pc].sort_values(key=np.abs, ascending=True).tail(10)
        colors = ['#C44E52' if v > 0 else '#4C72B0' for v in top.values]
        ax.barh(top.index, top.values, color=colors)
        ax.set_title(f'{pc} 로딩 (기여도) 상위 10개 feature')
        ax.axvline(0, color='k', linewidth=0.8)
    fig.suptitle(f'{tag} PCA 로딩', y=1.02)
    save(fig, f'pc_loadings_{"all" if tag=="전체 20개" else "no89"}.png')

    # (c) PC1(t), PC2(t) 시계열 트레이스
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    for ax, pc in zip(axes, [f'PC1_{tag}', f'PC2_{tag}']):
        for mid in normal_ids:
            sub = df[df.MaterialID == mid].sort_values('duration_ms')
            ax.plot(sub['duration_ms'], sub[pc], color='#4C72B0', alpha=0.7, label='정상' if mid == normal_ids[0] else None)
        for mid in abn_ids:
            sub = df[df.MaterialID == mid].sort_values('duration_ms')
            ax.plot(sub['duration_ms'], sub[pc], color='#C44E52', alpha=0.7, label='이상' if mid == abn_ids[0] else None)
        ax.axvline(0.6, color='gray', linestyle='--', linewidth=1)
        ax.set_ylabel(pc)
        ax.set_title(f'{pc} 시계열 트레이스 ({tag}, 정상 3개 vs 이상 3개 자재)')
        ax.legend()
    axes[-1].set_xlabel('duration_ms (정규화 시간)')
    save(fig, f'pc_timeseries_{"all" if tag=="전체 20개" else "no89"}.png')

    # (d) 자재 단위 검증 + 분류 성능
    test_df_now = df[df.is_test == 1]
    agg = test_df_now.groupby('MaterialID')[[f'PC1_{tag}', f'PC2_{tag}']].mean()
    y = test_df_now.groupby('MaterialID')['target'].first()
    print(f'[{tag}] PC1_mean corr={agg.iloc[:,0].corr(y):.3f}, PC2_mean corr={agg.iloc[:,1].corr(y):.3f}')
    for clf, name in [(LogisticRegression(max_iter=2000), '로지스틱'),
                       (RandomForestClassifier(n_estimators=300, random_state=42), 'RandomForest')]:
        pred = cross_val_predict(clf, agg, y, cv=StratifiedKFold(5, shuffle=True, random_state=42))
        print(f'[{tag}] {name} (5-fold CV): acc={accuracy_score(y,pred):.4f} f1={f1_score(y,pred):.4f}')


# ---------------------------------------------------------
# 1) VIF 진단 (20개 전체)
# ---------------------------------------------------------
agg_all = df.groupby('MaterialID')[feat_cols].mean()
Xv = pd.DataFrame(StandardScaler().fit_transform(agg_all), columns=feat_cols)
vif = pd.DataFrame({'feature': feat_cols,
                     'VIF': [variance_inflation_factor(Xv.values, i) for i in range(Xv.shape[1])]}).sort_values('VIF')
fig, ax = plt.subplots(figsize=(9, 8))
colors = ['#C44E52' if v > 10 else ('#DD8452' if v > 5 else '#4C72B0') for v in vif['VIF']]
ax.barh(vif['feature'], vif['VIF'], color=colors)
ax.axvline(5, color='gray', linestyle='--', linewidth=1, label='VIF=5')
ax.axvline(10, color='black', linestyle='--', linewidth=1, label='VIF=10')
ax.set_xlabel('VIF (분산팽창인자)'); ax.set_title('feature별 초기 VIF (다중공선성 진단)')
ax.legend()
save(fig, 'vif_initial.png')

# ---------------------------------------------------------
# 2~4) PCA (전체 20개, feature_8/9 포함)
# ---------------------------------------------------------
pca_pipeline(feat_cols, '전체 20개')

# ---------------------------------------------------------
# 5~7) PCA (feature_8/9 제외 18개)
# ---------------------------------------------------------
pca_pipeline(feat_cols_no89, 'feature89제외')

# ---------------------------------------------------------
# 8) 결정 경계 비교 (feature_10 vs feature_14, 로지스틱 vs RandomForest)
# ---------------------------------------------------------
agg = test_df.groupby('MaterialID')[feat_cols].mean()
y = test_df.groupby('MaterialID')['target'].first().values
X = agg[['feature_10', 'feature_14']].values
xx, yy = np.meshgrid(np.linspace(X[:, 0].min() - 0.3, X[:, 0].max() + 0.3, 300),
                      np.linspace(X[:, 1].min() - 0.1, X[:, 1].max() + 0.1, 300))
grid = np.c_[xx.ravel(), yy.ravel()]

fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
for ax, clf, name in zip(axes,
                          [LogisticRegression(max_iter=2000), RandomForestClassifier(n_estimators=300, random_state=42)],
                          ['로지스틱회귀 (직선 하나)', 'RandomForest (구불구불한 경계)']):
    clf.fit(X, y)
    Z = clf.predict(grid).reshape(xx.shape)
    ax.contourf(xx, yy, Z, alpha=0.25, cmap='RdBu_r', levels=[-0.5, 0.5, 1.5])
    ax.scatter(X[y == 0, 0], X[y == 0, 1], c='#4C72B0', s=18, label='정상', edgecolor='white', linewidth=0.3)
    ax.scatter(X[y == 1, 0], X[y == 1, 1], c='#C44E52', s=18, label='이상', edgecolor='white', linewidth=0.3)
    acc = clf.score(X, y)
    ax.set_title(f'{name}\n(학습 정확도 {acc:.1%})')
    ax.set_xlabel('feature_10 (자재 단위 평균)'); ax.set_ylabel('feature_14 (자재 단위 평균)')
    ax.legend(loc='upper left', fontsize=9)
fig.suptitle('정상/이상 구분 경계: 직선(로지스틱) vs 구불구불한 경계(RandomForest)', y=1.02, fontsize=14)
save(fig, 'decision_boundary_compare.png')

print('\n모든 이미지 생성 완료 ->', os.path.abspath(OUT_DIR))