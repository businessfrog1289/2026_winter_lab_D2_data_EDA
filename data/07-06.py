# -*- coding: utf-8 -*-
"""
D2.csv 후속 분석 2 - feature_7 스파이크, feature_9/8 severity 관계
실행: python eda_D2_followup2.py (data/D2.csv 필요)
결과: eda_outputs/ 에 15~17 이미지 저장
"""
import os, platform
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
from scipy import stats

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
test_df = df[df['is_test'] == 1]
y = test_df.groupby('MaterialID')['target'].first()

# ---------------------------------------------------------
# 15. feature_7 스파이크 분석
# ---------------------------------------------------------
agg_max = test_df.groupby('MaterialID')[feat_cols].max()
idx = test_df.groupby('MaterialID')['feature_7'].idxmax()
peak = df.loc[idx, ['MaterialID', 'duration_ms', 'feature_7']].set_index('MaterialID')
peak['target'] = y

TH = 3
spike = (agg_max['feature_7'] > TH).astype(int)
table = pd.crosstab(spike, y)
chi2, p, dof, exp = stats.chi2_contingency(table)
print('=== feature_7 스파이크(>%.0f) 발생 여부 vs target 카이제곱 검정 ===' % TH)
print(table)
print(f'chi2={chi2:.3f}, p-value={p:.6f}')
print('정상 스파이크 비율:', (spike[y == 0]).mean())
print('이상 스파이크 비율:', (spike[y == 1]).mean())

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
# (좌) 스파이크 발생 여부별 비율 막대그래프
rates = pd.DataFrame({
    'target': ['정상(0)', '이상(1)'],
    'spike_rate': [(spike[y == 0]).mean(), (spike[y == 1]).mean()]
})
axes[0].bar(rates['target'], rates['spike_rate'], color=['#4C72B0', '#C44E52'])
axes[0].set_ylabel('스파이크(feature_7 max>3) 발생 비율')
axes[0].set_title(f'정상/이상별 스파이크 발생 비율 (카이제곱 p={p:.1e})')

# (우) 스파이크 발생 시점 vs 크기 산점도
spiked = peak[peak['feature_7'] > TH]
for label, color, name in [(0, '#4C72B0', '정상'), (1, '#C44E52', '이상')]:
    sub = spiked[spiked.target == label]
    axes[1].scatter(sub['duration_ms'], sub['feature_7'], c=color, label=name, alpha=0.7)
axes[1].set_xlabel('스파이크 발생 시점 (duration_ms)')
axes[1].set_ylabel('스파이크 크기 (feature_7 max)')
axes[1].set_title('스파이크 발생 시점 vs 크기')
axes[1].legend()
save(fig, '15_feature7_spike_analysis.png')

# ---------------------------------------------------------
# 16. feature_9 / feature_8 severity 관계
# ---------------------------------------------------------
agg_mean = test_df.groupby('MaterialID')[feat_cols].mean()
sub = agg_mean.loc[y == 1, ['feature_8', 'feature_9']].copy()
sub['f9_level'] = pd.cut(sub['feature_9'], bins=[-40, -28, -22, -16, -8],
                          labels=['A(-28~-40,심각)', 'B(-22~-28)', 'C(-16~-22)', 'D(-8~-16,경미)'])

level_summary = sub.groupby('f9_level')[['feature_8', 'feature_9']].agg(['mean', 'count'])
print('\n=== feature_9 레벨 구간별 feature_8 관계 ===')
print(level_summary)

fig, ax = plt.subplots(figsize=(8, 7))
palette = {'A(-28~-40,심각)': '#8B0000', 'B(-22~-28)': '#C44E52',
           'C(-16~-22)': '#DD8452', 'D(-8~-16,경미)': '#4C72B0'}
for lvl, color in palette.items():
    s = sub[sub.f9_level == lvl]
    ax.scatter(s['feature_9'], s['feature_8'], c=color, label=f'{lvl} (n={len(s)})', alpha=0.7)
ax.set_xlabel('feature_9 (자재 단위 평균)')
ax.set_ylabel('feature_8 (자재 단위 평균)')
ax.set_title('이상 자재 내 feature_8 vs feature_9 — 연속적 severity 관계 확인')
ax.legend()
save(fig, '16_feature9_severity_scatter.png')

# 레벨별 다른 dynamic 변수(feature_14)도 같이 severity에 따라 변하는지 확인
sub2 = agg_mean.loc[y == 1, ['feature_14']].copy()
sub2['f9_level'] = sub['f9_level']
fig, ax = plt.subplots(figsize=(8, 5))
sns.boxplot(data=sub2, x='f9_level', y='feature_14', ax=ax,
            order=['A(-28~-40,심각)', 'B(-22~-28)', 'C(-16~-22)', 'D(-8~-16,경미)'])
ax.set_title('severity 레벨별 feature_14(dynamic 대표변수) 분포')
save(fig, '17_severity_vs_feature14.png')

print('\n모든 후속 분석2 완료 ->', os.path.abspath(OUT_DIR))