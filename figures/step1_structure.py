import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'Malgun Gothic'  # 한글 폰트 (Windows)
plt.rcParams['axes.unicode_minus'] = False

df = pd.read_csv('data/D2.csv')

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('D2.csv — 데이터 구조 개요 (Step 1)', fontsize=15, fontweight='bold')

# (1) 컬럼별 결측치 개수
ax = axes[0, 0]
missing = df.isnull().sum()
ax.barh(missing.index, missing.values, color='#2a78d6')
ax.set_title('컬럼별 결측치 개수')
ax.set_xlabel('결측치 수')
ax.set_xlim(0, max(1, missing.max() * 1.2 if missing.max() > 0 else 1))
for i, v in enumerate(missing.values):
    ax.text(v + 0.02, i, str(v), va='center', fontsize=7)
ax.tick_params(axis='y', labelsize=7)

# (2) MaterialID당 행 수 분포
ax = axes[0, 1]
rows_per_mat = df.groupby('MaterialID').size()
ax.hist(rows_per_mat, bins=range(rows_per_mat.min(), rows_per_mat.max()+2), color='#eb6834', edgecolor='white')
ax.set_title(f'MaterialID당 행(샘플) 수 분포\n(min={rows_per_mat.min()}, median={int(rows_per_mat.median())}, max={rows_per_mat.max()}, n={rows_per_mat.shape[0]}개 재료)')
ax.set_xlabel('재료 1개당 행 수')
ax.set_ylabel('재료(MaterialID) 개수')

# (3) dtype 구성
ax = axes[1, 0]
dtype_counts = df.dtypes.value_counts()
ax.bar(dtype_counts.index.astype(str), dtype_counts.values, color=['#2a78d6', '#eb6834'])
ax.set_title('컬럼 dtype 구성 (전체 25개 컬럼)')
ax.set_ylabel('컬럼 수')
for i, v in enumerate(dtype_counts.values):
    ax.text(i, v + 0.3, str(v), ha='center', fontweight='bold')

# (4) StepID별 행 수 + 중복행 요약 텍스트
ax = axes[1, 1]
step_counts = df['StepID'].value_counts().sort_index()
ax.bar(['Step ' + str(i) for i in step_counts.index], step_counts.values, color=['#2a78d6', '#eb6834'])
ax.set_title('StepID별 전체 행 수')
ax.set_ylabel('행 수')
for i, v in enumerate(step_counts.values):
    ax.text(i, v + 500, f'{v:,}', ha='center', fontweight='bold')
dup_count = df.duplicated().sum()
ax.text(0.5, -0.28, f'전체 중복 행: {dup_count}건 / 전체 결측치: {int(missing.sum())}건',
        transform=ax.transAxes, ha='center', fontsize=10, color='#444')

plt.tight_layout(rect=[0, 0.02, 1, 0.96])
plt.savefig('figures/step1_structure_overview.png', dpi=150)
print("saved figures/step1_structure_overview.png")

# 콘솔 요약도 출력
print("\n=== 요약 ===")
print("shape:", df.shape)
print("MaterialID 범위:", df['MaterialID'].min(), "~", df['MaterialID'].max(), "| 개수:", df['MaterialID'].nunique())
print("StepID 값:", sorted(df['StepID'].unique()))
print("rows_per_material describe:\n", rows_per_mat.describe())
print("중복행:", dup_count)
print("결측치 총합:", missing.sum())
