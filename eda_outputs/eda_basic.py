import pandas as pd
import numpy as np

pd.set_option('display.max_columns', 50)
pd.set_option('display.width', 200)

df = pd.read_csv('data/D2.csv')

print("="*80)
print("SHAPE:", df.shape)
print("="*80)
print("\nDTYPES:")
print(df.dtypes)

print("\n" + "="*80)
print("HEAD:")
print(df.head())

print("\n" + "="*80)
print("MISSING VALUES:")
print(df.isnull().sum())

print("\n" + "="*80)
print("DUPLICATE ROWS:", df.duplicated().sum())

print("\n" + "="*80)
print("DESCRIBE (all numeric):")
print(df.describe().T)

print("\n" + "="*80)
print("MaterialID unique count:", df['MaterialID'].nunique())
print("StepID unique values:", sorted(df['StepID'].unique()))
print("StepID value counts:")
print(df['StepID'].value_counts().sort_index())

print("\n" + "="*80)
print("is_test value counts:")
print(df['is_test'].value_counts())

print("\n" + "="*80)
print("target value counts (top 20):")
print(df['target'].value_counts().head(20))
print("target dtype unique count:", df['target'].nunique())

print("\n" + "="*80)
print("rows per MaterialID - describe:")
print(df.groupby('MaterialID').size().describe())

print("\n" + "="*80)
print("Is target constant within MaterialID?")
tg = df.groupby('MaterialID')['target'].nunique()
print(tg.value_counts())

print("\n" + "="*80)
print("Is is_test constant within MaterialID?")
tt = df.groupby('MaterialID')['is_test'].nunique()
print(tt.value_counts())

print("\n" + "="*80)
print("relationship between is_test and target (per material):")
mat_level = df.groupby('MaterialID').agg(is_test=('is_test','first'), target=('target','first'))
print(mat_level.groupby('is_test')['target'].describe())
print(mat_level['is_test'].value_counts())
