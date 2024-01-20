import numpy as np
import pandas as pd


file_path = r"YOURFILEPATH\case2_data.xlsx"
years = np.arange(3)
worker_types = np.arange(3)
education_types = np.arange(2)
degration_types = np.arange(2)

df_initial_workers = pd.read_excel(file_path, sheet_name="initial_workers", header=0, index_col=0)
df_workforce_requirement = pd.read_excel(file_path, sheet_name="workforce_requirement", header=0, index_col=0)
df_resignation_rate = pd.read_excel(file_path, sheet_name="resignation_rate", header=0, index_col=0)
df_hiring_limit = pd.read_excel(file_path, sheet_name="hiring_limit", header=0, index_col=0)
df_education_cost = pd.read_excel(file_path, sheet_name="education_cost", header=0, index_col=0)
df_hiring_cost = pd.read_excel(file_path, sheet_name="hiring_cost", header=0, index_col=0)
df_idleness_cost = pd.read_excel(file_path, sheet_name="idleness_cost", header=0, index_col=0)
df_outsource_cost = pd.read_excel(file_path, sheet_name="outsource_cost", header=0, index_col=0)
df_parttime_cost = pd.read_excel(file_path, sheet_name="parttime_cost", header=0, index_col=0)
df_education_limit = pd.read_excel(file_path, sheet_name="education_limit", header=0, index_col=0)
df_degraded_resignation_rate = pd.read_excel(file_path, sheet_name="degraded_resignation_rate", header=0, index_col=0)

initial_workers = {}
workforce_requirements = {}
resignation_rate = {}
hiring_limit = {}
hiring_cost = {}
education_cost = {}
idleness_cost = {}
outsource_cost = {}
parttime_cost = {}
education_limit = {}
degraded_resignation_rate = {}

for w in worker_types:
    initial_workers=df_initial_workers.to_numpy()[0]
    resignation_rate = df_resignation_rate.to_numpy()[0]
    hiring_limit = df_hiring_limit.to_numpy()[0]
    hiring_cost = df_hiring_cost.to_numpy()[0]
    idleness_cost = df_idleness_cost.to_numpy()[0]
    outsource_cost = df_outsource_cost.to_numpy()[0]
    parttime_cost = df_parttime_cost.to_numpy()[0]
    for t in years:
        workforce_requirements[(w,t)] = df_workforce_requirement.iloc[w,t]
for e in education_types:
    education_limit = df_education_limit.to_numpy()[0]
    education_cost = df_education_cost.to_numpy()[0]
for d in degration_types:
    degraded_resignation_rate = df_degraded_resignation_rate.to_numpy()[0]


       
