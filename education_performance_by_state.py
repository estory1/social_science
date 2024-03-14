#%%[markdown]
## US per-state education performance comparison
#
# Date created: 20240313

#%%
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import io
import re

pd.options.display.max_rows = 200
pd.options.display.max_columns = 50

#%%[markdown]
# src: [Which State Has The Best Test Scores? Analyzing Standardized Testing Trends – Forbes Advisor](https://www.forbes.com/advisor/education/student-resources/which-states-have-the-highest-standardized-test-scores/)
#%%
data1 = """Rank	State	Grade 4 Math % at/above Proficient	Grade 4 Reading % at/above Proficient	Grade 8 Math % at/above Proficient	Grade 8 Reading % at/above Proficient	Average SAT Score	Average ACT Score	Average MCAT Score
1	Massachusetts	42.90%	42.61%	35.06%	39.80%	1,112	26	515
5	Connecticut	37.01%	34.62%	29.95%	34.77%	1,007	26	514
40	District of Columbia	24.27%	26.49%	16.45%	22.16%	969	26	513
23	California	30.08%	30.96%	22.97%	29.88%	1,083	26	514
22	New York	28.42%	29.65%	28.39%	32.30%	1,039	25	513
4	New Hampshire	39.96%	37.02%	28.96%	32.82%	1,035	25	516
25	Maine	31.94%	29.19%	24.37%	29.29%	1,080	25	514
42	Delaware	25.66%	25.29%	18.29%	23.84%	958	25	513
7	Virginia	37.72%	31.82%	31.21%	30.98%	1,113	25	513
8	Colorado	36.40%	37.59%	27.82%	34.16%	996	25	514
15	Washington	34.97%	33.72%	27.81%	31.78%	1,081	25	513
19	Illinois	37.52%	33.30%	26.52%	32.37%	970	25	512
21	Rhode Island	33.65%	33.81%	23.65%	31.32%	958	25	515
26	Maryland	31.09%	30.64%	24.67%	32.78%	1,008	25	513
3	New Jersey	39.42%	38.02%	33.14%	41.58%	1,066	24	514
20	Michigan	32.26%	28.29%	25.43%	28.15%	967	24	511
14	Pennsylvania	40.24%	33.98%	27.41%	30.57%	1,078	24	513
10	Vermont	34.43%	33.64%	26.92%	34.46%	1,099	24	514
27	Idaho	36.33%	31.91%	32.43%	32.12%	970	23	511
18	Indiana	39.57%	32.85%	30.11%	30.62%	971	23	513
34	Georgia	33.95%	31.54%	23.71%	30.60%	1,054	21	509
11	South Dakota	40.28%	32.40%	32.23%	31.12%	1,208	21	510
36	Oregon	28.61%	27.99%	22.03%	27.79%	1,125	21	512
12	Minnesota	40.70%	32.17%	31.53%	29.66%	1,201	21	511
13	Iowa	40.42%	33.03%	28.08%	28.81%	1,208	21	512
49	West Virginia	22.84%	22.28%	15.09%	21.66%	923	20	508
43	Alaska	27.70%	24.42%	23.27%	25.97%	1,082	20	510
50	New Mexico	19.12%	20.97%	12.69%	18.43%	901	20	507
2	Utah	42.08%	36.83%	34.50%	35.66%	1,239	20	514
31	Missouri	34.17%	30.29%	23.91%	28.48%	1,191	20	511
28	North Dakota	40.35%	30.97%	28.20%	27.07%	1,287	20	506
6	Wisconsin	42.90%	32.60%	33.22%	32.39%	1,236	19	512
32	Kansas	34.87%	30.55%	23.22%	25.81%	1,245	19	510
39	Texas	38.13%	29.94%	23.77%	23.22%	978	19	512
9	Nebraska	43.20%	34.01%	30.96%	28.82%	1,252	19	512
17	Ohio	39.99%	34.51%	29.02%	33.13%	1,044	19	512
16	Wyoming	44.14%	38.34%	31.38%	29.74%	1,200	19	510
30	Florida	40.80%	38.99%	22.95%	29.36%	966	19	511
24	Montana	38.03%	33.74%	28.54%	29.11%	1,193	19	510
41	South Carolina	33.96%	32.45%	22.02%	26.56%	1,028	19	510
37	Kentucky	32.59%	31.07%	21.50%	29.02%	1,208	19	508
44	Arkansas	27.94%	29.66%	18.93%	25.79%	1,192	19	507
33	North Carolina	35.40%	32.32%	25.38%	25.69%	1,127	19	512
35	Tennessee	36.05%	30.17%	24.78%	28.04%	1,191	18	510
45	Louisiana	27.16%	28.29%	18.89%	26.86%	1,194	18	508
48	Alabama	27.17%	28.26%	18.69%	22.03%	1,161	18	509
29	Hawaii	37.00%	35.22%	22.21%	30.67%	1,114	18	512
51	Oklahoma	26.83%	24.02%	15.93%	21.28%	953	18	509
38	Arizona	32.31%	31.37%	23.80%	28.18%	1,183	18	511
47	Mississippi	32.07%	30.64%	17.75%	21.98%	1,184	18	507
46	Nevada	28.39%	26.92%	20.82%	28.80%	1,166	17	509
"""

#%%[markdown]
# src: [Per Pupil Spending by State 2024](https://worldpopulationreview.com/state-rankings/per-pupil-spending-by-state)
#%%
data2 = """State	Public Spending Per K-12 Student	Public Funding Per K-12 Student	Cost Of Living Index	K-12 Spending As Perc Taxpayer Income	Teachers As A Percent Of Staff Salaries	Public Education Average Post Secondary
Alabama	$10,108	$11,377	88.80	3.42%	56.9%	$21,760
Alaska	$18,392	$19,553	124.40	5.2%	46.1%	$27,266
Arizona	$8,770	$10,392	107.20	2.88%	46.5%	$25,166
Arkansas	$10,414	$11,736	90.30	3.79%	48.6%	$26,246
California	$13,642	$16,015	134.50	3.2%	45.6%	$17,946
Colorado	$11,070	$13,559	105.50	2.84%	46.3%	$25,203
Connecticut	$21,146	$22,703	113.10	4.9%	44.6%	$20,284
Delaware	$15,931	$16,522	102.60	4.12%	52.4%	$23,146
District of Columbia	$22,832	$29,121	148.70	3.4%	48.1%	$27,142
Florida	$9,983	$11,109	102.30	2.5%	52.8%	$23,834
Georgia	$11,203	$12,766	91.00	3.83%	50.4%	$43,420
Hawaii	$16,128	$17,233	118.40	3.58%	52%	$39,372
Idaho	$8,041	$9,369	106.10	2.99%	57.6%	$27,760
Illinois	$16,277	$15,746	90.80	4.29%	49.9%	$32,881
Indiana	$10,256	$12,810	91.50	3.27%	37.7%	$41,705
Iowa	$11,935	$14,129	89.70	3.72%	49%	$28,380
Kansas	$11,327	$13,937	87.70	3.58%	54.6%	$25,958
Kentucky	$11,278	$12,762	93.80	3.87%	42.9%	$27,277
Louisiana	$11,917	$13,235	92.00	3.81%	54.3%	$30,196
Maine	$15,691	$16,772	111.50	4.11%	42.2%	$30,293
Maryland	$15,582	$18,049	119.50	3.54%	51.4%	$38,453
Massachusetts	$19,193	$19,734	148.40	3.6%	56.1%	$18,813
Michigan	$12,053	$14,321	92.70	3.65%	46.4%	$22,316
Minnesota	$13,302	$16,205	94.10	3.53%	47.8%	$26,646
Mississippi	$9,255	$10,398	85.30	3.73%	47.5%	$26,423
Missouri	$11,349	$13,297	88.40	3.44%	52.5%	$27,160
Montana	$11,983	$13,739	103.70	3.3%	48.8%	$25,055
Nebraska	$12,741	$14,270	90.10	3.89%	49.3%	$24,316
Nevada	$9,124	$11,017	101.30	2.81%	85.9%	$34,646
New Hampshire	$17,456	$18,758	115.00	3.51%	46.2%	$36,914
New Jersey	$21,334	$23,613	114.10	4.73%	48.6%	$32,421
New Mexico	$10,469	$12,362	94.20	3.8%	57.8%	$38,516
New York	$24,881	$28,625	125.10	4.65%	55.3%	$33,760
North Carolina	$9,798	$10,298	96.10	3%	52.1%	$27,505
North Dakota	$14,037	$16,361	94.60	3.62%	51.1%	$21,746
Ohio	$13,437	$15,214	94.00	3.84%	31.5%	$31,465
Oklahoma	$9,200	$10,479	86.00	3.42%	49.4%	$39,061
Oregon	$12,460	$15,309	115.10	3.18%	44.1%	$33,289
Pennsylvania	$16,897	$19,369	99.00	3.9%	50%	$18,141
Rhode Island	$17,539	$18,732	110.50	4.18%	54.6%	$18,524
South Carolina	$10,991	$14,149	96.50	3.62%	64.3%	$29,045
South Dakota	$10,326	$12,269	93.80	2.96%	49.3%	$21,522
Tennessee	$9,942	$10,881	90.40	2.98%	51.8%	$29,665
Texas	$9,871	$12,324	93.00	3.46%	50.3%	$26,102
Utah	$7,951	$9,673	101.50	3.37%	50.5%	$22,187
Vermont	$21,219	$20,820	114.90	5.33%	45.9%	$31,103
Virginia	$12,638	$13,656	103.10	3.16%	50.5%	$26,561
Washington	$14,348	$17,276	115.10	3.22%	61.1%	$38,416
West Virginia	$12,266	$13,499	90.30	4.35%	51.1%	$20,519
Wisconsin	$12,694	$14,793	95.00	3.49%	57.5%	$40,038
Wyoming	$16,231	$19,152	92.80	4.23%	44.3%	$33,111
"""

#%%
# read per-state standardized scores
df1 = pd.read_csv(io.StringIO(
    re.sub( r"(\d)%","\\1", data1.replace("$","").replace(",","") )
    ), sep='\t', index_col="State")
print(df1.shape)
df1
#%%
df1.describe()
# %%
# read per-state expense per-student
df2 = pd.read_csv(io.StringIO(
    re.sub( r"(\d)%","\\1", data2.replace("$","").replace(",","") )
    ), sep='\t', index_col="State")
print(df2.shape)
df2
#%%
df2.describe()

#%%
# src: [State Tax Collections Per Capita: 1980-2021](https://www.census.gov/library/visualizations/interactive/state-tax-collections-per-capita-1980-2021.html)
# read per-state total tax collections per-capita
import locale
locale.setlocale(locale.LC_ALL,"")
df3 = pd.read_csv(
    "20240313 - census.gov - 2021 state-tax-collections-per-capita - All 51.csv", 
    index_col="Name",
    sep="\t", 
    encoding='utf-16').dropna()
df3["Revenue per Capita"] = df3["Revenue per Capita"].replace('[\$,]', '', regex=True).astype(float)
df3.rename(columns={"Revenue per Capita": "Total Tax Revenue per Capita"}, inplace=True)
df3

# %%
dfm = df1.join(df2, how='inner').join(df3["Total Tax Revenue per Capita"], how="left")
print(dfm.shape)
dfm

#%%[markdown]
### Dataset looks like...
dfm.head()

#%%[markdown]
### IL, NH, & MA are...
#%%
dfm.loc[["Illinois","New Hampshire","Massachusetts"]]

#%%[markdown]
### Basic 5num summary of dataset.
dfm.describe()

# %%[markdown]
### Compare IL to basic descriptive stats (mean, 25/50/75th percentiles, etc.) aggregating all US states.
# IL's values for each column are divided by the national aggregate for each column, and reported as a percentage. Therefore:
# *   100: IL equivalent to the national aggregate.
# * &lt; 100: IL underperforms the national aggregate (bad! Except for COL and total tax revenue).
# * &gt; 100: IL outperforms the national aggregate (good! Except for COL and total tax revenue).
#%%
round(
    (dfm.loc["Illinois"] / dfm.describe() * 100).loc[
        ["mean","std","min","25%","50%","75%","max"]
        ] [sorted(set(dfm.columns)-set(["Rank"]))],
    1)

# %%[markdown]
### What states maximize student-outcome-for-collected-per-capita-tax-dollar, as measured by Average MCAT?
# %%
dfm_vfm_mcat = (dfm["Average MCAT Score"] / dfm["Total Tax Revenue per Capita"]).sort_values(ascending=False).dropna()
print("* Illinois' rank on MCAT value-for-money (= state avg MCAT / state per-capita tax revenue) is #{}. Top 10 states:".format(dfm_vfm_mcat.index.get_loc("Illinois")+1))
dfm_vfm_mcat.head(10)
#%%
# bottom 10 states
dfm_vfm_mcat.tail(10)

# %%[markdown]
### What states maximize student-outcome-for-collected-per-capita-tax-dollar, as measured by Average ACT?
# %%
dfm_vfm_act = (dfm["Average ACT Score"] / dfm["Total Tax Revenue per Capita"]).sort_values(ascending=False).dropna()
print("* Illinois' rank on ACT value-for-money (= state avg ACT / state per-capita tax revenue) is #{}. Top 10 states:".format(dfm_vfm_act.index.get_loc("Illinois")+1))
dfm_vfm_act.head(10)
#%%
# bottom 10 states
dfm_vfm_act.tail(10)

# %%[markdown]
### What states maximize student-outcome-for-collected-per-capita-tax-dollar, as measured by Average SAT?
# %%
dfm_vfm_sat = (dfm["Average SAT Score"] / dfm["Total Tax Revenue per Capita"]).sort_values(ascending=False).dropna()
print("* Illinois' rank on SAT value-for-money (= state avg SAT / state per-capita tax revenue) is #{}. Top 10 states:".format(dfm_vfm_sat.index.get_loc("Illinois")+1))
dfm_vfm_sat.head(10)
#%%
# bottom 10 states
dfm_vfm_sat.tail(10)


# %%[markdown]
## Okay, but different states allocate total tax revenues differently.
### Perhaps a fairer comparison is grade 8 math % proficiency to per-state K-12 spending instead.

# %%[markdown]
### What states maximize student-outcome-for-collected-per-capita-tax-dollar, as measured by Grade 8 Math and Public Spending Per K-12 Student?
# %%
dfm_vfm_g8math = (dfm["Grade 8 Math % at/above Proficient"] / dfm["Public Spending Per K-12 Student"]).sort_values(ascending=False).dropna()
print("* Illinois' rank on Grade 8 Math value-for-money (= state avg Grade 8 Math / state Public Spending Per K-12 Student) is #{}. Top 10 states:".format(dfm_vfm_g8math.index.get_loc("Illinois")+1))
dfm_vfm_g8math.head(10)
#%%
# bottom 10 states
dfm_vfm_g8math.tail(10)

# %%[markdown]
### What states maximize student-outcome-for-collected-per-capita-tax-dollar, as measured by Grade 8 Reading and Public Spending Per K-12 Student?
# %%
dfm_vfm_g8read = (dfm["Grade 8 Reading % at/above Proficient"] / dfm["Public Spending Per K-12 Student"]).sort_values(ascending=False).dropna()
print("* Illinois' rank on Grade 8 Reading value-for-money (= state avg Grade 8 Reading / state Public Spending Per K-12 Student) is #{}. Top 10 states:".format(dfm_vfm_g8read.index.get_loc("Illinois")+1))
dfm_vfm_g8read.head(10)
#%%
# bottom 10 states
dfm_vfm_g8read.tail(10)



#%%[markdown]
### Out of curiosity, what is the most-informative feature in the dataset?
#%%
from sklearn.feature_selection import SelectKBest, mutual_info_regression

# select the 3 most informative features
kbest = SelectKBest(mutual_info_regression, k=3)
dfm_selected = kbest.fit(
    dfm.dropna()[list(set(dfm.columns)-set(["Grade 8 Math % at/above Proficient"]))], 
    dfm.dropna()["Grade 8 Math % at/above Proficient"])
print(dfm_selected.get_feature_names_out())
dfm_selected
# %%
# Train a random forest on dfm to predict "Grade 8 Math % at/above Proficient", and in sorted order, print the best features found by the RF.
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# Load the dataset
df = dfm

# Split the dataset into training and testing sets
X = df.dropna()[list(set(df.columns)-set(["Grade 8 Math % at/above Proficient"]))]
y = df.dropna()['Grade 8 Math % at/above Proficient']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a random forest classifier on the training set
rf = RandomForestRegressor(n_estimators=10, random_state=42)
rf.fit(X_train, y_train)

# Print the best features found by the RF
pd.DataFrame(zip(rf.feature_names_in_, rf.feature_importances_),columns=["name","importance"]).sort_values(by="importance", ascending=False)

# %%
# Same now, except for Grade 8 Reading.
# Split the dataset into training and testing sets
X = df.dropna()[list(set(df.columns)-set(["Grade 8 Reading % at/above Proficient"]))]
y = df.dropna()['Grade 8 Reading % at/above Proficient']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a random forest classifier on the training set
rf = RandomForestRegressor(n_estimators=10, random_state=42)
rf.fit(X_train, y_train)

# Print the best features found by the RF
pd.DataFrame(zip(rf.feature_names_in_, rf.feature_importances_),columns=["name","importance"]).sort_values(by="importance", ascending=False)

# %%
