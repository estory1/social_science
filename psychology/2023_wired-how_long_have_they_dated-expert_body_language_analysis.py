#%%[markdown]
# Descriptive analysis of the predictive performances of experts in: [Watch How Long Have They Dated? Expert Body Language Analysis | WIRED](https://www.wired.com/video/watch/3-experts-spy-on-dates-and-guess-how-long-couples-have-been-together)

#%%
import pandas as pd
df = pd.DataFrame({ "bs": [ ("m",4) , ("m",1) , ("fd",0) , ("d",1) , ("d",5) , ], "fbi": [ ("m",4) , ("d",5) , ("fd",0) , ("m",1) , ("m",4) , ], "comm": [ ("m",1) , ("d",5) , ("m",1) , ("m",4) , ("fd",0) , ], "label": [ ("m",1) , ("d",5) , ("fd",0) , ("d",1) , ("m",4) ,] })
df
# [ df[c][0] for c in [set(df.columns.tolist()) - set("label")] ]
# [ df[c].values[0] for c in [set(df.columns.tolist()) - set("label")] ]
# [ df[c].values for c in [set(df.columns.tolist()) - set("label")] ]
# [ df[c][0] for c in [list(set(df.columns.tolist()) - set("label"))] ]
# [ df[c].values for c in [list(set(df.columns.tolist()) - set("label"))] ]
# [ df[c].values[0] for c in [list(set(df.columns.tolist()) - set("label"))] ]
# df["bs"][0]
# df["bs"].values
# [ r[0] for r in df["bs"].values ]
# zip( [ r[0] for r in df["bs"].values ] , [ r[0] for r in df["label"].values ] )
# list( zip( [ r[0] for r in df["bs"].values ] , [ r[0] for r in df["label"].values ] ) )
# [ x[0] == x[1] for x in zip( [ r[0] for r in df["bs"].values ] , [ r[0] for r in df["label"].values ] ) ]

#%%
bs_type_predvacc = [ x[0] == x[1] for x in zip( [ r[0] for r in df["bs"].values ] , [ r[0] for r in df["label"].values ] ) ]
fbi_type_predvacc = [ x[0] == x[1] for x in zip( [ r[0] for r in df["fbi"].values ] , [ r[0] for r in df["label"].values ] ) ]
comm_type_predvacc = [ x[0] == x[1] for x in zip( [ r[0] for r in df["comm"].values ] , [ r[0] for r in df["label"].values ] ) ]
# bs_t_predvacc = [ x[0] == x[1] for x in zip( [ r[1] for r in df["bs"].values ] , [ r[1] for r in df["label"].values ] ) ]
# fbi_t_predvacc = [ x[0] == x[1] for x in zip( [ r[1] for r in df["fbi"].values ] , [ r[1] for r in df["label"].values ] ) ]
# comm_t_predvacc = [ x[0] == x[1] for x in zip( [ r[1] for r in df["comm"].values ] , [ r[1] for r in df["label"].values ] ) ]
# # df_type_predvacc = pd.DataFrame({ "bs": bs_type_predvacc, "fbi": fbi_type_predvacc, "comm": comm_t_predvacc })
# df_t_predvacc = pd.DataFrame({ "bs": bs_t_predvacc, "fbi": fbi_t_predvacc, "comm": comm_t_predvacc })
df_type_predvacc = pd.DataFrame({ "bs": bs_type_predvacc, "fbi": fbi_type_predvacc, "comm": comm_type_predvacc })
df_type_predvacc
# df_t_predvacc
#%%
bs_t_predvacc = [ x[0] - x[1] for x in zip( [ r[1] for r in df["bs"].values ] , [ r[1] for r in df["label"].values ] ) ]
fbi_t_predvacc = [ x[0] - x[1] for x in zip( [ r[1] for r in df["fbi"].values ] , [ r[1] for r in df["label"].values ] ) ]
comm_t_predvacc = [ x[0] - x[1] for x in zip( [ r[1] for r in df["comm"].values ] , [ r[1] for r in df["label"].values ] ) ]
df_t_predvacc = pd.DataFrame({ "bs": bs_t_predvacc, "fbi": fbi_t_predvacc, "comm": comm_t_predvacc })
df_t_predvacc
# df_type_predvacc
#%%
df_type_predvacc.describe()
#%%
df_t_predvacc.describe()
