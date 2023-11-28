import tgx
from utils import * 

"""
1. load a dataset
2. load into a graph
3. discretize the graph
4. save the graph back to a csv
"""

#! load the datasets
# dataset = tgx.builtin.uci()

data_name = "tgbl-review" #"tgbl-wiki"
dataset = tgx.tgb_data(data_name)
# dataset = tgx.tgb_data("tgbl-review")
# dataset = tgx.tgb_data("tgbl-coin") 


ctdg = tgx.Graph(dataset)
# ctdg.save2csv("ctdg")

time_scale = "monthly" #"weekly" #"daily"  #"hourly" #"minutely" 
dtdg, ts_list = ctdg.discretize(time_scale=time_scale, store_unix=True)
print ("discretize to ", time_scale)
print ("there is time gap, ", dtdg.check_time_gap())
list2csv(ts_list, data_name + "_ts" + "_" + time_scale + ".csv")






# dtdg = ctdg.discretize(time_scale="hourly")
# # dtdg.save2csv("dtdg_daily")
# print ("discretize to hourly")
# print ("there is time gap, ", dtdg.check_time_gap())


# dtdg = ctdg.discretize(time_scale="daily")
# # dtdg.save2csv("dtdg_daily")
# print ("discretize to daily")
# print ("there is time gap, ", dtdg.check_time_gap())

# dtdg = ctdg.discretize(time_scale="weekly")
# # dtdg.save2csv("dtdg_weekly")
# print ("discretize to weekly")
# print ("there is time gap, ", dtdg.check_time_gap())
