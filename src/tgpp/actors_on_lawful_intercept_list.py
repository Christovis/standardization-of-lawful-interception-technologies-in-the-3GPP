import glob
from tqdm import tqdm
import numpy as np
import pandas as pd

from bigbang import listserv
from bigbang.analysis.listserv import ListservArchive
from bigbang.analysis.listserv import ListservList
from bigbang.analysis.utils import (
    get_index_of_msgs_with_subject,
    get_index_of_msgs_with_datetime,
)


files = glob.glob("../../bigbang/archives/3GPP/*.mbox")

domains = []
localparts = {}
filepath = "../../bigbang-archives/3GPP/3GPP_TSG_SA_WG3_LI.mbox"
mlist = ListservList.from_mbox(
    name="---",
    filepath=filepath,
    include_body=False,
)

addrs = []
for addr in mlist.df["from"].values:
    try:
        _s = ("@").join(ListservList.get_name_localpart_domain(addr)[-2:])
        addrs.append(_s)
    except:
        print(f">>> ERROR>>> {addr}")

print(f"In total there are {mlist.get_messagescount()} messages")
addrs, msg_count = np.unique(addrs, return_counts=True)
print(f"Of those, {np.sum(msg_count)} were written by {len(addrs)} members")

df = pd.DataFrame({
    'email': addrs,
    'nr_of_msgs': msg_count,
})
df = df.sort_values(by=['nr_of_msgs'], ascending=False)
df.to_csv("/home/christovis/InternetGov/3GPP_TSG_SA_WG3_LI.csv", index=False)
#
#
#df = pd.read_csv(
#    '/home/christovis/InternetGov/3GPP_TSG_SA_WG3_LI.csv',
#    sep=',',
#    header=0,
#)
#
#df = df[df["nr_of_localparts"] > int(np.percentile(df["nr_of_localparts"].values, 90))]
#df.to_csv("/home/christovis/InternetGov/3GPP_TSG_SA_WG3_LI_90perc_most_active.csv", index=False)


