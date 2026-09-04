'''
Viirs fire data downloader script
'''

import pandas as pd
import viirs 


region = viirs.build_bounding_box_region("uruguay")
dataset = "VIIRS_NOAA20_SP"
data_folder = "dataset/viirs_data/"


dates = pd.date_range(
    start="2025-11-15",
    end="2026-02-15",
)


#Save data for each day as a csv file
for d in dates:
    viirs_fires = viirs.get_viirs_fires(region,dataset,d)
    filename = str(d.strftime("%Y%m%d_%H%M")) + ".csv"
    viirs_fires.to_csv(data_folder+filename,index = False)