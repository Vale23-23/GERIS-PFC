'''
VIIRS data retrieval module
'''
import pandas as pd


MAP_KEY = "726e2860ec89af63969f678db66c2bab"

def build_bounding_box_str(north:float,east:float,south:float,west:float) -> str:
    return f"{west},{south},{east},{north}"

def build_bounding_box_region(region:str) -> str:

    if(region  == "uruguay"):
        box = build_bounding_box_str(west  = -58.5,south = -35.2,east  = -53.0,north = -30.0)
    else:
        raise ValueError("Invalid region name.")

    return box

def get_viirs_fires(region: str, dataset: str, date:str) -> pd.DataFrame:
    date = pd.Timestamp(date).strftime("%Y-%m-%d")
    url = (
    f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/"
    f"{MAP_KEY}/{dataset}/{region}/1/{date}"
    )
    viirs_fires = pd.read_csv(url)[
    [
        "latitude",
        "longitude",
        "bright_ti4",
        "bright_ti5",
        "frp",
        "confidence",
        "acq_date",
        "acq_time",
        "daynight",
    ]
    ]
    return viirs_fires

def get_goes_filenames(viirs_data:pd.DataFrame, timespan: int) -> pd.DataFrame:
    
    band7_dir = "dataset/uruguay/ABI-L1b-Rad-B07/"
    band14_dir = "dataset/uruguay/ABI-L1b-Rad-B14/"
    fire_masks_dir = "dataset/uruguay/ABI-L2-FDCF-Mask/"

    #Build VIIRS dates for filename formating
    viirs_data["acq_datetime"] = pd.to_datetime(
        viirs_data["acq_date"].astype(str)
        + " "
        + viirs_data["acq_time"].astype(str).str.zfill(4),
        format="%Y-%m-%d %H%M",
        utc=True
    )

    viirs_data["goes_band7_files"] = None
    viirs_data["goes_band14_files"] = None
    viirs_data["goes_fire_files"] = None


    #For each VIIRS observation, store GOES filenames for the previous timespan hours
    #Examples:
    #VIIRS = 17:32 -> last GOES = 17:20
    #VIIRS = 17:30 -> last GOES = 17:20
    for index, row in viirs_data.iterrows():
        viirs_time = row["acq_datetime"]



        end = viirs_time.floor("10min") - pd.Timedelta(minutes=10)
        start = end - pd.Timedelta(hours=timespan)

        timestamps = pd.date_range(
            start=start,
            end=end,
            freq="10min"
        )

        band7_files = []
        band14_files= []
        goes_fire_files = []

        for timestamp in timestamps:

            filename = timestamp.strftime("%Y%m%d_%H%M") + ".npy"

            b7_filepath = band7_dir + filename
            b14_filepath = band14_dir + filename
            goes_fire_filepath = fire_masks_dir + filename

            band7_files.append(b7_filepath)
            band14_files.append(b14_filepath)
            goes_fire_files.append(goes_fire_filepath)

        viirs_data.at[index, "goes_band7_files"] = band7_files
        viirs_data.at[index, "goes_band14_files"] = band14_files
        viirs_data.at[index, "goes_fire_files"] = goes_fire_files

