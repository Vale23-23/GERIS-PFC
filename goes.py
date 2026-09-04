'''
GOES data retrieval module.
Data is retrieved from the Hugging Face repository. A Hugging Face token might be required and must be added to the .env file should it
be necessary.
'''

from huggingface_hub import snapshot_download, hf_hub_download
import numpy as np
import pandas as pd
from dotenv import load_dotenv
import os
from typing import Tuple

'''Load Hugging Face data'''
def get_hf_token():
    load_dotenv()
    token = os.getenv("HF_TOKEN")
    return token


'''Define time interval for which to retrieve GOES data'''
def get_date_range(date_ini:str,date_end:str,time_step:int) -> pd.DatetimeIndex:

    dates = pd.date_range(
        start=date_ini,
        end=date_end,
        freq=f"{time_step}min"
    )
    return dates

'''Download GOES data from HF'''
def download_goes_data(dates:pd.DatetimeIndex, repository:str, token:str):
    for date in dates:
        file_name = date.strftime("%Y%m%d_%H%M") + ".npy"
        for band in ["ABI-L1b-Rad-B07", "ABI-L1b-Rad-B14", "ABI-L2-FDCF-Mask"]:

            file_path = f"uruguay/{band}/{file_name}"

            hf_hub_download(
                repo_id= repository,
                repo_type="dataset",
                filename=file_path,
                token=token,
                local_dir="dataset"
            )
            print(f"Downloaded: {file_path}")

'''Retrieve downloaded GOES data'''
def get_goes_data_from_timestamp(timestamp:str) -> Tuple[np.ndarray,np.ndarray,np.ndarray]:
    '''Retrieve from single timestamp'''
    band7_dir = "dataset/uruguay/ABI-L1b-Rad-B07/"
    band14_dir = "dataset/uruguay/ABI-L1b-Rad-B14/"
    fire_masks_dir = "dataset/uruguay/ABI-L2-FDCF-Mask/"
    band7, band14 = [],[]
    fire_masks = []

    file_name = timestamp.strftime("%Y%m%d_%H%M") + ".npy"
    band7 = np.load(band7_dir+file_name)
    band14  = np.load(band14_dir+file_name)
    fire_mask = np.load(fire_masks_dir+file_name)

    return band7,band14,fire_mask

def get_goes_data_from_range(time_ini:str,time_end:str,timestep:int) -> Tuple[list[np.ndarray],list[np.ndarray],list[np.ndarray]]:
    '''Retrieve from time range'''
    band7_dir = "dataset/uruguay/ABI-L1b-Rad-B07/"
    band14_dir = "dataset/uruguay/ABI-L1b-Rad-B14/"
    fire_masks_dir = "dataset/uruguay/ABI-L2-FDCF-Mask/"
    band7, band14 = [],[]
    fire_masks = []


    dates = pd.date_range(
        start=time_ini,
        end=time_end,
        freq=f"{timestep}min"
    )

    for d in dates:
        file_name = d.strftime("%Y%m%d_%H%M") + ".npy"
        band7.append(np.load(band7_dir+file_name))
        band14.append(np.load(band14_dir+file_name))
        fire_masks.append(np.load(fire_masks_dir+file_name))

    return band7,band14,fire_masks