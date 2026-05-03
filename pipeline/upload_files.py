from dotenv import load_dotenv
load_dotenv()
from loader import LoaderStorage
from s3fs import S3FileSystem
import pandas as pd
 
import os
year_start = 19
year_end = 19
if __name__ == "__main__":
  storage = LoaderStorage(root="s3://data-mining/")
  dir_path = "C:\\Users\\simon schumacher\\OneDrive\\Desktop\\FSS_26_Master_1\\DataMining\\Project\\OTS_Data\\"
  files = os.listdir(dir_path)
 
 
  for j in range(year_start, year_end + 1):
    for i in range(1, 13):
        print(f"Processing Year: 20{j:02d}, Month: {i}")
        df_name = f"On_Time_Reporting_Carrier_On_Time_Performance_(1987_present)_20{j}_{i}.csv"
        fp = os.path.join(dir_path, df_name)
        df = pd.read_csv(fp)
        storage.write_csv(df, f"data/raw/{df_name}")