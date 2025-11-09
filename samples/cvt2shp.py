import os
import sys
sys.path.append('.')
import pandas as pd
import geopandas as gpd

from utils import *
from shapely.geometry import Point

def csv_to_points(sample_files, output_path):
    points = []
    for file_path in sample_files:
        try:
            df = pd.read_csv(file_path)
            latitude, longitude = df['latitude'].iloc[0], df['longitude'].iloc[0]

            point = Point(longitude, latitude)
            points.append(point)
            
        except Exception as e:
            print(f"error-{file_path}: {e}")
    
    gdf = gpd.GeoDataFrame(geometry=points)
    gdf.crs = 'EPSG:4326'
    
    # 保存为矢量文件
    gdf.to_file(output_path)
    print(f"saved to {output_path}")
    
if __name__ == '__main__':
    sample_files, _ = get_all_files_in_samples('.\\samples', split_rate=1)
    csv_to_points(sample_files, '.\\samples\\AAA_shapefiles\\samples.shp')
    