import rasterio
import geopandas as gpd
import rasterio
import geopandas as gpd
import csv

def sample_raster_around_point(raster_path, point_geopackage_path):
    # Open point GeoPackage
    list = []
    list2 = []
    points = gpd.read_file(point_geopackage_path)

    # Open raster file
    with rasterio.open(raster_path) as src:
        for index, point in points.iterrows():
            # Sample raster pixel values around the point location
            for x_offset in range(-1, 2):
                for y_offset in range(-1, 2):
                    sample_point = (point.geometry.x + x_offset * src.transform[0],
                                    point.geometry.y + y_offset * src.transform[4])
                    for val in src.sample([sample_point]):
                        # print( f"Point {index + 1}: Pixel value = {val} ")
                        for i in val:
                            list.append(i)
            list2.append(list)
            list = []
    #print(list2)

    with open("path_to_sample_window.csv", 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)

        titles = []  # Replace with your column titles
        titles = ["11_band1", "11_band2", "11_band3", "11_band4", "11_band5", "21_band1", "21_band2", "21_band3",
                  "21_band4", "21_band5", "31_band1", "31_band2", "31_band3", "31_band4", "31_band5", "12_band1",
                  "12_band2", "12_band3", "12_band4", "12_band5", "22_band1", "22_band2", "22_band3", "22_band4",
                  "22_band5", "32_band1", "32_band2", "32_band3", "32_band4", "32_band5", "13_band1", "13_band2",
                  "13_band3", "13_band4", "13_band5", "23_band1", "23_band2", "23_band3", "23_band4", "23_band5",
                  "33_band1", "33_band2", "33_band3", "33_band4", "33_band5", ]
        writer.writerow(titles)
        writer.writerows(list2)
# Example usage:
raster_file_path = "path_to_raster.tif"
point_geopackage_path = "path_to_points.gpkg"
sample_raster_around_point(raster_file_path, point_geopackage_path)
