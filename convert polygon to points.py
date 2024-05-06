import geopandas as gpd
import rasterio
from rasterio.features import geometry_mask
from shapely.geometry import Point
import numpy as np
from tqdm import tqdm

# Read polygon GeoPackage file
polygon_gpkg_path = "path_to_polygon.gpkg"
polygon_gdf = gpd.read_file(polygon_gpkg_path)

# Read raster TIFF file
raster_tif_path = "path_to_raster.tif"
with rasterio.open(raster_tif_path) as src:
    raster_bounds = src.bounds
    raster_crs = src.crs
    raster_data = src.read(1)  # assuming single band raster
    transform = src.transform

# Iterate through each pixel
points = []
points2=[]
for geom in tqdm(polygon_gdf.geometry, desc="Processing pixels"):
    # Convert polygon to raster mask
    id_value = polygon_gdf.loc[polygon_gdf.geometry == geom, 'id'].iloc[0]
    print("id:", id_value)
    mask = geometry_mask([geom.buffer(0)],
                          out_shape=raster_data.shape,
                          transform=transform,
                          invert=True)

    # Iterate through each pixel and check if it falls within the polygon
    for row in range(mask.shape[0]):
        for col in range(mask.shape[1]):
            if mask[row, col]:
                # Convert pixel coordinates to map coordinates
                x, y = transform * (col, row)
                # Create point geometry
                point = Point(x, y)
                point2=Point(x, y)
                points2.append(point2)
                points.append(point)

    points_gdf = gpd.GeoDataFrame(geometry=points2, crs=raster_crs)
    points2=[]
    # Save points to a new GeoPackage file
    points_gpkg_path = "Path_to_points"+str(id_value)+".gpkg"  # This save points file for individual polygon
    points_gdf.to_file(points_gpkg_path, driver="GPKG")


# Create GeoDataFrame from points
points_gdf = gpd.GeoDataFrame(geometry=points, crs=raster_crs)

# Save points to a new GeoPackage file
points_gpkg_path = "path_to_points_common.gpkg"  # This save common points file for all polygons
points_gdf.to_file(points_gpkg_path, driver="GPKG")
