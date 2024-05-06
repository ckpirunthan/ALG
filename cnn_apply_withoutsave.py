import rasterio
import pandas as pd
import numpy as np
import rasterio
import numpy as np
import csv
import pandas as pd
import math
import pickle
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import os


# dfxx=pd.DataFrame({})
# dfxx.to_csv('Y:/Python/csu_p3_HPC/databatch_' + "test" + '.csv', index=False)
def get_surrounding_pixels(raster, row, col):
    """Get the values of the surrounding 8 pixels."""
    pixel_values = []
    for i in range(row - 1, row + 2):
        for j in range(col - 1, col + 2):
            if 0 <= i < raster.shape[0] and 0 <= j < raster.shape[1]:
                pixel_values.append(raster[i, j])
            else:
                pixel_values.append(0.000000001)
    return pixel_values


def create_dataframe(raster_file):
    """Create a pandas DataFrame with pixel values and surrounding values."""
    with rasterio.open(raster_file) as src:
        with open("xgboost_model_all_nonveg.pkl", "rb") as model_file:
            xgbr = pickle.load(model_file)
            raster_data_band5 = src.read(5)

            print(len(raster_data_band5))

            # Initialize lists to store pixel values and surrounding pixel values
            pixels = []
            surrounding_pixels1 = []
            surrounding_pixels2 = []
            surrounding_pixels3 = []
            surrounding_pixels4 = []
            surrounding_pixels5 = []

            # Iterate over each pixel
            print(src.height)
            for batch in range(1 + math.floor(src.height / 1000)):

                # dft.to_csv('Y:/Python/csu_p3_HPC/databatch_' + str(batch) + '.csv', index=False)
                df = pd.read_csv('site2/databatch_' + str(batch) + '.csv')

                for i in range(1, 4):
                    for j in range(1, 4):
                        df[str(i) + str(j) + "_" + "band6"] = (df[str(i) + str(j) + "_" + "band5"] -
                                                               df[str(i) + str(j) + "_" + "band3"]) / (
                                                                      df[
                                                                          str(i) + str(j) + "_" + "band5"] +
                                                                      df[str(i) + str(j) + "_" + "band3"])
                        df[str(i) + str(j) + "_" + "band7"] = (df[str(i) + str(j) + "_" + "band2"] -
                                                               df[str(i) + str(j) + "_" + "band5"]) / (
                                                                      df[
                                                                          str(i) + str(j) + "_" + "band2"] +
                                                                      df[str(i) + str(j) + "_" + "band5"])
                        df[str(i) + str(j) + "_" + "band8"] = (df[str(i) + str(j) + "_" + "band5"] /
                                                               df[str(i) + str(j) + "_" + "band2"]) - 1
                        df[str(i) + str(j) + "_" + "band9"] = (2 * df[str(i) + str(j) + "_" + "band2"] -
                                                               df[str(i) + str(j) + "_" + "band1"] -
                                                               df[str(i) + str(j) + "_" + "band3"]) / (
                                                                      2 * df[
                                                                  str(i) + str(j) + "_" + "band2"] +
                                                                      df[
                                                                          str(i) + str(j) + "_" + "band1"] +
                                                                      df[str(i) + str(j) + "_" + "band3"])
                        df[str(i) + str(j) + "_" + "band10"] = (df[str(i) + str(j) + "_" + "band5"] -
                                                                df[str(i) + str(j) + "_" + "band4"]) / (
                                                                       df[
                                                                           str(i) + str(j) + "_" + "band5"] +
                                                                       df[
                                                                           str(i) + str(j) + "_" + "band4"])

                        df[str(i) + str(j) + "_" + "band11"] = (df[str(i) + str(j) + "_" + "band1"]) / (
                                df[str(i) + str(j) + "_" + "band1"] + df[
                            str(i) + str(j) + "_" + "band2"] + df[str(i) + str(j) + "_" + "band3"] + df[
                                    str(i) + str(j) + "_" + "band4"] + df[str(i) + str(j) + "_" + "band5"])
                        df[str(i) + str(j) + "_" + "band12"] = (df[str(i) + str(j) + "_" + "band2"]) / (
                                df[str(i) + str(j) + "_" + "band1"] + df[
                            str(i) + str(j) + "_" + "band2"] + df[str(i) + str(j) + "_" + "band3"] + df[
                                    str(i) + str(j) + "_" + "band4"] + df[str(i) + str(j) + "_" + "band5"])
                        df[str(i) + str(j) + "_" + "band13"] = (df[str(i) + str(j) + "_" + "band3"]) / (
                                df[str(i) + str(j) + "_" + "band1"] + df[
                            str(i) + str(j) + "_" + "band2"] + df[str(i) + str(j) + "_" + "band3"] + df[
                                    str(i) + str(j) + "_" + "band4"] + df[str(i) + str(j) + "_" + "band5"])
                        df[str(i) + str(j) + "_" + "band14"] = (df[str(i) + str(j) + "_" + "band4"]) / (
                                df[str(i) + str(j) + "_" + "band1"] + df[
                            str(i) + str(j) + "_" + "band2"] + df[str(i) + str(j) + "_" + "band3"] + df[
                                    str(i) + str(j) + "_" + "band4"] + df[str(i) + str(j) + "_" + "band5"])
                        df[str(i) + str(j) + "_" + "band15"] = (df[str(i) + str(j) + "_" + "band5"]) / (
                                df[str(i) + str(j) + "_" + "band1"] + df[
                            str(i) + str(j) + "_" + "band2"] + df[str(i) + str(j) + "_" + "band3"] + df[
                                    str(i) + str(j) + "_" + "band4"] + df[str(i) + str(j) + "_" + "band5"])
                        print(str(i) + str(j) + "_done")
                df2 = df.iloc[:, 45:].values
                df2_reshaped = df2.reshape(-1, 3, 3, 10)
                print("hi")
                model = tf.keras.models.load_model('modelXY_epoch_81.h5')
                pred = model.predict(df2_reshaped)
                print(np.argmax(pred, axis=1))
                print(pred)

                if batch == 0:
                    est = np.argmax(pred, axis=1)
                else:
                    est = np.concatenate([est, np.argmax(pred, axis=1)])

            est = est.reshape(src.height, src.width)
            print("hi")
            # Calculate the average pixel values across all other bands

            average_values = est
            print(average_values)

            # Update the metadata to reflect the new band
            new_count = src.count + 1
            new_profile = src.profile
            new_profile.update(count=new_count)

            # Create a new raster with the additional band
            with rasterio.open("site2/prediction_site2_10si_cnn_81epoch.tif", 'w', **new_profile) as dst:
                # Copy existing bands
                # for band_num in range(1, src.count + 1):
                # dst.write(src.read(band_num), band_num)

                # Add the new band with average values
                dst.write(average_values, 1)
    return df


# Example usage
raster_file = "site2/merges_micasense_site2_modified.tif"  # Replace with the path to your raster file
df = create_dataframe(raster_file)
print(df.head())
# df.to_csv('dataframe.csv', index=False)