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
#dfxx=pd.DataFrame({})
#dfxx.to_csv('Y:/Python/csu_p3_HPC/databatch_' + "test" + '.csv', index=False)
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
            raster_data_band1 = src.read(1)
            raster_data_band2 = src.read(2)
            raster_data_band3 = src.read(3)
            raster_data_band4 = src.read(4)
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
                surrounding_pixels1 = []
                surrounding_pixels2 = []
                surrounding_pixels3 = []
                surrounding_pixels4 = []
                surrounding_pixels5 = []
                if ((src.height - batch * 1000) > 1000):
                    limL = batch * 1000
                    limH = (1 + batch) * 1000

                else:
                    limL = batch * 1000
                    limH = src.height
                    print(limL)
                    print(limH)
                for row in range(limL, limH):
                    for col in range(src.width):
                        # pixel_value = [raster_data_band1[row, col],raster_data_band2[row, col],raster_data_band3[row, col],raster_data_band4[row, col],raster_data_band5[row, col]]
                        # Check if the pixel is not a border pixel
                        if row >= 0 and col >= 0 and row <= src.height - 1 and col <= src.width - 1:
                            surrounding_pixel_values_band1 = get_surrounding_pixels(raster_data_band1, row, col)
                            surrounding_pixel_values_band2 = get_surrounding_pixels(raster_data_band2, row, col)
                            surrounding_pixel_values_band3 = get_surrounding_pixels(raster_data_band3, row, col)
                            surrounding_pixel_values_band4 = get_surrounding_pixels(raster_data_band4, row, col)
                            surrounding_pixel_values_band5 = get_surrounding_pixels(raster_data_band5, row, col)
                            # Append pixel value and surrounding pixel values to lists
                            # pixels.append(pixel_value)
                            surrounding_pixels1.append(surrounding_pixel_values_band1)
                            surrounding_pixels2.append(surrounding_pixel_values_band2)
                            surrounding_pixels3.append(surrounding_pixel_values_band3)
                            surrounding_pixels4.append(surrounding_pixel_values_band4)
                            surrounding_pixels5.append(surrounding_pixel_values_band5)
                            # print(len(surrounding_pixel_values_band1))
                    print(row)

                band = [[], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [],
                        [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []]
                #     11 11 11 11 11 12 12 12 12 12 13 13 13 13 13

                for i in range(1, 4):  # 123
                    for j in range(1, 4):  # 123
                        # for k in range(len(surrounding_pixels1)):
                        #     band[((i-1)*3+(j-1))*5+0].append( surrounding_pixels1[k][(i-1)*3+(j-1)])
                        #     band[((i - 1) * 3 + (j - 1) )* 5 + 1].append(surrounding_pixels2[k][(i - 1) * 3 + (j - 1)])
                        #     band[((i - 1) * 3 + (j - 1) )* 5 + 2].append(surrounding_pixels3[k][(i - 1) * 3 + (j - 1)])
                        #     band[((i - 1) * 3 + (j - 1)) * 5 + 3].append(surrounding_pixels4[k][(i - 1) * 3 + (j - 1)])
                        #     band[((i - 1) * 3 + (j - 1) )* 5 + 4].append(surrounding_pixels5[k][(i - 1) * 3 + (j - 1)])
                        print("checkpoint2")
                        band[((i - 1) * 3 + (j - 1)) * 5 + 0] = [row[(i - 1) * 3 + (j - 1)] for row in
                                                                 surrounding_pixels1]
                        print("checkpoint3")
                        band[((i - 1) * 3 + (j - 1)) * 5 + 1] = [row[(i - 1) * 3 + (j - 1)] for row in
                                                                 surrounding_pixels2]
                        band[((i - 1) * 3 + (j - 1)) * 5 + 2] = [row[(i - 1) * 3 + (j - 1)] for row in
                                                                 surrounding_pixels3]
                        band[((i - 1) * 3 + (j - 1)) * 5 + 3] = [row[(i - 1) * 3 + (j - 1)] for row in
                                                                 surrounding_pixels4]
                        band[((i - 1) * 3 + (j - 1)) * 5 + 4] = [row[(i - 1) * 3 + (j - 1)] for row in
                                                                 surrounding_pixels5]

                print("checkpoint_0")

                # Create DataFrame
                dft = pd.DataFrame({'11_band1': band[0], '11_band2': band[1], '11_band3': band[2], '11_band4': band[3],
                                    '11_band5': band[4],
                                    '12_band1': band[5], '12_band2': band[6], '12_band3': band[7], '12_band4': band[8],
                                    '12_band5': band[9],
                                    '13_band1': band[10], '13_band2': band[11], '13_band3': band[12],
                                    '13_band4': band[13], '13_band5': band[14],
                                    '21_band1': band[15], '21_band2': band[16], '21_band3': band[17],
                                    '21_band4': band[18], '21_band5': band[19],
                                    '22_band1': band[20], '22_band2': band[21], '22_band3': band[22],
                                    '22_band4': band[23], '22_band5': band[24],
                                    '23_band1': band[25], '23_band2': band[26], '23_band3': band[27],
                                    '23_band4': band[28], '23_band5': band[29],
                                    '31_band1': band[30], '31_band2': band[31], '31_band3': band[32],
                                    '31_band4': band[33], '31_band5': band[34],
                                    '32_band1': band[35], '32_band2': band[36], '32_band3': band[37],
                                    '32_band4': band[38], '32_band5': band[39],
                                    '33_band1': band[40], '33_band2': band[41], '33_band3': band[42],
                                    '33_band4': band[43], '33_band5': band[44],
                                    })
                #dft.to_csv('Y:/Python/csu_p3_HPC/databatch_' + str(batch) + '.csv', index=False)
                df=dft

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
                                                                      2*df[
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
                pred=model.predict(df2_reshaped)
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
            with rasterio.open("site3/prediction_site3_10si_cnn_81epoch.tif", 'w', **new_profile) as dst:
                # Copy existing bands
                # for band_num in range(1, src.count + 1):
                # dst.write(src.read(band_num), band_num)

                # Add the new band with average values
                dst.write(average_values, 1)
    return df


# Example usage
raster_file = "site3/site3_micasense.tif"  # Replace with the path to your raster file
df = create_dataframe(raster_file)
print(df.head())
#df.to_csv('dataframe.csv', index=False)
