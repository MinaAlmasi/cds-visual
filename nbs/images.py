import os
import cv2
import pandas as pd
from pathlib import Path

def image_dir(directory):
    '''
    Function which saves paths for all images in directory
    '''

    #list all paths
    paths = os.listdir(directory)

    # only keep paths which are images
    images = [image for image in paths if image.endswith(".jpg")]

    return images

def image_hist_normalized(image):
    '''
    Function which reads in an image, creates histogram and normalizes histogram
    '''

    # read image
    image = cv2.imread(image)

    # create hist 
    hist = cv2.calcHist([image], [0,1,2], None, [256,256,256], [0,256, 0,256, 0,256])

    # normalize hist
    hist = cv2.normalize(hist, hist, 0, 1.0, cv2.NORM_MINMAX)

    return hist

def image_search(chosen_image, directory):
    # all images
    images = image_dir(directory)

    # remove chosen image from list of all paths
    if chosen_image in images:
        images.remove(chosen_image)

    # chosen image hist
    chosen_hist = image_hist_normalized(os.path.join(directory, chosen_image))

    data_imgs = []

    for image in images: 
        # normalize hist 
        image_hist = image_hist_normalized(os.path.join(directory, image))

        # create data frame
        data = pd.DataFrame()

        # define image file name
        data["Image"] = [image]
        
        #calculate distance score 
        data["Distance"] = [cv2.compareHist(chosen_hist, image_hist, cv2.HISTCMP_CHISQR)]

        #append dataframe to data
        data_imgs.append(data)

    #concatenate
    final_data = pd.concat(data_imgs, ignore_index = True)

    # sort data
    final_data = final_data.sort_values(by=["Distance"], ascending = True, ignore_index = True)

    # round value after sorting
    final_data["Distance"] = final_data["Distance"].round(decimals = 2)

    return final_data.head(5)

if __name__ == "__main__":
    # define paths 
    path = Path(__file__) # define path to current file

    directory = path.parents[2] / "flowers" 

    image = "image_0002.jpg"

    distances = image_search(image, directory)
    print(distances)
