import sys
import glob
import csv
import cv2
import numpy as np
import sklearn


#dataFolder = "MyTestData"
#dataFolder = "NewTestData"
#dataFolder = "01_01_2017_TestData"
dataFolder = "data"
offset = 0.2


def read_images_info_from_csv(csvPath):
    """
    read information from csv file
    """
    images_info = []

    with open(csvPath) as csvFile:
        reader = csv.reader(csvFile)
        # skip the first line
        next(csvFile)
        for line in reader:
            images_info.append(line)
    
    return images_info

def generte_images(images_info, add_left = False, add_right = False, batch_size = sys.maxsize - 1, augmentImage = False):
    """
    This serves as the generator function to read images from the given image list info
    """
    count = 0
    batch_images = []
    batch_images_path = []
    steering = []
    if (images_info is None):
        batch_images_array = np.array(batch_images)    
        steering_array =  np.array(steering) 
        return sklearn.utils.shuffle(batch_images_array, steering_array)

    global dataFolder
    batch_size = min(batch_size, len(images_info))
    while 1: #Loop forever so that the generator never terminates
        sklearn.utils.shuffle(images_info)
        for line in images_info:
            line[0] = line[0].replace("/","\\")
            line[1] = line[1].replace("/","\\")
            line[2] = line[2].replace("/","\\")
            # use absolute path, so that when new recovery data is generated,
            # i can just copy the new data to the driving_log.csv file without
            # copy all the new images data into the data folder
            # I run my code on my local computer            
            img_path         =  line[0]
            left_image_path  =  line[1]
            right_image_path =  line[2]
            angle = float(line[3])

            if count < batch_size :
                count += 1 
                batch_images_path.append(img_path);
                batch_images.append(cv2.imread(img_path))
                steering.append(angle)

                if (add_left):
                    batch_images_path.append(left_image_path)
                    batch_images.append(cv2.imread(left_image_path))
                    steering.append(angle + offset)

                if (add_right):
                    batch_images_path.append(right_image_path)
                    batch_images.append(cv2.imread(right_image_path))
                    steering.append(angle - offset)    

            if (count == batch_size):            
                if (augmentImage == True):
                    batch_images, steering = Augment_Images(batch_images, steering)
                print("  ", len(batch_images), len(steering))
                batch_images_array = np.array(batch_images)
                #batch_images_path = np.array(batch_images_path)
                steering_array =  np.array(steering) 
                yield sklearn.utils.shuffle(batch_images_array, steering_array)

                #reset the count and image list
                batch_images.clear()
                batch_images_path.clear() 
                steering.clear()
                count = 0

    if (augmentImage == True):
        batch_images, steering = Augment_Images(batch_images, steering)
    print(len(batch_images), len(steering))
    batch_images_array = np.array(batch_images)
    #batch_images_path = np.array(batch_images_path)
    steering_array =  np.array(steering) 
    return sklearn.utils.shuffle(batch_images_array, steering_array)


def Augment_Images(X_train, y_train):
    '''
    Augment the image by fliping
    '''
    X_train_new  = []
    y_train_new = []
    for image, value in zip(X_train, y_train):
        X_train_new.append(image)
        y_train_new.append(value)
        X_train_new.append(cv2.flip(image, 1))
        y_train_new.append(value * -1.0)

    return X_train_new, y_train_new

def Test():
    global dataFolder
    csvFilePath = dataFolder + "\\driving_log.csv"
    image_paths = read_images_info_from_csv(csvFilePath)
    image_generator = generte_images(image_paths, add_left = True, add_right = True, batch_size=256);
    
    for i in range(100):
        print("==============================================       ", i)
        batch_images, steerings = next(image_generator)
        print(len(batch_images), len(steerings))

    print("==============================================")
    data = next(image_generator)
    print(type(data))
    print(len(data[0]))
    print(len(data[1]))

if __name__ == "__main__":
    '''
    To test the functions 
    '''
    try:
        #Test()
        right_image_path = "H:\\SelfDrivingCar\\CarND-Behavioral-Cloning-P3\\examples\\right_1.jpg"
        original = cv2.imread(right_image_path)
        flipped = cv2.flip(original, 1)
        cv2.imwrite("flip_right_1.jpg", flipped)
    except:
        print("Exception info: ",  sys.exc_info()[0])

    print("end of file")