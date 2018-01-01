from keras.layers import Flatten, Dense, Activation, Input, Lambda, Dropout
from keras.layers import Convolution2D, MaxPooling2D, Cropping2D
from keras.models import Model
from keras import backend as K
import sys
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from image_util import *



def BuildNetwork(input_shape, nb_output, network_name):
    '''
    It is used to test for different network model
    '''   
    model = None
    if network_name == "Nvidia" or network_name is None:
        model = BuildNetwork_Nvidia(input_shape, nb_output)
    if network_name =="Lenet":
        model = BuildNetwork_LeNet(input_shape, nb_output)
    if network_name == "Test":
        model = BuildNetwork_Test(input_shape, nb_output)

    return model


def BuildNetwork_Nvidia(input_shape, nb_output):
    '''
    Build the network use Keras functional API
    '''
    input = Input(shape = input_shape)
    net = input
    # normalization
    net = Lambda(lambda x: x / 127.5 - 1.0)(net)
    # Cropping the reduce the noise of images
    net = Cropping2D(cropping = ((70, 25), (0, 0)))(net)
    # Use 5 convolutional layers
    net = Convolution2D(24, 5,5, subsample=(2, 2), activation="relu")(net)
    net = Convolution2D(36, 5,5, subsample=(2, 2), activation="relu")(net)
    net = Convolution2D(48, 5,5, subsample=(2, 2), activation="relu")(net)
    net = Convolution2D(64, 3,3, activation="relu")(net)
    net = Convolution2D(64, 3,3, activation="relu")(net)
    # Flatten layer
    flat_net = Flatten()(net)
    # 3 full connected layers
    net = Dense(100)(flat_net)
    # dropout to reduce overfit
    net = Dropout(0.2)(net)
    net = Dense(50)(net)
    net = Dropout(0.2)(net)
    net = Dense(10)(net)
    output = Dense(nb_output)(net)
    
    model = Model(input, output);    
    model.compile(optimizer="adam", loss="mse")                
    print(model.summary())
    return model

    

def BuildNetwork_LeNet(input_shape, nb_output = None):
    """
    This is the Lenet network
    """
    input = Input(shape = input_shape)
    net = input
    # Preprocss the data
    net = Cropping2D(cropping = ((70, 25), (0, 0)))(net)

    # Convolutional layer
    net = Convolution2D(6, 5, 5, activation="relu")(net)
    net = MaxPooling2D()(net)
    net = Convolution2D(6, 5, 5, activation="relu")(net)
    net = MaxPooling2D()(net)

    net = Flatten()(net)
    # normalization
    net = Lambda(lambda x: x / 255.0 - 0.5)(net)

    # full connected layer
    net = Dense(120)(net)
    net = Dense(84)(net)
    net = Dense(nb_output)(net)     

    # output
    output = net
    model = Model(input, output);    
    model.compile(optimizer="adam", loss="mse")                
    
    return model


def BuildNetwork_Test(input_shape, nb_output = None):
    """
    This is the simplest network to test
    """
    input = Input(shape = input_shape)
    net = Flatten()(input)

    net = Lambda(lambda x: x / 255.0 - 0.5)(net)
    if nb_output is not None:
        fc1 = Dense(nb_output)(net)
    else:
        fc1 = Dense()(net)

    output = fc1
    model = Model(input, output);
    
    model.compile(optimizer="adam", loss="mse")                
    
    return model


def plot_lossdata(history_object):
    '''
    plot the loss data
    ''' 
    plt.plot(history_object.history["loss"])
    plt.plot(history_object.history["val_loss"])
    plt.title("model mean squared error")
    plt.ylabel("mean squared error loss")
    plt.xlabel("nb_epoch")
    plt.legend(['traning set', "validation_set"], loc="upper right")
    plt.show()


# train the network
def Test_Network():
    global dataFolder
    csvFilePath = ".\\" + dataFolder + "\\driving_log.csv"
    images_path = read_images_info_from_csv(csvFilePath);
    train_samples_path, validation_samples_path = train_test_split(images_path, test_size=0.2)    

    batch_size = 64
    add_left = True
    add_right = True
    augment_image = True

    train_image_generator = generte_images(train_samples_path, 
                                add_left = add_left, 
                                add_right = add_right,
                                batch_size=batch_size, 
                                augmentImage = augment_image);

    valid_image_generator = generte_images(validation_samples_path, 
                                add_left = add_left, 
                                add_right = add_right, 
                                batch_size=batch_size, 
                                augmentImage = augment_image);

    input_shape = (160, 320,3)
    nb_epoch = 3
    nb_output = 1
    print("input_shape is: ", input_shape, ", batch_size = ", batch_size)
    network_name = "Nvidia"
    # network_name = "Lenet"
    # network_name = "Test"
    model = BuildNetwork(input_shape, nb_output, network_name)
        
    
    if batch_size is None:
        '''
        Test the whole image set for small dataset
        '''
        X_train, y_train = next(train_image_generator)
        validation_data = next(valid_image_generator)
        model.fit(X_train, y_train, validation_data=validation_data, nb_epoch=2, shuffle=True)
    else:
        '''
        Test the image data sources using generator
        '''
        train_samples_len = len(train_samples_path)
        valid_samples_len = len(validation_samples_path)
        if (add_left is True): 
            train_samples_len += len(train_samples_path)
            valid_samples_len += len(validation_samples_path)
        if (add_right is True):
            train_samples_len += len(train_samples_path)
            valid_samples_len += len(validation_samples_path)
        if (augment_image is True):
             train_samples_len *= 2
             valid_samples_len *= 2
        print(train_samples_len, valid_samples_len)
        # To Test the generator function
        #X_train, y_train = next(train_image_generator)
        #validation_data = next(valid_image_generator)
        history_object = model.fit_generator(train_image_generator, samples_per_epoch = train_samples_len,
             validation_data=valid_image_generator, nb_val_samples=valid_samples_len,
             nb_epoch = nb_epoch)

    modelname = network_name + "_" + dataFolder + "_model.h5"
    model.save(modelname)
    print("     Model saved:   ", modelname)

    plot_lossdata(history_object)
    print("end of training")

if __name__ == "__main__":
    try:
        Test_Network()
    except Exception as e: 
        print("Exception info:  ", sys.exc_info()[0], "  ", str(e))

    print("End of run")
