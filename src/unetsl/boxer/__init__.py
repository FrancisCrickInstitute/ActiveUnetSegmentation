#!/usr/bin/env python3

class BoxerDataSource:
    """
        This will use the normal indexed data source with a backing volume
    """
    def __init__(self, image, labels, input_shape):
        """
            Args:
                image: an image with channels (n, c, z, y, x) n & z can be
                    swapped as the image is treated 2d.
                labels: corresponding image that contains the skeleton of the 
                    image input.
        """
        self.image = image
        self.labels = labels
    def prepare(self):
        for i, stack in enumerate(self.image):
            

class BoxerModel:
    def __init__(self):
        self.input_shape=((1, 128, 128));
        
        pass
    
    def createModel(self):
        data_format = "channels_first"

        box_branch = keras.layers.Input((4, ))
        box_branch = keras.layers.Dense((1024, ))(box_branch)
        box_branch = keras.layers.Reshape((1, 32, 32))(box_branch)

        con_branch = keras.layers.Input(self.input_shape)
        
        
        for i in range(3):
            conv0 = keras.layers.Conv3D(
                        32*2**i, 
                        (1, 4, 4), 
                        padding='same', 
                        strides=(1, 2, 2), 
                        activation="relu",
                        data_format=data_format,
                        name = "contraction-%s"%i ) 
            con_branch = conv0(con_branch)
        

        merged = keras.layers.Concatenate([con_branch, box_branch])
        
        for i in range(5):
            steady = keras.layers.Conv3D(
                        256, 
                        ( 1, 3, 3),
                        padding='same', 
                        strides=(1, 1, 1), 
                        activation="relu",
                        data_format=data_format, 
                        name = "steady-%s"%i )
            c = steady(c)
            drp = keras.layers.SpatialDropout3D(rate = 0.1, data_format=data_format);
            merged = drp(merged)
        
        for i in range(3):
            tconv = keras.layers.Conv3DTranspose(
                    filters=64,
                    kernel_size=(1, 4, 4), 
                    strides=(1, 2, 2), 
                    data_format = data_format,
                    activation = "relu",
                    name = "expansion-%s",
                    padding = "same"
                    )
            merged = tconv(merged)
        
        
        opl = keras.layers.Conv3D(
                    2, 
                    (1, 1, 1),
                    padding='same', 
                    strides=(1,1,1),
                    data_format=data_format, 
                    name = "final", 
                    activate = "sigmoid")
        merged = opl(merged)
        
        #activation = keras.layers.Activation("sigmoid", name="output")

        self.model = keras.models.Model(inputs=[inp], outputs=[c])
        print(self.model.summary())
        pass
    
    def loadModel(self):
        pass
    
    def loadWeights(self):
        pass
    
    def saveModel(self):
        pass
    
    def trainModel(self):
        pass
    
    def predict(self, image):
        pass
    
    