import tensorflow as tf
import numpy as np
import time
import matplotlib.pyplot as plt

class DuplicateChannelsLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs):
        # Duplicate the grayscale channel across the channel dimension
        return tf.repeat(inputs, repeats=3, axis=-1)

    def get_config(self):
        base_config = super().get_config()
        return {**base_config}

class training_cnn_model():
    def __init__(self, LEARNING_RATE, LOAD_DATA_TO_MODEL=str, LOAD_TEST_DATA=str, CONV_FILTER_1=int, CONV_FILTER_2=int, CONV_FILTER_3=int, DENSE_UNIT_1=int, DENSE_UNIT_2=int, CONV_DROPOUT=float, DENSE_DROPOUT_1=float, DENSE_DROPOUT_2=float, EPOCHS=int, IMAGE_SIZE=tuple, CROPPED_IMAGE_SIZE=tuple, BATCH_SIZE=int, TRAINING_PATIENCE=int):

        self.load_data = LOAD_DATA_TO_MODEL
        self.load_test_data = LOAD_TEST_DATA

        self.conv_filter_1 = CONV_FILTER_1
        self.conv_filter_2 = CONV_FILTER_2
        self.conv_filter_3 = CONV_FILTER_3
        self.dense_unit_1 = DENSE_UNIT_1
        self.dense_unit_2 = DENSE_UNIT_2

        self.conv_dropout = CONV_DROPOUT
        self.dense_dropout_1 = DENSE_DROPOUT_1
        self.dense_dropout_2 = DENSE_DROPOUT_2

        self.epochs = EPOCHS
        self.learning_rate = LEARNING_RATE

        # Images are organized in subdirectories under 'LOAD_DATA' for each class
        self.image_size = IMAGE_SIZE # Resize ?
        self.cropped_image_size = CROPPED_IMAGE_SIZE
        self.batch_size = BATCH_SIZE
        self.training_patience = TRAINING_PATIENCE

        # Model + Figure connection with timestamp
        self.time_stamp_name = int(time.time())


    ### Preprocess

    def preprocess_for_training(self):
        """
        Preprocess for training, splits the data into training set and validation set.
        """
        ### Define data generators
        datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            rescale=1./255,
            validation_split=0.2
            ) # normalize images and split data for validation

        datagen_test = tf.keras.preprocessing.image.ImageDataGenerator(
            rescale=1./255
            ) # normalize images and split data for validation

        ### Load datasets
        train_generator = datagen.flow_from_directory(
            directory = self.load_data,
            target_size = self.image_size,
            batch_size = self.batch_size,
            color_mode = "grayscale",       # grayscale for 1 channel data
            class_mode = "categorical",     # 'categorical' for multiclass classification
            subset = "training"
            )

        validation_generator = datagen.flow_from_directory(
            directory = self.load_data,
            target_size = self.image_size,
            batch_size = self.batch_size,
            color_mode='grayscale',
            class_mode = 'categorical',
            subset = 'validation'
            ) 
        
        test_generator = datagen_test.flow_from_directory(
            directory = self.load_test_data,
            target_size = self.image_size,
            batch_size = self.batch_size,
            color_mode='grayscale',
            class_mode = 'categorical'     # 'categorical' for multiclass classification
        )

        return train_generator, validation_generator, test_generator

    def build_cnn_cnl_model(self):
        """
        Defines the architechture of pretrained convolutional neural network called ConvNeXtLarge (notop) with additional Dense layers.
        """
        # New input layer with the original image size
        input_shape = (self.image_size[0], self.image_size[1], 3)  # Replace with your desired input shape
        input_tensor = tf.keras.layers.Input(shape=input_shape)

        # Duplicate the grayscale channel to mimic 3-channel RGB
        duplicated_input = DuplicateChannelsLayer()(input_tensor)

        # For example, to crop to (128, 128, 3), adjust according to your needs
        cropped_input = tf.keras.layers.CenterCrop(height=self.cropped_image_size[0], width=self.cropped_image_size[1])(duplicated_input)
        

        # Initialize ConvNeXtLarge with the cropped input
        conv_next_large_model = tf.keras.applications.VGG19(
            include_top=False,
            weights="imagenet",
            input_tensor=cropped_input,  # input_tensor is optional input for the model, use the cropped input here
            input_shape=(self.cropped_image_size[0], self.cropped_image_size[1], 3),  # Updated to match the cropped size
            pooling=None,
            classes=1000,
            classifier_activation='softmax'
        )

        # Freeze the layers of the base model
        for layer in conv_next_large_model.layers:
            layer.trainable = False

        # Output of the ConvNextLarge model
        x = conv_next_large_model.output
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dense(1024, activation='relu', kernel_initializer="he_normal")(x)
        x = tf.keras.layers.Dropout(0.4)(x)
        x = tf.keras.layers.Dense(512, activation='relu', kernel_initializer="he_normal")(x)
        x = tf.keras.layers.Dropout(0.2)(x)
        x = tf.keras.layers.Dense(128, activation='relu', kernel_initializer="he_normal")(x)
        x = tf.keras.layers.Dropout(0.2)(x)

        # Output layer
        outputs = tf.keras.layers.Dense(9, activation='softmax')(x)  # 9 neurons in the final layer for 9 classes

        # Model
        model = tf.keras.models.Model(inputs=input_tensor, outputs=outputs)

        return model


    def build_cnn_model(self):
        """
        Defines the architechture of convolutional neural network with a spatial attention mechanism.
        """

        # input layer (use cropped or non cropped version of input layer, but NOT both)
        
        input_layer = tf.keras.Input(shape=(self.image_size[0], self.image_size[1], 1)) # Last number (1) meaning grayscale while (3) is RGB.

        # Duplicate the grayscale channel to mimic 3-channel RGB
        #duplicated_input = DuplicateChannelsLayer()(input_layer)

        # Crop layer
        x = tf.keras.layers.CenterCrop(height=self.cropped_image_size[0], width=self.cropped_image_size[1])(input_layer)

        # hidden layers
        x = tf.keras.layers.Conv2D(self.conv_filter_1, kernel_size=(4, 4), activation='relu', kernel_initializer="he_normal" )(x)
        x = tf.keras.layers.MaxPooling2D(3, 3)(x)
        x = tf.keras.layers.Dropout(self.conv_dropout)(x)
        x = tf.keras.layers.Conv2D(self.conv_filter_2, kernel_size=(3, 3), activation='relu', kernel_initializer="he_normal" )(x)
        x = tf.keras.layers.MaxPooling2D(2, 2)(x)
        x = tf.keras.layers.Dropout(self.conv_dropout)(x)
        x = tf.keras.layers.Conv2D(self.conv_filter_3, kernel_size=(3, 3), activation='relu', kernel_initializer="he_normal" )(x)
        x = tf.keras.layers.MaxPooling2D(2, 2)(x)
        x = tf.keras.layers.Dropout(self.conv_dropout)(x)

        # Flatten and results
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(self.dense_unit_1, activation='relu', kernel_initializer="he_normal" , kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
        x = tf.keras.layers.Dropout(self.dense_dropout_1)(x)
        x = tf.keras.layers.Dense(self.dense_unit_2, activation='relu', kernel_initializer="he_normal" )(x)
        x = tf.keras.layers.Dropout(self.dense_dropout_2)(x)

        # output layer
        outputs = tf.keras.layers.Dense(9, activation='softmax')(x) # 9 neurons in the final layer for 9 classes
        model = tf.keras.models.Model(inputs=input_layer, outputs=outputs)

        return model

    def __adam_optimizer(self):
        """Adam optimizer combines two advantages of two other extensions of stochastic gradient descent:
        1 ) Momentum, optimizer accelerate in directions with consistent gradient signals
        2 ) Adaptive learning rate, adjusts the learning rate for each parameter individually
        """
        # Instantiate the Adam optimizer with custom hyperparameters
        custom_adam_optimizer = tf.keras.optimizers.Adam(
            lr=self.learning_rate,                # Learning rate, default is 0.001
            beta_1=0.9,                             # Exponential decay rate for the 1st moment estimates, default is 0.9
            beta_2=0.999,                           # Exponential decay rate for the 2nd moment estimates, default is 0.999
            epsilon=1e-07,                          # A small constant for numerical stability, default is 1e-07
            decay=0.0,                              # Learning rate decay over each update, default is 0
            # amsgrad=False                         # Whether to apply the AMSGrad variant of this algorithm, default is False
        )
        return custom_adam_optimizer

    def build_cnn_cnl_compile_train(self, train_generator, validation_generator):
        """
        Building and training the model
        """
        # Build the model
        self.model = self.build_cnn_cnl_model()

        # Start learning
        self.model.compile(
            optimizer = self.__adam_optimizer(),
            loss = "categorical_crossentropy",
            metrics = ["accuracy"]
            )

        # Early stopping
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=self.training_patience, restore_best_weights=True) # Auto Stop

        # Fit
        self.history = self.model.fit(
            train_generator,
            steps_per_epoch=train_generator.samples // self.batch_size,
            epochs=self.epochs,
            validation_data=validation_generator,
            validation_steps=validation_generator.samples // self.batch_size,
            callbacks=[early_stopping])
        
        return self.model, self.history

    def build_compile_train(self, train_generator, validation_generator):
        """
        Building and training the model
        """
        # Build the model
        self.model = self.build_cnn_model()

        # Start learning
        self.model.compile(
            optimizer = self.__adam_optimizer(),
            loss = "categorical_crossentropy",
            metrics = ["accuracy"]
            )

        # Early stopping
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=self.training_patience, restore_best_weights=True) # Auto Stop

        # Fit
        self.history = self.model.fit(
            train_generator,
            steps_per_epoch=train_generator.samples // self.batch_size,
            epochs=self.epochs,
            validation_data=validation_generator,
            validation_steps=validation_generator.samples // self.batch_size,
            callbacks=[early_stopping])
        
        return self.model, self.history

    def save_model(self, model):
        """
        Saves the model
        """
        #model.save(f"models\\CNN_model_{self.batch_size}batsz_{self.time_stamp_name}.h5")
        model.save(f"models/CNN_model_{self.batch_size}batsz_{self.time_stamp_name}.h5")
        saved_model_name = f"CNN_model_{self.batch_size}batsz_{self.time_stamp_name}"
        print("")
        print(f"Model saved: CNN_model_{self.batch_size}batsz_{self.time_stamp_name}.h5")
        print("")
        return saved_model_name

    def validation_evaluation_visualization(self, validation_generator, history):
        """
        Evaluates the performance of trainset vs validation set. Plots the graph and prints final val_loss, val_accuracy values into title.
        """
        # Evaluate the model
        eval_result = self.model.evaluate(validation_generator)
        print(f'Validation Loss: {eval_result[0]}, Validation Accuracy: {eval_result[1]}')

        # Visualization
        # Plot training & validation accuracy values
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title(f'Model accuracy - val_loss:{round(eval_result[0], 4)} - val_acc: {round(eval_result[1], 4)}')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')
        plt.savefig(f"results/Figure_{self.batch_size}batsz_{self.time_stamp_name}.png")
        
