import tensorflow as tf
from main_datamaker import create_data_all
from CNN_training import training_cnn_model
from CNN_training import DuplicateChannelsLayer
from predicting import prediction_and_results

seed_value = 1

#### SETTINGS ####
# Adjust settings
# Module settings
#1) Create data module
CREATE_DATA = 1                 # 1 = CREATE DATA, 0 = DO NOT CREATE NEW DATA
GENERATE_TEST_DATA = 0          # 1 = PREPROCESS TEST DATA
#2) Train module
TRAIN_MODEL = 0                 # 1 = TRAIN NEW MODEL, 2 = Train pretrained model,  0 = LOAD READY MODEL
#3) Prediction module
TESTSET_PREDICTION_RESULTS = 0   # 1 = RUN PREDICTION, 0 = DO NOT RUN PREDICTION

# LOAD AND SAVE DIRS
ORIGINAL_DIR = "Bardus_thick_labeled_data/Original_traindata"               # Path where files will be loaded for data augmentation
ORIGINAL_TEST_DIR = "Bardus_thick_labeled_data/Test"                        # Path where files will be loaded for data augmentation
AUGMENTED_DIR = "Bardus_thick_labeled_data/random"              # Path where the newly generated files will be saved
AUGMENTED_TEST_DIR = "Bardus_thick_labeled_data/Test_preprosessed"           # Path where the newly generated files will be saved
LOAD_DATA_TO_MODEL = "Bardus_thick_labeled_data/gray_traindata"         # This data will be loaded for model training.
LOAD_MODEL_FILE = "models/CNN_model_20batsz_1711018502.h5"                  # This path will be needed only if TRAIN_MODEL = 0
LOAD_TEST_DATA = "Bardus_thick_labeled_data/Test"                           # Path for test data which will be used for end results

### 1) Crete data module settings
# How much augmented data is wanted
CREATE_AUGMENTED_SAMPLES_PER_CLASS = 900

#NOISE_LEVEL
NOISE_LEVEL = 0 # 0 is no noise, 0.1 is alot of noise.
# BRIGHTNESS
ADD_BRIGHTNESS = 0 # 0 is no extra brightness
# CLAHE
ADD_CLAHE = 0 # 0 is no CLAHE, 1 is CLAHE
# ROTATION YES/NO
ANGLE_ROTATION = 1 # 0 no angle rotation, 1 = yes rotation (+/- 20degrees). If chosen 0 = no, image will only be rotated randomly 90, 180 degree or mirrored vertically or horizontally.

# CREATE_DATA = 1, Do you want to generate all data? # 1 = generate all data, 0 do not generate all data
GENE_ALL_DATA = 1
# GENE_ALL_DATA = 0, If not all data is wanted, what data is wanted? generate 1, do not generate 0
BLACK = 1
BLACKINGLASS = 1
BROKENFEEDTHROUGH = 1
DRILL = 1
FEEDTHROUGH= 1
METAL = 1
MISSINGFEEDTHROUGH = 1
SCRATCH = 1
SILICON = 1


### 2) Train data module settings
# For basic model
CONV_FILTER_1 = 128
CONV_FILTER_2 = 256
CONV_FILTER_3 = 512
DENSE_UNIT_1 = 128
DENSE_UNIT_2 = 64

CONV_DROPOUT = 0.3
DENSE_DROPOUT_1 = 0.4
DENSE_DROPOUT_2 = 0.2

EPOCHS = 20                         # int
LEARNING_RATE = 0.001                # default value is 0.001

IMAGE_SIZE = (128, 128)             # tuple
CROPPED_IMAGE_SIZE = (84, 84)       # tuple
BATCH_SIZE = 100                    # int

TRAINING_PATIENCE = 5               # int


### Do not adjust code under this line ###

tf.random.set_seed(seed_value)
# Data creation module
#GENERATE_NOISE = 1
obj_create_all_data = create_data_all(CREATE_AUGMENTED_SAMPLES_PER_CLASS, IMAGE_SIZE, ORIGINAL_DIR, AUGMENTED_DIR, NOISE_LEVEL, ADD_BRIGHTNESS, ADD_CLAHE, ANGLE_ROTATION)

if GENERATE_TEST_DATA == 1:
    # for test set preparation
    obj_create_test_all_data = create_data_all(CREATE_AUGMENTED_SAMPLES_PER_CLASS, IMAGE_SIZE, ORIGINAL_TEST_DIR, AUGMENTED_TEST_DIR, NOISE_LEVEL, ADD_BRIGHTNESS, ADD_CLAHE, ANGLE_ROTATION)
    obj_create_test_all_data.preprocess_all_test_data()

if CREATE_DATA == 1:

    if GENE_ALL_DATA == 1:
        obj_create_all_data.black_data()
        obj_create_all_data.blackinglass()
        obj_create_all_data.brokenfeed()
        obj_create_all_data.drill()
        obj_create_all_data.feed_data()
        obj_create_all_data.metal()
        obj_create_all_data.missingfeed()
        obj_create_all_data.scratch()
        obj_create_all_data.silicon()

    else:
        if BLACK == 1:
            obj_create_all_data.black_data()
        if BLACKINGLASS == 1:
            obj_create_all_data.blackinglass()
        if BROKENFEEDTHROUGH == 1:
            obj_create_all_data.brokenfeed()
        if DRILL == 1:
            obj_create_all_data.drill()
        if FEEDTHROUGH == 1:
            obj_create_all_data.feed_data()
        if METAL == 1:
            obj_create_all_data.metal()
        if MISSINGFEEDTHROUGH == 1:
            obj_create_all_data.missingfeed()
        if SCRATCH == 1:
            obj_create_all_data.scratch()
        if SILICON == 1:
            obj_create_all_data.silicon()


# Training module
#self, 
# LOAD_DATA_TO_MODEL=str, CONV_FILTER_1=int, CONV_FILTER_2=int, CONV_FILTER_3=int, DENSE_UNIT_1=int, DENSE_UNIT_2=int, CONV_DROPOUT=float, DENSE_DROPOUT_1=float, DENSE_DROPOUT_2=float, EPOCHS=int, WEIGHT_INITIALIZER=str, LEARNING_RATE=float, IMAGE_SIZE=tuple, BATCH_SIZE=int, TRAINING_PATIENCE=int
if TRAIN_MODEL != 0:
    obj_training_cnn_model = training_cnn_model(
        LEARNING_RATE,
        LOAD_DATA_TO_MODEL,
        LOAD_TEST_DATA,
        CONV_FILTER_1,
        CONV_FILTER_2,
        CONV_FILTER_3,
        DENSE_UNIT_1,
        DENSE_UNIT_2,
        CONV_DROPOUT,
        DENSE_DROPOUT_1,
        DENSE_DROPOUT_2,
        EPOCHS,
        IMAGE_SIZE,
        CROPPED_IMAGE_SIZE,
        BATCH_SIZE,
        TRAINING_PATIENCE
        )
    # Test_generator is not used anywhere in this version
    train_generator, validation_generator, test_generator = obj_training_cnn_model.preprocess_for_training()


if TRAIN_MODEL == 1:
    model, history = obj_training_cnn_model.build_compile_train(train_generator, validation_generator)
    model_name = obj_training_cnn_model.save_model(model)
    obj_training_cnn_model.validation_evaluation_visualization(validation_generator, history)
    # preparation for the next step
    if TESTSET_PREDICTION_RESULTS == 1:
        save_trained_model = tf.keras.models.load_model(f"models/{model_name}.h5", custom_objects={"DuplicateChannelsLayer": DuplicateChannelsLayer})


elif TRAIN_MODEL == 2:
    # Train using ConvNextLarge pretrained model
    model, history = obj_training_cnn_model.build_cnn_cnl_compile_train(train_generator, validation_generator)
    model_name = obj_training_cnn_model.save_model(model)
    obj_training_cnn_model.validation_evaluation_visualization(validation_generator, history)
    # preparation for the next step
    if TESTSET_PREDICTION_RESULTS == 1:
        save_trained_model = tf.keras.models.load_model(f"models/{model_name}.h5", custom_objects={"DuplicateChannelsLayer": DuplicateChannelsLayer})
        

elif TRAIN_MODEL == 0:
    if TESTSET_PREDICTION_RESULTS == 1:
        # Load the pre-trained model for prediction
        save_trained_model = tf.keras.models.load_model(LOAD_MODEL_FILE, custom_objects={"DuplicateChannelsLayer": DuplicateChannelsLayer})
        model_name = LOAD_MODEL_FILE[-31:-3]
        print("")
        print("Loaded pre-trained model", LOAD_MODEL_FILE[-32:])
        print("")

else:
    print("")
    print("Choose TRAIN_MODEL 1 or 0 to be able to predict testset")
    print("")


# Prediction module and visualization module

if TESTSET_PREDICTION_RESULTS == 1:
    obj_prediction_results = prediction_and_results(save_trained_model, model_name, LOAD_TEST_DATA)
    obj_prediction_results.predictions_confidences_true_labels_to_csv()
    obj_prediction_results.plot_cm_heatmap()
    print("")
    print("Heatmap's and confidence csv file created for", model_name + ".h5")
    print("")
