from create_augmented_data import data_maker

class create_data_all():
    def __init__(self, create_augmented_samples_per_class=int, IMAGE_SIZE=tuple, ORIGINAL_DIR=str, AUGMENTED_DIR=str, GENERATE_NOISE=int, NOISE_LEVEL=float, ADD_BRIGHTNESS=int, ADD_CLAHE=int, ANGLE_ROTATION=int):

        self.create_augmented_samples_per_class = create_augmented_samples_per_class
        self.image_size = IMAGE_SIZE
        self.ORIGINAL_DIR = ORIGINAL_DIR
        self.AUGMENTED_DIR = AUGMENTED_DIR
        self.GENERATE_NOISE = GENERATE_NOISE
        self.NOISE_LEVEL = NOISE_LEVEL
        self.ADD_BRIGHTNESS = ADD_BRIGHTNESS
        self.ADD_CLAHE = ADD_CLAHE
        self.ANGLE_ROTATION = ANGLE_ROTATION

    def black_data(self):
        black_data_maker = data_maker("Black", self.create_augmented_samples_per_class, self.image_size, self.ORIGINAL_DIR, self.AUGMENTED_DIR, self.NOISE_LEVEL, self.ADD_BRIGHTNESS, self.ADD_CLAHE, self.ANGLE_ROTATION)
        black_data_maker.generate()
        print("1/9 black complete")

    def blackinglass(self):
        blackinglass_data_maker = data_maker("BlackInGlass", self.create_augmented_samples_per_class, self.image_size, self.ORIGINAL_DIR, self.AUGMENTED_DIR, self.NOISE_LEVEL, self.ADD_BRIGHTNESS, self.ADD_CLAHE, self.ANGLE_ROTATION)
        blackinglass_data_maker.generate()
        print("2/9 blackinglass complete")

    def brokenfeed(self):
        brokenfeed_data_maker = data_maker("BrokenFeedthrough", self.create_augmented_samples_per_class, self.image_size, self.ORIGINAL_DIR, self.AUGMENTED_DIR, self.NOISE_LEVEL, self.ADD_BRIGHTNESS, self.ADD_CLAHE, self.ANGLE_ROTATION)
        brokenfeed_data_maker.generate()
        print("3/9 brokenfeed complete")

    def drill(self):
        drill_data_maker = data_maker("Drill", self.create_augmented_samples_per_class, self.image_size, self.ORIGINAL_DIR, self.AUGMENTED_DIR, self.NOISE_LEVEL, self.ADD_BRIGHTNESS, self.ADD_CLAHE, self.ANGLE_ROTATION)
        drill_data_maker.generate()
        print("4/9 drill complete")

    def feed_data(self):
        feed_data_maker = data_maker("Feedthrough", self.create_augmented_samples_per_class, self.image_size, self.ORIGINAL_DIR, self.AUGMENTED_DIR, self.NOISE_LEVEL, self.ADD_BRIGHTNESS, self.ADD_CLAHE, self.ANGLE_ROTATION)
        feed_data_maker.generate()
        print("5/9 feed complete")

    def metal(self):
        metal_data_maker = data_maker("Metal", self.create_augmented_samples_per_class, self.image_size, self.ORIGINAL_DIR, self.AUGMENTED_DIR, self.NOISE_LEVEL, self.ADD_BRIGHTNESS, self.ADD_CLAHE, self.ANGLE_ROTATION)
        metal_data_maker.generate()
        print("6/9 metal complete")

    def missingfeed(self):
        missingfeed_data_maker = data_maker("MissingFeedthrough", self.create_augmented_samples_per_class, self.image_size, self.ORIGINAL_DIR, self.AUGMENTED_DIR, self.NOISE_LEVEL, self.ADD_BRIGHTNESS, self.ADD_CLAHE, self.ANGLE_ROTATION)
        missingfeed_data_maker.generate()
        print("7/9 missingfeed complete")

    def scratch(self):
        scratch_data_maker = data_maker("Scratch", self.create_augmented_samples_per_class, self.image_size, self.ORIGINAL_DIR, self.AUGMENTED_DIR, self.GENERATE_NOISE, self.NOISE_LEVEL, self.ADD_BRIGHTNESS, self.ADD_CLAHE, self.ANGLE_ROTATION)
        scratch_data_maker.generate()
        print("8/9 scratch complete")

    def silicon(self):
        silicon_data_maker = data_maker("Silicon", self.create_augmented_samples_per_class, self.image_size, self.ORIGINAL_DIR, self.AUGMENTED_DIR, self.GENERATE_NOISE, self.NOISE_LEVEL, self.ADD_BRIGHTNESS, self.ADD_CLAHE, self.ANGLE_ROTATION)
        silicon_data_maker.generate()
        print("9/9 silicon complete")

    def preprocess_all_test_data(self):
        preprocess_black_test_data = data_maker("Black", self.create_augmented_samples_per_class, self.image_size, self.ORIGINAL_DIR, self.AUGMENTED_DIR, self.GENERATE_NOISE, self.NOISE_LEVEL, self.ADD_BRIGHTNESS, self.ADD_CLAHE, self.ANGLE_ROTATION)
        preprocess_black_test_data.preprocess_and_save_test_data(self.ORIGINAL_DIR, self.AUGMENTED_DIR)
        preprocess_blackinglass_data_maker = data_maker("BlackInGlass", self.create_augmented_samples_per_class, self.image_size, self.ORIGINAL_DIR, self.AUGMENTED_DIR, self.GENERATE_NOISE, self.NOISE_LEVEL, self.ADD_BRIGHTNESS, self.ADD_CLAHE, self.ANGLE_ROTATION)
        preprocess_blackinglass_data_maker.preprocess_and_save_test_data(self.ORIGINAL_DIR, self.AUGMENTED_DIR)
        preprocess_brokenfeed_data_maker = data_maker("BrokenFeedthrough", self.create_augmented_samples_per_class, self.image_size, self.ORIGINAL_DIR, self.AUGMENTED_DIR, self.GENERATE_NOISE, self.NOISE_LEVEL, self.ADD_BRIGHTNESS, self.ADD_CLAHE, self.ANGLE_ROTATION)
        preprocess_brokenfeed_data_maker.preprocess_and_save_test_data(self.ORIGINAL_DIR, self.AUGMENTED_DIR)
        preprocess_drill_data_maker = data_maker("Drill", self.create_augmented_samples_per_class, self.image_size, self.ORIGINAL_DIR, self.AUGMENTED_DIR, self.GENERATE_NOISE, self.NOISE_LEVEL, self.ADD_BRIGHTNESS, self.ADD_CLAHE, self.ANGLE_ROTATION)
        preprocess_drill_data_maker.preprocess_and_save_test_data(self.ORIGINAL_DIR, self.AUGMENTED_DIR)
        preprocess_feed_data_maker = data_maker("Feedthrough", self.create_augmented_samples_per_class, self.image_size, self.ORIGINAL_DIR, self.AUGMENTED_DIR, self.GENERATE_NOISE, self.NOISE_LEVEL, self.ADD_BRIGHTNESS, self.ADD_CLAHE, self.ANGLE_ROTATION)
        preprocess_feed_data_maker.preprocess_and_save_test_data(self.ORIGINAL_DIR, self.AUGMENTED_DIR)
        preprocess_metal_data_maker = data_maker("Metal", self.create_augmented_samples_per_class, self.image_size, self.ORIGINAL_DIR, self.AUGMENTED_DIR, self.GENERATE_NOISE, self.NOISE_LEVEL, self.ADD_BRIGHTNESS, self.ADD_CLAHE, self.ANGLE_ROTATION)
        preprocess_metal_data_maker.preprocess_and_save_test_data(self.ORIGINAL_DIR, self.AUGMENTED_DIR)
        preprocess_missingfeed_data_maker = data_maker("MissingFeedthrough", self.create_augmented_samples_per_class, self.image_size, self.ORIGINAL_DIR, self.AUGMENTED_DIR, self.GENERATE_NOISE, self.NOISE_LEVEL, self.ADD_BRIGHTNESS, self.ADD_CLAHE, self.ANGLE_ROTATION)
        preprocess_missingfeed_data_maker.preprocess_and_save_test_data(self.ORIGINAL_DIR, self.AUGMENTED_DIR)
        preprocess_scratch_data_maker = data_maker("Scratch", self.create_augmented_samples_per_class, self.image_size, self.ORIGINAL_DIR, self.AUGMENTED_DIR, self.GENERATE_NOISE, self.NOISE_LEVEL, self.ADD_BRIGHTNESS, self.ADD_CLAHE, self.ANGLE_ROTATION)
        preprocess_scratch_data_maker.preprocess_and_save_test_data(self.ORIGINAL_DIR, self.AUGMENTED_DIR)
        preprocess_silicon_data_maker = data_maker("Silicon", self.create_augmented_samples_per_class, self.image_size, self.ORIGINAL_DIR, self.AUGMENTED_DIR, self.GENERATE_NOISE, self.NOISE_LEVEL, self.ADD_BRIGHTNESS, self.ADD_CLAHE, self.ANGLE_ROTATION)
        preprocess_silicon_data_maker.preprocess_and_save_test_data(self.ORIGINAL_DIR, self.AUGMENTED_DIR)