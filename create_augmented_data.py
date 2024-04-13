import tensorflow as tf
import tensorflow_addons as tfa
import os
import numpy as np
import cv2

class data_maker:
    def __init__(self, class_name=str, create_augmented_samples_per_class=int, IMAGE_SIZE=tuple, ORIGINAL_DIR=str, AUGMENTED_DIR=str, NOISE_LEVEL=float, ADD_BRIGHTNESS=int, ADD_CLAHE=int, ANGLE_ROTATION=int):

        self.class_name = class_name
        self.image_size = IMAGE_SIZE
        self.create_augmented_samples_per_class = create_augmented_samples_per_class

        self.ORIGINAL_DIR = os.path.join(ORIGINAL_DIR, f"{self.class_name}")
        self.AUGMENTED_DIR = os.path.join(AUGMENTED_DIR, f"aug_{self.class_name}")
        self.NOISE_LEVEL = NOISE_LEVEL
        self.ADD_BRIGHTNESS = ADD_BRIGHTNESS
        self.ADD_CLAHE = ADD_CLAHE
        self.ANGLE_ROTATION = ANGLE_ROTATION

        if not os.path.exists(self.AUGMENTED_DIR):
            os.makedirs(self.AUGMENTED_DIR)


    def _add_noise_to_image(self, image):
        # Convert image to float32
        image_float = tf.cast(image, tf.float32)
        
        # Normalize image to 0-1 if it's originally in 0-255
        image_normalized = image_float / 255.0 if tf.reduce_max(image_float) > 1.0 else image_float
        
        stddev = self.NOISE_LEVEL  # Adjust the standard deviation
        noise = tf.random.normal(shape=tf.shape(image_normalized), mean=0.0, stddev=stddev, dtype=tf.float32)
        
        # Add the noise
        image_noisy = image_normalized + noise
        
        # Clip values to ensure they're between 0 and 1
        image_noisy_clipped = tf.clip_by_value(image_noisy, 0.0, 1.0)
        
        # Convert back to uint8
        image_noisy_uint8 = tf.cast(image_noisy_clipped * 255, tf.uint8)
        return image_noisy_uint8

    def _apply_clahe(self, image: np.ndarray, clahe_iterations=1) -> np.ndarray:
        """Apply CLAHE to a grayscale image.
           adaptive histogram equalization can solve problems with high contrast differences"""
        # Create a CLAHE object (Arguments are optional)
        clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8, 8))

        for _ in range(clahe_iterations):
            clahe_img = clahe.apply(image)
        return clahe_img
    

    def image_rotation(self, image):
        """ Apply a random rotation of 0, 90, or 180 degrees or a vertical flip."""

        # Rotate by chance
        choice = np.random.choice(['None', 'rotate_90', 'rotate_180', 'vertical_flip', 'horizontal_flip']) 

        #These specific classes we dont want to rotate 90 degrees
        if choice == 'rotate_90':    #can be added: and self.class_name != "MissingFeedthrough" and self.class_name != "BrokenFeedthrough"
            image = tf.image.rot90(image, k=1)  # Rotate 90 degrees
        elif choice == 'rotate_180':
            image = tf.image.rot90(image, k=2)  # Rotate 180 degrees , 2 times 90 degree flip
        elif choice == 'vertical_flip':
            image = tf.image.flip_up_down(image)  # Vertical flip
        elif choice == 'horizontal_flip':
            image = tf.image.flip_left_right(image)
        else:
            pass

        return image


    def angle_rotate(self, img_tensor, degree_range=(-20, 20)):
        # Generate a random angle in radians
        angle_degrees = np.random.uniform(degree_range[0], degree_range[1])
        angle_radians = angle_degrees * np.pi / 180.0

        # Rotate the image
        rotated_img_tensor = tfa.image.rotate(img_tensor, angles=angle_radians, interpolation='BILINEAR')

        return rotated_img_tensor

    def shear_image(self, image, shear_range=(0, 0.1), axis=0):
        """
        Apply a shearing transformation to an image.
        """
        #image_shape = tf.shape(image)
        #height, width = image_shape[0], image_shape[1]

        shear_factor = np.random.uniform(shear_range[0], shear_range[1])
        
        if axis == 0:  # Horizontal shear
            shear_matrix = [1, shear_factor, 0, 0, 1, 0, 0, 0]
        else:  # Vertical shear
            shear_matrix = [1, 0, 0, shear_factor, 1, 0, 0, 0]
        
        # Apply the transformation
        sheared_image = tfa.image.transform(
            images=image,
            transforms=shear_matrix,
            interpolation='BILINEAR'
        )
        
        return sheared_image

    def random_zoom_image(self, image, zoom_range=(0.6, 1.0)):
        """Random amout of zoom, 1 is no zoom"""
        
        original_size = tf.cast(tf.shape(image)[:2], tf.float32)
    
        # Generate a random zoom factor from zoom range
        zoom_factor = np.random.uniform(zoom_range[0], zoom_range[1])
        
        # Crop the central portion of the image
        image_cropped = tf.image.central_crop(image, zoom_factor)
        
        # Resize the cropped portion back to the original size
        image_zoomed_in = tf.image.resize(image_cropped, size=tf.cast(original_size, tf.int32))
        
        return image_zoomed_in

    def adjust_brightness(self, image, brightness_delta=0.1):
        """ Adjust brightness of the image"""
        # Load the image
        #image = tf.io.read_file(image_path)
        #image = tf.image.decode_image(image, channels=3, expand_animations=False)
        image = tf.image.convert_image_dtype(image, tf.float32)  # Convert to float values in [0, 1]
        
        # Adjust brightness
        brighter_image = tf.image.adjust_brightness(image, delta=brightness_delta)
        
        # Convert back to uint8
        brighter_image_uint8 = tf.image.convert_image_dtype(brighter_image, tf.uint8)
        
        # Use TensorFlow IO or another method to save or display the image
        return brighter_image_uint8

    def read_and_preprocess_image(self, file_path):
        """Read and preprocess image using TensorFlow."""
        # Read the image file
        img = tf.io.read_file(file_path)
        # Decode the image to tensor and apply color conversion
        image = tf.image.decode_jpeg(img, channels=1)  # Use channels=1 for grayscale, and  =3 for rbg

        if self.ADD_BRIGHTNESS == 1:
            # Optionally adjust brightness
            image = self.adjust_brightness(image)

        if self.ADD_CLAHE == 1:
            #Convert tensor to numpy for CLAHE
            img_numpy = image.numpy().squeeze()  # Remove batch dimension and convert to numpy
            img_numpy = img_numpy.astype(np.uint8)  # Scale to 0-255 for CLAHE
            # Apply CLAHE
            img_clahe = self._apply_clahe(img_numpy)

            # Convert back to tensor, rescale to 0-1, and add channel dimension
            img_clahe = tf.convert_to_tensor(img_clahe, dtype=tf.float32) / 255.0
            image = tf.expand_dims(img_clahe, axis=-1)  # Add the channel dimension back

        # Resize images if necessary
        img = tf.image.resize(image, [self.image_size[0], self.image_size[1]])

        # Noise to image
        img = self._add_noise_to_image(img)

        return img

    def _generate_unique_filename(self, original_filename, augmentation_index):
        """Generate unique filename"""
        base_filename = original_filename.split('.')[0]
        counter = 1  # Start with 1 for the first augmented file
        while True:
            augmented_filename = f"{base_filename}_aug_{augmentation_index}_{counter}.jpg"
            augmented_img_path = os.path.join(self.AUGMENTED_DIR, augmented_filename)
            if not os.path.exists(augmented_img_path):
                return augmented_filename
            counter += 1  # Increment the counter and try with a new suffix if the file exists

    def generate(self):
        """Dynamically make as same amount of images for each class"""
        # Calculate the current number of images in the class.
        current_image_count = len([name for name in os.listdir(self.ORIGINAL_DIR) if name.endswith(".jpg")])
        
        # Calculate how many times each image needs to be augmented to reach the target.
        # Using max to avoid division by zero and ensure at least one augmentation for very small classes.
        augmentations_needed = max(self.create_augmented_samples_per_class // current_image_count, 1)
        
        for filename in os.listdir(self.ORIGINAL_DIR):
            if filename.endswith(".jpg"):
                original_img_path = os.path.join(self.ORIGINAL_DIR, filename)

                # Initialize a counter for how many augmentations have been generated for this image.
                counter = 0
                for augmentation_index in range(augmentations_needed):
                    img = self.read_and_preprocess_image(original_img_path)
                    
                    img = self.image_rotation(img) # 90, 180, hor, verz degree rotation
                    img = self.shear_image(img)
                    img = self.random_zoom_image(img)
                    
                    if self.ANGLE_ROTATION == 1:
                        img = self.angle_rotate(img)  # +/- 20 degree rotation 
                    
                    # Prepare the image for saving.
                    img_uint8 = tf.cast(img, tf.uint8)
                    # Encode the image as JPEG.
                    img_encoded = tf.io.encode_jpeg(img_uint8, quality=100)
                    
                    # Generate a unique file name for each augmented image.
                    augmented_filename = self._generate_unique_filename(filename, augmentation_index)
                    augmented_img_path = os.path.join(self.AUGMENTED_DIR, augmented_filename)

                    # Save the augmented image.
                    tf.io.write_file(augmented_img_path, img_encoded)

                    counter += 1
    
    def preprocess_and_save_test_data(self, source_dir, destination_dir):
        """Preprocess test images from categorized folders and save them elsewhere."""
        if not os.path.exists(destination_dir):
            os.makedirs(destination_dir)

        categories = [d for d in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, d))]

        for category in categories:
            category_path = os.path.join(source_dir, category)
            dest_category_path = os.path.join(destination_dir, "aug_" + category)

            if not os.path.exists(dest_category_path):
                os.makedirs(dest_category_path)

            for filename in os.listdir(category_path):
                if filename.endswith(".jpg"):
                    original_img_path = os.path.join(category_path, filename)
                    img = self.read_and_preprocess_image(original_img_path)

                    # Prepare image for saving
                    img_uint8 = tf.cast(img, tf.uint8)
                    # Encode the image as JPEG
                    img_encoded = tf.io.encode_jpeg(img_uint8, quality=100, format='grayscale')

                    # Save the preprocessed image
                    test_img_path = os.path.join(dest_category_path, filename)
                    tf.io.write_file(test_img_path, img_encoded)
            

        print(f"{category} - Test data preprocessing complete and saved to:", destination_dir)