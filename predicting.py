import os
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

class prediction_and_results():
    """
    This module summarizes the results in .csv file and visualizes confusion matrix, and classification report
    """
    def __init__(self, trained_model, model_name=str, LOAD_TEST_DATA=str):

        # Load your model
        self.model = trained_model
        self.model_name = model_name

        # Label name and corresponding value (label)
        self.true_label_name_dict = {"Black": 0, "BlackInGlass": 1, "BrokenFeedthrough": 2, "Drill": 3, "Feedthrough": 4, "Metal": 5, "MissingFeedthrough": 6, "Scratch": 7, "Silicon": 8}

        # Load data
        self.load_test_data = LOAD_TEST_DATA

        self.predictions, self.confidences, self.true_labels = self.__predict_images_in_folder(self.load_test_data)

        # Convert true_labels to indices if they are not numeric
        self.label_to_index = {label: index for index, label in enumerate(sorted(set(self.true_labels)))}
        self.true_label_indices = [self.label_to_index[label] for label in self.true_labels]


    def __preprocess_image(self, image_path, target_size=(128, 128)):
        # Read the image file
        img = tf.io.read_file(image_path)
        # Decode the image to tensor and apply color conversion
        img = tf.image.decode_jpeg(img, channels=1)
        # uint to float to be able to calculate
        img = tf.cast(img, tf.float32) / 255.0

        # model expects batch size
        img = np.expand_dims(img, axis=0)   # add a batch dimension

        # Resize
        img = tf.image.resize(img, [target_size[0], target_size[1]])
        return img

        # # Load the image
        # img = cv2.imread(image_path)
        # # Resize the image
        # img = cv2.resize(img, target_size)
        # # Convert the image to grayscale
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)   # For GRAYSCALE images
        # # Convert the image to a 0-1 range
        # img = img / 255.0
        # Reshape the image 
        #img = np.expand_dims(img, axis=0)   # add a batch dimension
        #img = np.expand_dims(img, axis=-1)  # Add channel dimension, NO NEEED IF WORKING WITH 3 channel images
        #return img

    def __predict_images_in_folder(self, folder_path):
        predictions = []
        confidences = []
        true_labels = []
        for class_folder in os.listdir(folder_path):
            class_folder_path = os.path.join(folder_path, class_folder)
            if os.path.isdir(class_folder_path):
                for img_file in os.listdir(class_folder_path):
                    img_path = os.path.join(class_folder_path, img_file)
                    img = self.__preprocess_image(img_path)
                    print(img.shape)
                    prediction = self.model.predict(img)
                    predicted_class = np.argmax(prediction)
                    confidence = np.max(prediction)
                    predictions.append(predicted_class)
                    confidences.append(confidence)
                    true_labels.append(class_folder) 
        return predictions, confidences, true_labels

    def __invert_dict(self, d):
            return {v: k for k, v in d.items()}

    def predictions_confidences_true_labels_to_csv(self):
        # gather up data
        #predictions, confidences, true_labels = self.__predict_images_in_folder(self.load_test_data)

        # Invert the true_label_name_dict to map indices back to names
        index_to_label_name = self.__invert_dict(self.true_label_name_dict)

        # Map predicted class indices to names
        prediction_labels = [index_to_label_name[pred] for pred in self.predictions]

        # Create a DataFrame from the lists
        data = {
            "Prediction": prediction_labels,
            "Confidence": self.confidences,
            "True Label": self.true_labels,
            "Correct": [prediction_labels[i] == self.true_labels[i] for i in range(len(self.predictions))]
        }

        df = pd.DataFrame(data)

        # Save the DataFrame to a CSV file
        csv_file_path = f"results/predictions_confidences_true_labels_{self.model_name[-31:-3]}.csv"
        df.to_csv(csv_file_path, index=False)


    def plot_cm_heatmap(self):
        # Generate the confusion matrix for heatmap
        self.cm = confusion_matrix(self.true_label_indices, self.predictions)

        plt.figure(figsize=(12, 10))  # Set figure size
        ax = sns.heatmap(self.cm, annot=False, fmt="d", cmap="Blues", xticklabels=self.label_to_index.keys(), yticklabels=self.label_to_index.keys(), cbar=True)
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted Labels")
        plt.ylabel("True Labels")
        plt.xticks(rotation=45, fontsize=8)
        plt.yticks(rotation=0, fontsize=8)
        plt.tight_layout()

        # Calculate the percentage and add to heatmap
        for i in range(self.cm.shape[0]):
            for j in range(self.cm.shape[1]):
                total = np.sum(self.cm[i])  # Support amount of the specific class
                percentage = f'{self.cm[i, j]/total:.1%}' if total > 0 else '0%'
                text_color = "white" if i == j else "black"  # White for diagonal (true positives), black for others
                ax.text(j + 0.5, i + 0.5, f'{self.cm[i, j]}\n({percentage})', ha="center", va="center", 
                        color=text_color, fontsize=8)

        plt.savefig(f"results/Heatmap_cm_{self.model_name[-31:-3]}")  # Save the figure to file

    
    def classification_report(self):
        # Classification report for detailed analysis
        report = classification_report(self.true_label_indices, self.predictions, target_names = self.label_to_index.keys(), output_dict=True)
        report_df = pd.DataFrame(report).transpose()

        # Drop the 'support' column for visualization purposes
        report_df.drop(columns='support', inplace=True, errors='ignore')

        # Plotting
        # Precision, how many correct answers
        # Recall how many true positives
        # f1 score evaluates both false negatives and true positives, mathematically f1 score = (2 * precision * recall) / (precision + recall)
        plt.figure(figsize=(10, 8))
        sns.heatmap(report_df, annot=True, cmap='Blues', fmt='.2f', cbar=True)
        plt.title('Classification Report Heatmap')
        plt.savefig(f"results\\Heatmap_report_{self.model_name[-31:-3]}")
