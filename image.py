import matplotlib.pyplot as plt
import imageio

# Read the image using imageio
image = imageio.imread(r'F:\WAVI_classifier\Bardus_thick_labeled_data\random\aug_Silicon\image-707659579_1.jpg')

# Display the image
plt.imshow(image)
plt.axis('off')  # Optional: Removes the axis with numbers
plt.show()