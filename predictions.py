# 1. Import Dependencies
import os
from matplotlib import pyplot as plt
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow import keras

# =======================================================
# load unseen testing data

test_data_dir = rf'{os.getcwd()}/data/2750-test'

batch_size = 32
test_dataset = image_dataset_from_directory(
    test_data_dir,
    image_size=(64, 64),
    batch_size=batch_size,
    shuffle=False
)

# =======================================================
# inspect tensorflow testing dataset

# get class names and number of classes
class_names = test_dataset.class_names
num_classes = len(class_names)
print(f'number of classes: {num_classes}', class_names)

# =======================================================
# load trained land cover classification model

# load keras model and print summary
model = keras.models.load_model(f'{os.getcwd()}/land_cover_model')
model.summary()

# =======================================================
# make predictions

# iterate over the test dataset batches
for dataset in iter(test_dataset):
    # unpack a single batch of images and labels
    image_batch, label_batch = dataset

    # make predictions on test dataset
    y_prob = model.predict(dataset, verbose=1)

    # visualize 10 images from dataset
    plt.figure()
    for i in range(10):
        # retrieve ith image from current batch and show
        ax = plt.subplot(2, 5, i + 1)
        image = image_batch[i].numpy().astype("uint8")
        plt.imshow(image)
        plt.axis("off")  # turn off axis for clarity

        # index of highest probability indicates predicted class
        y_class = y_prob[i].argmax()

        # display image title with actual and predicted labels
        plt.title(f'Actual: {class_names[label_batch[i]][:10]},'
                  f'\nPredicted: {class_names[y_class][:10]}')

# show all figures
plt.show() 