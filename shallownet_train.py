from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imageToArrayPreprocessor import ImageToArrayPreprocessor
from simplePreprocessor import SimplePreprocessor
from simpleDatasetLoader import  SimpleDatasetLoader

from MyNetworks.lenet import LeNet
from tensorflow.keras.optimizers import SGD
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np

path='animals//animals'
model_path= 'lenet_animal_weights.hdf'
print("[INFO] Loading images...")
image_paths= list(paths.list_images(path))

sp =SimplePreprocessor(32, 32)
iap= ImageToArrayPreprocessor()
sdl= SimpleDatasetLoader(preprocessors=[sp, iap])

(data, labels)= sdl.load(image_paths, verbose=500)
data= data.astype('float')/255.0

(trainX, testX, trainY, testY)= train_test_split(data, labels, test_size= 0.25, random_state= 42)

trainY= LabelBinarizer().fit_transform(trainY)
testY= LabelBinarizer().fit_transform(testY)

print('[INFO] Compiling model...')
model= LeNet.build(width=32, height=32, depth=3, classes=3)
model.compile(loss='categorical_crossentropy', optimizer=SGD(0.005), metrics=['accuracy'])

print('[INFO] Training...')
H= model.fit(trainX, trainY, validation_data=(testX, testY), batch_size=32, epochs=100, verbose=1)

print('[INFO] Serializing...')
model.save(model_path)

print('[INFO] Evaluating...')
preds= model.predict(testX, batch_size=32)

print(classification_report(testY.argmax(axis=1),
 preds.argmax(axis=1),
 target_names=["cat", "dog", "panda"]))

plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, 100), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, 100), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, 100), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, 100), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.show()