from imageToArrayPreprocessor import ImageToArrayPreprocessor
from simplePreprocessor import SimplePreprocessor
from simpleDatasetLoader import  SimpleDatasetLoader
from keras.models import load_model
from imutils import paths
import numpy as np
import cv2

model_path= 'shallow_weights.hdf'
path= 'animals/animals'
class_labels= ['cat', 'dog', 'panda']

print('[INFO] sampling images...')
image_paths= np.array(list(paths.list_images(path)))
idxs= np.random.randint(0, len(image_paths), size=(30,))
image_paths= image_paths[idxs]

sp= SimplePreprocessor(32, 32)
iap= ImageToArrayPreprocessor()

loader= SimpleDatasetLoader([sp, iap])
(data, labels)= loader.load(image_paths)

data= data.astype('float')/255

print('[INFO] loading model...')
model= load_model(model_path)

print('[INFO] predicting...')
preds= model.predict(data, batch_size=32).argmax(axis=1)

for (i, image_path) in enumerate(image_paths):
    image= cv2.imread(image_path)

    cv2.putText(image, f'Label: {class_labels[preds[i]]}',
        (10, 30), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0,255,0), 2)
    cv2.imshow('Image', image)
    cv2.waitKey(0)
cv2.destroyAllWindows()