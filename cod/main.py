import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import os

dataPath = "../dataset/n"

dataGen = ImageDataGenerator(rescale = 1./255,validation_split =0.2)

train_generator =dataGen.flow_from_directory(
    dataPath,
    target_size=(28,28),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

validation_generator = dataGen.flow_from_directory(
    dataPath,
    target_size=(28, 28),
    batch_size=32,
    class_mode='categorical',  
    subset='validation'
)

model = Sequential()


model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))


model.add(Dense(15, activation='softmax'))

# Compilar o modelo
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


model.summary()


history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // validation_generator.batch_size,
    epochs=10 
)


loss, accuracy = model.evaluate(validation_generator)
print(f'Acurácia do modelo: {accuracy * 100:.2f}%')





img_dir = '../test/'

# caminho das imagens
img_paths = [os.path.join(img_dir, img_name) for img_name in os.listdir(img_dir)] 


#array para guardar valores preditos 
predictions = []

#carregar e prever valores da imagens
for img_path in img_paths:
  
    img = image.load_img(img_path, target_size=(28, 28))
    
  
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  
    img_array /= 255.0 

   
    prediction = model.predict(img_array)
    
   
    predicted_class = np.argmax(prediction)
    
   
    predictions.append(predicted_class)



for i, img_path in enumerate(img_paths):
    print(f"Imagem: {img_path}, Classe prevista: {predictions[i]}")




n = -1
i = 0
n1 = ''
n2 = ''

#loop para pegar dígitos até chegar no símbolo de operação
while n <=10:
    n1 = n1 + str(predictions[i])
    n = predictions[i+1]
    i +=1


i +=1
#loop para pegar dígitos depois do símbolo de operação
for j in range(i, len(predictions)):  
    n2 += str(predictions[j])
    if predictions[j] >= 10:
        n = -1  
        break 


n1 = int(n1)
n2 = int(n2)
op = n



#dictionary para definir operação a ser feita 
operations = {
    10: lambda a, b: a + b,
    11: lambda a, b: a / b if b != 0 else "Divisão por zero",
    12: lambda a, b: a == b,
    13: lambda a, b: a * b,
    14: lambda a, b: a - b
}
   
n3 = operations.get(op, lambda a, b: "Operação inválida")(n1, n2)


print(f'resultado: {n3}' )


