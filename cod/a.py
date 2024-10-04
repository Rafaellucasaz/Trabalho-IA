import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

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



for batch_images, batch_labels in train_generator:
    # Exemplo: Acessando a primeira imagem do lote
    first_image = batch_images[0]  # A primeira imagem (forma: (28, 28, 3))
    
    # Verificando os dados da primeira imagem
    print("Dados da primeira imagem:")
    print(first_image[:28])  # Exibirá os valores normalizados entre 0.0 e 1.0
    print("Forma da imagem:", first_image.shape)  # Deve ser (28, 28, 3)
    
    # Exibindo os rótulos correspondentes
    first_label = batch_labels[0]  # O rótulo correspondente
    print("Rótulo da primeira imagem:", first_label)
    break;  # Um vetor de 10 elementos representando a classe