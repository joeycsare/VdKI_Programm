import tensorflow as tf
import matplotlib.pyplot as plt
import random as rnd
from tensorflow.keras import layers
import numpy as np
import os

batch_size = 4 # Anzahl der Bilder die in einem Batch verarbeitet werden
# Bildgröße beim Import (in Pixel)
img_height = 255
img_width = 255

# Trainingsdaten vom Unterordner "bilder" einlesen
#path = os.path.join("C:\\","Users","ENSMM","Desktop","Master WS 22","Master WS 21","VdKI","KI_Projekt Abgaben","Bilder"),
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
  "Bilder",
  validation_split=0.2, # Anteil der Bilder, die zum Trainieren verwendet werden (hier: 20%)
  subset="training",
  seed=123, # seed for shuffling and transformations
  image_size=(img_height, img_width),
  batch_size=batch_size),

# Validierungsdaten vom Unterordner "bilder" einlesen
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
  "Bilder",
  validation_split=0.2, # Anteil der Bilder, die zum Trainieren verwendet werden (hier: 20%)
  subset="validation",
  seed=123, # seed for shuffling and transformations
  image_size=(img_height, img_width),
  batch_size=batch_size)

# Klassenbezeichnungen definieren und ausgeben
# Klassen werden über die Unterordnernamen im Ordner "bilder" definiert

class_names = train_ds.class_names
print(class_names)

# die ersten 9 eingelesenen Bilder darstellen (nur Kontrolle ob richtig eingelesen; wird für CNN nicht benötigt)
"""
plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
  for i in range(8):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(images[i].numpy().astype("uint8"))
    plt.title(class_names[labels[i]])
    plt.axis("off")
plt.show()
"""

# Ausgeben der Batch-Size und der Bildergröße (nur Kontrolle ob richtig eingelesen; wird für CNN nicht benötigt)
for image_batch, labels_batch in train_ds:
  print(image_batch.shape)
  print(labels_batch.shape)
  break

# Standardisierung der Daten (Angleichung des Farbraums auf Werte von 0 bis 1) --> Ziel: kleinere Eingabewerte
normalization_layer = tf.keras.layers.experimental.preprocessing.Rescaling(1./255)

# Leistung verbessern durch Laden der Bilder in den Speicher
AUTOTUNE = tf.data.experimental.AUTOTUNE
train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# Modelldefinition
# 2 Convolutional Layers mit jeweils folgender Max-Pooling-Schicht, dann Flatten und zwei Dense-Layers
# Aktivierungsfunktion relu: Ausgabe aller positiven Werte; negative Werte --> 0)
num_classes = 2 # Anzahl der Klassen 2 (Locher und Tacker)
model = tf.keras.Sequential([
  layers.experimental.preprocessing.Rescaling(1./255),
  layers.Conv2D(32, 3, activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(64, 3, activation='relu'),
  layers.MaxPooling2D(),
  layers.Flatten(),
  layers.Dense(128, activation='relu'),
  layers.Dense(num_classes)
])

# Modell kompilieren
model.compile(optimizer='adam', loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])

# Modell trainieren und Testen anhand der oben definierten Validierungsdaten, Anzahl der Durchläufe durch Epochenzahl festgelegt
model.fit(train_ds, validation_data=val_ds, epochs=1)


# Vorhersage erstellen für alle Bilder im Validierungssatz und in das Array predictions[] speichern
probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
predictions = probability_model.predict(val_ds)
# Batch für Testbilder
batch_number = 14
# Anzeigen des zufällig ausgewählten Bildes
plt.figure(figsize=(5, 2))
plt.axis("off") # Keine Achsen im Plot anzeigen
ax = plt.subplot(1, 3, 1)  # Erstellung eines Subplots/Platzhalter für das Bild

# Testbilder Anzeigen mit Vorhersage: Immer ein Batch
for i in range(0,4):
  for images, labels in val_ds.take(batch_number):     # .take(?) --> ?: Batch Nummer (Alle Bilder werden in Batches unterteilt abhängig von der Variable batch_size
    plt.imshow(images[i].numpy().astype("uint8"))  # Einfügen/Plotten des Bildes
    title = " Prob(Locher): " + str(round(predictions[((batch_number)*batch_size)+i,0],2)) + " Prob(Tacker): " + str(round(predictions[((batch_number)*batch_size)+i,1],2)) # Titel des Bildes für Ausgabe der Wahrscheinlichkeiten
    plt.title(title, loc="left")
    plt.suptitle("Es sollte sein: " + str(class_names[labels[i]]))  # Subtitel für Ausgabe der Vorraussage
  plt.show()  # Fenster anzeigen
