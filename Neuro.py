import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator


class PlantDiseaseClassifier:
    def __init__(self):
        self.database_path = r"C:\Users\user\Desktop\the_database"
        self.camera_path = r"C:\Users\user\Desktop\photos_from_the_camera"
        self.img_size = 150
        self.batch_size = 32
        self.epochs = 15
        self.model = None
        self.class_names = ['chlorosis', 'dandelion', 'healthy_leaf', 'plantain', 'powdery_mildew']

        self.create_model()

    def load_and_preprocess_data(self):
        """Загрузка и подготовка данных"""
        images = []
        labels = []

        for class_name in self.class_names:
            class_path = os.path.join(self.database_path, f'{class_name}.jpg')
            img = cv2.imread(class_path)
            if img is not None:
                img = cv2.resize(img, (self.img_size, self.img_size))
                img = img / 255.0
                images.append(img)
                labels.append(self.class_names.index(class_name))

        datagen = ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest'
        )

        images = np.array(images)
        labels = np.array(labels)

        X_train, X_test, y_train, y_test = train_test_split(
            images, labels, test_size=0.2, random_state=42)

        return (X_train, y_train), (X_test, y_test), datagen

    def create_model(self):
        """Создание модели CNN"""
        self.model = models.Sequential([
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=(self.img_size, self.img_size, 3)),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dense(len(self.class_names), activation='softmax')
        ])

        self.model.compile(optimizer='adam',
                           loss='sparse_categorical_crossentropy',
                           metrics=['accuracy'])

    def train_model(self):
        """Обучение модели"""
        (X_train, y_train), (X_test, y_test), datagen = self.load_and_preprocess_data()

        train_generator = datagen.flow(X_train, y_train, batch_size=self.batch_size)

        history = self.model.fit(
            train_generator,
            steps_per_epoch=len(X_train) // self.batch_size,
            epochs=self.epochs,
            validation_data=(X_test, y_test)
        )

        self.model.save('plant_disease_model.h5')
        return history

    def load_trained_model(self):
        """Загрузка предварительно обученной модели"""
        if os.path.exists('plant_disease_model.h5'):
            self.model = tf.keras.models.load_model('plant_disease_model.h5')
            return True
        return False

    def predict_image(self, img_path):
        """Предсказание для нового изображения"""
        if not self.model:
            if not self.load_trained_model():
                print("Модель не обучена. Сначала обучите модель.")
                return None

        img = cv2.imread(img_path)
        if img is None:
            print(f"Не удалось загрузить изображение: {img_path}")
            return None

        img = cv2.resize(img, (self.img_size, self.img_size))
        img = img / 255.0
        img = np.expand_dims(img, axis=0)

        predictions = self.model.predict(img)
        predicted_class = np.argmax(predictions[0])
        confidence = np.max(predictions[0])

        return self.class_names[predicted_class], confidence

    def analyze_camera_images(self):
        """Анализ всех изображений из папки камеры"""
        results = []

        for img_name in os.listdir(self.camera_path):
            img_path = os.path.join(self.camera_path, img_name)
            if img_path.lower().endswith(('.png', '.jpg', '.jpeg')):
                prediction, confidence = self.predict_image(img_path)
                if prediction:
                    results.append({
                        'image': img_name,
                        'prediction': prediction,
                        'confidence': float(confidence),
                        'description': self.get_description(prediction)
                    })

        return results

    def get_description(self, class_name):
        """Получение описания по названию класса"""
        descriptions = {
            'chlorosis': "Заболевание: хлороз (желтые листья)",
            'dandelion': "Сорняк: одуванчик",
            'healthy_leaf': "Здоровое растение",
            'plantain': "Сорняк: подорожник",
            'powdery_mildew': "Заболевание: мучнистая роса (белые пятна)"
        }
        return descriptions.get(class_name, "Неизвестное состояние растения")


def main():
    classifier = PlantDiseaseClassifier()

    if not classifier.load_trained_model():
        print("Обучение модели...")
        classifier.train_model()
        print("Модель успешно обучена и сохранена.")

    results = classifier.analyze_camera_images()

    print("\nРезультаты анализа:")
    for result in results:
        print(f"\nИзображение: {result['image']}")
        print(f"Результат: {result['description']}")
        print(f"Точность: {result['confidence']:.2%}")
        print(f"Класс: {result['prediction']}")


if __name__ == "__main__":
    main()