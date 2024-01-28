## Этот проект направлен на применение моделей YOLO5, Autoencoder и Unet

**Поставленные задачи**
  - Обучение и последующее применение модели YOLOv5 для ***детекции и классификации опухулей мозга***
  - Обучение и последующее применение модели автоэнкодера для ***очищения документов от шумов***
  - Обучение и последующее применение модели Unet для ***семантической сегментации леса с аэрокосмических снимков***

---
**Реализация:**
- Для обучения модели YOLOv5 был использован [датасет](https://www.kaggle.com/datasets/davidbroberts/brain-tumor-object-detection-datasets)
  
  ####
  
  - **Number of images for training:** 310 элементов
  - **Batch size:** 16
  - **Epoches:** 800

- Для обучения модели автоэнкодера был использован [датасет](https://drive.google.com/file/d/1LsHSn8dM8BTZ7EoWU6-n1I1BvR0p5tIx/view?usp=drive_link)
  
  ####
  
  - **Number of images for training:** 982 элемента
  - **Batch size:** 16
  - **Epoches:** 800
  
- Для обучения модели Unet был использован [датасет](https://www.kaggle.com/datasets/quadeer15sh/augmented-forest-segmentation)
  
  ####
  
  - **Number of images for training:** 5109 элементов
  - **Model:** однослойная
  - **Batch size:** 16
  - **Epoches:** 10
  - **Learning rate:** адаптивный

- **Деплой моделей на платформу для визуализации [Streamlit]()**

---
#### Проект был реализован командой [YerlanBaidildin](https://github.com/YerlanBaidildin), [Awlly](https://github.com/Awlly) и [sakoser](https://github.com/sakoser)
