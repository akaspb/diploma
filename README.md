# Изучение нейронной сети PatchmatchNet

## Начало работы

Установить необходимые библиотеки:
```
pip install -r requirements.txt
```

Для работы потребуется подготовленный набор данных, ссылка для скачивания архива с набором данных DTU: 

[Скачать набор данных DTU](https://drive.google.com/file/d/1Al4BauJ25jqVShFGz60hgCDX7-ZNMJGo/view?usp=sharing)


## Демонстрация работы обученной модели

Параметры обученной нами модели располагаются в каталоге `diploma/model/`


Для демонстрации работы обученной нами модели необходимо выбрать сцену с одним освещением и запустить скрипт:
```
python eval.py --input_folder dtu/scan1/ --output_folder dtu_evalled/scan1/ --checkpoint_path diploma/model/params_000000.ckpt
```

Для демонстрации опорного изображения, истинной глубины и её оценки вызвать скрипт:
```
python diploma/show.py --input_folder dtu/scan1/ --output_folder dtu_evalled/scan1/ --image_idx 0
```

Можно добавить сохранение получаемых диаграмм:
```
python diploma/show.py --input_folder dtu/scan1/ --output_folder dtu_evalled/scan1/ --image_idx 0 --save=examples/scan1_0_0.png
```
