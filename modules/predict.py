# <YOUR_IMPORTS>
import json
import os
import dill
import pandas as pd

path = os.environ.get('PROJECT_PATH', '..')
def predict():
    # Загрузка Модели:
    file_name = f'{path}/data/models/cars_pipe.pkl'
    with open(file_name, 'rb') as file:
        object_to_load = dill.load(file)
    #print(object_to_load)

    # Перебор с=всех json testовых файлов:
    directory_path = f'{path}/data/test'  # Замените путь на вашу директорию

    # Получаем список файлов в директории
    file_list = os.listdir(directory_path)

    # Лист из словорей, чтоб в конце возвести в ДатаФрейм и передать, как csv:
    result = []

    # Перебираем файлы:
    for file_name in file_list:
        # Полный путь к файлу:
        file_path = os.path.join(directory_path, file_name)

        # Проверка на файл:
        if os.path.isfile(file_path):
            print(f"Найден файл: {file_path}")
            with open(file_path, 'r') as file:
                data = json.load(file)
                #print(data)

            # Возвёл в ДатаФрейм:
            df = pd.DataFrame([data])
            #print(df)

            # Предсказание:
            y = object_to_load.predict(df)
            #print(y)

            # Теперь всё в result.append() для csv:
            result.append({'ID': data['id'], 'Predict': y[0]}) # [0] - чтоб без квадратных скобок сохранять

    #print(result)

    # Ответ: предсказания в один Dataframe и сохраняет их в csv-формате в папку data/predictions
    otv = pd.DataFrame(result)
    #print(otv)

    # Сохранить ответ в data/predictions:
    otv.to_csv(f'{path}/data/predictions/result.csv', index=False)


if __name__ == '__main__':
    predict()
