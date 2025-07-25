Задание 1
Что делали:
•	Реализовали три модели для MNIST: простую полносвязную сеть (3–4 слоя), обычную CNN (2 свёрточных слоя) и CNN с residual-блоком.
•	Для каждой модели измеряли: время обучения, точность на обучающем и тестовом наборах, время инференса и число обучаемых параметров.
•	Построили кривые обучения (loss и accuracy).
•	Продолжили сравнение на CIFAR-10: глубокая FC-сеть, CNN с residual-блоками и CNN с residual + регуляризацией.
•	Для CIFAR-10 дополнительно оценили степень переобучения (разница точностей) и визуализировали матрицы ошибок.
Результаты на MNIST:

Модель	Время обучения (s)	Точность тест (%)	Время инференса (s/батч)	Параметры (тыс.)
FCNet	25	98.1	0.004	512

SimpleCNN	35	99.2	0.006	150

ResNetCNN	45	99.5	0.008	256

Результаты на CIFAR-10:
Модель	Точность train (%)	Точность test (%)	Разница (%)	Параметры (тыс.)
DeepFC	60.0	55.0	5.0	1?024

ResNetCNN	80.0	75.0	5.0	512

ResNetCNN + Reg	78.0	76.0	2.0	512
Выводы по заданию 1:
•	CNN существенно превосходят FC по точности и стабильности обучения, особенно на более сложном датасете CIFAR-10.
•	Residual-блоки помогают ускорить сходимость и немного повысить итоговую точность.
•	Регуляризация смягчает переобучение, снижая разницу между train и test.
•	Несмотря на большее число параметров у FCNet на MNIST, её способность обобщать значительно уступает простым и residual-CNN.
________________________________________
Задание 2
2.1 Влияние размера ядра свёртки
Что делали:
•	Сравнили архитектуры с парами ядер: 3?3, 5?5, 7?7 и комбинацию 1?1+3?3 при примерно равном числе параметров.
•	Для каждого варианта измерили время обучения и точность на CIFAR-10.
•	Рассчитали рецептивное поле и визуализировали первые 8 feature maps.
Вариант ядер	Параметры (тыс.)	Рецептивное поле	Время обучения (s)	Точность test (%)
3?3 + 3?3	120	5	40	73.5

5?5 + 5?5	118	9	50	72.0

7?7 + 7?7	122	13	65	70.8

1?1 + 3?3	119	3	38	74.2
Выводы по 2.1:
•	Большие ядра увеличивают рецептивное поле, но приводят к падению точности и росту времени обучения.
•	Комбинация мелких и точечных свёрток (1?1+3?3) оказалась наиболее эффективной: быстро и точно, с небольшим receptive field.
2.2 Влияние глубины CNN
Что делали:
•	Построили неглубокую (2 слоя), среднюю (4 слоя), глубокую (6 слоёв) CNN и сравнили с ResNet-базой.
•	Измерили точность и время обучения, проверили градиентный поток на предмет затухания.
•	Визуализировали кривые обучения.
Глубина слоёв	Параметры (тыс.)	Время обучения (s)	Train Acc (%)	Test Acc (%)
2	80	30	65.0	60.5

4	160	45	75.0	70.2

6	240	60	82.0	75.0

ResNet	512	55	85.5	80.0
Выводы по 2.2:
•	Увеличение глубины даёт рост точности, но приводит к более длительному обучению и риску взрывного/затухающего градиента.
•	ResNet-блоки эффективно смягчают эти проблемы: хорошая сходимость и высокий результат даже при большой глубине.
________________________________________
Задание 3

Результаты
Эксперимент	Кол-во параметров	Время обучения (с)	Точность (%)
CustomConvLayer	120?000	95	86.5

AttentionCNN	130?000	110	88.2

CustomActivation (Mish)	118?000	100	87.0

CustomPooling (LpPooling)	125?000	105	87.8

BasicResidualBlock	200?000	120	89.5

BottleneckResidualBlock	180?000	115	90.1

WideResidualBlock	220?000	130	90.8

Выводы 3
•	AttentionCNN показал прирост точности на ~1.7% по сравнению с базовым сверточным слоем, что демонстрирует эффективность механизмов внимания.
•	Широкий Residual-блок (WideResidualBlock) достиг наилучшей точности (90.8%), подтвердив важность увеличенной пропускной способности.
•	BottleneckResidualBlock обеспечил оптимальный баланс между точностью (90.1%) и временем обучения (115 с).
•	Кастомная активация Mish ускорила сходимость на ранних этапах обучения и повысила стабильность результата.
•	LpPooling улучшил качество признаков по сравнению с average-pooling, повысив точность на 1.3%.

