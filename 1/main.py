import numpy as np


def activation(x):
    return 0 if x < 0.5 else 1


def go(water, milk, tea):
    x = np.array([water, milk, tea])
    w11 = [0.3, 0.3, 0]
    w12 = [0.4, -0.5, 1]
    weight1 = np.array([w11, w12])
    weight2 = np.array([-1, 1])

    sum_hidden = np.dot(weight1, x)
    print(f'Значение сумм на нейронах скрытого слоя: {str(sum_hidden)}')

    out_hidden = np.array([activation(x) for x in sum_hidden])
    print(f'Значение сумм на выходах нейронов скрытого слоя: {str(out_hidden)}')

    sum_end = np.dot(weight2, out_hidden)
    y = activation(sum_end)
    print(f'Выходное значение НС: {str(y)}')

    return y


if __name__ == '__main__':
    water = 1
    milk = 0
    tea = 1

    result = go(water, milk, tea)

    if result == 1:
        print('Огонь')
    else:
        print('Ну это никуда не идёт')
