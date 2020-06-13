# 问题： N个矩阵相乘，求一个计算顺序让总计算次数最少。
# 如矩阵$m_1$、$m_2$、$m_3$、$m_4$，分别是5x20、20x50、50x1、1x100的矩阵。那么：
# - 如果是$(((m_1 * m_2) * m_3) * m_4)$的顺序，则需要5x20x50+5x50+5x100次乘法运算，即5750次
# - 如果是$((m_1 * (m_2 * m_3)) * m_4)$的顺序，则需要20x50x1+5x20x1+5x1x100次乘法运算，即1600次
# - 其他以此类推

import copy
import numpy as np
import time


# 该方法用的BFS算法实现的，每次都将计算结果放入队列，然后每次pop出最前面的结果，最后再对所有的结果排序，最后找出计算次数最少的那个计算顺序
class Solution1(object):
    @staticmethod
    def minimize_matrix_computation(matrix_list):
        matrix_shape_list = [matrix.shape for matrix in matrix_list]
        original_sequence = [i for i in range(0, len(matrix_list))]

        all_list = [[matrix_shape_list, original_sequence, 0]]
        saved_list = []
        while(all_list):
            current_original_list = all_list.pop()

            shape_list = current_original_list[0]
            sequence_list = current_original_list[1]
            computation_nums = current_original_list[2]

            # print("original_shape_list: {}".format(shape_list))
            # print("sequence_list: {}".format(sequence_list))
            # print("com_nums: {}".format(computation_nums))

            for k in range(0, len(shape_list) - 1):
                # print("shape_list: {}".format(shape_list))
                # print("k: {}".format(k))
                temp_shape_list = copy.copy(shape_list)
                # print("1: {}".format(temp_shape_list))
                temp_sequence_list = copy.copy(sequence_list)
                temp_computation_nums = computation_nums
                temp_shape_list.insert(k, (temp_shape_list[k][0], temp_shape_list[k+1][1]))
                # print("2: {}".format(temp_shape_list))
                temp_sequence_list.insert(k, [temp_sequence_list[k], temp_sequence_list[k + 1]])

                temp_computation_nums = temp_computation_nums + \
                    temp_shape_list[k + 1][0] * temp_shape_list[k + 1][1] * temp_shape_list[k + 2][1]
                # print("computation_nums: {}".format(temp_computation_nums))

                temp_shape_list.pop(k + 1)
                # print("3: {}".format(temp_shape_list))
                temp_sequence_list.pop(k + 1)
                temp_shape_list.pop(k + 1)
                # print("shape: {}".format(temp_shape_list))
                temp_sequence_list.pop(k + 1)
                # print("sequence: {}".format(temp_sequence_list))

                current_processed_list = [temp_shape_list, temp_sequence_list, temp_computation_nums]

                all_list.insert(0, current_processed_list)
                # print("all_list: {}".format(all_list))

                if len(shape_list) == 2:
                    saved_list.append(current_processed_list)

        print("final saved list: {}".format(saved_list))
        min_num = saved_list[0][2]
        sequence = saved_list[0][1]
        for i in range(1, len(saved_list)):
            if min_num > saved_list[i][2]:
                min_num = saved_list[i][2]
                sequence = saved_list[i][1]
            # print("\nmin_num: {}".format(min_num))
            # print("sequence: {}".format(sequence))

        print("\nmin_num: {}".format(min_num))
        print("sequence: {}".format(sequence))

        return sequence


# 该方法使用动态规划的思想来实现，大致思路和上述BFS算法的思路非常相似，都是保存上一次任意两个矩阵的相乘结果，然后再对剩下的矩阵进行迭代，
# 一直到计算完整个矩阵链，只不过，solution1用的队列的方式，这次用递归的方法。这个方法消耗的时间大概为solution1的1/20.
# 时间复杂度为:
class Solution2(object):
    @staticmethod
    def minimize_matrix_computation(matrix_list):

        def process_list(current_all_list):
            """
            process current matrix chain
            :param current_all_list: [matrix_shape_list, computation_sequence_list, computation_nums]
            :return: next need calculated list
            """
            processed_list = []

            for current_original_list in current_all_list:
                current_processed_all_list = []

                shape_list = current_original_list[0]
                sequence_list = current_original_list[1]
                computation_nums = current_original_list[2]

                # print("original_shape_list: {}".format(shape_list))
                # print("sequence_list: {}".format(sequence_list))
                # print("com_nums: {}".format(computation_nums))

                for k in range(0, len(shape_list) - 1):
                    # print("shape_list: {}".format(shape_list))
                    # print("k: {}".format(k))
                    temp_shape_list = copy.copy(shape_list)
                    # print("1: {}".format(temp_shape_list))
                    temp_sequence_list = copy.copy(sequence_list)
                    temp_computation_nums = computation_nums
                    temp_shape_list.insert(k, (temp_shape_list[k][0], temp_shape_list[k+1][1]))
                    # print("2: {}".format(temp_shape_list))
                    temp_sequence_list.insert(k, [temp_sequence_list[k], temp_sequence_list[k + 1]])

                    temp_computation_nums = temp_computation_nums + \
                        temp_shape_list[k + 1][0] * temp_shape_list[k + 1][1] * temp_shape_list[k + 2][1]
                    # print("computation_nums: {}".format(temp_computation_nums))

                    temp_shape_list.pop(k + 1)
                    # print("3: {}".format(temp_shape_list))
                    temp_sequence_list.pop(k + 1)
                    temp_shape_list.pop(k + 1)
                    # print("shape: {}".format(temp_shape_list))
                    temp_sequence_list.pop(k + 1)
                    # print("sequence: {}".format(temp_sequence_list))

                    current_processed_list = [temp_shape_list, temp_sequence_list, temp_computation_nums]

                    current_processed_all_list.append(current_processed_list)
                    # print("current_processed_all_list: {}".format(current_processed_all_list))

                processed_list.extend(current_processed_all_list)
                # print("processed_list: {}".format(processed_list))

            if len(current_all_list[0][0]) == 2:
                return processed_list

            return process_list(processed_list)

        matrix_shape_list = [matrix.shape for matrix in matrix_list]
        original_sequence = [i for i in range(0, len(matrix_list))]
        initial_all_list = [[matrix_shape_list, original_sequence, 0]]
        saved_list = process_list(initial_all_list)

        print("final saved list: {}".format(saved_list))
        min_num = saved_list[0][2]
        sequence = saved_list[0][1]
        for i in range(1, len(saved_list)):
            if min_num > saved_list[i][2]:
                min_num = saved_list[i][2]
                sequence = saved_list[i][1]
            # print("\nmin_num: {}".format(min_num))
            # print("sequence: {}".format(sequence))

        print("\nmin_num: {}".format(min_num))
        print("sequence: {}".format(sequence))

        return sequence


if __name__ == '__main__':
    '''
    m1 = np.array([[1, 1, 1],
                   [1, 1, 1]])

    m2 = np.array([[2, 2],
                   [2, 2],
                   [2, 2]])

    m3 = np.array([[3, 3, 3, 3],
                   [3, 3, 3, 3]])

    m4 = np.array([[4],
                   [4],
                   [4],
                   [4]])
    '''
    m1 = np.linspace(0, 100, 5000)
    m2 = np.linspace(0, 200, 100000)
    m3 = np.linspace(0, 30, 1000)
    m4 = np.linspace(0, 40, 50)
    m5 = np.linspace(0, 100, 50)
    m6 = np.linspace(0, 200, 100)
    m7 = np.linspace(0, 30, 400)
    m8 = np.linspace(0, 40, 200)
    m9 = np.linspace(0, 100, 100)
    m10 = np.linspace(0, 200, 200)
    '''
    m11 = np.linspace(0, 30, 400)
    m12 = np.linspace(0, 40, 200)
    m13 = np.linspace(0, 100, 100)
    m14 = np.linspace(0, 200, 200)
    m15 = np.linspace(0, 30, 400)
    m16 = np.linspace(0, 40, 200)
    '''

    m1 = m1.reshape(50, 100)
    m2 = m2.reshape(100, 1000)
    m3 = m3.reshape(1000, 1)
    m4 = m4.reshape(1, 50)
    m5 = m5.reshape(50, 1)
    m6 = m6.reshape(1, 100)
    m7 = m7.reshape(100, 4)
    m8 = m8.reshape(4, 50)
    m9 = m9.reshape(50, 2)
    m10 = m10.reshape(2, 100)
    '''
    m11 = m11.reshape(100, 4)
    m12 = m12.reshape(4, 50)
    m13 = m13.reshape(50, 2)
    m14 = m14.reshape(2, 100)
    m15 = m15.reshape(100, 4)
    m16 = m16.reshape(4, 50)
    '''

    # matrix_list = [m1, m2, m3, m4, m5, m6, m7, m8, m9, m10, m11, m12, m13, m14, m15, m16]
    matrix_list = [m1, m2, m3, m4, m5, m6, m7, m8, m9, m10]
    # matrix_list = [m1, m2, m3]

    startTime = time.time()
    matrix_multiply_sequences = Solution2.minimize_matrix_computation(matrix_list)
    print("\ntime = {}".format(time.time() - startTime))
    print(matrix_multiply_sequences)
