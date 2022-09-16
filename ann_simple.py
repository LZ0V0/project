import tensorflow as tf
import operator
import numpy as np
import math
import os
import matplotlib
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
import cv2


def clean_data(dict):  # remove some data by their location
    target_value = 'Cappadocia'
    key_list = list(dict.keys())
    value_list = list(dict.values())
    ind = value_list.index(target_value)
    del dict[key_list[ind]]
    return dict


def get_coin_imgs(number_list):     # # get coin images by numbers corresponding
    coin_imgs = []
    for num in number_list:
        # coin_imgs.append(cv2.imread('./imgs_prepared/{}.jpg'.format(num)))
        coin_imgs.append(cv2.imread('./imgs_prepared/{}.jpg'.format(num), cv2.IMREAD_GRAYSCALE))
        # coin_imgs.append(tf.image.rgb_to_grayscale(cv2.imread('./imgs_prepared/{}.jpg'.format(num))))
    return np.array(coin_imgs)


def label_to_lst(lst):
    # location order in label_lst is 'Phoenicia', 'Mesopotamia', 'Syria', 'Cyprus','Judaea', 'Egypt'
    label_lst = []
    for loc in lst:
        if loc == 'Phoenicia':    label_lst.append([1, 0, 0, 0, 0, 0])
        if loc == 'Mesopotamia':  label_lst.append([0, 1, 0, 0, 0, 0])
        if loc == 'Syria':        label_lst.append([0, 0, 1, 0, 0, 0])
        if loc == 'Cyprus':       label_lst.append([0, 0, 0, 1, 0, 0])
        if loc == 'Judaea':       label_lst.append([0, 0, 0, 0, 1, 0])
        if loc == 'Egypt':        label_lst.append([0, 0, 0, 0, 0, 1])
    return label_lst


# 展开list
def forfor(a):
    return [item for sublist in a for item in sublist]


def leakyrelu(xx, leak=0.2, name="LeakyRelu"):
    with tf.variable_scope(name):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * xx + f2 * tf.abs(xx)

# 局部响应归一化层（LRN）
def norm(x, lsize=4):
    return tf.nn.lrn(x, lsize, bias=1.0, alpha=0.001 / 9.0, beta=0.75)


# 从总数据中随机选取batch_size个使用，若全部使用则会爆显存
# train_data训练集特征(光谱)，train_target训练集对应的标签（含量信息），batch_size
def next_batch(train_data1, train_target1, batch_size):
    if train_data1.shape[0] >= batch_size:
        # 打乱数据集
        index = [iii for iii in range(0, len(train_target1))]
        np.random.shuffle(index)
        # 建立batch_data与batch_target的空列表
        batch_data = []
        batch_target = []
        # 向空列表加入训练集及标签
        for nb_i in range(0, batch_size):
            batch_data.append(train_data1[index[nb_i]])
            batch_target.append(train_target1[index[nb_i]])
    else:  # 如果随机抽取的数目大于所有的光谱数，则直接返回所有光谱
        batch_data = train_data1
        batch_target = train_target1
    return batch_data, batch_target  # 返回



# 初始化权值、偏置
def weight_variable(shape):  # 权值初始化函数
    initial = tf.truncated_normal(shape, stddev=0.1)  # 截断正态分布函数，stddev为标准差
    return tf.Variable(initial)


def bias_variable(shape):  # 偏置初始化函数
    initial = tf.constant(0.1, shape=shape)  # 构建一个形状为shape，数值为0.1的常量
    return tf.Variable(initial)


# 卷积、池化
def conv2d(cx, w):
    return tf.nn.conv2d(cx, w, strides=[1, 1, 1, 1], padding='SAME')


def max_pooling_30x30(mpx):
    return tf.nn.max_pool(mpx, ksize=[1, 30, 30, 1], strides=[1, 1, 1, 1], padding='SAME')


# 构建卷积层
x = tf.placeholder(tf.float32, [None, 120, 240])  # 定义x的形式，x为输入
y_ = tf.placeholder(tf.float32, [None, 6])  # 定义y_的形式，y_为数据的标签
keep_prob = tf.placeholder(tf.float32)
x_image = tf.reshape(x, [-1, 120, 240, 1])  # 输入图像像素为64*96，第一个-1代表样本参数不固定，最后一个1代表使用灰度图

# 全连接层（两层隐含层分别为600,300神经元）
h_norm3_flat = tf.reshape(x_image, [-1, 120 * 240 * 1])  # 将卷积输出的4D矩阵转换为1D向量以输入全连接层，-1代表不限制样本数量
W_fcl = weight_variable([120 * 240 * 1, 600])
b_fcl = bias_variable([600])  # 每个神经元一个偏置
h_fcl = leakyrelu(tf.matmul(h_norm3_flat, W_fcl) + b_fcl)  # 实施全连接计算，本质即是两个矩阵相乘
h_fcl_drop = tf.nn.dropout(h_fcl, keep_prob)

W_fc2 = weight_variable([600, 500])
b_fc2 = bias_variable([500])  # 每个神经元一个偏置
h_fc2 = leakyrelu(tf.matmul(h_fcl_drop, W_fc2) + b_fc2)  # 实施全连接计算，本质即是两个矩阵相乘
h_fc2_drop = tf.nn.dropout(h_fc2, keep_prob)  # dropout输入为全连接层的输出

# Readout层，数据读取层
W_fc3 = weight_variable([500, 6])  # 1024为全连接层输出，对应本层输入，10代表数字分类为10（即0到9）
b_fc3 = bias_variable([6])  # 本层10个神经元对应10个偏置
# y_conv = tf.nn.tanh(tf.matmul(h_fcl_drop, W_fc3) + b_fc3)
y_conv = tf.matmul(h_fc2_drop, W_fc3) + b_fc3

# 参数训练与模型评估
cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_, logits=y_conv))
train_step = tf.train.AdamOptimizer(0.0004).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# 测试集参数评估设置
yy_ = tf.placeholder(tf.float32, [None, 6])  # 数据的标签
yy_conv = tf.placeholder(tf.float32, [None, 6])  # 预测值
testing_correct_prediction = tf.equal(tf.argmax(yy_conv, 1), tf.argmax(yy_, 1))
testing_accuracy = tf.reduce_mean(tf.cast(testing_correct_prediction, tf.float32))


# model configs
os.environ["CUDA_VISIBLE_DEVICES"] = '0'  # '0'指定第一块GPU可用,若不填（''）则使用CPU计算
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.9  # 程序最多只能占用指定gpu90%的显存
config.gpu_options.allow_growth = True  # 程序按需申请内存

batch_size = 50  # 随机取batch_size个训练样本
descs = clean_data(np.load('describes_prepared.npy', allow_pickle=True).item())
decs_keys = np.array(list(descs.keys()))
# KF = KFold(n_splits=5, shuffle=True)   # KFold cross validation
KF = KFold(n_splits=8, shuffle=True)


run_num = 0   # 表示当下是交叉验证的第几次

# run model
for train_index, test_index in KF.split(decs_keys):
    train_keys, test_keys = decs_keys[train_index], decs_keys[test_index]
    train = get_coin_imgs(train_keys)
    train_label = np.array(label_to_lst(operator.itemgetter(*train_keys)(descs)))
    test = get_coin_imgs(test_keys)
    test_label = np.array(label_to_lst(operator.itemgetter(*test_keys)(descs)))  # get location labels by numbers corresponding

    recond_train = []  # 用于记录每次训练集的数据
    recond_test = []  # 用于记录每次测试集的数据
    run_num = run_num + 1
    # print(test)

    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(1200):
            # 训练
            batch = next_batch(train, train_label, batch_size)
            train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})  # 训练时保留概率0.5
            train_accuracy = accuracy.eval(feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
            # 显示训练过程，每i次显示一次
            if i % 1 == 0:
                # train_accuracy = accuracy.eval(feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
                print('NO.%d, step %d, training accuracy %g' % (run_num, i+1, train_accuracy))
                # print(sess.run(y_, feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0}))
                # print(sess.run(y_conv, feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0}))

                # 以下代码实现将测试集依次全部放入模型，再将预测结果比对标签数据，得出总体的预测准确度
                number = 100
                y_conv_all = []
                y_label_all = []
                for num_i in range(0, int(math.ceil(test.shape[0] / number))):  # 将整个训练集放入训练好的模型，根据预测出的含量分类
                    if ((num_i + 1) * number) <= test.shape[0]:
                        end_number = ((num_i + 1) * number) - 1
                        batch_num1 = test[num_i * number: end_number]
                        batch_num2 = test_label[num_i * number: end_number]
                    else:
                        end_number = test.shape[0] - 1
                        batch_num1 = test[num_i * number: end_number]
                        batch_num2 = test_label[num_i * number: end_number]

                    label_test = y_.eval(feed_dict={x: batch_num1, y_: batch_num2})
                    label_test_conv = y_conv.eval(feed_dict={x: batch_num1, y_: batch_num2, keep_prob: 1.0})
                    y_label_all.append(label_test)
                    y_conv_all.append(label_test_conv)

                y_label_all = forfor(y_label_all)
                y_conv_all = forfor(y_conv_all)
                # RMSE_P = rmse.eval(feed_dict={yy_: y_label_all, yy_conv: y_conv_all})
                test_accuracy = testing_accuracy.eval(feed_dict={yy_: y_label_all, yy_conv: y_conv_all})
                print('Testing accuracy %g' % test_accuracy)

                # 记录当前数据
                recond_train.append(train_accuracy)
                recond_test.append(test_accuracy)

        recond_train = np.array(recond_train, dtype=float)  # list转化为narray
        recond_test = np.array(recond_test, dtype=float)

        smallest = max(enumerate(recond_test), key=operator.itemgetter(1))  # 寻找最大预测
        smallest_value = round(smallest[1], 3)  # 保留三位小数
        print(smallest_value)  # 输出最小的RMSEP和其对应训练次数

        # 画图设置
        plt.ion()
        matplotlib.rcParams['xtick.direction'] = 'in'  # 将x轴的刻度线方向设置向内
        matplotlib.rcParams['ytick.direction'] = 'in'  # 将y轴的刻度方向设置向内

        # 设置输出的图片大小
        figsize = 21, 9
        figure, ax = plt.subplots(figsize=figsize)

        # 设置坐标轴标签及字体大小
        font2 = {'family': 'Times New Roman', 'weight': 'normal', 'size': 30}
        plt.ylabel('Accuracy', font2)
        plt.xlabel('Times', font2)

        # 设置坐标刻度值的大小以及刻度值的字体
        plt.tick_params(labelsize=30)
        labels = ax.get_xticklabels() + ax.get_yticklabels()
        [label.set_fontname('Times New Roman') for label in labels]

        # 画图展示训练，测试结果变化
        pic_x = range(0, int(len(recond_train)))
        # plt.plot(pic_x[10:], recond_train[10:])  # 前几个数过大，不利于作图，跳过前10个开始画图
        # plt.plot(pic_x[10:], recond_test[10:])
        plt.plot(pic_x, recond_train)
        plt.plot(pic_x, recond_test)
        plt.plot(smallest[0], smallest[1], '*-r', markersize=15)  # 标注RMSEP最小点
        leg = plt.legend(["Train", "Test", "Largest {}({})".format(smallest_value, smallest[0])],
                         loc='upper right', prop={'family': 'Times New Roman', 'size': 30})
        leg.get_frame().set_linewidth(0.0)  # 去除图例边框
        plt.savefig('./pictures/ann-No {}.png'.format(run_num), bbox_inches='tight')
        plt.pause(3)  # 图片显示3秒
        plt.close()
        # print('Testing accuracy %g' % accuracy.eval(feed_dict={x: test, y_: test_label, keep_prob: 1.0}))
        sess.close()

        # 增加画图，数据记录       程序后台跑着，写论文
