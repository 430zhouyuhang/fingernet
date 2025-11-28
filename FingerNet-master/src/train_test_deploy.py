# coding=utf-8
# ===== 1) 标准库优先，越轻的越靠前 =====
import os, sys, argparse
from multiprocessing import Pool
from functools import partial, reduce
from time import time
from datetime import datetime

# ===== 2) 后台绘图后端要在导入 pyplot 前设置 =====
import matplotlib
matplotlib.use('Agg')  # 必须早于 `import matplotlib.pyplot as plt`
import matplotlib.pyplot as plt

# ===== 3) 只用标准库先解析命令行（便于提前设置 GPU 环境）=====
parser = argparse.ArgumentParser(description='Train-Test-Deploy')
parser.add_argument('GPU',  type=str, nargs='?', default='0',
                    help='GPU id(s), e.g. "0", "0,1" or "cpu"')
parser.add_argument('mode', type=str, nargs='?', default='train',
                    choices=['train', 'test', 'deploy'],
                    help='train, test or deploy')
parser.add_argument('--image_format', type=str, default=None,
                    help='图像格式（如.bmp, .png, .jpg），None表示自动检测')
parser.add_argument('--train_set', type=str, nargs='+', default=['../datasets/CISL24218/'],
                    help='训练数据集路径列表')
parser.add_argument('--train_sample_rate', type=float, nargs='+', default=None,
                    help='训练数据集采样率列表，None表示不采样')
parser.add_argument('--test_set', type=str, nargs='+', default=['../datasets/NISTSD27/'],
                    help='测试数据集路径列表')
parser.add_argument('--deploy_set', type=str, nargs='+', 
                    default=['../datasets/out_collect/DB1_A/denoised1','../datasets/out_collect/DB2_A/denoised2','../datasets/out_collect/DB3_A/denoised3','../datasets/out_collect/DB4_A/denoised4'],
                    help='部署数据集路径列表')
parser.add_argument('--pretrain', type=str, default='../models/released_version/Model.model',
                    help='预训练模型路径')
parser.add_argument('--output_dir', type=str, default=None,
                    help='输出目录，None表示使用时间戳自动生成')
args = parser.parse_args()

# 支持的图像格式列表（按优先级排序）
SUPPORTED_IMAGE_FORMATS = ['.bmp', '.png', '.jpg', '.jpeg', '.tiff', '.tif']

# ===== 4) 在导入 TensorFlow 之前设置可见 GPU =====
if args.GPU.strip().lower() == 'cpu':
    os.environ['CUDA_VISIBLE_DEVICES'] = ''     # 禁用 GPU
else:
    os.environ['CUDA_VISIBLE_DEVICES'] = args.GPU  # 例如 "0" 或 "0,1"

# 降低 TF 日志噪音
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')

# ===== 5) 其余第三方库（非 TF）再导入 =====
import cv2, pickle
import imageio.v2 as imageio
import numpy as np
from scipy import ndimage, signal, sparse, io

# 本地工具（理想情况下改成显式导入具体符号，避免 * ）
from utils import *

# ===== 6) 最后导入 TensorFlow / Keras（此时 GPU 可见性已生效）=====
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Activation, Lambda,
    Conv2D, MaxPooling2D, UpSampling2D,
    BatchNormalization, PReLU
)
from tensorflow.keras.regularizers import l2
try:
    # TF 2.x + Keras 2.15.* 存在 legacy 优化器；若无则回退到现行优化器
    from tensorflow.keras.optimizers import legacy as legacy_optimizers
except Exception:
    from tensorflow.keras import optimizers as legacy_optimizers
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import ModelCheckpoint

# ===== 7) 保持你当前工程所需的 TF1 图模式（如需）=====
tf.compat.v1.disable_eager_execution()
config = tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(allow_growth=True))
sess = tf.compat.v1.Session(config=config)

# 注意：Keras 3 不支持 set_session；如果你使用 keras==2.15.* 则可保留
try:
    tf.compat.v1.keras.backend.set_session(sess)
except Exception:
    # 如果环境是 Keras 3，这里不会生效也不影响后续
    pass

# 运行配置
batch_size = 2
use_multiprocessing = False

# 数据集与输出
train_set = args.train_set
train_sample_rate = args.train_sample_rate
test_set = args.test_set
deploy_set = args.deploy_set
pretrain = args.pretrain
if args.output_dir is None:
    output_dir = '../output/'+datetime.now().strftime('%Y%m%d-%H%M%S')
else:
    output_dir = args.output_dir
logging = init_log(output_dir)
# 复制当前脚本到输出目录，便于复现实验
copy_file(os.path.abspath(__file__), output_dir+'/')

# 图像归一化（保持与原实现一致）
def img_normalization(img_input, m0=0.0, var0=1.0):
    m = K.mean(img_input, axis=[1,2,3], keepdims=True)
    var = K.var(img_input, axis=[1,2,3], keepdims=True)
    after = K.sqrt(var0*tf.compat.v1.square(img_input-m)/var)
    image_n = tf.compat.v1.where(tf.compat.v1.greater(img_input, m), m0+after, m0-after)
    return image_n

# atan2：以 y/x 的象限关系计算角度
def atan2(y_x):
    y, x = y_x[0], y_x[1]+K.epsilon()
    atan = tf.compat.v1.atan(y/x)
    angle = tf.compat.v1.where(tf.compat.v1.greater(x,0.0), atan, tf.compat.v1.zeros_like(x))
    angle = tf.compat.v1.where(tf.compat.v1.logical_and(tf.compat.v1.less(x,0.0),  tf.compat.v1.greater_equal(y,0.0)), atan+np.pi, angle)
    angle = tf.compat.v1.where(tf.compat.v1.logical_and(tf.compat.v1.less(x,0.0),  tf.compat.v1.less(y,0.0)), atan-np.pi, angle)
    return angle

# 传统方向估计（Sobel+Gaussian）作为网络 Lambda 的一部分
def orientation(image, stride=8, window=17):
    with tf.compat.v1.name_scope('orientation'):
        assert image.get_shape().as_list()[3] == 1, 'Images must be grayscale'
        strides = [1, stride, stride, 1]
        E = np.ones([window, window, 1, 1])
        sobelx = np.reshape(np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=float), [3, 3, 1, 1])
        sobely = np.reshape(np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=float), [3, 3, 1, 1])
        gaussian = np.reshape(gaussian2d((5, 5), 1), [5, 5, 1, 1])
        with tf.compat.v1.name_scope('sobel_gradient'):
            Ix = tf.compat.v1.nn.conv2d(image, sobelx, strides=[1,1,1,1], padding='SAME', name='sobel_x')
            Iy = tf.compat.v1.nn.conv2d(image, sobely, strides=[1,1,1,1], padding='SAME', name='sobel_y')
        with tf.compat.v1.name_scope('eltwise_1'):
            Ix2 = tf.compat.v1.multiply(Ix, Ix, name='IxIx')
            Iy2 = tf.compat.v1.multiply(Iy, Iy, name='IyIy')
            Ixy = tf.compat.v1.multiply(Ix, Iy, name='IxIy')
        with tf.compat.v1.name_scope('range_sum'):
            Gxx = tf.compat.v1.nn.conv2d(Ix2, E, strides=strides, padding='SAME', name='Gxx_sum')
            Gyy = tf.compat.v1.nn.conv2d(Iy2, E, strides=strides, padding='SAME', name='Gyy_sum')
            Gxy = tf.compat.v1.nn.conv2d(Ixy, E, strides=strides, padding='SAME', name='Gxy_sum')
        with tf.compat.v1.name_scope('eltwise_2'):
            Gxx_Gyy = tf.compat.v1.subtract(Gxx, Gyy, name='Gxx_Gyy')
            theta = atan2([2*Gxy, Gxx_Gyy]) + np.pi
        # 低通平滑（Gaussian）后再除以 2 得到最终方向
        with tf.compat.v1.name_scope('gaussian_filter'):
            phi_x = tf.compat.v1.nn.conv2d(tf.compat.v1.cos(theta), gaussian, strides=[1,1,1,1], padding='SAME', name='gaussian_x')
            phi_y = tf.compat.v1.nn.conv2d(tf.compat.v1.sin(theta), gaussian, strides=[1,1,1,1], padding='SAME', name='gaussian_y')
            theta = atan2([phi_y, phi_x])/2
    return theta

# 将上面的传统方向估计封装为 Keras 模型（供数据管道调用预测）
def get_tra_ori():
    img_input=Input(shape=(None, None, 1))
    theta = Lambda(orientation)(img_input)
    model = Model(inputs=[img_input,], outputs=[theta,])
    return model
tra_ori_model = get_tra_ori()

# 查找图像文件（支持多种格式）
def find_image_files(folder, image_format=None):
    """
    查找文件夹中的图像文件
    
    Args:
        folder: 文件夹路径
        image_format: 指定的图像格式（如'.bmp'），None表示自动检测
        
    Returns:
        (文件路径数组, 文件名数组（不含扩展名）, 检测到的格式)
    """
    if image_format is not None:
        # 使用指定的格式
        files, names = get_files_in_folder(folder, image_format)
        if len(files) > 0:
            return files, names, image_format
        else:
            logging.warning(f"未找到 {image_format} 格式的文件，尝试自动检测...")
    
    # 自动检测格式（按优先级）
    for fmt in SUPPORTED_IMAGE_FORMATS:
        files, names = get_files_in_folder(folder, fmt)
        if len(files) > 0:
            detected_format = fmt
            logging.info(f"在 {folder} 中检测到图像格式: {detected_format} ({len(files)} 个文件)")
            return files, names, detected_format
    
    # 未找到任何图像文件
    logging.warning(f"在 {folder} 中未找到支持的图像格式文件")
    return np.array([]), np.array([]), None

# 加载图像文件（自动检测格式）
def load_image_file(file_path, name, folder, image_format=None):
    """
    加载图像文件，支持多种格式
    
    Args:
        file_path: 文件路径（不含扩展名）
        name: 文件名（不含扩展名）
        folder: 文件夹路径
        image_format: 指定的图像格式，None表示自动检测
        
    Returns:
        图像数组，如果未找到则返回None
    """
    if image_format is not None:
        # 使用指定的格式
        full_path = os.path.join(folder, name + image_format)
        if os.path.exists(full_path):
            return imageio.imread(full_path)
    
    # 自动检测格式
    for fmt in SUPPORTED_IMAGE_FORMATS:
        full_path = os.path.join(folder, name + fmt)
        if os.path.exists(full_path):
            return imageio.imread(full_path)
    
    return None

# 扫描数据集，得到最大图像尺寸（按 8 对齐）与样本名
def get_maximum_img_size_and_names(dataset, sample_rate=None, image_format=None):
    if sample_rate is None:
        sample_rate = [1]*len(dataset)
    img_name, folder_name, img_size = [], [], []
    detected_format = image_format
    
    for folder, rate in zip(dataset, sample_rate):
        files, names, fmt = find_image_files(folder+'images/', image_format)
        if len(files) == 0:
            continue
        
        if detected_format is None:
            detected_format = fmt
        
        img_name.extend(names.tolist()*rate)
        folder_name.extend([folder]*names.shape[0]*rate)
        
        # 读取第一个文件获取尺寸
        img = load_image_file('', names[0], folder+'images/', fmt)
        if img is not None:
            img_size.append(np.array(img.shape))
    
    if len(img_name) == 0:
        raise ValueError("未找到任何图像文件！请检查数据集路径和图像格式。")
    
    img_name = np.asarray(img_name)
    folder_name = np.asarray(folder_name)
    img_size = np.max(np.asarray(img_size), axis=0)
    # 让尺寸能被 8 整除（下采样/上采样步幅为 8）
    img_size = np.array(np.ceil(img_size/8)*8,dtype=np.int32)
    return img_name, folder_name, img_size, detected_format

# 单样本加载与增强（可随机旋转/平移），并对齐输出尺寸
def sub_load_data(data, img_size, aug, image_format=None): 
    img_name, dataset, fmt = data
    if fmt is None:
        fmt = image_format
    
    # 加载主图像
    img = load_image_file('', img_name, dataset+'images/', fmt)
    if img is None:
        raise FileNotFoundError(f"未找到图像文件: {dataset}images/{img_name}")
    
    # 加载分割标签（通常是PNG格式）
    seg_path = os.path.join(dataset+'seg_labels/', img_name+'.png')
    if os.path.exists(seg_path):
        seg = imageio.imread(seg_path)
    else:
        # 尝试其他格式
        seg = None
        for ext in ['.png', '.bmp', '.jpg', '.jpeg']:
            seg_path = os.path.join(dataset+'seg_labels/', img_name+ext)
            if os.path.exists(seg_path):
                seg = imageio.imread(seg_path)
                break
        if seg is None:
            seg = np.zeros_like(img)
    
    # 加载方向标签（尝试多种格式）
    ali = None
    for ext in ['.bmp', '.png', '.jpg', '.jpeg']:
        ali_path = os.path.join(dataset+'ori_labels/', img_name+ext)
        if os.path.exists(ali_path):
            ali = imageio.imread(ali_path)
            break
    if ali is None:
        ali = np.zeros_like(img)
    mnt = np.array(mnt_reader(dataset+'mnt_labels/'+img_name+'.mnt'), dtype=float)
    if any(img.shape != img_size):
        # 填充到目标尺寸（随机/居中）
        if np.random.rand()<aug:
            tra = np.int32(np.random.rand(2)*(np.array(img_size)-np.array(img.shape)))
        else:
            tra = np.int32(0.5*(np.array(img_size)-np.array(img.shape)))
        img_t = np.ones(img_size)*np.mean(img)
        seg_t = np.zeros(img_size)
        ali_t = np.ones(img_size)*np.mean(ali)
        img_t[tra[0]:tra[0]+img.shape[0],tra[1]:tra[1]+img.shape[1]] = img
        seg_t[tra[0]:tra[0]+img.shape[0],tra[1]:tra[1]+img.shape[1]] = seg
        ali_t[tra[0]:tra[0]+img.shape[0],tra[1]:tra[1]+img.shape[1]] = ali
        img = img_t
        seg = seg_t
        ali = ali_t
        mnt = mnt+np.array([tra[1],tra[0],0]) 
    if np.random.rand()<aug:
        # 随机旋转和平移
        rot = np.random.rand() * 360
        tra = (np.random.rand(2)-0.5) / 2 * img_size 
        img = ndimage.rotate(img, rot, reshape=False, mode='reflect')
        img = ndimage.shift(img, tra, mode='reflect')
        seg = ndimage.rotate(seg, rot, reshape=False, mode='constant')
        seg = ndimage.shift(seg, tra, mode='constant')
        ali = ndimage.rotate(ali, rot, reshape=False, mode='reflect')
        ali = ndimage.shift(ali, tra, mode='reflect') 
        mnt_r = point_rot(mnt[:, :2], rot/180*np.pi, img.shape, img.shape)  
        mnt = np.column_stack((mnt_r+tra[[1, 0]], mnt[:, 2]-rot/180*np.pi))
    # 仅保留图内且非边界的细节点
    mnt = mnt[(8<=mnt[:,0])*(mnt[:,0]<img_size[1]-8)*(8<=mnt[:, 1])*(mnt[:,1]<img_size[0]-8), :]
    return img, seg, ali, mnt   

# 数据加载器（生成器）：按 batch 输出图像、标签与名称
def load_data(dataset, tra_ori_model, rand=False, aug=0.0, batch_size=1, sample_rate=None, image_format=None):
    if type(dataset[0]) == str:
        img_name, folder_name, img_size, detected_format = get_maximum_img_size_and_names(dataset, sample_rate, image_format)
        if image_format is None:
            image_format = detected_format
    else:
        img_name, folder_name, img_size = dataset
        detected_format = image_format
    if rand:
        rand_idx = np.arange(len(img_name))
        np.random.shuffle(rand_idx)
        img_name = img_name[rand_idx]
        folder_name = folder_name[rand_idx]
    if batch_size > 1 and use_multiprocessing==True:
        p = Pool(batch_size)        
    p_sub_load_data = partial(sub_load_data, img_size=img_size, aug=aug, image_format=image_format)
    for i in range(0,len(img_name), batch_size):
        have_alignment = np.ones([batch_size, 1, 1, 1])
        image = np.zeros((batch_size, img_size[0], img_size[1], 1))
        segment = np.zeros((batch_size, img_size[0], img_size[1], 1))
        alignment = np.zeros((batch_size, img_size[0], img_size[1], 1))
        minutiae_w = np.zeros((batch_size, img_size[0]//8, img_size[1]//8, 1))-1
        minutiae_h = np.zeros((batch_size, img_size[0]//8, img_size[1]//8, 1))-1
        minutiae_o = np.zeros((batch_size, img_size[0]//8, img_size[1]//8, 1))-1
        batch_name = [img_name[(i+j)%len(img_name)] for j in range(batch_size)]
        batch_f_name = [folder_name[(i+j)%len(img_name)] for j in range(batch_size)]
        batch_format = [detected_format]*batch_size
        if batch_size > 1 and use_multiprocessing==True:    
            results = p.map(p_sub_load_data, zip(batch_name, batch_f_name, batch_format))
        else:
            results = list(map(p_sub_load_data, zip(batch_name, batch_f_name, batch_format)))
        for j in range(batch_size):
            img, seg, ali, mnt = results[j]
            if np.sum(ali) == 0:
                have_alignment[j, 0, 0, 0] = 0
            image[j, :, :, 0] = img / 255.0
            segment[j, :, :, 0] = seg / 255.0
            alignment[j, :, :, 0] = ali / 255.0
            minutiae_w[j, (mnt[:, 1]//8).astype(int), (mnt[:, 0]//8).astype(int), 0] = mnt[:, 0] % 8
            minutiae_h[j, (mnt[:, 1]//8).astype(int), (mnt[:, 0]//8).astype(int), 0] = mnt[:, 1] % 8
            minutiae_o[j, (mnt[:, 1]//8).astype(int), (mnt[:, 0]//8).astype(int), 0] = mnt[:, 2]
        # 生成分割标签与细节点存在性掩码
        label_seg = segment[:, ::8, ::8, :]
        label_seg[label_seg>0] = 1
        label_seg[label_seg<=0] = 0
        minutiae_seg = (minutiae_o!=-1).astype(float)
        # 通过传统方向估计模型得到方向标签（角度），并构造高斯分布标签
        orientation = tra_ori_model.predict(alignment)        
        orientation = orientation/np.pi*180+90
        orientation[orientation>=180.0] = 0.0 # orientation [0, 180)
        minutiae_o = minutiae_o/np.pi*180+90 # [90, 450)
        minutiae_o[minutiae_o>360] = minutiae_o[minutiae_o>360]-360 # to current coordinate system [0, 360)
        minutiae_ori_o = np.copy(minutiae_o) # copy one
        minutiae_ori_o[minutiae_ori_o>=180] = minutiae_ori_o[minutiae_ori_o>=180]-180 # for strong ori label [0,180)      
        # ori 2 gaussian
        gaussian_pdf = signal.windows.gaussian(361, 3)
        y = np.reshape(np.arange(1, 180, 2), [1,1,1,-1])
        delta = np.array(np.abs(orientation - y), dtype=int)
        delta = np.minimum(delta, 180-delta)+180
        label_ori = gaussian_pdf[delta]
        # ori_o 2 gaussian
        delta = np.array(np.abs(minutiae_ori_o - y), dtype=int)
        delta = np.minimum(delta, 180-delta)+180
        label_ori_o = gaussian_pdf[delta] 
        # mnt_o 2 gaussian
        y = np.reshape(np.arange(1, 360, 2), [1,1,1,-1])
        delta = np.array(np.abs(minutiae_o - y), dtype=int)  
        delta = np.minimum(delta, 360-delta)+180
        label_mnt_o = gaussian_pdf[delta]         
        # w 2 gaussian
        gaussian_pdf = signal.windows.gaussian(17, 2)
        y = np.reshape(np.arange(0, 8), [1,1,1,-1])
        delta = (minutiae_w-y+8).astype(int)
        label_mnt_w = gaussian_pdf[delta]
        # h 2 gaussian
        delta = (minutiae_h-y+8).astype(int)
        label_mnt_h = gaussian_pdf[delta]
        # mnt cls label -1:neg, 0:no care, 1:pos（邻域平滑将部分负样本设为 0）
        label_mnt_s = np.copy(minutiae_seg)
        label_mnt_s[label_mnt_s==0] = -1 # neg to -1
        label_mnt_s = (label_mnt_s+ndimage.maximum_filter(label_mnt_s, size=(1,3,3,1)))/2 # around 3*3 pos -> 0
        # 按分割与对齐掩码裁剪有效区域
        label_ori = label_ori * label_seg * have_alignment
        label_ori_o = label_ori_o * minutiae_seg
        label_mnt_o = label_mnt_o * minutiae_seg
        label_mnt_w = label_mnt_w * minutiae_seg
        label_mnt_h = label_mnt_h * minutiae_seg
        yield image, label_ori, label_ori_o, label_seg, label_mnt_w, label_mnt_h, label_mnt_o, label_mnt_s, batch_name
    if batch_size > 1 and use_multiprocessing==True:
        p.close()
        p.join()
    return

# 若干 Lambda/辅助算子
def merge_mul(x):
    return reduce(lambda x,y:x*y, x)

def merge_sum(x):
    return reduce(lambda x,y:x+y, x)

def reduce_sum(x):
    return K.sum(x,axis=-1,keepdims=True) 

def merge_concat(x):
    return tf.concat(x,3)

def select_max(x):
    # 在通道内归一化后，仅保留最接近 1 的候选，其余置零，再归一化为 one-hot
    x = x / (K.max(x, axis=-1, keepdims=True)+K.epsilon())
    x = tf.compat.v1.where(tf.compat.v1.greater(x, 0.999), x, tf.compat.v1.zeros_like(x)) # select the biggest one
    x = x / (K.sum(x, axis=-1, keepdims=True)+K.epsilon()) # prevent two or more ori is selected
    return x  

# 卷积+BN（+PReLU）模块封装
def conv_bn(bottom, w_size, name, strides=(1,1), dilation_rate=(1,1)):
    top = Conv2D(w_size[0], (w_size[1],w_size[2]),
        kernel_regularizer=l2(5e-5),
        padding='same', 
        strides=strides,
        dilation_rate=dilation_rate,
        name='conv-'+name)(bottom)
    top = BatchNormalization(name='bn-'+name)(top)
    return top

def conv_bn_prelu(bottom, w_size, name, strides=(1,1), dilation_rate=(1,1)):
    if dilation_rate == (1,1):
        conv_type = 'conv'
    else:
        conv_type = 'atrousconv'
    top = Conv2D(w_size[0], (w_size[1],w_size[2]),
        kernel_regularizer=l2(5e-5),
        padding='same', 
        strides=strides,
        dilation_rate=dilation_rate,
        name=conv_type+name)(bottom)
    top = BatchNormalization(name='bn-'+name)(top)
    top=PReLU(alpha_initializer='zero', shared_axes=[1,2], name='prelu-'+name)(top)
    return top

# 主网络：VGG-like 主干 + 多尺度分支（ori/seg）+ 增强（Gabor）+ 细节点四头
def get_main_net(input_shape=(512,512,1), weights_path=None):
    img_input=Input(input_shape)
    bn_img=Lambda(img_normalization, name='img_norm')(img_input)
    # 特征提取（VGG 风格）
    conv=conv_bn_prelu(bn_img, (64,3,3), '1_1') 
    conv=conv_bn_prelu(conv, (64,3,3), '1_2')
    conv=MaxPooling2D(pool_size=(2,2),strides=(2,2))(conv)

    conv=conv_bn_prelu(conv, (128,3,3), '2_1') 
    conv=conv_bn_prelu(conv, (128,3,3), '2_2') 
    conv=MaxPooling2D(pool_size=(2,2),strides=(2,2))(conv)

    conv=conv_bn_prelu(conv, (256,3,3), '3_1') 
    conv=conv_bn_prelu(conv, (256,3,3), '3_2') 
    conv=conv_bn_prelu(conv, (256,3,3), '3_3')   
    conv=MaxPooling2D(pool_size=(2,2),strides=(2,2))(conv)

    # 多尺度 ASPP 分支（dilation = 1/4/8）
    scale_1=conv_bn_prelu(conv, (256,3,3), '4_1', dilation_rate=(1,1))
    ori_1=conv_bn_prelu(scale_1, (128,1,1), 'ori_1_1')
    ori_1=Conv2D(90, (1,1), padding='same', name='ori_1_2')(ori_1)
    seg_1=conv_bn_prelu(scale_1, (128,1,1), 'seg_1_1')
    seg_1=Conv2D(1, (1,1), padding='same', name='seg_1_2')(seg_1)

    scale_2=conv_bn_prelu(conv, (256,3,3), '4_2', dilation_rate=(4,4))
    ori_2=conv_bn_prelu(scale_2, (128,1,1), 'ori_2_1')
    ori_2=Conv2D(90, (1,1), padding='same', name='ori_2_2')(ori_2)    
    seg_2=conv_bn_prelu(scale_2, (128,1,1), 'seg_2_1')
    seg_2=Conv2D(1, (1,1), padding='same', name='seg_2_2')(seg_2)

    scale_3=conv_bn_prelu(conv, (256,3,3), '4_3', dilation_rate=(8,8))
    ori_3=conv_bn_prelu(scale_3, (128,1,1), 'ori_3_1')
    ori_3=Conv2D(90, (1,1), padding='same', name='ori_3_2')(ori_3)  
    seg_3=conv_bn_prelu(scale_3, (128,1,1), 'seg_3_1')
    seg_3=Conv2D(1, (1,1), padding='same', name='seg_3_2')(seg_3)

    # sum 融合（ori/seg）
    ori_out=Lambda(merge_sum)([ori_1, ori_2, ori_3]) 
    ori_out_1=Activation('sigmoid', name='ori_out_1')(ori_out)
    ori_out_2=Activation('sigmoid', name='ori_out_2')(ori_out)

    seg_out=Lambda(merge_sum)([seg_1, seg_2, seg_3])
    seg_out=Activation('sigmoid', name='seg_out')(seg_out)
    # ----------------------------------------------------------------------------
    # 增强部分（Gabor 卷积 + 方向 one-hot 加权 + 上采样的 seg 掩码）
    filters_cos, filters_sin = gabor_bank(stride=2, Lambda=8)
    filter_img_real = Conv2D(filters_cos.shape[3],(filters_cos.shape[0],filters_cos.shape[1]),
        weights=[filters_cos, np.zeros([filters_cos.shape[3]])], padding='same',
        name='enh_img_real_1')(img_input)
    filter_img_imag = Conv2D(filters_sin.shape[3],(filters_sin.shape[0],filters_sin.shape[1]),
        weights=[filters_sin, np.zeros([filters_sin.shape[3]])], padding='same',
        name='enh_img_imag_1')(img_input)
    ori_peak = Lambda(ori_highest_peak)(ori_out_1)
    ori_peak = Lambda(select_max)(ori_peak) # select max ori and set it to 1
    upsample_ori = UpSampling2D(size=(8,8))(ori_peak)
    seg_round = Activation('softsign')(seg_out)      
    upsample_seg = UpSampling2D(size=(8,8))(seg_round)
    mul_mask_real = Lambda(merge_mul)([filter_img_real, upsample_ori])
    enh_img_real = Lambda(reduce_sum, name='enh_img_real_2')(mul_mask_real)
    mul_mask_imag = Lambda(merge_mul)([filter_img_imag, upsample_ori])
    enh_img_imag = Lambda(reduce_sum, name='enh_img_imag_2')(mul_mask_imag)
    enh_img = Lambda(atan2, name='phase_img')([enh_img_imag, enh_img_real])
    enh_seg_img = Lambda(merge_concat, name='phase_seg_img')([enh_img, upsample_seg])
    # ----------------------------------------------------------------------------
    # 细节点分支：方向/偏移/置信度
    mnt_conv=conv_bn_prelu(enh_seg_img, (64,9,9), 'mnt_1_1') 
    mnt_conv=MaxPooling2D(pool_size=(2,2),strides=(2,2))(mnt_conv)

    mnt_conv=conv_bn_prelu(mnt_conv, (128,5,5), 'mnt_2_1') 
    mnt_conv=MaxPooling2D(pool_size=(2,2),strides=(2,2))(mnt_conv)

    mnt_conv=conv_bn_prelu(mnt_conv, (256,3,3), 'mnt_3_1')  
    mnt_conv=MaxPooling2D(pool_size=(2,2),strides=(2,2))(mnt_conv)    

    mnt_o_1=Lambda(merge_concat)([mnt_conv, ori_out_1])
    mnt_o_2=conv_bn_prelu(mnt_o_1, (256,1,1), 'mnt_o_1_1')
    mnt_o_3=Conv2D(180, (1,1), padding='same', name='mnt_o_1_2')(mnt_o_2)
    mnt_o_out=Activation('sigmoid', name='mnt_o_out')(mnt_o_3)

    mnt_w_1=conv_bn_prelu(mnt_conv, (256,1,1), 'mnt_w_1_1')
    mnt_w_2=Conv2D(8, (1,1), padding='same', name='mnt_w_1_2')(mnt_w_1)
    mnt_w_out=Activation('sigmoid', name='mnt_w_out')(mnt_w_2)

    mnt_h_1=conv_bn_prelu(mnt_conv, (256,1,1), 'mnt_h_1_1')
    mnt_h_2=Conv2D(8, (1,1), padding='same', name='mnt_h_1_2')(mnt_h_1)
    mnt_h_out=Activation('sigmoid', name='mnt_h_out')(mnt_h_2) 

    mnt_s_1=conv_bn_prelu(mnt_conv, (256,1,1), 'mnt_s_1_1')
    mnt_s_2=Conv2D(1, (1,1), padding='same', name='mnt_s_1_2')(mnt_s_1)
    mnt_s_out=Activation('sigmoid', name='mnt_s_out')(mnt_s_2)

    if args.mode == 'deploy':
        model = Model(inputs=[img_input,], outputs=[enh_img_real, ori_out_1, ori_out_2, seg_out, mnt_o_out, mnt_w_out, mnt_h_out, mnt_s_out])
    else:
        model = Model(inputs=[img_input,], outputs=[ori_out_1, ori_out_2, seg_out, mnt_o_out, mnt_w_out, mnt_h_out, mnt_s_out])     
    if weights_path:
        model.load_weights(weights_path, by_name=True)
    return model   

# 方向向量合成：从 90 通道分布得到 sin(2θ)、cos(2θ) 及模值
kernal2angle = np.reshape(np.arange(1, 180, 2, dtype=float), [1,1,1,90])/90.*np.pi #2angle = angle*2
sin2angle, cos2angle = np.sin(kernal2angle), np.cos(kernal2angle)
def ori2angle(ori):
    sin2angle_ori = K.sum(ori*sin2angle, -1, keepdims=True)
    cos2angle_ori = K.sum(ori*cos2angle, -1, keepdims=True)
    modulus_ori = K.sqrt(K.square(sin2angle_ori)+K.square(cos2angle_ori))
    return sin2angle_ori, cos2angle_ori, modulus_ori

# 各项损失（保留原论文实现）：方向交叉熵+一致性、分割加权交叉熵+平滑、细节点置信度加权交叉熵
def ori_loss(y_true, y_pred, lamb=1.):
    # clip
    y_pred = tf.clip_by_value(y_pred, K.epsilon(), 1 - K.epsilon())
    # get ROI
    label_seg = K.sum(y_true, axis=-1, keepdims=True)
    label_seg = tf.cast(tf.greater(label_seg, 0), tf.float32) 
    # weighted cross entropy loss
    lamb_pos, lamb_neg = 1., 1. 
    logloss = lamb_pos*y_true*K.log(y_pred)+lamb_neg*(1-y_true)*K.log(1-y_pred)
    logloss = logloss*label_seg # apply ROI
    logloss = -K.sum(logloss) / (K.sum(label_seg) + K.epsilon())
    # coherence loss, nearby ori should be as near as possible
    mean_kernal = np.reshape(np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]], dtype=np.float32)/8, [3, 3, 1, 1])    
    sin2angle_ori, cos2angle_ori, modulus_ori = ori2angle(y_pred)
    sin2angle = K.conv2d(sin2angle_ori, mean_kernal, padding='same')
    cos2angle = K.conv2d(cos2angle_ori, mean_kernal, padding='same')
    modulus = K.conv2d(modulus_ori, mean_kernal, padding='same')
    coherence = K.sqrt(K.square(sin2angle) + K.square(cos2angle)) / (modulus + K.epsilon())
    coherenceloss = K.sum(label_seg) / (K.sum(coherence*label_seg) + K.epsilon()) - 1
    loss = logloss + lamb*coherenceloss
    return loss

def ori_o_loss(y_true, y_pred):
    # clip
    y_pred = tf.clip_by_value(y_pred, K.epsilon(), 1 - K.epsilon())
    # get ROI
    label_seg = K.sum(y_true, axis=-1, keepdims=True)
    label_seg = tf.cast(tf.greater(label_seg, 0), tf.float32) 
    # weighted cross entropy loss
    lamb_pos, lamb_neg= 1., 1. 
    logloss = lamb_pos*y_true*K.log(y_pred)+lamb_neg*(1-y_true)*K.log(1-y_pred)
    logloss = logloss*label_seg # apply ROI
    logloss = -K.sum(logloss) / (K.sum(label_seg) + K.epsilon())
    return logloss

def seg_loss(y_true, y_pred, lamb=1.):
    # clip
    y_pred = tf.clip_by_value(y_pred, K.epsilon(), 1 - K.epsilon())
    # weighted cross entropy loss
    total_elements = K.sum(tf.ones_like(y_true))
    label_pos = tf.cast(tf.greater(y_true, 0.0), tf.float32)   
    lamb_pos = 0.5 * total_elements / K.sum(label_pos)
    lamb_neg = 1 / (2 - 1/lamb_pos)
    logloss = lamb_pos*y_true*K.log(y_pred)+lamb_neg*(1-y_true)*K.log(1-y_pred)
    logloss = -K.mean(K.sum(logloss, axis=-1))
    # smooth loss
    smooth_kernal = np.reshape(np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype=np.float32)/8, [3, 3, 1, 1])
    smoothloss = K.mean(K.abs(K.conv2d(y_pred, smooth_kernal)))
    loss = logloss + lamb*smoothloss
    return loss

def mnt_s_loss(y_true, y_pred):
    # clip
    y_pred = tf.clip_by_value(y_pred, K.epsilon(), 1 - K.epsilon())
    # get ROI
    label_seg = tf.cast(tf.not_equal(y_true, 0.0), tf.float32) 
    y_true = tf.where(tf.less(y_true,0.0), tf.zeros_like(y_true), y_true) # set -1 -> 0
    # weighted cross entropy loss       
    total_elements = K.sum(label_seg) + K.epsilon()  
    lamb_pos, lamb_neg = 10., .5
    logloss = lamb_pos*y_true*K.log(y_pred)+lamb_neg*(1-y_true)*K.log(1-y_pred)
    # apply ROI
    logloss = logloss*label_seg
    logloss = -K.sum(logloss) / total_elements
    return logloss    

# 高斯平滑后取峰值的位置（用于选择主方向）
def ori_highest_peak(y_pred, length=180):
    glabel = gausslabel(length=length,stride=2).astype(np.float32)
    ori_gau = K.conv2d(y_pred,glabel,padding='same')
    return ori_gau

# NumPy 版本的高斯平滑峰值选择（用于 deploy 模式，避免计算图累积）
def ori_highest_peak_numpy(y_pred, length=180):
    """使用 NumPy 实现，避免在 TensorFlow 图模式中累积节点"""
    glabel = gausslabel(length=length, stride=2).astype(np.float32)
    # glabel shape: [1, 1, 90, 90], y_pred shape: [batch, H, W, 90]
    # K.conv2d 在这里相当于对通道维度进行加权求和
    # 使用 tensordot 实现: [batch, H, W, 90] x [90, 90] -> [batch, H, W, 90]
    kernel = glabel[0, 0, :, :]  # [90, 90]
    ori_gau = np.tensordot(y_pred, kernel, axes=([-1], [0]))  # [batch, H, W, 90]
    return ori_gau

# 评价指标：角度误差在 k 度内的准确率（含 ROI 掩码）
def ori_acc_delta_k(y_true, y_pred, k=10, max_delta=180):
    # get ROI
    label_seg = K.sum(y_true, axis=-1)
    label_seg = tf.cast(tf.greater(label_seg, 0), tf.float32) 
    # get pred angle    
    angle = K.cast(K.argmax(ori_highest_peak(y_pred, max_delta), axis=-1), dtype=tf.float32)*2.0+1.0
    # get gt angle
    angle_t = K.cast(K.argmax(y_true, axis=-1), dtype=tf.float32)*2.0+1.0
    # get delta
    angle_delta = K.abs(angle_t - angle)
    acc = tf.less_equal(K.minimum(angle_delta, max_delta-angle_delta), k)
    acc = K.cast(acc, dtype=tf.float32)
    # apply ROI
    acc = acc*label_seg
    acc = K.sum(acc) / (K.sum(label_seg)+K.epsilon())
    return acc
def ori_acc_delta_10(y_true, y_pred):
    return ori_acc_delta_k(y_true, y_pred, 10)
def ori_acc_delta_20(y_true, y_pred):
    return ori_acc_delta_k(y_true, y_pred, 20)
def mnt_acc_delta_10(y_true, y_pred):
    return ori_acc_delta_k(y_true, y_pred, 10, 360)
def mnt_acc_delta_20(y_true, y_pred):
    return ori_acc_delta_k(y_true, y_pred, 20, 360)    

# 分割精度（正样本/负样本/总体）
def seg_acc_pos(y_true, y_pred):
    y_true = tf.where(tf.less(y_true,0.0), tf.zeros_like(y_true), y_true)
    acc = K.cast(K.equal(y_true, K.round(y_pred)), dtype=tf.float32)
    acc = K.sum(acc * y_true) / (K.sum(y_true)+K.epsilon())
    return acc    
def seg_acc_neg(y_true, y_pred):
    y_true = tf.where(tf.less(y_true,0.0), tf.zeros_like(y_true), y_true)
    acc = K.cast(K.equal(y_true, K.round(y_pred)), dtype=tf.float32)
    acc = K.sum(acc * (1-y_true)) / (K.sum(1-y_true)+K.epsilon())
    return acc
def seg_acc_all(y_true, y_pred):
    y_true = tf.where(tf.less(y_true,0.0), tf.zeros_like(y_true), y_true)
    return K.mean(K.equal(y_true, K.round(y_pred)))  

# 细节点偏移的平均误差
def mnt_mean_delta(y_true, y_pred):
    # get ROI
    label_seg = K.sum(y_true, axis=-1)
    label_seg = tf.cast(tf.greater(label_seg, 0), tf.float32) 
    # get pred pos    
    pos = K.cast(K.argmax(y_pred, axis=-1), dtype=tf.float32)
    # get gt pos
    pos_t = K.cast(K.argmax(y_true, axis=-1), dtype=tf.float32)
    # get delta
    pos_delta = K.abs(pos_t - pos)
    # apply ROI
    pos_delta = pos_delta*label_seg
    mean_delta = K.sum(pos_delta) / (K.sum(label_seg)+K.epsilon())
    return mean_delta

# 训练流程（每若干步测试一次并保存权重）
def train(input_shape=(512,512,1)):
    img_name, folder_name, img_size, detected_format = get_maximum_img_size_and_names(train_set, train_sample_rate, args.image_format)
    if args.image_format is None:
        args.image_format = detected_format  
    main_net_model = get_main_net((img_size[0],img_size[1],1), pretrain)
    plot_model(main_net_model, to_file=output_dir+'/model.png',show_shapes=True)
    adam = legacy_optimizers.Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)    
    main_net_model.compile(optimizer=adam, 
        loss={'ori_out_1':ori_loss, 'ori_out_2':ori_o_loss, 'seg_out':seg_loss, 
                'mnt_o_out':ori_o_loss, 'mnt_w_out':ori_o_loss, 'mnt_h_out':ori_o_loss, 'mnt_s_out':mnt_s_loss}, 
        loss_weights={'ori_out_1':.1, 'ori_out_2':.1, 'seg_out':10., 
                'mnt_w_out':.5, 'mnt_h_out':.5, 'mnt_o_out':.5,'mnt_s_out':200.},
        metrics={'ori_out_1':[ori_acc_delta_10,],
                 'ori_out_2':[ori_acc_delta_10,],
                 'seg_out':[seg_acc_pos, seg_acc_neg, seg_acc_all],
                 'mnt_o_out':[mnt_acc_delta_10,],
                 'mnt_w_out':[mnt_mean_delta,],
                 'mnt_h_out':[mnt_mean_delta,],
                 'mnt_s_out':[seg_acc_pos, seg_acc_neg, seg_acc_all]})
    for epoch in range(100):
        for i, train in enumerate(load_data((img_name, folder_name, img_size), tra_ori_model, rand=True, aug=0.7, batch_size=batch_size, image_format=args.image_format)):
            loss = main_net_model.train_on_batch(train[0], 
                {'ori_out_1':train[1], 'ori_out_2':train[2], 'seg_out':train[3],
                'mnt_w_out':train[4], 'mnt_h_out':train[5], 'mnt_o_out':train[6], 'mnt_s_out':train[7]})  
            if i%(20//batch_size) == 0:
                logging.info("epoch=%d, step=%d", epoch, i)
                logging.info("%s", " ".join(["%s:%.4f\n"%(x) for x in zip(main_net_model.metrics_names, loss)]))
            if i%(10000//batch_size) == (10000//batch_size)-1:
                # 每 10000 张图片做一次测试并保存权重
                outdir = "%s/%03d_%05d/"%(output_dir, epoch, i)
                re_mkdir(outdir)
                savedir = "%s%s"%(outdir, str(epoch)+'_'+str(i))
                main_net_model.save_weights(savedir, True)
                for folder in test_set:
                    test([folder,], savedir, outdir, test_num=10, draw=False)
    return

# 将网络输出转为细节点列表（坐标、方向、得分）
def label2mnt(mnt_s_out, mnt_w_out, mnt_h_out, mnt_o_out, thresh=0.5):
    mnt_s_out = np.squeeze(mnt_s_out)
    mnt_w_out = np.squeeze(mnt_w_out)
    mnt_h_out = np.squeeze(mnt_h_out)
    mnt_o_out = np.squeeze(mnt_o_out)
    assert len(mnt_s_out.shape)==2 and len(mnt_w_out.shape)==3 and len(mnt_h_out.shape)==3 and len(mnt_o_out.shape)==3 
    # get cls results
    mnt_sparse = sparse.coo_matrix(mnt_s_out>thresh)
    mnt_list = np.array(list(zip(mnt_sparse.row, mnt_sparse.col))),
    mnt_list = np.array(mnt_list[0], dtype=np.int32)
    if mnt_list.shape[0] == 0:
        return np.zeros((0, 4))
    # get regression results
    mnt_w_out = np.argmax(mnt_w_out, axis=-1)
    mnt_h_out = np.argmax(mnt_h_out, axis=-1)
    mnt_o_out = np.argmax(mnt_o_out, axis=-1) # TODO: use ori_highest_peak(np version)
    # get final mnt
    mnt_final = np.zeros((len(mnt_list), 4))
    mnt_final[:, 0] = mnt_sparse.col*8 + mnt_w_out[mnt_list[:,0], mnt_list[:,1]]
    mnt_final[:, 1] = mnt_sparse.row*8 + mnt_h_out[mnt_list[:,0], mnt_list[:,1]]
    mnt_final[:, 2] = (mnt_o_out[mnt_list[:,0], mnt_list[:,1]]*2-89.)/180*np.pi
    mnt_final[mnt_final[:, 2]<0.0, 2] = mnt_final[mnt_final[:, 2]<0.0, 2]+2*np.pi
    mnt_final[:, 3] = mnt_s_out[mnt_list[:,0], mnt_list[:, 1]]  # confidence score
    return mnt_final

# 测试流程：预测、计算指标、NMS 与可视化
def test(test_set, model, outdir, test_num=10, draw=True):
    logging.info("Testing %s:"%(test_set))
    img_name, folder_name, img_size, detected_format = get_maximum_img_size_and_names(test_set, image_format=args.image_format)
    if args.image_format is None:
        args.image_format = detected_format  
    main_net_model = get_main_net((img_size[0],img_size[1],1), model)
    nonsense = legacy_optimizers.SGD(learning_rate=0.0, momentum=0.0, nesterov=False)    
    main_net_model.compile(optimizer=nonsense,
        loss={'ori_out_1':ori_loss, 'ori_out_2':ori_o_loss, 'seg_out':seg_loss, 
                'mnt_o_out':ori_o_loss, 'mnt_w_out':ori_o_loss, 'mnt_h_out':ori_o_loss, 'mnt_s_out':mnt_s_loss}, 
        loss_weights={'ori_out_1':.1, 'ori_out_2':.1, 'seg_out':10., 
                'mnt_w_out':.5, 'mnt_h_out':.5, 'mnt_o_out':.5,'mnt_s_out':200.},        
        metrics={'ori_out_1':[ori_acc_delta_10,ori_acc_delta_20],
                 'ori_out_2':[ori_acc_delta_10,ori_acc_delta_20],
                 'seg_out':[seg_acc_pos, seg_acc_neg, seg_acc_all],
                 'mnt_o_out':[mnt_acc_delta_10,mnt_acc_delta_20],
                 'mnt_w_out':[mnt_mean_delta,],
                 'mnt_h_out':[mnt_mean_delta,],
                 'mnt_s_out':[seg_acc_pos, seg_acc_neg, seg_acc_all]})
    ave_loss, ave_prf_nms = [], []
    for j, test in enumerate(load_data((img_name, folder_name, img_size), tra_ori_model, rand=False, aug=0.0, batch_size=1, image_format=args.image_format)):      
        if j < test_num:
            logging.info("%d / %d: %s"%(j+1, len(img_name), img_name[j]))    
            ori_out_1, ori_out_2, seg_out, mnt_o_out, mnt_w_out, mnt_h_out, mnt_s_out  = main_net_model.predict(test[0])
            metrics = main_net_model.train_on_batch(test[0], 
                {'ori_out_1':test[1], 'ori_out_2':test[2], 'seg_out':test[3],
                'mnt_w_out':test[4], 'mnt_h_out':test[5], 'mnt_o_out':test[6], 'mnt_s_out':test[7]})  
            ave_loss.append(metrics)
            logging.info("%s", " ".join(["%s:%.4f\n"%(x) for x in zip(main_net_model.metrics_names, metrics)]))
            mnt_gt = label2mnt(test[7], test[4], test[5], test[6])
            mnt_s_out = mnt_s_out * test[3]
            mnt = label2mnt(mnt_s_out, mnt_w_out, mnt_h_out, mnt_o_out, thresh=0.5)
            mnt_nms = nms(mnt)
            
            # 输出详细信息
            logging.info("GT细节点数量: %d, 预测细节点数量: %d (NMS前: %d)"%(len(mnt_gt), len(mnt_nms), len(mnt)))
            if len(mnt_nms) > 0:
                avg_confidence = np.mean(mnt_nms[:, 3]) if mnt_nms.shape[1] >= 4 else 0.0
                max_confidence = np.max(mnt_nms[:, 3]) if mnt_nms.shape[1] >= 4 else 0.0
                logging.info("预测细节点置信度 - 平均: %.3f, 最大: %.3f"%(avg_confidence, max_confidence))
            
            p, r, f, l, o = mnt_P_R_F(mnt_gt, mnt_nms)
            logging.info("After_nms:\nprecision: %f\nrecall: %f\nf1-measure: %f\nlocation_dis: %f\norientation_delta:%f\n----------------\n"%(
                p, r, f, l, o))
            ave_prf_nms.append([p, r, f, l, o])            
            if draw:                         
                angval = sess.run(ori_highest_peak(ori_out_1))                           
                angval = (np.argmax(angval, axis=-1)*2-90)/180.*np.pi
                draw_ori_on_img(test[0], angval, seg_out, "%s%s_ori.png"%(outdir, test[8][0]))
                draw_minutiae(test[0], mnt_nms[:,:3], "%s%s_mnt.png"%(outdir, test[8][0]))
                draw_minutiae(test[0], mnt_gt[:,:3], "%s%s_mnt_gt.png"%(outdir, test[8][0]))
        else:
            break
    logging.info("Average testing results:")
    ave_loss = np.mean(np.array(ave_loss), 0)
    ave_prf_nms = np.mean(np.array(ave_prf_nms), 0)
    logging.info("\n%s\n", " ".join(["%s:%.4f\n"%(x) for x in zip(main_net_model.metrics_names, ave_loss)]))
    logging.info("After_nms:\nprecision: %f\nrecall: %f\nf1-measure: %f\nlocation_dis: %f\norientation_delta:%f\n----------------\n"%(
                    ave_prf_nms[0],ave_prf_nms[1],ave_prf_nms[2],ave_prf_nms[3],ave_prf_nms[4]))     
    return

# 部署：对未标注数据进行预测，导出可视化与结构化结果
def deploy(deploy_set, set_name=None, image_format=None):
    if set_name is None:
        # 从路径中提取数据集名称（处理末尾斜杠）
        deploy_set_normalized = deploy_set.rstrip('/')
        set_name = os.path.basename(deploy_set_normalized)
        # 如果提取失败，使用默认名称
        if not set_name:
            set_name = deploy_set_normalized.split('/')[-2] if '/' in deploy_set_normalized else 'dataset'
    mkdir(output_dir+'/'+set_name+'/')
    logging.info("Predicting %s:"%(set_name)) 
    
    # 查找图像文件
    files, img_name, detected_format = find_image_files(deploy_set, image_format)
    if len(img_name) == 0:
        deploy_set = deploy_set+'images/'
        files, img_name, detected_format = find_image_files(deploy_set, image_format)
    
    if len(img_name) == 0:
        raise ValueError(f"在 {deploy_set} 中未找到任何图像文件！")
    
    if image_format is None:
        image_format = detected_format
    
    logging.info(f"检测到图像格式: {image_format} ({len(img_name)} 个文件)")
    
    # 读取第一个文件获取尺寸
    img = load_image_file('', img_name[0], deploy_set, image_format)
    if img is None:
        raise FileNotFoundError(f"无法加载图像文件: {deploy_set}{img_name[0]}")
    
    img_size = img.shape
    # 使用整除保持整数尺寸，避免传入 Keras 的浮点形状
    img_size = (np.array(img_size, dtype=np.int32)//8*8).astype(np.int32)      
    main_net_model = get_main_net((img_size[0],img_size[1],1), pretrain)
    
    time_c = []
    for i in range(0,len(img_name)):
        logging.info("%s %d / %d: %s"%(set_name, i+1, len(img_name), img_name[i]))
        time_start = time()    
        image = load_image_file('', img_name[i], deploy_set, image_format)
        if image is None:
            logging.warning(f"跳过无法加载的文件: {img_name[i]}")
            continue
        image = image / 255.0
        image = image[:img_size[0],:img_size[1]]      
        image = np.reshape(image,[1, image.shape[0], image.shape[1], 1])
        enhance_img, ori_out_1, ori_out_2, seg_out, mnt_o_out, mnt_w_out, mnt_h_out, mnt_s_out = main_net_model.predict(image) 
        time_afterconv = time()
        round_seg = np.round(np.squeeze(seg_out))
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(5, 5))
        seg_out = cv2.morphologyEx(round_seg, cv2.MORPH_OPEN, kernel)
        mnt = label2mnt(np.squeeze(mnt_s_out)*np.round(np.squeeze(seg_out)), mnt_w_out, mnt_h_out, mnt_o_out, thresh=0.5)
        mnt_nms = nms(mnt)
        # 使用 NumPy 版本避免计算图累积
        ori_gau = ori_highest_peak_numpy(ori_out_1)
        ori = (np.argmax(ori_gau, axis=-1)*2-90)/180.*np.pi  
        time_afterpost = time()
        
        # 输出详细信息
        logging.info("检测到细节点数量: %d (NMS前: %d)"%(len(mnt_nms), len(mnt)))
        if len(mnt_nms) > 0:
            avg_confidence = np.mean(mnt_nms[:, 3]) if mnt_nms.shape[1] >= 4 else 0.0
            max_confidence = np.max(mnt_nms[:, 3]) if mnt_nms.shape[1] >= 4 else 0.0
            logging.info("细节点置信度 - 平均: %.3f, 最大: %.3f"%(avg_confidence, max_confidence))
        
        # 保存细节点文件（包含confidence信息）
        mnt_writer(mnt_nms, img_name[i], img_size, "%s/%s/%s.mnt"%(output_dir, set_name, img_name[i]))        
        
        # 生成可视化结果
        draw_ori_on_img(image, ori, np.ones_like(seg_out), "%s/%s/%s_ori.png"%(output_dir, set_name, img_name[i]))        
        draw_minutiae(image, mnt_nms[:,:3], "%s/%s/%s_mnt.png"%(output_dir, set_name, img_name[i]))
        
        # 保存增强图像和分割结果
        imageio.imwrite("%s/%s/%s_enh.png"%(output_dir, set_name, img_name[i]), (np.squeeze(enhance_img)*ndimage.zoom(np.round(np.squeeze(seg_out)), [8,8], order=0)).astype(np.uint8))
        imageio.imwrite("%s/%s/%s_seg.png"%(output_dir, set_name, img_name[i]), (ndimage.zoom(np.round(np.squeeze(seg_out)), [8,8], order=0)*255).astype(np.uint8)) 
        
        # 保存.mat文件（包含方向信息和分布图）
        io.savemat("%s/%s/%s.mat"%(output_dir, set_name, img_name[i]), {
            'orientation': ori, 
            'orientation_distribution_map': ori_out_1,
            'minutiae': mnt_nms,
            'segmentation': seg_out,
            'confidence_scores': mnt_nms[:, 3] if mnt_nms.shape[1] >= 4 else None
        })
        
        time_afterdraw = time()
        time_c.append([time_afterconv-time_start, time_afterpost-time_afterconv, time_afterdraw-time_afterpost])
        logging.info("load+conv: %.3fs, seg-postpro+nms: %.3f, draw: %.3f"%(time_c[-1][0],time_c[-1][1],time_c[-1][2]))
    time_c = np.mean(np.array(time_c),axis=0)
    logging.info("Average: load+conv: %.3fs, oir-select+seg-post+nms: %.3f, draw: %.3f"%(time_c[0],time_c[1],time_c[2]))
    return  

# 入口：根据模式运行
def main():
    logging.info("="*60)
    logging.info("FingerNet 指纹识别系统启动")
    logging.info("模式: %s" % args.mode)
    logging.info("GPU: %s" % args.GPU)
    logging.info("输出目录: %s" % output_dir)
    logging.info("="*60)
    
    if args.mode == 'train':
        logging.info("开始训练模式...")
        train()
    elif args.mode == 'test':        
        logging.info("开始测试模式...")
        for folder in test_set:
            logging.info("测试数据集: %s" % folder)
            test([folder,], pretrain, output_dir+"/", test_num=258, draw=False) 
    elif args.mode == 'deploy':
        logging.info("开始部署模式...")
        for i, folder in enumerate(deploy_set):
            logging.info("处理数据集 %d: %s" % (i, folder))
            deploy(folder, None, args.image_format)  # 传入 None 让 deploy 函数自动提取数据集名称
    else:
        logging.error("未知模式: %s" % args.mode)
        pass
    
    logging.info("="*60)
    logging.info("FingerNet 处理完成")
    logging.info("="*60)

if __name__ =='__main__':
    main()
