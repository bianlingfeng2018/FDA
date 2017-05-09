# FDA
Fisher线性判别——基于手写数字
## FDA_function.py
这个文件中包含着关于线性判别的相关函数，其中还包括手写数字的提取，在使用Fisher线性判别辨别两组手写数字时先运行该文件<br>
## FDA_MNIST.py
该文件中 K 是可以改变的，其余不行，其中 K 只能在0-9中任取两个数<br>
例如：<br>
K = [0,1]<br>
输出：accuary =  0.9832<br>

K = [0,8]<br>
输出：accuary =  0.9947<br>

K = [6,8]<br>
输出：accuary =  0.9959<br>

* 手写二进制数据集再 LSC 文件夹中获取（4个）
