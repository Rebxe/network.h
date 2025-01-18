- 2025.01.18 更新：去除了预设层级和优化器的成员变量 `int bs`，用户不再需要为每个层级和优化器都指定批大小，仅需在 `auto_dao::init()` 中指定即可；

### 简介

基于 C++14 的仅头文件的神经网络库，代码可读，方便研究神经网络的实现。

计算加速：

- CPU 加速：基于 OpenMP 和 AVX256 指令集；
- GPU 加速：基于微软 `amp.h`；（效果不太好）

CPU 代码支持大部分编译器，可用基于 GCC 的 DEV-C++ 编译；GPU 代码仅支持 VS 系列编译器编译。

支持自动求偏导（反向传播），用户仅需定义前向过程。

支持读取/保存图片文件，使用了开源库 [stb](https://github.com/nothings/stb)。

本文仅介绍库的使用方法，关于机器学习的原理部分请移步：[咕咕咕]()



### 基本概念

#### 训练阶段

在此阶段，数据以批为单位进入神经网络，执行前向过程，计算出结果；再根据结果对应的损失值，反向传播得出各参数偏导，通过优化器更新参数。



#### 测试阶段

在测试阶段，数据一个一个进入神经网络，执行前向过程，计算出结果。

该阶段中反向传播会被禁用，且某些层级算子的行为会改变（例如批归一化层）。



### 张量、自动反向传播

头文件 `auto_dao.h`。



#### 命名空间 `auto_dao`

|            成员            |                          含义/作用                           |
| :------------------------: | :----------------------------------------------------------: |
|      `int Batch_Size`      |      批大小，若为 $0$ 则表示当前为测试阶段而非训练阶段       |
|       `struct node`        |                   内部结构体，用户无需访问                   |
|  `std::vector<node*> tmp`  | 内部变量，记录当前申请的所有 `node` 的地址，方便释放内存及初始化反向传播 |
| `void init(int BatchSize)` | 前向过程前必须执行的函数，释放现有所有张量占用的内存空间，初始化 `Batch_Size` |
|     `init_backward()`      |   反向传播前必须执行的函数，初始化现有所有张量以便反向传播   |



#### 三维张量 `val3d`

定义了可自动求偏导（反向传播）的三维张量类型 `val3d`。在训练阶段，张量会同时存储整个批次中的数据（所以它实际上是四维的）。`val3d` 会自动记录前向过程，方便反向传播。

|                   成员                    |                          含义/作用                           |
| :---------------------------------------: | :----------------------------------------------------------: |
|                  `int d`                  |                         张量的通道数                         |
|                  `int h`                  |                          张量的高度                          |
|                  `int w`                  |                          张量的宽度                          |
|                `float* a`                 | 张量数值的起始地址（数值按照 `auto_dao::Batch_Size*d*h*w` 的方式存储） |
|                `float* da`                | 张量偏导的起始地址（偏导按照 `auto_dao::Batch_Size*d*h*w` 的方式存储，与数值一一对应） |
|                 `val3d()`                 |                默认构造函数（不进行任何操作）                |
| `val3d(int td,int th,int tw,float val=0)` | 构造函数，初始化一个 `td*th*tw` 的张量，其每个位置的数值都是 `val` |
| `val3d(int td,int th,int tw,float *dat)`  | 构造函数，根据 `dat[0]` 到 `dat[max(auto_dao::Batch_Size,1)*td*th*tw-1]` 中的数值初始化一个 `td*th*tw` 的张量 |
|               `backward()`                |            将该张量的偏导传递下去[*](#backward())            |
|           `auto_dao::node *dat`           | 内部变量，不使用 `private` 关键字仅仅是为了增加代码可读性，避免大量 `friend` 关键字 |

<span id="backward()">*</span>：调用 `backward` 时，会将该张量标记为”偏导计算完成“状态，并将该张量的偏导反向传播至对其有影响的张量处。若本次操作导致某个张量的偏导计算完成（影响到的所有张量的偏导都已传播至该张量），则会自动调用其 `backward` 函数（类似 DAG 上反向 bfs）。故**用户仅需在手动计算所有输出端张量的偏导后，手动调用所有输出端张量的 `backward` 函数**。



#### 张量相关函数

|                    函数                    |                          含义/作用                           |
| :----------------------------------------: | :----------------------------------------------------------: |
| `val3d reshape(val3d x,int d,int h,int w)` | 软塑性：创建一个新的三维张量，`d,h,w` 为传入的 `d,h,w`，数据从 `x.a` 拷贝（需保证 `d*h*w==x.d*x.h*x.w`） |
| `val3d toshape(val3d x,int d,int h,int w)` | 硬塑性：创建一个新的三维张量，`d,h,w` 为传入的 `d,h,w`，数据从 `x.a` 循环拷贝，即 `i,j,k` 处的数值为 `x` 的 `i%x.d,j%x.h,k%x.w` 处的数值 |
|     `val3d operator+(val3d x,val3d y)`     | 创建一个新的三维张量，其每一位的数值都是 `x` 对应位置的数值和 `y` 对应位置的数值相加（需保证 `x` 和 `y` 形状相同） |
|     `val3d operator-(val3d x,val3d y)`     |                      同上，相加变为相减                      |
|     `val3d operator*(val3d x,val3d y)`     |                          同上，相乘                          |
|     `val3d operator/(val3d x,val3d y)`     |                  同上，相除（`x` 为被除数）                  |
|       `val3d dcat(val3d x,val3d y)`        | 创建一个新的三维张量，其是 `x` 和 `y` 按照 `d` 这一维拼接起来的结果（`x` 占用 `[0,x.d-1]`，`y` 占用 `[x.d,x.d+y.d-1]`，需保证 `x.h==y.h` 且 `x.w==y.w`） |
|  `float MSEloss(val3d x,float* realout)`   | 使用 `realout[0]` 到 `realout[max(auto_dao::Batch_Size,1)*x.d*x.h*x.w]` 中的数据为三维张量 `x` 计算[均方差损失](#MSEloss)，同时为 `x` 计算偏导 |
|  `float BCEloss(val3d x,float* realout)`   |          同上，但计算的是[二元交叉熵损失](#BCEloss)          |

##### <span id="MSEloss">均方差损失（MSEloss）</span>

$x$ 是输出，$r$ 是真实数据：
$$
loss=\frac{1}{n}\sum\limits_{i=1}^n(x_i-r_i)^2
$$

##### <span id="BCEloss">二元交叉熵损失（BSEloss）</span>

$x$ 是输出，$r$ 是真实数据：
$$
loss=-\frac{1}{n}\sum\limits_{i=1}^nr_i\ln(x_i)+(1-r_i)\ln(1-x_i)
$$
注意 $x_i$ 会被限制在 $[10^{-7},1-10^{-7}]$ 内以防止出现 `inf` 或 `nan`，若超出范围则偏导为 $0$。



#### Demo

```cpp
float arr[2*2*2]={0.1,0.2,
                  0.3,0.4,

                  0.1,0.2,
                  0.3,0.4};
auto_dao::init(1);  // 准备前向过程，批大小 1，当前是训练阶段 
val3d x(1,2,2,0.5); // 创建一个 1*2*2 的三维张量，每一位的值都是 0.5 
val3d y(1,2,2,0.6); // 创建一个 1*2*2 的三维张量，每一位的值都是 0.6
val3d z(2,2,2,arr); // 根据 arr 中的值创建一个 2*2*2 的三维张量
val3d xy=dcat(x,y); // 按 d 一维拼接 x 和 y 两个张量，结果为 2*2*2 的三维张量 
val3d res1=xy+z;    // 将 xy 和 z 对应的每一位加起来 
val3d res2=xy*z;    // 将 xy 和 z 对应的每一位乘起来 
for(int i=0;i<2*2*2;i++) printf("%.2f ",res1.a[i]);printf("\n");
for(int i=0;i<2*2*2;i++) printf("%.2f ",res2.a[i]);printf("\n");
/*
输出应为：
0.60 0.70 0.80 0.90 0.70 0.80 0.90 1.00
0.05 0.10 0.15 0.20 0.06 0.12 0.18 0.24
这里以第二行第一个数字为例，0.5*0.1 确实是 0.05 
*/
auto_dao::init_backward(); // 准备反向传播 
for(int i=0;i<2*2*2;i++) res1.da[i]=1,res2.da[i]=-1; // 给出输出端张量的偏导
res1.backward(),res2.backward(); // 将偏导传递下去
for(int i=0;i<1*2*2;i++) printf("%.2f ",x.da[i]);printf("\n");
for(int i=0;i<1*2*2;i++) printf("%.2f ",y.da[i]);printf("\n");
for(int i=0;i<2*2*2;i++) printf("%.2f ",z.da[i]);printf("\n");
/*
输出应为：
0.90 0.80 0.70 0.60
0.90 0.80 0.70 0.60
0.50 0.50 0.50 0.50 0.40 0.40 0.40 0.40
这里以第一行第一个数字为例：
    - 这是 x 中第一个数对应的偏导，不妨设其为 w；
    - w 影响到 res1 中的第一个数 r1 和 res2 的第一个数 r2；
    - 其中 r1 = w + 0.1, r2 = w * 0.1；
    - r1 的偏导为 1，r2 的偏导为 -1，故 w 的偏导为 1 + (-1) * 0.1 = 0.9，结果正确。 
*/
```



### 预设优化器

头文件：`Optimizer/*.h`

命名规则：全大写命名

在本库中，权重及其在反向传播中求得的偏导统一存储于优化器中，方便统一更新。

统一公有成员：

|               成员                |                          含义/作用                           |
| :-------------------------------: | :----------------------------------------------------------: |
|              `int m`              |                           权重数量                           |
|            `float lrt`            |                            学习率                            |
| `void init(float Learn_Rate,...)` | 初始化优化器参数，该函数的第一个参数及含义固定，为学习率，根据不同优化器具体情况可能有更多参数 |
|          `void build()`           |           初始化优化器，为权重及其偏导分配内存空间           |
|  `void save(std::ofstream& ouf)`  |        将优化器参数及权重保存到二进制文件流 `ouf` 中         |
|  `void load(std::ifstream& inf)`  |         从二进制文件流 `inf` 中读取优化器参数及权重          |
|         `void delthis()`          |                      释放申请的内存空间                      |
|          `float* _wei()`          |                     获取权重数组起始地址                     |
|          `float* _tmp()`          |                     获取偏导数组起始地址                     |
|      `void init_backward()`       |        清空累计的偏导，即将偏导数组置零，准备反向传播        |
|          `void flush()`           |       利用当前累计的偏导更新权重，在反向传播完成后调用       |
|           默认构造函数            | 仅在启用宏 `ENABLE_AUTO_SL` 时才有默认构造函数，用于自动生成神经网络的保存、读取和空间释放函数（实现静态反射） |

参数命名规则和主流的神经网络库大致相同。



#### SGD 优化器

头文件：`Optimizer/SGD.h`

定义了优化器类型 `SGD`，其所有公有成员均无特殊。

参数更新方式：（$w$ 为参数，$\Delta$ 为偏导）
$$
w_i\to w_i-lrt\times \Delta_i
$$



#### Adam 优化器

头文件：`Optimizer/Adam.h`

定义了优化器类型 `ADAM`，其特殊成员如下：

|                             成员                             |                          含义/作用                           |
| :----------------------------------------------------------: | :----------------------------------------------------------: |
|                          `float b1`                          |                  参数更新公式中的 $\beta_1$                  |
|                          `float b2`                          |                  参数更新公式中的 $\beta_2$                  |
|                         `float eps`                          | 参数更新公式中的 $\epsilon$，一个很小的非负实数，防止除以 $0$ |
| `void init(float Learn_Rate,float beta1=0.9,float beta2=0.999,float Eps=1e-8)` | 初始化优化器参数，`b1`，`b2` 和 `eps` 分别初始化为 `beta1`，`beta2` 和 `Eps` |

参数更新方式：（$w$ 为参数，$\Delta$ 为偏导）

- 初始 $v_i\to 0,s_i\to 0$；

- 对于第 $t$ 次更新：
  $$
  v_i\to \beta_1v_i+(1-\beta_1)\Delta_i\\
  s_i\to \beta_2s_i+(1-\beta_2)\Delta_i^2\\
  \bar v_i=\frac{v_i}{1-\beta_1^t}\\
  \bar s_i=\frac{s_i}{1-\beta_2^t}\\
  w_i\to w_i-lrt\times \frac{\bar v_i}{\sqrt{\bar s_i}+\epsilon}
  $$



### 预设网络层级（层级算子）

头文件：`Layers/*.h`

命名规则：全大写命名

统一公有成员：（有可训练权重）

|                          成员                           |                          含义/作用                           |
| :-----------------------------------------------------: | :----------------------------------------------------------: |
|                 `void init(int& m,...)`                 | 初始化层级参数，该函数第一个参数及其含义固定，为权重计数器（用于统计权重数量，一般传入优化器的 `m`）。根据不同层级的具体情况可能有更多参数 |
|        `void build(float*& wei,float*& tmp,...)`        | 为层级分配权重、偏导储存空间并初始化权重，其中 `wei` 为权重储存起始地址，`tmp` 为偏导储存起始地址 |
|             `void save(std::ofstream& ouf)`             |   将层级参数保存到二进制文件流 `ouf` 中，权重并不会被保存    |
| `void load(std::ifstream& inf,float*& wei,float*& tmp)` | 从二进制文件流 `inf` 中读取层级参数，并根据 `wei` 和 `tmp` 为层级分配权重[*](#load()) |
|               `val3d operator()(val3d x)`               |         在三维张量 `x` 上应用该层级的操作并返回结果          |
|                      默认构造函数                       | 仅在启用宏 `ENABLE_AUTO_SL` 时才有默认构造函数，用于自动生成神经网络的保存、读取和空间释放函数（实现静态反射） |

<span id="load()">*</span>：`wei` 为权重数组起始地址，`tmp` 为偏导数组起始地址，层级将会从 `wei` 中获取其权重并分配空间（这要求优化器的 `load()` 函数已经被调用）。

统一公有成员：（无可训练权重）

|              成员               |                          含义/作用                           |
| :-----------------------------: | :----------------------------------------------------------: |
|        `void init(...)`         |     初始化层级参数，根据不同层级的具体情况可能有更多参数     |
| `void save(std::ofstream& ouf)` |            将层级参数保存到二进制文件流 `ouf` 中             |
| `void load(std::ifstream& inf)` |             从二进制文件流 `inf` 中读取层级参数              |
|   `val3d operator()(val3d x)`   |         在三维张量 `x` 上应用该层级的操作并返回结果          |
|          默认构造函数           | 仅在启用宏 `ENABLE_AUTO_SL` 时才有默认构造函数，用于自动生成神经网络的保存、读取和空间释放函数（实现静态反射） |



#### 全连接层（FC）

头文件：`Layers/fc.h`

定义了全连接层类型 `FC`，其特殊成员如下：

|                            成员                            |                          含义/作用                           |
| :--------------------------------------------------------: | :----------------------------------------------------------: |
|                         `int ins`                          |                          输入值个数                          |
|                         `int ous`                          |                          输出值个数                          |
|                         `float* w`                         |                       权重存储起始地址                       |
|            `void init(int& m,int INS,int OUS)`             | 初始化层级参数，额外将 `ins` 和 `ous` 初始化为 `INS` 和 `OUS` |
| `void build(float*& wei,float*& tmp,int InitType=INIT_HE)` | 为层级分配权重、偏导储存空间并按照 `InitType` 的方式初始化 `w`（`Xavier` 或 `HE`） |

`FC` 层将会接受大小满足 `d*h*w=ins` 的三维张量输入，并按照 `w` 加权求和后变换为大小为 `ous*1*1` 的三维张量。



#### 偏置层（BIAS）

头文件：`Layers/bias.h`

定义了偏置层类型 `BIAS`，其特殊成员如下：

|                 成员                  |                         含义/作用                         |
| :-----------------------------------: | :-------------------------------------------------------: |
|                `int d`                |                     输入张量的通道数                      |
|                `int h`                |                      输入张量的高度                       |
|                `int w`                |                      输入张量的宽度                       |
|              `float* b`               |                     权重存储起始地址                      |
|   `void init(int& m,SHAPE3D Input)`   | 初始化层级参数，额外利用 `Input` 的三维大小初始化 `d,h,w` |
| `void build(float*& wei,float*& tmp)` |    为层级分配权重、偏导储存空间，将 `b` 初始化为全 $0$    |

`BIAS` 层将会接受大小为 `d*h*w` 的三维张量输入，并为第 $i$ 个通道的所有值增加 `b[i]` 的偏置后输出。



#### 卷积层（CONV）

头文件：`Layers/conv.h`

定义了卷积层类型 `CONV`，其特殊成员如下：

|                            成员                            |                          含义/作用                           |
| :--------------------------------------------------------: | :----------------------------------------------------------: |
|                         `int ind`                          |                       输入张量的通道数                       |
|                         `int inh`                          |                        输入张量的高度                        |
|                         `int inw`                          |                        输入张量的宽度                        |
|                         `int cnt`                          |               卷积核的个数（输出张量的通道数）               |
|                          `int ch`                          |                         卷积核的高度                         |
|                          `int cw`                          |                         卷积核的宽度                         |
|                         `int stx`                          |                   卷积核在高度方向上的步长                   |
|                         `int sty`                          |                   卷积核在宽度方向上的步长                   |
|                         `int pdx`                          |  高度方向上的 Padding 大小（上下都会补充 `pdx` 个 `pdval`）  |
|                         `int pdy`                          |  宽度方向上的 Padding 大小（左右都会填充 `pdy` 个 `pdval`）  |
|                       `float pdval`                        |                         Padding 的值                         |
|                         `int ouh`                          |                        输出张量的高度                        |
|                         `int ouw`                          |                        输出张量的宽度                        |
|                         `float* w`                         |                       权重存储起始地址                       |
|                      `void init(...)`                      |               初始化层级参数[*](#CONV::init())               |
| `void build(float*& wei,float*& tmp,int InitType=INIT_HE)` | 为层级分配权重、偏导储存空间并按照 `InitType` 的方式初始化 `w`（`Xavier` 或 `HE`） |

<span id="CONV::init()">*</span>：`init()` 函数将初始化卷积层参数并计算出 `ouh` 和 `ouw`，详细声明及特殊参数含义如下：

```cpp
void init(int& m,
    SHAPE3D Input,
    int CoreCnt,std::pair<int,int> Core,
    std::pair<int,int> Stride={1,1},
    std::pair<int,int> Padding={0,0},float PaddingVal=0)
```

|             参数             |                          含义/作用                           |
| :--------------------------: | :----------------------------------------------------------: |
|       `SHAPE3D Input`        |        使用 `Input` 三维的值分别初始化 `ind,inh,inw`         |
|        `int CoreCnt`         |                使用该值初始化卷积核个数 `cnt`                |
|  `std::pair<int,int> Core`   |      初始化卷积核大小，`ch=Core.first, cw=Core.second`       |
| `std::pair<int,int> Stride`  |    初始化步幅大小，`stx=Stride.first, sty=Stride.second`     |
| `std::pair<int,int> Padding` | 初始化 Padding 大小，`pdx=Padding.first, pdy=Padding.second` |
|      `float PaddingVal`      |                    使用该值初始化 `pdval`                    |

调用 `init()` 函数后将自动初始化 `ouh=(inh+pdx*2-ch)/stx+1, ouw=(inw+pdy*2-cw)/sty+1`。

`CONV` 层接受大小为 `ind*inh*inw` 的三维张量输入，并做卷积操作后输出大小为 `cnt*ouh*ouw` 的三维张量。



#### 反卷积层（DECONV）

头文件：`Layers/deconv.h`

定义了反卷积层类型 `DECONV`，其特殊成员及其含义与 `CONV` 类型相同，但没有 `pdval` 及 `PaddingVal`，即输入张量的每个值乘上卷积核后叠加到输出张量上，`cnt*ouh*oud` 的三维张量经过参数（仅交换 `cnt` 和 `ind`）一样的 `CONV` 后会变为 `ind*inh*inw` 的三维张量。

具体细节不再赘述，详见[咕咕咕]()。



#### 池化层（POOLING）

头文件：`Layers/pooling.h`

定义了池化层类型 `POOLING`，其特殊成员如下：

|       成员       |              含义/作用              |
| :--------------: | :---------------------------------: |
|    `int ind`     |          输入张量的通道数           |
|    `int inh`     |           输入张量的高度            |
|    `int inw`     |           输入张量的宽度            |
|     `int ch`     |            池化核的高度             |
|     `int cw`     |            池化核的宽度             |
|    `int tpe`     |  池化操作类型（最大池化/均值池化）  |
|    `int stx`     |      池化核在高度方向上的步长       |
|    `int sty`     |      池化核在宽度方向上的步长       |
|    `int ouh`     |           输出张量的高度            |
|    `int ouw`     |           输出张量的宽度            |
| `void init(...)` | 初始化层级参数[*](#POOLING::init()) |

<span id="POOLING::init()">*</span>：`init()` 函数将初始化池化层参数并计算出 `ouh` 和 `ouw`，详细声明及特殊参数含义如下：

```cpp
inline void init(SHAPE3D Input,
    std::pair<int,int> Core,
    int Type=MAX_POOLING,
    std::pair<int,int> Stride={-1,-1})
```

|            参数             |                          含义/作用                           |
| :-------------------------: | :----------------------------------------------------------: |
|       `SHAPE3D Input`       |        使用 `Input` 三维的值分别初始化 `ind,inh,inw`         |
|  `std::pair<int,int> Core`  |      初始化池化核大小，`ch=Core.first, cw=Core.second`       |
|         `int Type`          | 初始化池化操作类型，`MAX_POOLING` 表示最大池化，`MEAN_POOLING` 表示均值池化 |
| `std::pair<int,int> Stride` | 初始化步幅大小，`stx=Stride.first, sty=Stride.second`，特别的，若某一项为 `-1` 则表示该项取池化核的对应参数 |

调用 `init()` 函数后将自动初始化 `ouh=(inh+stx-1)/stx, ouw=(inw+sty-1)/sty`。

`POOLING` 层接受大小为 `ind*inh*inw` 的三维张量输入，并做池化操作后输出大小为 `ind*ouh*ouw` 的三维张量。



#### 拓展层（EXT）

头文件：`Layers/ext.h`

定义了拓展层类型 `EXT`，其效果是将每个位置的值在原地复制若干份（变胖），特殊成员如下：

|                        成员                        |                          含义/作用                           |
| :------------------------------------------------: | :----------------------------------------------------------: |
|                     `int ind`                      |                       输入张量的通道数                       |
|                     `int inh`                      |                        输入张量的高度                        |
|                     `int inw`                      |                        输入张量的宽度                        |
|                     `int filx`                     |                           填充高度                           |
|                     `int fily`                     |                           填充宽度                           |
|                     `int ouh`                      |                        输出张量的高度                        |
|                     `int ouw`                      |                        输出张量的宽度                        |
| `void init(SHAPE3D Input,std::pair<int,int> Fill)` | 初始化层级参数，使用 `Input` 三维的值分别初始化 `ind,inh,inw`，使用 `Fill` 两维的值分别初始化 `filx` 和 `fily` |

调用 `init()` 函数后将自动初始化 `ouh=inh*filx, ouw=inw*fily`。

`EXT` 层接受大小为 `ind*inh*inw` 的三维张量输入，将每个值在原地变为 `filx*fily` 的值相等的矩形后输出大小为 `ind*ouh*ouw` 的三维张量。



#### 批归一化层（BN）

头文件：`Layers/bn.h`

定义了批归一化层类型 `BN`，特殊成员如下：

|                             成员                             |                          含义/作用                           |
| :----------------------------------------------------------: | :----------------------------------------------------------: |
|                           `int d`                            |                       输入张量的通道数                       |
|                           `int h`                            |                        输入张量的高度                        |
|                           `int w`                            |                        输入张量的宽度                        |
|                        `float delta`                         |                    滑动平均参数 $\delta$                     |
|                         `float eps`                          |      极小量 $\epsilon$，防止让方差变得很小以至于除以零       |
|                          `float* k`                          |                       系数数组起始地址                       |
|                          `float* b`                          |                       偏置数组起始地址                       |
|                        `float* e_avg`                        |             均值的滑动平均，用于测试时的前向过程             |
|                        `float* e_var`                        |             方差的滑动平均，用于测试时的前向过程             |
| `void init(int& m,SHAPE3D Input,float Delta=0.9,float EPS=1e-4)` | 初始化层级参数，使用 `Input` 三维的值分别初始化 `d,h,w`，使用 `Delta` 和 `EPS` 分别初始化 `delta` 和 `eps` |
|            `void build(float*& wei,float*& tmp)`             | 为层级分配权重、偏导储存空间并将 `k` 和 `e_var` 初始化为全 $1$，`b` 和 `e_avg` 初始化为全 $0$ |

`BN` 层将会**对所有批的输入数据一起操作**，为输入张量的每个通道做批归一化，而输出的三维张量形状不变。具体的，假设这是第 $i$ 个通道，将通道内部所有位置所有批的值拿出来放入数组 $a$ 中（假设共 $n$ 个），则在训练阶段得到对应的输出 $\overline{a}$ 的流程为：
$$
\mu_i=\frac{1}{n}\sum\limits_{j=1}^{n} a_i\\
\sigma_i=\frac{1}{n}\sum\limits_{j=1}^{n} (a_j-\mu_i)^2\\
\overline{a}_j=k_i\frac{a_j-\mu_i}{\sqrt{\sigma_i+\epsilon}}+b_i
$$
并且 `e_avg` 和 `e_var` 在每次训练阶段的前向过程都会做如下更新：
$$
e\_avg_i=\delta\times e\_avg_i +(1-\delta)\times \mu_i\\
e\_var_i=\delta\times e\_var_i +(1-\delta)\times \sigma_i\\
$$
在测试阶段，由于数据量较小，均值和方差往往不够准确，故采用之前的滑动平均来计算输出：
$$
\overline{a}_j=k_i\frac{a_j-e\_avg_i}{\sqrt{e\_var_i+\epsilon}}+b_i
$$


#### 组归一化层（GN）

头文件：`Layers/gn.h`

定义了组归一化层类型 `GN`，特殊成员如下：

|                       成员                       |                          含义/作用                           |
| :----------------------------------------------: | :----------------------------------------------------------: |
|                     `int d`                      |                       输入张量的通道数                       |
|                     `int h`                      |                        输入张量的高度                        |
|                     `int w`                      |                        输入张量的宽度                        |
|                     `int g`                      |                         每组的通道数                         |
|                   `float eps`                    |      极小量 $\epsilon$，防止让方差变得很小以至于除以零       |
|                    `int cnt`                     |                             组数                             |
|                    `float* k`                    |                       系数数组起始地址                       |
|                    `float* b`                    |                       偏置数组起始地址                       |
| `void init(int& m,SHAPE3D Input,float EPS=1e-4)` | 初始化层级参数，使用 `Input` 三维的值分别初始化 `d,h,w`，使用 `EPS` 初始化 `eps` |
|      `void build(float*& wei,float*& tmp)`       | 为层级分配权重、偏导储存空间并将 `k` 初始化为全 $1$，`b` 初始化为全 $0$ |

调用 `init()` 函数后将自动初始化 `cnt=d/g+(d%g!=0)`。

`GN` 层将会**对每个批的数据分别操作**，将输入张量每连续的至多 $g$ 个通道分为一组，共 $cnt$ 组，每组内做和 `BN` 层大致相同的归一化操作，而输出的三维张量形状不变。

由于是对每个批的数据分别操作，故测试时的前向过程和训练时一致。



#### Softmax 归一化层（Softmax）

头文件：`Layers/Softmax.h`

定义了 Softmax 归一化层类型 `SOFTMAX`，特殊成员如下：

|            成员            |                        含义/作用                        |
| :------------------------: | :-----------------------------------------------------: |
|          `int d`           |                    输入张量的通道数                     |
|          `int h`           |                     输入张量的高度                      |
|          `int w`           |                     输入张量的宽度                      |
| `void init(SHAPE3D Input)` | 初始化层级参数，使用 `Input` 三维的值分别初始化 `d,h,w` |

`SOFTMAX` 层将会对输入张量沿着通道做 Softmax 操作，输出张量三维形状不变，即对于位置 $x,y$ 上的 $d$ 个值，设其分别为 $a_{[1,d]}$，则输出 $\overline a_{[1,d]}$ 的计算方法如下：
$$
\overline{a}_i=\frac{e^{a_i}}{\sum\limits_{j=1}^d e^{a_j}}
$$


#### 各种激活函数层

公共特殊成员：

|           成员           |                          含义/作用                           |
| :----------------------: | :----------------------------------------------------------: |
|        `int siz`         |                  输入张量的大小（`d*h*w`）                   |
| `void init(int Siz,...)` | 初始化层级参数，第一个参数及其含义固定，使用 `Siz` 初始化 `siz`，若有更多参数将给出说明 |

各种激活函数层将对输入张量的每个数值分别应用对应的激活函数 $f$，即 $x_{\text{out}}=f(x_{\text{in}})$，输出张量三维形状不变。



##### ReLU 层（RELU）

头文件：`Layers/ReLU.h`

定义了 ReLU 层类型 `RELU`：
$$
f(x)=\max(x,0)
$$

##### Leaky_ReLU 层（LEAKY_RELU）

头文件：`Layers/Leaky_ReLU.h`

定义了 Leaky_ReLU 层类型 `LEAKY_RELU`，其特殊成员如下：

|                 成员                  |                  含义/作用                  |
| :-----------------------------------: | :-----------------------------------------: |
|               `float a`               |       激活函数 $f$ 中的参数 $\alpha$        |
| `void init(int Siz,float Alpha=0.01)` | 初始化层级参数，额外使用 `Alpha` 初始化 `a` |

激活函数表达式如下：
$$
f(x)=\begin{cases}\alpha x&x<0\\x&x\ge 0\end{cases}
$$

##### Sigmoid 层（SIGMOID）

头文件：`Layers/Sigmoid.h`

定义了 Sigmoid 层类型 `SIGMOID`：
$$
f(x)=\frac{1}{1+e^{-x}}
$$

##### Tanh 层（TANH）

头文件：`Layers/Tanh.h`

定义了 Tanh 层类型 `TANH`：
$$
f(x)=\frac{e^x-e^{-x}}{e^x+e^{-x}}
$$


### 自动保存/读取/释放空间

头文件：`auto_saveload.h`，定义宏 `ENABLE_AUTO_SL` 以启用。

原理是使用构造函数创建反射，依次执行预设优化器和所有预设层级的对应函数。



#### 基础用法

在用户定义的神经网络类中，将预设优化器的声明放置于所有预设层级声明的前面，并且不能有多个预设优化器。

在预设优化器声明前加上 `AUTO_SL_BEG` 关键字，在所有预设层级声明的末尾加上 `AUTO_SL_BEG` 关键字。

将会自动定义类的三个成员变量 `save`、`load` 和 `delthis`，并自动定义其 `()` 运算符。

例子：

```cpp
class network
{
AUTO_SL_BEG
    ADAM opt;
    FC fc1,fc2,fc3;
    CONV c1,c2;
AUTO_SL_END
}test;
```

使用 `test.save(path)` 和 `test.load(path)` 以保存到文件或从文件中读取。

使用 `test.delthis()` 以释放该神经网络类预设优化器及各预设层级占用的内存空间。



#### 进阶用法

对于用户自己定义的层级/优化器，可以使用如下几个宏自动生成对应的构造函数：

```cpp
AUTO_SL_LAYER_CONSTRUCTER_WEIGHT_DELTHISFUNC(classname)
AUTO_SL_LAYER_CONSTRUCTER_WEIGHT(classname)
AUTO_SL_LAYER_CONSTRUCTER_DELTHISFUNC(classname)
AUTO_SL_LAYER_CONSTRUCTER(classname)

AUTO_SL_OPTIMIZER_CONSTRUCTER(classname)
```

具体详见 `auto_saveload.h` 中的注释及源代码。



### 图片读写及其它文件操作

头文件 `file_io.h`

定义了若干文件操作（包含图片读写）函数：

|                             函数                             |                          含义/作用                           |
| :----------------------------------------------------------: | :----------------------------------------------------------: |
|            `void readf(std::ifstream& inf,T& x)`             |     从二进制输入流 `inf` 中读取类型 `T` 的数据到 `x` 中      |
|        `void readf(std::ifstream& inf,T* x,int siz)`         | 从二进制输入流 `inf` 中读取连续的 `siz` 个类型 `T` 的数据到 `x` 起始的数组中 |
|               `writf(std::ofstream& ouf,T x)`                |      将类型 `T` 的数据 `x` 输出到二进制输出流 `ouf` 中       |
|           `writf(std::ofstream& ouf,T* x,int siz)`           | 将从 `x` 起始的连续 `siz`个类型 `T` 的数据依次输出到二进制输出流 `ouf` 中 |
| `void readimg(std::string path,int& d,int& h,int& w,float* img,float l=-1,float r=1)` | 读取 `path` 对应的图片文件，将其通道数和高度、宽度分别储存在 `d,h,w` 中，将其每个像素点每个通道的值归一化到 $[l,r]$ 后按 `d*h*w` 的格式储存在 `img` 中 |
| `void readimg(std::string path,float* img,float l=-1,float r=1)` |                含义基本同上，但不储存三维大小                |
| `void savejpg(std::string path,int d,int h,int w,float* img,float l=-1,float r=1,int quality=100)` | 将起始地址为 `img` 的一张归一化到 $[l,r]$ 的 `d*h*w` 的图片以 `jpg` 的格式保存在文件 `path` 中，图片质量为 `quality` |
| `void savepng(std::string path,int d,int h,int w,float* img,float l=-1,float r=1)` |      含义基本同上，但保存格式为 `png` 且无图片质量参数       |
| `void savebmp(std::string path,int d,int h,int w,float* img,float l=-1,float r=1)` |               含义基本同上，但保存格式为 `bmp`               |
| `void getfiles(std::string path,std::vector<std::string>& files)` | 获取目录 `path` 下的所有文件，并将其路径保存到 `files` 中（也会获取更深的所有子目录中的文件） |



### 其它头文件

|    头文件     |                          含义/作用                           |
| :-----------: | :----------------------------------------------------------: |
|   `stb/*.h`   | 开源库 [stb](https://github.com/nothings/stb) 中的若干头文件 |
| `fast_calc.h` |                    定义了快速矩阵乘法函数                    |
|  `defines.h`  |                   各头文件的公共定义、引用                   |
|  `network.h`  |                            库入口                            |



### 基于 [MNIST](https://github.com/Rebxe/MNIST) 的 demo

```cpp
#include <iostream>
#include <cstdlib>
#include <cstdio>
#include <chrono>

#define ENABLE_OPENMP
#define THREAD_NUM 6
#define ENABLE_AUTO_SL

#include "../../network_h/network.h"

using namespace std;

const int T = 60000, TEST_T = 10000;
const int Batch_Size = 60;
const float lrt = 0.001;
const int total_batch = 5000, calctme = 5;

struct NETWORK
{
	AUTO_SL_BEG
		ADAM opt;
		
		CONV c1;
		BN b1;
		LEAKY_RELU a1;
		POOLING p1;
		
		CONV c2;
		BN b2;
		LEAKY_RELU a2;
		POOLING p2;
		
		FC fc1;
		BIAS bi1;
		LEAKY_RELU a3;
		
		FC fc2;
		BIAS bi2;
		SOFTMAX sfm1;
	AUTO_SL_END
	
	float in[Batch_Size * 28 * 28];
	val3d out;
	
	inline void init()
	{
		opt.init(lrt);
		
		c1.init(opt.m,(SHAPE3D){1,28,28},8,{3,3},{1,1},{1,1},0);
		b1.init(opt.m,(SHAPE3D){8,28,28});
		a1.init(8*28*28);
		p1.init((SHAPE3D){8,28,28},{2,2});
		
		c2.init(opt.m,(SHAPE3D){8,14,14},16,{3,3},{1,1},{1,1},0);
		b2.init(opt.m,(SHAPE3D){16,14,14});
		a2.init(16*14*14);
		p2.init((SHAPE3D){16,14,14},{2,2});
		
		fc1.init(opt.m,16*7*7,128);
		bi1.init(opt.m,(SHAPE3D){128,1,1});
		a3.init(128);
		
		fc2.init(opt.m,128,10);
		bi2.init(opt.m,(SHAPE3D){10,1,1});
		sfm1.init((SHAPE3D){10,1,1});
		
		opt.build();
		
		float *wei=opt._wei(),*tmp=opt._tmp();
		c1.build(wei,tmp),b1.build(wei,tmp);	
		c2.build(wei,tmp),b2.build(wei,tmp);
		fc1.build(wei,tmp),bi1.build(wei,tmp);
		fc2.build(wei,tmp,INIT_XAVIER),bi2.build(wei,tmp); 
	}
	inline void forward(bool test)
	{
		auto_dao::init(test?0:Batch_Size);
		val3d x(1,28,28,in);
		x=c1(x),x=b1(x),x=a1(x),x=p1(x);
		x=c2(x),x=b2(x),x=a2(x),x=p2(x);
		x=fc1(x),x=bi1(x),x=a3(x);
		x=fc2(x),x=bi2(x),x=sfm1(x);
		out=x;
	}
	inline float backward(float *rout)
	{
		opt.init_backward();
		auto_dao::init_backward();
		float res=MSEloss(out,rout);
		out.backward();
		opt.flush();
		return res;
	}
};

float casin[T + 5][28 * 28];
int casans[T + 5];

float outs[Batch_Size * 10];
float total_loss;

NETWORK brn;

inline void loaddata(string imgpath,string anspath,int T)
{
	FILE* fimg = fopen(imgpath.c_str(), "rb");
	FILE* fans = fopen(anspath.c_str(), "rb");
	if (fimg == NULL)
	{
		puts("加载图片数据失败\n");
		system("pause");
		exit(1);
	}
	if (fans == NULL)
	{
		puts("加载答案数据失败\n");
		system("pause");
		exit(1);
	}
	fseek(fimg, 16, SEEK_SET);
	fseek(fans, 8, SEEK_SET);
	unsigned char* img = new unsigned char[28 * 28];
	for (int cas = 1; cas <= T; cas++)
	{
		fread(img, 1, 28 * 28, fimg);
		for (int i = 0; i < 28 * 28; i++) casin[cas][i] = img[i] / (float)255;
		unsigned char num;
		fread(&num, 1, 1, fans);
		casans[cas] = num;
	}
	delete[] img;
	fclose(fimg), fclose(fans);
}

void train()
{
	int cins = 0, couts = 0;
	for (int tb = 1; tb <= Batch_Size; tb++)
	{
		int cas = (rand() * (RAND_MAX + 1) + rand()) % T + 1;
		for (int i = 0; i < 28 * 28; i++) brn.in[cins++] = casin[cas][i];
		for (int i = 0; i < 10; i++) outs[couts++] = casans[cas] == i;
	}
	brn.forward(false);
	total_loss+=brn.backward(outs);
}

inline bool test(int cas)
{
	for (int i = 0; i < 28 * 28; i++) brn.in[i] = casin[cas][i];
	brn.forward(true);
	int mxid = 0;
	for (int i = 1; i < 10; i++) if (brn.out.a[i] > brn.out.a[mxid]) mxid = i;
	return mxid == casans[cas];
}

int main()
{
	printf("模式选择：\n");
	printf("[1] 加载 AI 并测试\n");
	printf("[2] 训练 AI（最好的 AI 模型将会保存到 ./best.ai）\n");
	int mode;
	scanf("%d", &mode);
	system("cls");
	string imgpath = "./MNIST/img", 
		   anspath = "./MNIST/ans", 
		   testimgpath = "./MNIST/testimg",
		   testanspath = "./MNIST/testans";
	printf("训练图片文件：%s\n", imgpath.c_str());
	printf("训练答案文件：%s\n", anspath.c_str());
	printf("评估图片文件：%s\n", testimgpath.c_str());
	printf("评估答案文件：%s\n\n", testanspath.c_str());
	if (mode == 1)
	{
		printf("请输入之前保存的 AI 路径\n");
		string path;
		cin >> path;
		brn.load(path);
	}
	else
	{
		printf("加载数据中...\n");
		loaddata(imgpath,anspath,T);
		printf("加载数据完成\n\n");
		brn.init();
		total_loss = 0;
		printf("开始训练...\n\n");
    	auto start = std::chrono::high_resolution_clock::now();
		for (int i = 1; i <= total_batch; i++)
		{
			train();
			if (i % calctme == 0)
			{
				total_loss /= (float)calctme;
				printf("[%.2f%%] 训练了 %d 组样本，平均 loss %f\n", i / (float)total_batch * 100, i * Batch_Size, total_loss);
				total_loss = 0;
			}
		}
	    auto stop = std::chrono::high_resolution_clock::now();
	    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count();
		brn.save("best.ai");
		printf("\n训练完成！共训练 %d ms，模型已保存到 best.ai\n\n",(int)duration);
	}
	printf("加载测试数据中...\n");
	loaddata(testimgpath,testanspath,TEST_T);
	printf("加载测试数据完成\n\n");
	printf("开始模型评估...\n");
	int tot = 0;
	for (int i = 1; i <= TEST_T; i ++) tot += test(i);
	printf("模型评估完成，正确率：%.2f%%\n\n", (float)tot / TEST_T * 100);
	brn.delthis();
	system("pause");
	return 0;
}
```

