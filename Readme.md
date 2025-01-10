### 简介

基于 C++14 的仅头文件的神经网络库，代码可读，方便研究神经网络的实现。

计算加速：

- CPU 加速：基于 OpenMP 和 AVX256 指令集；
- GPU 加速：基于微软 `amp.h`；（效果不太好）

CPU 代码支持大部分编译器，可用基于 GCC 的 DEV-C++ 编译；GPU 代码仅支持 VS 系列编译器编译。

支持自动求偏导（反向传播），用户仅需定义前向过程。

支持读取/保存图片文件，使用了开源库 [stb](https://github.com/nothings/stb)。

本文仅介绍库的使用方法，关于机器学习的原理部分请移步：[咕咕咕]()

### 张量、自动反向传播

头文件 `auto_dao.h`。

#### 命名空间 `auto_dao`

|                   成员                   |                          含义/作用                           |
| :--------------------------------------: | :----------------------------------------------------------: |
|             `int Batch_Size`             |                            批大小                            |
|               `bool test`                | 前向过程中使用，为 `true` 则代表当前为推导阶段而非训练阶段，会影响批归一化算子的具体操作 |
|              `struct node`               |                   内部结构体，用户无需访问                   |
|         `std::vector<node*> tmp`         | 内部变量，记录当前申请的所有 `node` 的地址，方便释放内存及初始化反向传播 |
| `void init(int BatchSize,bool flg_test)` | 前向过程前必须执行的函数，释放现有所有张量占用的内存空间，初始化 `Batch_Size` 及 `test` |
|            `init_backward()`             |   反向传播前必须执行的函数，初始化现有所有张量以便反向传播   |

#### 三维张量 `val3d`

定义了可自动求偏导（反向传播）的三维张量类型 `val3d`，张量会同时存储整个批次中的数据（所以它实际上是四维的）：

|                    成员                    |                          含义/作用                           |
| :----------------------------------------: | :----------------------------------------------------------: |
|                  `int d`                   |                         张量的通道数                         |
|                  `int h`                   |                          张量的高度                          |
|                  `int w`                   |                          张量的宽度                          |
|                 `float* a`                 | 张量数值的起始地址（数值按照 `auto_dao::Batch_Size*d*h*w` 的方式存储） |
|                `float* da`                 | 张量偏导的起始地址（偏导按照 `auto_dao::Batch_Size*d*h*w` 的方式存储，与数值一一对应） |
|                 `val3d()`                  |                默认构造函数（不进行任何操作）                |
| `val3d(int td,int th,int tw,float val=0)`  | 构造函数，初始化一个 `td*th*tw` 的张量，其每个位置的数值都是 `val` |
|  `val3d(int td,int th,int tw,float *dat)`  | 构造函数，根据 `dat[0]` 到 `dat[bs*td*th*tw-1]` 中的数值初始化一个 `td*th*tw` 的张量，其中若 `auto_dao::test` 为 `true` 则 `bs=1`，否则 `bs=auto_dao::Batch_Size` |
|                `backward()`                |                   将该张量的偏导传递下去*                    |
|           `auto_dao::node *dat`            |                    内部变量，用户无需访问                    |

*：调用 `backward` 时，会将该张量标记为”偏导计算完成“状态，并将该张量的偏导反向传播至对其有影响的张量处。若本次操作导致某个张量的偏导计算完成（影响到的所有张量的偏导都已传播至该张量），则会自动调用其 `backward` 函数（类似 DAG 上反向 bfs）。故**用户仅需在手动计算所有输出端张量的偏导后，手动调用所有输出端张量的 `backward` 函数**。

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
|  `float MSEloss(val3d x,float* realout)`   |                                                              |

#### demo

```cpp
float arr[2*2*2]={0.1,0.2,
                  0.3,0.4,

                  0.1,0.2,
                  0.3,0.4};
auto_dao::init(1,false); // 准备前向过程，批大小 1，当前是训练阶段 
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

头文件：`Optimizer\*.h`

命名规则：全大写命名

在本库中，权重及其在反向传播中求得的偏导统一存储于优化器中，方便统一更新。

统一公有成员：

|                       成员                       |                          含义/作用                           |
| :----------------------------------------------: | :----------------------------------------------------------: |
|                     `int bs`                     |                            批大小                            |
|                     `int m`                      |                           权重数量                           |
|                   `float lrt`                    |                            学习率                            |
| `void init(int Batch_Size,float Learn_Rate,...)` | 初始化优化器参数，该函数的前两个参数及含义固定（批大小和学习率），根据不同优化器具体情况可能有更多参数 |
|                  `void build()`                  |           初始化优化器，为权重及其偏导分配内存空间           |
|         `void save(std::ofstream& ouf)`          | 将优化器参数（不包括批大小）及权重保存到二进制文件流 `ouf` 中 |
|  `void load(std::ifstream& inf,int Batch_Size)`  | 从二进制文件流 `inf` 中读取优化器参数及权重，并将批大小初始化为 `Batch_Size` |
|                 `void delthis()`                 |                      释放申请的内存空间                      |
|                 `float* _wei()`                  |                     获取权重数组起始地址                     |
|                 `float* _tmp()`                  |                     获取偏导数组起始地址                     |
|              `void init_backward()`              |        清空累计的偏导，即将偏导数组置零，准备反向传播        |
|                  `void flush()`                  |       利用当前累计的偏导更新权重，在反向传播完成后调用       |
|                   默认构造函数                   | 仅在启用宏 `ENABLE_AUTO_SL` 时才有默认构造函数，用于自动生成神经网络的保存和读取函数 |

参数命名规则和主流的神经网络库大致相同。

#### SGD 优化器

头文件：`Optimizer\SGD.h`

定义了优化器类型 `SGD`，其所有公有成员均无特殊。

参数更新方式：（$w$ 为参数，$\Delta$ 为偏导）
$$
w_i\to w_i-lrt\times \Delta_i
$$

#### Adam 优化器

头文件：`Optimizer\Adam.h`

定义了优化器类型 `ADAM`，其特殊成员如下：

|                             成员                             |                          含义/作用                           |
| :----------------------------------------------------------: | :----------------------------------------------------------: |
|                          `float b1`                          |                  参数更新公式中的 $\beta_1$                  |
|                          `float b2`                          |                  参数更新公式中的 $\beta_2$                  |
|                         `float eps`                          | 参数更新公式中的 $\epsilon$，一个很小的非负实数，防止除以 $0$ |
| `void init(int Batch_Size,float Learn_Rate,float beta1=0.9,float beta2=0.999,float Eps=1e-8)` | 初始化优化器参数，`b1`，`b2` 和 `eps` 分别初始化为 `beta1`，`beta2` 和 `Eps` |

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

### 预设网络层级（单层算子）

头文件：`Layers\*.h`

命名规则：全大写命名

统一公有成员：

|      |      |
| :--: | :--: |
|      |      |
|      |      |
|      |      |
|      |      |
|      |      |
|      |      |
|      |      |
|      |      |
|      |      |



- `init`：初始化算子参数；
- `build`：为算子分配权重、偏导储存空间，并初始化权重（没有权重的层算子就没有该函数）；
- `operator()`：传入一个 `val3d`，返回一个 `val3d`，表示在传入的对象上应用该算子；
- `save/load`：保存到文件流中、从文件流中读取；
- `delthis`：释放空间；

参数命名规则和主流的神经网络库大致相同，`init` 的第一个参数 `int &m` 传入的是权重数量的计数器，一般传入 优化器的成员变量 `m`；`build` 的两个 `float*` 参数传入的分别是权重存储起始地址、偏导存储起始地址。

### 自动保存/读取器

定义宏 `ENABLE_AUTO_SL` 以启用。

在用户定义的神经网络类中，将优化器的声明放置于所有层级声明的前面，并且不能有多个优化器。

在优化器声明前加上 `AUTO_SL_BEG` 关键字，在所有层级声明的末尾加上 `AUTO_SL_BEG` 关键字。

将会自动定义类的两个成员函数 `save` 和 `load`。

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

使用 `test.save(path)` 和 `test.load(path,Batch_Size)` 以保存到文件或从文件中读取。

### 其它头文件

- `fast_calc.h`：快速矩阵乘法相关；
- `file_io.h`：文件读写相关；
- `defines.h`：公共部分，定义、链接；
- `network.h`：库入口；

### 基于 MNIST 的例子

```cpp
#include <iostream>
#include <cstdlib>
#include <cstdio>
#include <chrono>

#define ENABLE_OPENMP 0
#define THREAD_NUM 6
#define ENABLE_AUTO_SL 0

#include "./network_h/network.h"

using namespace std;

const int HEI = 28, WID = 28;
const int N = HEI * WID, Q = 10;
const int T = 60000, TESTT = 10000;
const int Batch_Size = 60, total_batch = 5000, calctme = 5;

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
		SIGMOID a4;
	AUTO_SL_END
	
	float in[Batch_Size * N];
	val3d out;
	
	inline void init()
	{
		opt.init(Batch_Size,0.001);
		
		c1.init(opt.m,Batch_Size,(SHAPE3D){1,28,28},8,{3,3},{1,1},{1,1},0);
		b1.init(opt.m,Batch_Size,(SHAPE3D){8,28,28});
		a1.init(Batch_Size,8*28*28);
		p1.init(Batch_Size,(SHAPE3D){8,28,28},{2,2});
		
		c2.init(opt.m,Batch_Size,(SHAPE3D){8,14,14},16,{3,3},{1,1},{1,1},0);
		b2.init(opt.m,Batch_Size,(SHAPE3D){16,14,14});
		a2.init(Batch_Size,16*14*14);
		p2.init(Batch_Size,(SHAPE3D){16,14,14},{2,2});
		
		fc1.init(opt.m,Batch_Size,16*7*7,128);
		bi1.init(opt.m,Batch_Size,(SHAPE3D){128,1,1});
		a3.init(Batch_Size,128);
		
		fc2.init(opt.m,Batch_Size,128,10);
		bi2.init(opt.m,Batch_Size,(SHAPE3D){10,1,1});
		a4.init(Batch_Size,10);
		
		opt.build();
		
		float *wei=opt._wei(),*tmp=opt._tmp();
		c1.build(wei,tmp),b1.build(wei,tmp);	
		c2.build(wei,tmp),b2.build(wei,tmp);
		fc1.build(wei,tmp),bi1.build(wei,tmp);
		fc2.build(wei,tmp,INIT_XAVIER),bi2.build(wei,tmp); 
	}
	inline void forward(bool test)
	{
		auto_dao::init(Batch_Size,test);
		val3d x(1,28,28,in);
		x=c1(x),x=b1(x),x=a1(x),x=p1(x);
		x=c2(x),x=b2(x),x=a2(x),x=p2(x);
		x=fc1(x),x=bi1(x),x=a3(x);
		x=fc2(x),x=bi2(x),x=a4(x);
		out=x;
	}
	inline void backward()
	{
		opt.init_backward();
		auto_dao::init_backward();
		out.backward();
		opt.flush();
	}
};

float casin[T + 5][N + 5];
int casans[T + 5];

float outs[Batch_Size * Q];
float total_loss;

NETWORK brn;

inline float lossfunc(float out[], float rout[], float dout[])
{
	float res = 0;
	for (int i = 0; i < Batch_Size * Q; i++)
	{
		res += 0.5 * (out[i] - rout[i]) * (out[i] - rout[i]);
		dout[i] = out[i] - rout[i];
	}
	return res;
}

void train()
{
	int cins = 0, couts = 0;
	for (int tme = 1; tme <= Batch_Size; tme++)
	{
		int cas = (rand() * (RAND_MAX + 1) + rand()) % T + 1;
		for (int i = 0; i < N; i++) brn.in[cins++] = casin[cas][i];
		for (int i = 0; i < Q; i++) outs[couts++] = casans[cas] == i;
	}
	brn.forward(false);
	total_loss+=lossfunc(brn.out.a,outs,brn.out.da)/Batch_Size;
	brn.backward();
}

inline bool test(int cas)
{
	for (int i = 0; i < N; i++) brn.in[i] = casin[cas][i];
	brn.forward(true);
	int mxid = 0;
	for (int i = 1; i < Q; i++) if (brn.out.a[i] > brn.out.a[mxid]) mxid = i;
	return mxid == casans[cas];
}

int main()
{
	printf("模式选择：\n");
	printf("[1] 加载 AI 并测试\n");
	printf("[2] 训练 AI（最好的 AI 模型将会保存到 .\\best.ai）\n");
	int mode;
	scanf("%d", &mode);
	system("cls");
	string imgpath = "./data/MNIST/img", 
		   anspath = "./data/MNIST/ans", 
		   testimgpath = "./data/MNIST/testimg",
		   testanspath = "./data/MNIST/testans";
	printf("\n训练图片文件：%s\n", imgpath.c_str());
	printf("训练答案文件：%s\n", anspath.c_str());
	printf("评估图片文件：%s\n", testimgpath.c_str());
	printf("评估答案文件：%s\n", testanspath.c_str());
	printf("加载配置文件完成\n\n");
	if (mode == 1)
	{
		printf("请输入之前保存的 AI 路径\n");
		string path;
		cin >> path;
		brn.load(path,Batch_Size);
		printf("加载用于模型评估的测试数据中...\n");
		FILE* fimg, * fans;
		fimg = fopen(testimgpath.c_str(), "rb");
		if (fimg == NULL)
		{
			puts("加载测试图片数据失败\n");
			system("pause");
			return -1;
		}
		fans = fopen(testanspath.c_str(), "rb");
		if (fans == NULL)
		{
			puts("加载测试答案数据失败\n");
			system("pause");
			return -1;
		}
		fseek(fimg, 16, SEEK_SET);
		fseek(fans, 8, SEEK_SET);
		unsigned char* img = new unsigned char[N + 5];
		for (int cas = 1; cas <= TESTT; cas++)
		{
			fread(img, 1, N, fimg);
			for (int i = 0; i < N; i++) casin[cas][i] = img[i] / (float)255;
			unsigned char num;
			fread(&num, 1, 1, fans);
			casans[cas] = num;
		}
		delete[] img;
		fclose(fimg), fclose(fans);
		printf("加载数据完成\n\n");
		printf("开始模型评估...\n");
		int tot = 0;
		for (int i = 1; i <= TESTT; i ++) tot += test(i);
		printf("模型评估完成，正确率：%.2f%%\n\n", (float)tot / TESTT * 100);
		system("pause");
	}
	else
	{
		printf("加载数据中...\n");
		FILE* fimg, * fans;
		fimg = fopen(imgpath.c_str(), "rb");
		if (fimg == NULL)
		{
			puts("加载图片数据失败\n");
			system("pause");
			return -1;
		}
		fans = fopen(anspath.c_str(), "rb");
		if (fans == NULL)
		{
			puts("加载答案数据失败\n");
			system("pause");
			return -1;
		}
		fseek(fimg, 16, SEEK_SET);
		fseek(fans, 8, SEEK_SET);
		unsigned char* img = new unsigned char[N + 5];
		for (int cas = 1; cas <= T; cas++)
		{
			fread(img, 1, N, fimg);
			for (int i = 0; i < N; i++) casin[cas][i] = img[i] / (float)255;
			unsigned char num;
			fread(&num, 1, 1, fans);
			casans[cas] = num;
		}
		delete[] img;
		fclose(fimg), fclose(fans);
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
		printf("\n训练完成！共训练 %d ms，模型已保存到 best.ai\n\n",(int)duration);
		brn.save("best.ai");
		printf("加载用于模型评估的测试数据中...\n");
		fimg = fopen(testimgpath.c_str(), "rb");
		if (fimg == NULL)
		{
			puts("加载测试图片数据失败\n");
			system("pause");
			return -1;
		}
		fans = fopen(testanspath.c_str(), "rb");
		if (fans == NULL)
		{
			puts("加载测试答案数据失败\n");
			system("pause");
			return -1;
		}
		fseek(fimg, 16, SEEK_SET);
		fseek(fans, 8, SEEK_SET);
		img = new unsigned char[N + 5];
		for (int cas = 1; cas <= TESTT; cas++)
		{
			fread(img, 1, N, fimg);
			for (int i = 0; i < N; i++) casin[cas][i] = img[i] / (float)255;
			unsigned char num;
			fread(&num, 1, 1, fans);
			casans[cas] = num;
		}
		delete[] img;
		fclose(fimg), fclose(fans);
		printf("加载数据完成\n\n");
		printf("开始模型评估...\n");
		int tot = 0;
		for (int i = 1; i <= TESTT; i ++) tot += test(i);
		printf("模型评估完成，正确率：%.2f%%\n\n", (float)tot / TESTT * 100);
		system("pause");
	}
	return 0;
}
```

