# QM

## 第 1 章： 从波粒二象性到量子力学

本章先在熟悉的坐标空间讨论如何从波粒二象性（量子物理）过渡到量子力学，为在第 2 章一般性的学习量子力学基础做准备。

### 1.1 波粒二象性 #第一次课

什么是波粒二象性？是指几何形状，还是指运动形态？

#### 1.1.1 用波动表示粒子物理量

$$
\varepsilon=h v, \quad \vec{p}=h / \lambda
$$

其中, $\varepsilon, \vec{p}$ 是粒子物理量, $v, \lambda$ 是波动物理量。波粒二象性是指力学量的取值既具有 粒子性, 也具有波动性。不是指几何形状, 也不是指运动形态。

#### 1.1.2 用粒子表示波动物理量

$$
(2)
$$

在原子的半经典描述中, 电子具有确定的分离轨道。“确定轨道”是经典的, “分 离轨道”是量子的。跃迂过程的能量守恒

$$
h v=\varepsilon_{m}-\varepsilon_{n}
$$

辐射的能量与两条轨道相关 $\rightarrow$ 力学量表现为一个矩阵, 取值与两个指标相关。

如何在力学 (粒子) 的框架内统一波与粒子两个方面? $\rightarrow$ 量子力学 $1.2$ 几率解释

#### 1.2.1 电子双缝衍射实验

实验结果:

只开缝 1, 屏上有波动条纹, 强度分布 $I_{1}(x)=\left|\psi_{1}(x)\right|^{2}, \psi_{1}(x)$ 为对应的波 只开缝 2, 屏上有波动条纹, 强度分布 $I_{2}(x)=\left|\psi_{2}(x)\right|^{2}, \psi_{2}(x)$ 为对应的波 同时开两缝, 屏上有波动条纹, 强度分布 $I(x)=\left|\psi_{1}(x)+\psi_{2}(x)\right|^{2} \neq I_{1}+I_{2}$, 电子具 有波的衍射特性。

实验分析:

一次只发射一个电子, 屏上出现位置不确定的光斑分布, 长时间后出现衍射条纹。 光斑说明粒子性, 但位置不确定说明统计性, 故不是经典粒子, 是统计意义上的粒 子 (位置不确定, 但位置的平均值确定)；衍射条纹说明波动性, 但长时间说明统计 性, 故不是经典波, 是统计意义上的波。说明粒子的位置这个力学量具有统计意义 上的波粒二象性。

一个电子有波粒二象性是微观粒子的固有特性, 不是多个粒子相互作用的结果, 不是经典的 (多粒子) 统计结果。每次发射一个电子, 即使初态完全相同, 也仍具 有上述统计性, 而每一次丢一枚硬币, 若初始条件完全相同, 则每一次结果同。

#### 1.2.2 Born 统计解释

引入几率波函数 $\psi(\vec{x}, t)$ 来表述粒子位置的几率幅，

衍射条纹的强度 $\propto \begin{cases}\text { 波的振幅的平方 }|\psi(\vec{x}, t)|^{2} & \text { (从波动性出发) } \\ \text { 粒子出现的几率 } \rho(\vec{x}, t) & \text { (从粒子性出发) }\end{cases}$

有

$$
\rho(\vec{x}, t) \propto|\psi(\vec{x}, t)|^{2}=\psi^{*}(\vec{x}, t) \psi(\vec{x}, t)
$$

注意已考虑几率波为复函数的一般情况。

用统计性统一了粒子与波。把波的效应包含在统计性中。关于统计意义上粒子 的力学就是量子力学。经典粒子的坐标有确定的值, 量子粒子的坐标不确定, 只有 平均值

$$
\langle\vec{x}\rangle=\frac{\int d^{3} \vec{x} \vec{x}|\psi(\vec{x}, t)|^{2}}{\int d^{3} \vec{x}|\psi(\vec{x}, t)|^{2}}
$$

确定。再次强调：经典多粒子系统具有统计性, 而量子力学的单粒子系统就有统计 性!

### 1.3 几率波的性质

#### 1.3.1几率归一化

几率解释要求几率归一。如果几率波平方 (几率) 可积

$$
\int d^{3} \vec{x}|\psi(\vec{x}, t)|^{2}=A<\infty
$$

则 $\frac{1}{\sqrt{A}} \psi(\vec{x}, t)$ 为归一化几率波。如果波函数平方不可积, 例如无相互作用粒子的平面

![](https://cdn.mathpix.com/cropped/2023_03_03_9fa09cf3c2310a66d23fg-2.jpg?height=43&width=851&top_left_y=1869&top_left_x=1002)
不影响相对几率,

$$
\frac{\left|\psi_{1}(\vec{x}, t)\right|^{2}}{\left|\psi_{2}(\vec{x}, t)\right|^{2}}
$$

与归一化无关。

在统计解释中, 几率波 $\psi(\vec{x}, t)$ 的意义是通过几率 $|\psi(\vec{x}, t)|^{2}$ 来定义的。但是, 归 一化后, $\psi(\vec{x}, t)$ 仍有相位不确定性，

$$
\psi(\vec{x}, t) \rightarrow e^{i \alpha} \psi(\vec{x}, t)
$$

并不改变几率分布。这导致问题：统计解释是否包含了波的全部信息?

经典波是不要求归一的。经典波 $|\psi(\vec{x}, t)|^{2}$ 是能量密度。 $\psi(\vec{x}, t)$ 和 $C \psi(\vec{x}, t)$ 是完全 不同的波, 后者能量密度是前者 $C^{2}$ 倍。

##### 2 几率的单值连续有限性

1) 几率分布 $|\psi(\vec{x}, t)|^{2}$ 单值 $\rightarrow$ 几率波 $\psi(\vec{x}, t)$ 单值;
2) 相互作用系统的几率可归一, 几率分布 $|\psi(\vec{x}, t)|^{2}$ 有限 $\rightarrow$ 几率波 $\psi(\vec{x}, t)$ 有限;
3) 几率分布 $|\psi(\vec{x}, t)|^{2}$ 是否连续, 或者几率波 $\psi(\vec{x}, t)$ 是否连续与相互作用有关, 下面会详细讨论。

总结: 单值、有限、连续是一般条件下几率解释对统计波函数的物理约束条件。

### 1.4 Schroedinger 方程 #第二次课

量子力学不同于经典力学之处就是波粒二象性的波对于粒子性质的影响，即力 学量 (坐标) 取值一般不确定, 而是按照统计几率取值: 位于 $\vec{x}$ 的几率为 $|\psi(\vec{x}, t)|^{2}$ 。 因此问题归结于如何计算几率波 $\psi(\vec{x}, t)$

#### 1.4.1几率波的演化方程 (Schroedinger 方程, 1926)

$$
i \hbar \frac{\partial}{\partial t} \psi(\vec{x}, t)=\left(-\frac{\hbar^{2} \vec{\nabla}^{2}}{2 m}+V(\vec{x}, t)\right) \psi(\vec{x}, t)
$$

$V(\vec{x}, t)$ 是相互作用势。

对于自由粒子，

$$
i \hbar \frac{\partial}{\partial t} \psi(\vec{x}, t)=-\frac{\hbar^{2} \vec{\nabla}^{2}}{2 m} \psi(\vec{x}, t)
$$

用分离变量法可以证明由粒子动量 $\vec{p}$ 和粒子能量 $\varepsilon=\frac{\vec{p}^{2}}{2 m}$ 构成的平面波

$$
\psi(\vec{x}, t)=A e^{\frac{i}{\hbar}(\vec{p} \cdot \vec{x}-\varepsilon)}=A e^{i(\vec{k} \cdot \vec{x}-\omega t)}, \quad \text { 波矢 } \vec{k}=\frac{\vec{p}}{\hbar}, \text { 频率 } \omega=\frac{\varepsilon}{\hbar}
$$

是方程的解

1) Schroedinger 方程是几率波的基本运动方程, 地位如同经典力学中的牛顿 方程, 是量子力学基本假定之一。

2) 在 Schroedinger 方程中包含虚因子 $i$, 要求波函数 $\psi(\vec{x}, t)$ 必为复函数。所以 平面几率波只能是 $A e^{(\vec{k} \cdot \vec{x}-\omega t)}$, 不能是 $A \cos (\vec{k} \cdot \vec{x}-\omega t)$ 。

3）显然, Schroedinger 方程无相对论协变性, 时间是一阶微分, 空间是二阶微 分, 描述的是非相对论体系 (本课程讨论的都是非相对论量子力学, 相对论量子力 学的基本方程是 Dirac 方程, 见高等量子力学课程)。

#### 1.4.2 几率守恒

因为相互作用势是一个实函数，有

$$
\begin{gathered}
i \hbar \frac{\partial}{\partial t} \psi(\vec{x}, t)=\left(-\frac{\hbar^{2} \vec{\nabla}^{2}}{2 m}+V(\vec{x}, t)\right) \psi(\vec{x}, t) \\
-i \hbar \frac{\partial}{\partial t} \psi^{*}(\vec{x}, t)=\left(-\frac{\hbar^{2} \vec{\nabla}^{2}}{2 m}+V(\vec{x}, t)\right) \psi^{*}(\vec{x}, t)
\end{gathered}
$$

将第一个方程乘 $\psi^{*}(\vec{x}, t)$, 将第二个方程乘 $\psi(\vec{x}, t)$, 然后相减, 得到关于几率的连续 性方程 (流守恒方程)

$$
\begin{gathered}
\frac{\partial}{\partial t} \rho(\vec{x}, t)+\vec{\nabla} \cdot \vec{\jmath}(\vec{x}, t)=0 \\
\rho(\vec{x}, t)=|\psi(\vec{x}, t)|^{2} \\
\vec{\jmath}(\vec{x}, t)=-\frac{i \hbar}{2 m}\left(\psi^{*}(\vec{x}, t) \vec{\nabla} \psi(\vec{x}, t)-\psi(\vec{x}, t) \vec{\nabla} \psi^{*}(\vec{x}, t)\right)
\end{gathered}
$$

$\rho$ 是几率密度, $\vec{J}$ 的物理意义是什么? 对连续性方程在有限空间 $V$ 积分:

![](https://cdn.mathpix.com/cropped/2023_03_03_377fced270eaf42233f5g-1.jpg?height=85&width=117&top_left_y=1999&top_left_x=410)

$$
\begin{gathered}
\int_{V} d^{3} \vec{x} \frac{\partial}{\partial t} \rho(\vec{x}, t)+\int_{V} d^{3} \vec{x} \vec{\nabla} \cdot \vec{\jmath}(\vec{x}, t)=0 \\
\frac{d}{d t} \int_{V} d^{3} \vec{x} \rho(\vec{x}, t)=-\int_{S} d \vec{S} \cdot \vec{\jmath}(\vec{x}, t)
\end{gathered}
$$

说明定域几率守恒: 体积 $V$ 内几率的变化=流出面积 $\vec{S}$ 的几率, 故 $\vec{\jmath}(\vec{x}, t)$ 是几率流密度。

若对整个空间积分, $V \rightarrow \infty$, 由几率波在无穷远处为零 (有相互作用情况) 或

者周期性条件 (自由粒子情况)，有

$$
\frac{d}{d t} \int_{\infty} d^{3} \vec{x} \rho(\vec{x}, t)=-\int_{\infty} d \vec{S} \cdot \vec{\jmath}(\vec{x}, t)=0
$$

$$
\int_{\infty} d^{3} \vec{x} \rho(\vec{x}, t)=\text { 常数 }
$$

说明总几率守恒。相互作用过程中无粒子的产生与消灭。若几率波是可以归一的, 则常数 $=1$, 且归一化与时间无关, Schroedinger方程保证了归一性不随时间而变。

几率守恒说明, 量子力学 (无论非相对论还是相对论形式) 不能描述粒子的产 生与消灭，原则上必须用量子场论处理。是否可以用量子力学来等效的处理粒子的 衰变呢? 假如相互作用势 $V$ 是复函数，则没有上面的连续性方程，几率不守恒，说明 可以用 Schroedinger 方程等效地描述粒子的衰变。设

$$
V=V_{R}-i V_{I}
$$

$V_{R}, V_{I}$ 均为 $(\vec{x}, t)$ 的实函数。重复上面的流守恒推导过程, 复势的 Schroedinger 方程 导致连续性方程

$$
\frac{\partial}{\partial t} \rho(\vec{x}, t)+\vec{\nabla} \cdot \vec{\jmath}(\vec{x}, t)=-\frac{2}{\hbar} V_{I}(\vec{x}, t) \rho(\vec{x}, t)
$$

对全空间积分,

$$
\frac{d}{d t} \int_{\infty} d^{3} \vec{x} \rho(\vec{x}, t)=-\frac{2}{\hbar} \int_{\infty} d^{3} \vec{x} V_{I}(\vec{x}, t) \rho(\vec{x}, t)
$$

当 $V_{I}>0$, 总几率减少, 粒子发生了衰变, 单位时间内减少的几率

$$
R(t)=\frac{2}{\hbar} \int_{\infty} d^{3} \vec{x} V_{I}(\vec{x}, t) \rho(\vec{x}, t)
$$

当 $V_{I}<0$, 总几率增加, 产生了粒子, $|R|$ 为单位时间内增加的几率。

#### 1.4.3 几率波的叠加

Schroedinger 方程是关于 $\psi(\vec{x}, t)$ 的线性方程（ $V$ 与几率波 $\psi$ 无关）。若 $\psi_{1}, \psi_{2}, \ldots \ldots \psi_{N}$ 是方程的解, 则它们的任意线性迭加

$$
\psi=\sum_{n=1}^{N} c_{n} \psi_{n}
$$

仍是方程的解, 仍是描述体系的几率波函数。

### 1.5 态函数

粒子处于位置 $\vec{x}$ 的几率分布 $\rho(\vec{x}, t)=|\psi(\vec{x}, t)|^{2}$ 。其他力学量，例如动量 $\vec{p}$ ，的取 值几率? 如果几率波 $\psi(\vec{x}, t)$ 只能给出位置的几率分布, 而不能给出其他力学量的几 率分布, 则几率波不能完全确定体系的力学量几率分布。如果几率波能给出所有力 学量的几率分布, 则可称几率波为体系的态函数。说明知道了几率波, 则知道了体 系所有力学量的几率分布。

将几率波 $\psi(\vec{x}, t)$ 由空间平面波展开 (平面波的完备性, 付里叶展开):

$$
\psi(\vec{x}, t)=\int \frac{d^{3} \vec{p}}{(2 \pi \hbar)^{3 / 2}} \varphi(\vec{p}, t) e^{\frac{i}{\hbar} \vec{p} \cdot \vec{x}}
$$

由平面波的正交归一,

$$
\begin{aligned}
\int d^{3} \vec{x}\left(\frac{1}{(2 \pi \hbar)^{3 / 2}} e^{\frac{i}{\hbar} \vec{p} \cdot \vec{x}}\right)\left(\frac{1}{(2 \pi \hbar)^{3 / 2}} e^{\frac{i}{\hbar^{\prime}} \cdot \vec{x}}\right)^{*} & =\int d^{3} \vec{x} \frac{1}{(2 \pi \hbar)^{3}} e^{\frac{i}{\hbar}\left(\vec{p}-\vec{p}^{\prime}\right) \cdot \vec{x}} \\
& =\delta\left(\vec{p}-\vec{p}^{\prime}\right)
\end{aligned}
$$

有

$$
\varphi(\vec{p}, t)=\int \frac{d^{3} \vec{x}}{(2 \pi \hbar)^{3 / 2}} \psi(\vec{x}, t) e^{-\frac{i}{\hbar} \vec{p} \cdot \vec{x}}
$$

问题: $\psi(\vec{x}, t)$ 是坐标几率幅, $\varphi(\vec{p}, t)$ 的物理意义是什么? 由

$$
\langle\vec{x}\rangle=\int d^{3} \vec{x} \vec{x}|\psi(\vec{x}, t)|^{2}=\int d^{3} \vec{x} \vec{x} \psi^{*}(\vec{x}, t) \psi(\vec{x}, t)
$$

有

$$
\frac{d\langle\vec{x}\rangle}{d t}=\int d^{3} \vec{x} \vec{x}\left(\frac{\partial \psi^{*}(\vec{x}, t)}{\partial t} \psi(\vec{x}, t)+\psi^{*}(\vec{x}, t) \frac{\partial \psi(\vec{x}, t)}{\partial t}\right)
$$

由 Schroedinger 方程, 有

$$
\frac{d\langle\vec{x}\rangle}{d t}=\frac{i \hbar}{2 m} \int d^{3} \vec{x} \vec{x} \vec{\nabla} \cdot\left(\psi^{*}(\vec{x}, t) \vec{\nabla} \psi(\vec{x}, t)-\psi(\vec{x}, t) \vec{\nabla} \psi^{*}(\vec{x}, t)\right)
$$

由矢量微分公式, 对于任意的矢量 $\vec{A}, \vec{B}$,

$$
\vec{\nabla} \times(\vec{A} \times \vec{B})=(\vec{B} \cdot \vec{\nabla}) \vec{A}-(\vec{A} \cdot \vec{\nabla}) \vec{B}+(\vec{\nabla} \cdot \vec{B}) \vec{A}-(\vec{\nabla} \cdot \vec{A}) \vec{B}
$$

取

$$
\vec{A}=\vec{x}, \quad \vec{B}=\psi^{*}(\vec{x}, t) \vec{\nabla} \psi(\vec{x}, t)-\psi(\vec{x}, t) \vec{\nabla} \psi^{*}(\vec{x}, t)
$$

有

$$
\vec{x}(\vec{\nabla} \cdot \vec{B})=\vec{\nabla} \times(\vec{x} \times \vec{B})-(\vec{B} \cdot \vec{\nabla}) \vec{x}+(\vec{x} \cdot \vec{\nabla}) \vec{B}+(\vec{\nabla} \cdot \vec{x}) \vec{B}
$$

右边第二项

$$
(\vec{B} \cdot \vec{\nabla}) \vec{x}=\vec{B}
$$

第三第四项

$$
(\vec{x} \cdot \vec{\nabla}) \vec{B}+(\vec{\nabla} \cdot \vec{x}) \vec{B}=\frac{\partial}{\partial x}(x \vec{B})+\frac{\partial}{\partial y}(y \vec{B})+\frac{\partial}{\partial z}(z \vec{B})
$$

故第一项和第三第四项均为全微分, 考虑到几率波在无穷远处为零或者周期性条件,

对于积分没贡献, 只有第二项有积分贡献,

$$
\frac{d\langle\vec{x}\rangle}{d t}=-\frac{i \hbar}{2 m} \int d^{3} \vec{x}\left(\psi^{*}(\vec{x}, t) \vec{\nabla} \psi(\vec{x}, t)-\psi(\vec{x}, t) \vec{\nabla} \psi^{*}(\vec{x}, t)\right)
$$

再对括号中第二部分进行分部积分, 有 (平均速度)

$$
\frac{d\langle\vec{x}\rangle}{d t}=-\frac{i \hbar}{m} \int d^{3} \vec{x} \psi^{*}(\vec{x}, t) \vec{\nabla} \psi(\vec{x}, t)
$$

平均动量 (量子力学的力学量平均值与经典力学的力学量对应)

$$
\langle\vec{p}\rangle=m \frac{d\langle\vec{x}\rangle}{d t}=\int d^{3} \vec{x} \psi^{*}(\vec{x}, t)(-i \hbar \vec{\nabla}) \psi(\vec{x}, t)
$$

代入 $\psi(\vec{x}, t)$ 的付里叶展开式

$$
\begin{aligned}
\langle\vec{p}\rangle & =\int \frac{d^{3} \vec{x} d^{3} \vec{p}_{1} d^{3} \vec{p}_{2}}{(2 \pi \hbar)^{3}} \varphi^{*}\left(\vec{p}_{1}, t\right) e^{-\frac{i}{\hbar} \vec{p}_{1} \cdot \vec{x}}(-i \hbar \vec{\nabla}) \varphi\left(\vec{p}_{2}, t\right) e^{\frac{i}{\hbar} \vec{p}_{2} \cdot \vec{x}} \\
& =\int \frac{d^{3} \vec{x} d^{3} \vec{p}_{1} d^{3} \vec{p}_{2}}{(2 \pi \hbar)^{3}} \varphi^{*}\left(\vec{p}_{1}, t\right) \vec{p}_{2} \varphi\left(\vec{p}_{2}, t\right) e^{\frac{i}{\hbar}\left(\vec{p}_{2}-\vec{p}_{1}\right) \cdot \vec{x}} \\
& =\int d^{3} \vec{p}_{1} d^{3} \vec{p}_{2} \varphi^{*}\left(\vec{p}_{1}, t\right) \vec{p}_{2} \varphi\left(\vec{p}_{2}, t\right) \delta\left(\vec{p}_{2}-\vec{p}_{1}\right) \\
& =\int d^{3} \vec{p} \varphi^{*}(\vec{p}, t) \vec{p} \varphi(\vec{p}, t)=\int d^{3} \vec{p} \vec{p}|\varphi(\vec{p}, t)|^{2}
\end{aligned}
$$

表明 $|\varphi(\vec{p}, t)|^{2}$ 是动量取值为 $\vec{p}$ 的几率, $\varphi(\vec{p}, t)$ 是动量几率幅。

由分析力学, 坐标动量是基本的力学量, 其它力学量可由坐标动量表示。有了 由 Schroedinger 方程确定的坐标空间的几率波 $\psi(\vec{x}, t)$, 所有力学量的几率分布就确 定了, 故称几率波 $\psi(\vec{x}, t)$ 为态函数。由于给定 $\varphi(\vec{p}, t)$, 与给定 $\psi(\vec{x}, t)$ 等价, 故动量空 间的 $\varphi(\vec{p}, t)$ 也可以称之为系统的态函数。

态函数满足的 Schroedinger 方程也就称为态方程。

### 1.6 算符与对易关系

比较坐标空间中力学量 $\vec{x}$ 和 $\vec{p}$ 的平均值,

$$
\begin{gathered}
\langle\vec{x}\rangle=\int d^{3} \vec{x} \psi^{*}(\vec{x}, t) \vec{x} \psi(\vec{x}, t) \\
\langle\vec{p}\rangle=\int d^{3} \vec{x} \psi^{*}(\vec{x}, t)(-i \hbar \vec{\nabla}) \psi(\vec{x}, t)
\end{gathered}
$$

可引入坐标算符和动量算符

$$
\hat{\vec{x}}=\vec{x}, \quad \hat{\vec{p}}=-i \hbar \vec{\nabla}
$$

有

$$
\langle\vec{x}\rangle=\int d^{3} \vec{x} \psi^{*}(\vec{x}, t) \hat{\vec{x}} \psi(\vec{x}, t)
$$

对于平面波 $\psi(\vec{x}, t)=A e^{\frac{i}{\hbar}(\vec{p} \cdot \vec{x}-\varepsilon)}$, 有

$$
\langle\vec{p}\rangle=\frac{\int d^{3} \vec{x} \psi^{*}(\vec{x}, t) \hat{\vec{p}} \psi(\vec{x}, t)}{\int d^{3} \vec{x} \psi^{*}(\vec{x}, t) \psi(\vec{x}, t)}=\vec{p}
$$

说明在平面波态, 动量有确定值, 能量有确定值 $\varepsilon=\frac{\vec{p}^{2}}{2 m}$ 。

对于一般经典力学量 $O(\vec{x}, \vec{p})$, 可定义对应的量子力学算符

$$
O(\vec{x}, \vec{p}) \rightarrow \hat{O}(\hat{\vec{x}}, \hat{\vec{p}})
$$

量子力学平均值

$$
\langle O\rangle=\int d^{3} \vec{x} \psi^{*}(\vec{x}, t) \hat{O} \psi(\vec{x}, t)
$$

例如哈密顿量

$$
H=\frac{\vec{p}^{2}}{2 m}+V(\vec{x}, t) \rightarrow \widehat{H}=\frac{\hat{\vec{p}}^{2}}{2 m}+V(\vec{x}, t)=-\frac{\hbar^{2} \vec{\nabla}^{2}}{2 m}+V(\vec{x}, t)
$$

Schroedinger 方程可以简写为

$$
i \hbar \frac{\partial}{\partial t} \psi=\left(-\frac{\hbar^{2} \vec{\nabla}^{2}}{2 m}+V\right) \psi=\widehat{H} \psi
$$

由于动量算符是空间的微分, 它和坐标是不对易的。例如在一维情形, 有

$$
\left(\hat{x} \hat{p}_{x}-\hat{p}_{x} \hat{x}\right) \psi(x, t)=-i \hbar\left(x \partial_{x}-\partial_{x} x\right) \psi(x, t)=i \hbar \psi(x, t)
$$

由于 $\psi(x, t)$ 是任意的态函数，有

$$
\hat{x} \hat{p}_{x}-\hat{p}_{x} \hat{x}=\left[\hat{x}, \hat{p}_{x}\right]=i \hbar
$$

对于三维情形，容易证明,

$$
\left[\hat{x}_{i}, \hat{x}_{j}\right]=0, \quad\left[\hat{p}_{i}, \hat{p}_{j}\right]=0, \quad\left[\hat{x}_{i}, \hat{p}_{j}\right]=i \hbar \delta_{i j}
$$

由于波粒二象性, 导致力学量取值不确定, 按照几率分布取值, 使得力学量不 再是一个 $\vec{x}$ 和 $\vec{p}$ 的函数（可对易），而是要用算符表示（不对易）

 #作业: Griffiths, Problems 1.7,1.8 and 1.15

### 1.7 定态 Schroedinger 方程 #第三次课

Schroedinger 方程

$$
i \hbar \frac{\partial}{\partial t} \psi(\vec{x}, t)=\widehat{H} \psi(\vec{x}, t)
$$

如果哈密顿量不显含时间, $\widehat{H}(\vec{x})$, 即相互作用势不显含时间, $V(\vec{x})$, 可将态函数分 离变量，

$$
\psi(\vec{x}, t)=\varphi(\vec{x}) f(t)
$$

则

$$
\frac{1}{f(t)} i \hbar \frac{d}{d t} f(t)=\frac{1}{\varphi(\vec{x})} \widehat{H} \varphi(\vec{x})=E
$$

$E$ 为不依赖于时空的分离变量常数。Schroedinger 方程的特解为

$$
\begin{gathered}
f(t)=A e^{-\frac{i}{\hbar} E t} \\
\widehat{H} \varphi(\vec{x})=E \varphi(\vec{x}) \\
\psi(\vec{x}, t)=\varphi(\vec{x}) e^{-\frac{i}{\hbar} E t}
\end{gathered}
$$

分离变量常数 $E$ 就是哈密顿量本征方程的本征值。求解 Schroedinger 方程归结于求 解哈密顿量本征方程, 即能量本征方程, 确定本征值 $E$ 和本征态 $\varphi(\vec{x})$ 。

注意: 能量本征态 $\psi_{E}(\vec{x}, t)$ 的线性叠加

$$
\psi(\vec{x}, t)=\sum_{E} C_{E} \psi_{E}(\vec{x}, t)=\sum_{E} C_{E} \varphi_{E}(\vec{x}) e^{-\frac{i}{\hbar} E t}
$$

仍然是 Schroedinger 方程的解

$$
i \hbar \frac{\partial}{\partial t} \psi(\vec{x}, t)=\sum_{E} C_{E} E \varphi_{E}(\vec{x}) e^{-\frac{i}{\hbar} E t}=\sum_{E} C_{E} \widehat{H} \varphi_{E}(\vec{x}) e^{-\frac{i}{\hbar} E t}=\widehat{H} \psi(\vec{x}, t)
$$

但是不再满足能量本征方程

$$
\widehat{H} \psi(\vec{x}, t)=\sum_{E} C_{E} \widehat{H} \varphi_{E}(\vec{x}) e^{-\frac{i}{\hbar} E t}=\sum_{E} C_{E} E \varphi_{E}(\vec{x}) e^{-\frac{i}{\hbar} E t} \neq E \psi(\vec{x}, t)
$$

不再是能量本征态

能量本征态的性质：

1) 几率密度 $\rho(\vec{x}, t)=|\psi(\vec{x}, t)|^{2}=|\varphi(\vec{x})|^{2}=\rho(\vec{x}, 0)$ 与时间无关。
	几率流密度

$$
\begin{array}
\vec{\jmath}(\vec{x}, t)=-\frac{i \hbar}{2 m}\left(\psi^{*}(\vec{x}, t) \vec{\nabla} \psi(\vec{x}, t)-\psi(\vec{x}, t) \vec{\nabla} \psi^{*}(\vec{x}, t)\right)\\=-\frac{i \hbar}{2 m}\left(\varphi^{*}(\vec{x}) \vec{\nabla} \varphi(\vec{x})-\varphi(\vec{x}) \vec{\nabla} \varphi^{*}(\vec{x})\right)=\vec{\jmath}(\vec{x}, 0)
\end{array}
$$
与时间无关。

任意不显含时间 $t$ 的力学量 $O(\vec{x}, \vec{p})$ 的平均值

$$
\begin{aligned}
\langle O\rangle(t) & =\int d^{3} \vec{x} \psi^{*}(\vec{x}, t) \hat{O}(\hat{\vec{x}}, \hat{\vec{p}}) \psi(\vec{x}, t) \\
& =\int d^{3} \vec{x} \varphi^{*}(\vec{x}) \hat{O}(\hat{\vec{x}}, \hat{\vec{p}}) \varphi(\vec{x})=\langle O\rangle(0)
\end{aligned}
$$

与时间无关。==说明哈密顿量不显含时间时, 任意力学量在能量本征态的性质不随时间改变, 故能量本征方程称为定态 Schroedinger 方程, 能量本征态称为定态。==

2) 若势 $V(\vec{x})$ 具有空间反演不变形 $V(-\vec{x})=V(\vec{x})$ ，则

$$
\begin{gathered}
\left(-\frac{\hbar^{2} \vec{\nabla}^{2}}{2 m}+V(\vec{x})\right) \varphi(\vec{x})=E \varphi(\vec{x}) \\
\left(-\frac{\hbar^{2} \vec{\nabla}^{2}}{2 m}+V(\vec{x})\right) \varphi(-\vec{x})=E \varphi(-\vec{x})
\end{gathered}
$$

说明 $\varphi(\vec{x})$ 和 $\varphi(-\vec{x})$ 是定态方程属于同一能量 $E$ 的两个解。若定态不简并 (即本征值与本征态一一对应, 简并是指一个本征值对应几个本征态),

$$
\varphi(-\vec{x})=C \varphi(\vec{x})
$$

即
$$
\varphi(\vec{x})=C \varphi(-\vec{x})=C^{2} \varphi(\vec{x}), \quad C^{2}=1, \quad C=\pm 1
$$

即定态满足
$$
\begin{array}{rlrl}
\varphi(-\vec{x}) & =\varphi(\vec{x}) & \text { 称为偶宇称态 } \\
\varphi(-\vec{x}) & =-\varphi(\vec{x}) & \text { 称为奇宇称态 }
\end{array}
$$

==当哈米顿量具有空间反演对称性时，无简并定态或为偶宇称态, 或为奇宇称态。==

### 1.8 束缚态

由于某种相互作用, 粒子只能在有限空间运动,

$$
\text { 几率密度 } \lim _{\vec{x} \rightarrow \infty} \rho(\vec{x}) \rightarrow 0, \text { 几率波 } \lim _{\vec{x} \rightarrow \infty} \varphi(\vec{x}) \rightarrow 0
$$

称态 $\varphi(\vec{x})$ 为束缚态。

#### 1.8.1 一维束缚态无简并

若 $\varphi_{1}(x), \varphi_{2}(x)$ 是属于同一能量 $E$ 的两个定态, 由能量本征方程

$$
\begin{aligned}
& \varphi_{1}^{\prime \prime}+\frac{2 m}{\hbar^{2}}(E-V(x)) \varphi_{1}=0 \\
& \varphi_{2}^{\prime \prime}+\frac{2 m}{\hbar^{2}}(E-V(x)) \varphi_{2}=0
\end{aligned}
$$

将第一个方程乘以 $\varphi_{2}$, 将第二个方程乘以 $\varphi_{1}$, 再相减,

$$
\begin{gathered}
\varphi_{1} \varphi_{2}^{\prime \prime}-\varphi_{2} \varphi_{1}^{\prime \prime}=0 \\
\left(\varphi_{1} \varphi_{2}^{\prime}-\varphi_{2} \varphi_{1}^{\prime}\right)^{\prime}=0 \\
\varphi_{1} \varphi_{2}^{\prime}-\varphi_{2} \varphi_{1}^{\prime}=C
\end{gathered}
$$

常数 $C$ 与坐标 $x$ 无关。对于束缚态,
$$
\varphi_{i}(x \rightarrow \infty) \rightarrow 0, \quad C=0
$$
有
$$
\varphi_{1} \varphi_{2}^{\prime}-\varphi_{2} \varphi_{1}^{\prime}=0
$$
在 $\varphi_{i}(x) \neq 0$ 的区间（感兴趣的区间）, 有
$$
\begin{gathered}
\frac{\varphi_{1}^{\prime}}{\varphi_{1}}=\frac{\varphi_{2}^{\prime}}{\varphi_{2}} \\
\left(\ln \frac{\varphi_{1}}{\varphi_{2}}\right)^{\prime}=0, \quad \ln \frac{\varphi_{1}}{\varphi_{2}}=\ln C^{\prime}, \quad \varphi_{1}=C^{\prime} \varphi_{2}
\end{gathered}
$$
说明 $\varphi_{1}, \varphi_{2}$ 为同一个态, 一维束缚态无简并。由此, 当哈密顿量具有空间反演对称 性, 即 $V(-x)=V(x)$, 一维束缚态有确定宇称, $C^{\prime}=\pm 1$ 。

#### 1.8.2 一维几率波的连续性

$$
\varphi^{\prime \prime}=-\frac{2 m}{\hbar^{2}}(E-V(x)) \varphi
$$
若 $V(x)$ 连续, 则 $\varphi^{\prime \prime}, \varphi^{\prime}, \varphi$ 均连续。

若 $V(x)$ 在 $x=a$ 不连续, 有突变 $\Delta V$, 则 $\varphi$ "不连续。在不连续点附近的邻域积分,
$$
\varphi^{\prime}(a+\epsilon)-\varphi^{\prime}(a-\epsilon)=-\frac{2 m}{\hbar^{2}} \int_{a-\epsilon}^{a+\epsilon}(E-V(x)) \varphi(x) d x
$$

若突变 $\Delta V$ 有限, 则积分 $\rightarrow 0, \varphi^{\prime}, \varphi$ 均连续。

若 $\Delta V \rightarrow \infty$, 积分
$$
\int_{a-\epsilon}^{a+\epsilon}(E-V(x)) \varphi(x) d x=\left\{\begin{array}{cc}
\infty, & \varphi^{\prime}, \varphi \text { 均不连续 } \\
\text { 有限, } & \varphi^{\prime} \text { 不连续, } \varphi \text { 连续 } \\
0, & \varphi^{\prime}, \varphi \text { 均连续 }
\end{array}\right.
$$

*例 1*: $\delta$ 势阱的束缚态
$$
V(x)=-\gamma \delta(x)
$$
在 $x \neq 0$ ， 定态方程
$$
\varphi^{\prime \prime}+\frac{2 m}{\hbar^{2}} E \varphi=0
$$
考虑 $E<0$ 的情形, 令 $k^{2}=-\frac{2 m E}{\hbar^{2}}$, 有
$$
\varphi(x)= \begin{cases}A e^{k x}+A^{\prime} e^{-k x} & x<0 \\ C e^{k x}+C^{\prime} e^{-k x} & x>0\end{cases}
$$
由于 $x<0$ 与 $x>0$ 两个区间被 $x=0$ 处的 $\delta$ 势阱分开, 几率波不一定相同, 故一般情 况下常数 $A, A^{\prime}$ 不同于 $C, C^{\prime}$ 。

要求束缚态条件 $\varphi(\pm \infty)=0$, 有
$$
A^{\prime}=C=0, \quad \varphi(x)=\left\{\begin{array}{cc}
A e^{k x} & x<0 \\
C^{\prime} e^{-k} & x>0
\end{array}\right.
$$
注意: 若取 $E>0$, 为振荡解, 不满足束缚态条件。

由于 $V(-x)=V(x)$, 具有空间反演对称性, 故一维束缚态有确定宇称, 即偶宇称态
$$
\varphi_{s}(x)=\varphi_{s}(-x)= \begin{cases}A e^{k x} & x<0 \\ A^{e^{-k}} & x>0\end{cases}
$$
或奇宇称态
$$
\varphi_{a}(x)=-\varphi_{a}(-x)= \begin{cases}A e^{k x} & x<0 \\ -A^{e^{-k}} & x>0\end{cases}
$$
由于 $\varphi_{s}(x)$ 在 $x=0$ 处连续, $\varphi_{s}\left(0^{+}\right)=\varphi_{s}\left(0^{-}\right)=A$, 满足几率单值的条件, 但 $\varphi_{a}(x)$ 在 $x=0$ 处不连续, $\varphi_{a}\left(0^{+}\right)=-A, \varphi_{a}\left(0^{-}\right)=A$, 不满足几率单值的条件。故不 存在 $\varphi_{a}(x)$, 只有 $\varphi_{s}(x)$ 。

常数 $A$ 由归一化决定。

现在由 $\varphi_{s}^{\prime}(x)$ 的不连续性确定束缚态能量 $E<0$ 。将 $\varphi_{s}(x)$ 在 $x=0$ 处的定态方程 积分，有
$$
\begin{gathered}
\varphi_{s}^{\prime}\left(0^{+}\right)-\varphi_{s}^{\prime}\left(0^{-}\right)=-\frac{2 m}{\hbar^{2}} \int_{0^{-}}^{0^{+}}(E+\gamma \delta(x)) \varphi_{s}(x) d x=-\frac{2 m}{\hbar^{2}} \gamma \varphi_{s}(0) \\
\text { 将 } \varphi_{s}(0)=A, \varphi_{s}^{\prime}\left(0^{+}\right)=-k A, \varphi_{s}^{\prime}\left(0^{-}\right)=k A \text { 代入, 有 } \\
-2 k A=-\frac{2 m}{\hbar^{2}} \gamma A
\end{gathered}
$$
则
$$
k=\frac{m}{\hbar^{2}} \gamma, \quad E=-\frac{\hbar^{2} k^{2}}{2 m}=-\frac{m \gamma^{2}}{2 \hbar^{2}}
$$
说明 $\delta$ 势阱中只有一个束缚态。 

*例 2*：方势阱的束缚态

![](https://cdn.mathpix.com/cropped/2023_03_03_9fa09cf3c2310a66d23fg-2.jpg?height=211&width=274&top_left_y=89&top_left_x=411)

定态方程
$$
\begin{cases}\varphi^{\prime \prime}+\frac{2 m}{\hbar^{2}} E \varphi=0 & |x|>a \\ \varphi^{\prime \prime}+\frac{2 m}{\hbar^{2}}\left(E+V_{0}\right) \varphi=0 & |x|<a\end{cases}
$$
取 $-V_{0}<E<0$ (束缚态能量)，令
$$
\begin{gathered}
k^{2}=-\frac{2 m E}{\hbar^{2}}, \quad k^{\prime 2}=\frac{2 m\left(E+V_{0}\right)}{\hbar^{2}} \\
\left\{\begin{array}{lr}
\varphi^{\prime \prime}-k^{2} \varphi=0 & |x|>a \\
\varphi "+k^{\prime 2} \varphi=0 & |x|<a
\end{array}\right. \\
\varphi(x)=\left\{\begin{array}{lr}
A e^{k x}+A^{\prime} e^{-k} & x<-a \\
B e^{i k^{\prime} x}+B^{\prime} e^{-i k^{\prime} x} & |x|<a \\
C e^{k x}+C^{\prime} e^{-k x} & x>a
\end{array}\right.
\end{gathered}
$$
要求束缚态条件 $\varphi(\pm \infty)=0$, 有
$$
\varphi(x)=\left\{\begin{array}{lr}
A e^{k x} & x<-a \\
B e^{i k^{\prime} x}+B^{\prime} e^{-i k^{\prime} x} & |x|<a \\
C^{\prime} e^{-k x} & x>a
\end{array}\right.
$$
由于 $V(-x)=V(x)$, 一维束缚态有确定宇称, 即偶宇称态和奇宇称态
$$\varphi_{s}(x)=\varphi_{s}(-x)=\left\{\begin{array}{lr}A e^{k x} & x<-a \\ B e^{i k^{\prime} x}+B e^{-i k^{\prime} x}=2 B \cos \left(k^{\prime} x\right) & |x|< a \\ A e^{-k x} & x>a\end{array}\right.$$

$$
\varphi_{a}(x)=-\varphi_{a}(-x)=\left\{\begin{array}{lc}
A e^{k x} & x<-a \\
B e^{i k^{\prime} x}-B e^{-i k^{\prime} x}=2 i B \sin \left(k^{\prime} x\right) & |x|<a \\
-A e^{-k x} & x>a
\end{array}\right.
$$
由于在 $x=-a, a$ 处势的突变 $\Delta V$ 有限, $\varphi, \varphi^{\prime}$ 连续。先考虑偶宇称态。由 $x=-a$ 处 $\varphi_{s}, \varphi_{s}^{\prime}$ 的连续性, 有
$$
A e^{-k a}=2 B \cos \left(k^{\prime} a\right)
$$
$$
A k e^{-k a}=2 B k^{\prime} \sin \left(k^{\prime} a\right)
$$
$x=a$ 处的连续性条件给出相同的方程。故有
$$
\begin{gathered}
k^{\prime} \operatorname{tg}\left(k^{\prime} a\right)=k \rightarrow \text { 分离能谱 } E_{n} \\
B=\frac{1}{2} A e^{-k a} \sec \left(k^{\prime} a\right)
\end{gathered}
$$
$$
\varphi_{s}(x)=A\left\{\begin{array}{lr}
e^{k x} & x<-a \\
e^{-k a} \sec \left(k^{\prime} a\right) \cos \left(k^{\prime} x\right) & |x|<a \\
e^{-k x} & x>a
\end{array}\right.
$$
归一化条件决定常数 $A$,
$$
\int_{-\infty}^{\infty}\left|\varphi_{s}(x)\right|^{2} d x=1
$$
类似, 对于奇宇称态有
$$
\begin{gathered}
k^{\prime} \operatorname{ctg}\left(k^{\prime} a\right)=-k \rightarrow \text { 分离能谱 } E_{n} \\
\varphi_{a}(x)=A\left\{\begin{array}{lr}
e^{k x} & x<-a \\
-e^{-k a} \csc \left(k^{\prime} a\right) \sin \left(k^{\prime} x\right) & |x|<a \\
-e^{-k x} & x>a
\end{array}\right.
\end{gathered}
$$

 #作业: Griffiths, Problems 2.1,2.2 and $2.4$

### 1.9 一维散射态 #第四次课

散射态是粒子可以在无穷远处运动的一种状态。
![](https://cdn.mathpix.com/cropped/2023_03_03_e89a976464253a89f023g-1.jpg?height=170&width=740&top_left_y=652&top_left_x=476)

逶射平面波

![](https://cdn.mathpix.com/cropped/2023_03_03_e89a976464253a89f023g-1.jpg?height=58&width=123&top_left_y=662&top_left_x=1479)

一维散射问题: 在 $x=-\infty$ 发射向右的平面波 $e^{i k x}$ (入射波矢 $\vec{k}=\vec{p} / \hbar$ ), 经过有 限力程的相互作用势 $V(x)$, 确定 $x=-\infty$ 与 $x=\infty$ （已无相互作用）的几率波。

散射包含有弹性散射和非弹性散射。弹性散射只改变入射粒子动量的方向, 非 弹性散射可以改变粒子动量, 能量, 甚至产生或者湮灭粒子。在量子力学中只考虑 弹性散射。

不同于束缚态问题中由定态方程+约束条件决定本征态和分离的本征值，散射 问题没有无穷远处几率为零的约束条件。弹性散射过程中的能量是不变的 $\varepsilon=\frac{p^{2}}{2 m}$, 由定态方程决定散射态。

例 $3: \delta$ 势阱的散射态

$$
V(x)=-\gamma \delta(x)
$$

$x \neq 0$ 的定态方程

$$
\varphi^{\prime \prime}+\frac{2 m}{\hbar^{2}} E \varphi=0
$$

考虑 $E>0$ (散射态) 的情形，令 $k^{2}=\frac{2 m E}{\hbar^{2}}$, 有

$$
\varphi(x)= \begin{cases}A e^{i k x}+A^{\prime e^{-i k x}} & x<0 \\ C e^{i k x}+C^{\prime} e^{-i k x} & x>0\end{cases}
$$

对于边界条件为在 $x=-\infty$ 发射向右入射几率波的散射问题, 在 $x>0$ 不可能有向左 的几率波 $C^{\prime} e^{-i k x}$, 故 $C^{\prime}=0$ 。

注意： $E<0$ 不可能有散射态。 由 $x=0$ 处 $\varphi(x)$ 的连续性条件, 有

$$
A+A^{\prime}=C
$$

再由在 $x=0$ 处（势函数突变点） $\varphi(x)$ 的一阶导数满足的方程,

$$
\begin{gathered}
\varphi^{\prime}\left(0^{+}\right)-\varphi^{\prime}\left(0^{-}\right)=-\frac{2 m}{\hbar^{2}} \int_{0^{-}}^{0^{+}}(E+\gamma \delta(x)) \varphi(x) d x=-\frac{2 m}{\hbar^{2}} \gamma \varphi(0) \\
i k\left(C-A+A^{\prime}\right)=-\frac{2 m}{\hbar^{2}} \gamma\left(A+A^{\prime}\right)
\end{gathered}
$$

有

$$
A^{\prime}=\frac{i \beta}{1-i \beta} A, \quad C=\frac{1}{1-i \beta} A, \quad \beta=\frac{m \gamma}{\hbar^{2} k}
$$

由几率流的公式

$$
\vec{J}=-\frac{i \hbar}{2 m}\left(\psi^{*} \partial_{x} \psi-\psi \partial_{x} \psi^{*}\right) \vec{e}_{x}
$$

对于入射波 $A e^{i k x}$, 反射波 $A^{\prime e^{-i k x}}$ 和透射波 $C e^{i k x}$, 有入射几率流, 反射几率流和透射 几率流

$$
\vec{J}_{入}=\frac{\hbar}{m} k|A|^{2} \vec{e}_{x}, \quad \vec{J}_{\text {反 }}=-\frac{\hbar}{m} k\left|A^{\prime}\right|^{2} \vec{e}_{x}, \quad \vec{J}_{\text {透 }}=\frac{\hbar}{m} k|C|^{2} \vec{e}_{x}
$$

定义反射几率

$$
R=\frac{\left|\vec{\jmath}_{反}\right|}{\left|\vec{\jmath}_{入}\right|}=\frac{\left|A^{\prime}\right|^{2}}{|A|^{2}}=\frac{\beta^{2}}{1+\beta^{2}}
$$

和透射几率

$$
T=\frac{\left|\vec{J}_{\text {透 }}\right|}{\left|\vec{\jmath}_{入}\right|}=\frac{|C|^{2}}{|A|^{2}}=\frac{1}{1+\beta^{2}}
$$

有散射过程几率守恒

$$
R+T=1
$$

例 4：方势阱的散射态

$$
V(x)= \begin{cases}0 & |x|>a \\ -V_{0} & |x|<a\end{cases}
$$

考虑 $E>0$ 的定态方程 (散射问题)

$$
\left\{\begin{array}{ll}
\varphi^{\prime \prime}+k^{2} \varphi=0 & |x|>a \\
\varphi^{\prime \prime}+k^{\prime 2} \varphi=0 & |x|<a
\end{array} \quad k^{2}=\frac{2 m E}{\hbar^{2}}, \quad k^{\prime 2}=\frac{2 m\left(E+V_{0}\right)}{\hbar^{2}}\right.
$$

解为

$$
\varphi(x)=\left\{\begin{array}{lr}
A e^{i k x}+A^{\prime} e^{-i k x} & x<-a \\
B e^{i k^{\prime} x}+B^{\prime} e^{-i k^{\prime} x} & |x|<a \\
C e^{i k x}+C^{\prime} e^{-i k x} & x>a
\end{array}\right.
$$

对于边界条件为在 $x=-\infty$ 发射入射几率波的散射问题, $C^{\prime}=0$ 。

由 $\Delta V$ 在 $x=-a, a$ 点有限, 故 $\varphi, \varphi^{\prime}$ 在 $x=-a, a$ 处连续,

$$
\begin{gathered}
A e^{-i k a}+A^{\prime} e^{i k a}=B e^{-i k^{\prime} a}+B^{\prime} e^{i k^{\prime} a} \\
-A i k e^{-i k a}+A^{\prime} i k e^{i k a}=-B i k^{\prime} e^{-i k^{\prime} a}+B^{\prime} i k^{\prime} e^{i k^{\prime} a} \\
B e^{i k^{\prime} a}+B^{\prime} e^{-i k^{\prime} a}=C e^{i k a} \\
B i k^{i k^{\prime} a}-B^{\prime} i k^{\prime} e^{-i k^{\prime} a}=C i k e^{i k a}
\end{gathered}
$$

可用 $A$ 来表示其它 4 个常数 $A^{\prime}, B, B^{\prime}, C$ 。若只对反射系数和透射系数感兴趣, 只求反 射平面波 $A^{\prime} e^{-i k x}$ 和透射平面波 $C e^{i k x}$ 的系数 $A^{\prime}, C$ :

$$
\begin{aligned}
A^{\prime} & =\frac{i \eta / 2 \sin \left(2 k^{\prime} a\right) e^{-2 i k a}}{\cos \left(2 k^{\prime} a\right)-i \epsilon / 2 \sin \left(2 k^{\prime} a\right)} A \\
C & =\frac{e^{-2 i k a}}{\cos \left(2 k^{\prime} a\right)-i \epsilon / 2 \sin \left(2 k^{\prime} a\right)} A \\
& \epsilon=\frac{k^{\prime}}{k}+\frac{k}{k^{\prime}}, \quad \eta=\frac{k^{\prime}}{k}-\frac{k}{k^{\prime}}
\end{aligned}
$$

反射系数和透射系数

$$
\begin{aligned}
R & =\frac{\left|A^{\prime}\right|^{2}}{|A|^{2}}=\frac{\eta^{2} / 4 \sin ^{2}\left(2 k^{\prime} a\right)}{\cos ^{2}\left(2 k^{\prime} a\right)+\epsilon^{2} / 4 \sin ^{2}\left(2 k^{\prime} a\right)} \\
T & =\frac{|C|^{2}}{|A|^{2}}=\frac{1}{\cos ^{2}\left(2 k^{\prime} a\right)+\epsilon^{2} / 4 \sin ^{2}\left(2 k^{\prime} a\right)}
\end{aligned}
$$

可以证明几率守恒

$$
R+T=1
$$

考虑两种特殊情形:

a) $V_{0}=0$, 无相互作用, $k^{\prime}=k, \varepsilon=2, \eta=0$, 则 $R=0, T=1$, 全透射。

b) $V_{0} \neq 0$, 但 $\sin \left(2 k^{\prime} a\right)=0, R=0, T=1$, 共振透射, 全透射。

共振能量: $2 k^{\prime} a=n \pi$,

$$
E_{n}=-V_{0}+\frac{n^{2} \pi^{2} \hbar^{2}}{8 m a^{2}}
$$

当然, $n$ 要充分大以保证 $E_{n}>0$ 的散射条件。

讨论:

经典力学运动区间: $E=T+V>V$, 粒子不能在 $E<V$ 的区间运动。

量子力学运动区间: 由 Schroedinger 方程, 在 $V \neq \infty$ 的区间, 几率幅 $\psi \neq 0$, 粒子可以运动, 在 $V=\infty$ 的区间, $\psi=0$, 粒子不能运动。

所以，经典粒子与量子粒子的运动区间一般不同。
![](https://cdn.mathpix.com/cropped/2023_03_03_e89a976464253a89f023g-4.jpg?height=720&width=412&top_left_y=1343&top_left_x=228)

经典与量子粒子有共同的运动区间, 阱内, 经典束缚态, 量子束缚态。

假设初始粒子在阱内，

则经典粒子只能在阱内运动,

量子粒子可以向所有区间运动, 量子遂道效应。

量子与经典粒子都可以在所有区间运动。

### 1.10 两体束缚态

由于两体问题牵涉到相互之间的转动, 先要讨论轨道角动量这个力学量。

#### 1 轨道角动量

由经典力学定义

$$
\vec{L}=\vec{r} \times \vec{p}
$$

得到坐标空间的量子力学轨道角动量算符

$$
\begin{gathered}
\hat{\vec{L}}=\hat{\vec{r}} \times \hat{\vec{p}}=\vec{r} \times \hat{\vec{p}} \\
\hat{L}_{x}=y \hat{p}_{z}-z \hat{p}_{y}=-i \hbar\left(y \partial_{z}-z \partial_{y}\right) \\
\hat{L}_{y}=z \hat{p}_{x}-x \hat{p}_{z}=-i \hbar\left(z \partial_{x}-x \partial_{z}\right) \\
\hat{L}_{z}=x \hat{p}_{y}-y \hat{p}_{x}=-i \hbar\left(x \partial_{y}-y \partial_{x}\right) \\
\hat{L}^{i}=\varepsilon^{i j k} x_{j} \hat{p}_{k}, \quad i, j, k=1,2,3(x, y, z)
\end{gathered}
$$

$\varepsilon^{i j k}$ 为全反对称张量 $\left(\varepsilon^{123}=1\right)$, 上下指标相同要求和。

由坐标动量对易关系

$$
\left[\hat{x}_{i}, \hat{x}_{j}\right]=0, \quad\left[\hat{p}_{i}, \hat{p}_{j}\right]=0, \quad\left[\hat{x}_{i}, \hat{p}_{j}\right]=i \hbar \delta_{i j}
$$

容易证明轨道角动量对易关系

$$
\begin{gathered}
{\left[\hat{L}_{x}, \hat{L}_{y}\right]=i \hbar \hat{L}_{z}, \quad\left[\hat{L}_{y}, \hat{L}_{z}\right]=i \hbar \hat{L}_{x}, \quad\left[\hat{L}_{z}, \hat{L}_{x}\right]=i \hbar \hat{L}_{y}} \\
{\left[\hat{L}^{i}, \hat{L}^{j}\right]=i \hbar \varepsilon^{i j k} \hat{L}_{k}, \quad \hat{\vec{L}} \times \hat{\vec{L}}=i \hbar \hat{\vec{L}}}
\end{gathered}
$$

引入

$$
\hat{\vec{L}}^{2}=\widehat{L}_{x}^{2}+\widehat{L}_{y}^{2}+\widehat{L}_{z}^{2}
$$

描述角动量的大小, 它与各个分量是对易的,

$$
\left[\hat{\vec{L}}^{2}, \hat{L}_{i}\right]=0
$$

采用球坐标系 $(x, y, z \rightarrow r, \theta, \varphi)$,

$$
\begin{gathered}
\hat{L}_{x}=i \hbar\left(\sin \varphi \partial_{\theta}+\operatorname{ctg} \theta \cos \varphi \partial_{\varphi}\right) \\
\hat{L}_{y}=-i \hbar\left(\cos \varphi \partial_{\theta}-\operatorname{ctg} \theta \sin \varphi \partial_{\varphi}\right) \\
\hat{L}_{z}=-i \hbar \partial_{\varphi} \\
\hat{\vec{L}}^{2}=-\hbar^{2}\left(\frac{1}{\sin \theta} \partial_{\theta}\left(\sin \theta \partial_{\theta}\right)+\frac{1}{\sin ^{2} \theta} \partial_{\varphi}^{2}\right)
\end{gathered}
$$

$\hat{\vec{L}}$ 只与 2 个变量 $\theta, \varphi$ 有关 (在直角坐标系与 3 个变量 $x, y, z$ 有关)。

先看 $\hat{L}_{z}$ 的本征方程。由于 $\hat{L}_{z}$ 与 $\theta$ 无关,

$$
\hat{L}_{z} \Phi(\varphi)=-i \hbar \frac{d}{d \varphi} \Phi(\varphi)=L_{z} \Phi(\varphi)
$$

一阶微分方程的解

$$
\Phi(\varphi)=A e^{\frac{i}{\hbar^{L}} \varphi}
$$

考虑几率波的单值条件

$$
\Phi(\varphi)=\Phi(\varphi+2 \pi)
$$

和归一化, 得到本征值和本征态

$$
\begin{aligned}
L_{z}=m \hbar, \quad m & =0, \pm 1, \pm 2, \cdots \pm \infty \\
\Phi_{m}(\varphi) & =\frac{1}{\sqrt{2 \pi}} e^{i m \varphi}
\end{aligned}
$$

统计波函数的单值条件导致 $\hat{L}_{z}$ 的本征值不连续。

再看 $\widehat{\vec{L}}^{2}$ 的本征方程。由于 $\hat{\vec{L}}^{2}$ 与 $\theta$ 和 $\varphi$ 有关,

$$
\hat{\vec{L}}^{2} Y(\theta, \varphi)=-\hbar^{2}\left(\frac{1}{\sin \theta} \partial_{\theta}\left(\sin \theta \partial_{\theta}\right)+\frac{1}{\sin ^{2} \theta} \partial_{\varphi}^{2}\right) Y(\theta, \varphi)=L^{2} Y(\theta, \varphi)
$$

由于 $\hat{\vec{L}}^{2}$ 对 $\theta, \varphi$ 的微分没有交叉项, 可以用分离变量法求解 (见参考书),

$$
Y(\theta, \varphi)=P(\theta) \Phi(\varphi)
$$

得到两个关于 $P(\theta)$ 和 $\Phi(\varphi)$ 的常微分方程, 考虑 $\varphi$ 方向的单值条件

$$
\Phi(\varphi)=\Phi(\varphi+2 \pi)
$$

和 $\theta$ 方向的有限条件

$$
P(0), P(\pi) \text { 不发散 }
$$

以及归一化条件

$$
\int|Y(\theta, \varphi)|^{2} \sin \theta d \theta d \varphi=1
$$

得到本征值和本征态（球谐函数）

$$
L^{2}=l(l+1) \hbar^{2}, \quad l=0,1,2, \cdots \infty
$$

$$
Y_{l m}(\theta, \varphi)=\sqrt{\frac{(l-|m|) !(2 l+1)}{(l+|m|) ! 4 \pi}} P_{l}^{m}(\cos \theta) e^{i m \varphi}, \quad m=0, \pm 1, \pm 2, \cdots \pm l
$$

$P_{l}^{m}(x)$ 为连带 Legendre 多项式。

统计波函数的单值条件和有限条件导致 $\widehat{\vec{L}}^{2}$ 的本征值不连续。本征值 $L^{2}$ 只与角量 子数 $l$ 有关, 但本征态还与磁量子数 $m$ 有关, 简并度 $g=2 l+1$ 。 由于 $\hat{\vec{L}}^{2}$ 的本征态 $Y_{l m}(\theta, \varphi)$ 中与 $\varphi$ 有关的部分就是 $\hat{L}_{z}$ 的本征态, 故 $Y_{l m}(\theta, \varphi)$ 是 $\hat{\vec{L}}^{2}, \hat{L}_{z}$ 的共同本征态:

$$
\begin{aligned}
& \hat{\vec{L}}^{2} Y_{l m}(\theta, \varphi)=l(l+1) \hbar^{2} Y_{l m}(\theta, \varphi), \quad l=0,1,2, \cdots, \infty \\
& \hat{L}_{z} Y_{l m}(\theta, \varphi)=m \hbar Y(\theta, \varphi), \quad m=0, \pm 1, \pm 2, \cdots \pm l
\end{aligned}
$$

角动量 $(l)$ 确定时, 在任意方向的分量 $(m)$ 受限制。

 #作业: Griffiths, Problems 2.27, 2.28, $2.33$ and $4.4$
