# BAMPS/MD 压强计算物理说明 (Physics of Pressure Calculation)

在本项目中，我们同时使用了三种压强计算方法：**器壁压强 ($P_{wall}$)**、**维里压强 ($P_{virial}$)** 以及 **理论预测压强 ($P_{theory}$)**。以下是它们的物理本质与计算公式。

---

## 1. 维里压强 ($P_{virial}$)

### 物理定义
维里压强是基于 **克劳修斯维里定理 (Clausius Virial Theorem)** 计算的。它通过系统内所有粒子的位置和相互作用力来表征系统的整体压强，而不需要依赖于边界碰撞。

### 计算公式
$$ P_{virial} = \rho k_B T + \frac{1}{3V} \sum_{i<j} \vec{F}_{ij} \cdot \vec{r}_{ij} $$

其中：
*   **第一项 ($\rho k_B T$)**：动能贡献（理想气体部分）。在代码中通过粒子的动量和总粒子数计算得出。
*   **第二项 ($\frac{1}{3V} \sum \vec{F} \cdot \vec{r}$)**：势能/力场修正（维里部分）。它描述了粒子间的 Lennard-Jones 力对压强的贡献。
    *   对于 **斥力** ($F \cdot r > 0$)，压强增加。
    *   对于 **引力** ($F \cdot r < 0$)，压强降低（这是 $\epsilon$ 扫描的关键）。

### 为什么这么算？
*   **稳健性**：在分子动力学 (MD) 中，统计壁面碰撞 ($P_{wall}$) 的噪声极大，特别是在平衡态演化初期。
*   **各向同性**：维里压反映了体相 (Bulk) 属性，适用于任何形状的模拟区域。

---

## 2. 理论压强 ($P_{theory}$)

### 物理定义
$P_{theory}$ 是基于 **范德华物态方程 (van der Waals EoS)** 的解析预测，用于验证模拟结果的正确性。

### 计算公式
$$ P_{theory} = \frac{n k_B T}{1 - nb} - a n^2 $$

### 参数映射 (Lennard-Jones to vdW)
我们将 Lennard-Jones 参数 ($\sigma, \epsilon$) 映射到 vdW 常数：
*   **排除体积效应 ($b$)**：
    $$ b = \frac{2}{3} \pi \sigma^3 $$
    代表粒子本身占据的空间，导致压强随 $\sigma$ 增加而迅速上升。
*   **吸引力修正 ($a$)**：
    $$ a = \frac{16}{9} \pi \epsilon \sigma^3 $$
    代表由于 $\epsilon$ 吸引井导致的压强下降。

---

## 3. 数值实现注意事项 (BAMPS/DSMC)

在我们的程序中，针对 **测试粒子方法 (Test Particle Method)** 进行了以下缩放修正：

1.  **动能项缩放**：
    $$ P_{kinetic} \propto \frac{1}{N_{test}} \sum p^2 $$
    因为每个真实粒子被分身成了 $N_{test}$ 个，所以总动能需要除以 $N_{test}$。
2.  **维里项缩放**：
    $$ P_{virial\_part} \propto \frac{1}{N_{test}^2} \sum \vec{F} \cdot \vec{r} $$
    这是最容易出错的地方。由于相互作用力是成对计算的，测试粒子数为 $N$ 时，对的数量是 $N^2$ 量级。因此，为了得到真实物理压强，维里和必须除以 $TESTPARTCL^2$。

---

## 4. 单位制 (Units)
本模拟使用高能物理单位制（自然单位制）：
*   **距离**：$fm$ (费米, $10^{-15} m$)
*   **能量/温度**：$GeV$
*   **压强**：$GeV/fm^3$
*   **时间**：$fm/c$
