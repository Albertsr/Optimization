# Optimization

- 贝叶斯优化
- 梯度下降法 / 牛顿法 / 拟牛顿法
- Adam / AdaDelta / RMSProp...

---

### 一、贝叶斯优化

#### 1.1 优化目标与适用场景

  ![opt_problem](https://github.com/Albertsr/Optimization/blob/master/pic/01opt_problem.jpg)

- **优化目标：** 运用尽可能少的搜索次数找到连续函数的**全局最优解（Global Optimum）** 
- **搜索空间：** 典型的搜索空间是超矩形（hyper-rectangle)，形如 

  ![02search_space](https://github.com/Albertsr/Optimization/blob/master/pic/02search_space.jpg)
  
- **更适用于不高于20维的连续搜索空间：** It is best-suited for optimization over continuous domains of less than 20 dimensions  
- **对随机噪声具备一定的容忍度：** tolerates stochastic noise in function evaluations


#### 1.2 目标函数f(x)的特点  
- **expensive to evaluate：** f(x)有较高的计算成本；
- **black box：** f(x)的凹凸性、线性与否等函数性质均未知；
- **derivative-free：** f(x)不一定可微，从而无法运用梯度下降法、牛顿法、拟牛顿法等基于梯度的优化算法；
- **continuous：** f(x)连续型函数，从而可以用高斯过程等模型来拟合；
- **i.i.d noise：**  默认噪音独立于参数评估与高斯过程，同时方差恒定；[What is the difference between i.i.d noise and white noise? ](https://dsp.stackexchange.com/questions/23881/what-is-the-difference-between-i-i-d-noise-and-white-noise)

#### 1.3 贝叶斯包含两个重要组成部分
- **Surrogate Model：** 作为先验分布，近似拟合参数点与优化函数之间的函数关系；
- **Acquisition Function：** 基于后验分布评估可行域内的候选点，并确定下一轮的最佳搜索点；

#### 1.4 总结
- 贝叶斯优化用于求解黑盒、不可微函数的全局最优化问题；
  - BayesOpt is designed for black-box derivative free global optimization
- 贝叶斯优化是“基于序列模型的优化方法”，它根据历史信息迭代模型后，再决定下一次的搜索点；
  - BayesOpt is a sequential model-based optimization (SMBO) approach
  - SMBO methods sequentially construct models to approximate the performance of hyperparameters based on historical measurements, and then subsequently choose new hyperparameters to test based on this model
  
---

### 二、Surrogate Model

#### 2.1 Surrogate Model在贝叶斯优化中的作用
- 作为先验分布函数，用于拟合参数点与优化目标之间的函数关系
- Surrogate Model确定之后，就能生成函数在候选点处取值f(x)的后验分布
- 例如，当高斯过程作为Surrogate Model时，对于新的候选点x_{n+1}，基于贝叶斯公式，生成后验概率分布
  ![posterior_distribution](https://github.com/Albertsr/Optimization/blob/master/pic/03posterior_distribution.jpg)

#### 2.2 常见的Surrogate Model及其对应的Package
- **Gaussian Process**  
  - Python Package : [Bayesian Optimization](https://github.com/fmfn/BayesianOptimization)
  - Published Paper : [Gaussian processes in machine learning [Carl Edward Rasmussen]](https://www.cs.ubc.ca/~hutter/EARG.shtml/earg/papers05/rasmussen_gps_in_ml.pdf)

- **Tree Parzen Estimator**
  - Python Package : [Hyperopt](https://github.com/hyperopt/hyperopt)
  - Published Paper : [Algorithms for Hyper-Parameter Optimization [Bergstra et.al.]](https://papers.nips.cc/paper/4443-algorithms-for-hyper-parameter-optimization.pdf)
  - Code Test : [hyopt.py](https://github.com/Albertsr/Optimization/blob/master/Code/hyopt.py)
  
- **SMAC**
  - Python Package : [Auto-Sklearn](https://automl.github.io/auto-sklearn/master/)
  - Published Paper : [Efficient and Robust Automated Machine Learning [Feurer et al.]](http://papers.nips.cc/paper/5872-efficient-and-robust-automated-machine-learning.pdf)
  
---

### 三、Acquisition Function
#### 3.1 Acquisition Function在贝叶斯优化中的作用
- AF函数是基于后验分布生成的，它在exploring与exploiting之间取得一个平衡，并用于确定下一轮的采样点；
- 优化问题如下：

  ![04_af](https://github.com/Albertsr/Optimization/blob/master/pic/04_af.jpg)
  
#### 3.2 exploitation与exploration的平衡
- **exploitation（开发）**
  - 倾向于选取后验分布中期望均值较大的点，期望均值与模型最终指标成正比；
  - 对当前已知的、表现良好的点附近进行更细致的寻优；
  
- **exploration（探索）**
  - 倾向于选取后验分布中方差较大的点；
  - 对未知领域进行更进一步的探索； 

#### 3.3 常见AF函数

##### 1) PI (Probability of Improvement)

   ![05_PI](https://github.com/Albertsr/Optimization/blob/master/pic/05_PI.jpg)
   
- 核心思想——衡量上升的概率
- 缺点
  - 过于关注增长的概率，对增长幅度有所忽略；
  - 更关注exploitation，对exploration缺乏关注
  
##### 2) EI (Expected Improvement)  [参考文献：Paper_02]
- 核心思想——衡量上升的幅度

  ![06_1](https://github.com/Albertsr/Optimization/blob/master/pic/06_1.jpg)
  
- 数学期望的求解

  ![06_2](https://github.com/Albertsr/Optimization/blob/master/pic/06_2.jpg)
  
- EI最终的表达式

   ![06_3](https://github.com/Albertsr/Optimization/blob/master/pic/06_3.jpg)
   
##### 3) UCB (upper confidence bound) 
   ![07_ucb](https://github.com/Albertsr/Optimization/blob/master/pic/07_ucb.jpg)

##### 4) entropy search
- [参考文献：Paper_02]
##### 5) knowledge gradient
- [参考文献：Paper_02]

---

### 四、贝叶斯优化的核心思想与流程总结

#### 4.1 核心思想

- 选择替代模型作为先验函数，用于构建参数与优化目标之间的函数关系
- 先验函数在每一轮搜索之后都会进行迭代，表现为先验函数的参数得到更新
- 后验分布会随着先验函数的更新而同步发生改变，进一步使得AF函数进行更新
- 基于AF函数最大化，进行下一轮的最优参数的选取

#### 4.2 贝叶斯优化流程
  
  ![08_process](https://github.com/Albertsr/Optimization/blob/master/pic/08_process.jpg)

- 基于高斯过程，确定函数f的先验分布
- 基于初始空间填充设计，确定n个初始观测点，并计算函数f在对应观测点上的取值
- 若寻优的累计次数不超过设定的最大次数阈值N，则重复以下过程：
  - 基于所有观测点更新f的后验分布
  - 基于上述后验分布，更新acquisition function，并将极大值点x_n作为下一次的观测点
  - 计算y_n = f(x_n)，将(x_n, y_n)添加至已知数据集
- 上传迭代完成之后，函数取最大值时对应的x_n即为贝叶斯优化得到的最佳参数

---

#### 参考文献
- [Paper_01] [A Tutorial on Bayesian Optimization [Peter I. Frazier]](https://arxiv.org/abs/1807.02811)

- [Paper_02] [A Tutorial on Bayesian Optimization of
Expensive Cost Functions, with Application to
Active User Modeling and Hierarchical
Reinforcement Learning [Brochu et.al.]](https://arxiv.org/abs/1012.2599)

---
