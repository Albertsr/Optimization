# Optimization

- 贝叶斯优化
- 梯度下降法
- 牛顿法
- 拟牛顿法
- Adam / AdaDelta / RMSProp...


## 一、贝叶斯优化概述

### 1.1 目标函数
- 运用尽可能少的搜索次数找到一组最佳的超参数`$ x^* $`使得`$f(x)$`最大
- 适用于不高于20维的连续搜索空间，并对随机噪声具备一定的容忍度
  - It is best-suited for optimization over continuous domains of less than 20
dimensions
  - tolerates stochastic noise in function evaluations


### 1.2 函数`$f(x)$`的特点 
  - **expensive to evaluate：** 有较高的计算成本；
  - **black box：** `$f(x)$`没有明确的函数表达式；
  - **derivative-free：** `$f(x)$`可能无法求导，从而无法运用梯度相关的优化算法；
  - **连续：** 从而可以用高斯过程等模型来拟合；
  
---

### 二、贝叶斯包含两个重要组成部分

#### 2.1 surrogate method 
- **目的：** 作为先验分布函数，用于拟合原函数
- **类型：**
  - Gaussian Process (bayesian_optimization)
  - Tree Parzen Estimator (hyperopt)
  - SMAC (autosklearn)
  - Random Forest

#### 2.2 acquisition function
- **目的：** 基于后验分布函数生成，用于确定下一轮的采样点 


---

### 三、贝叶斯优化的流程（基于高斯过程）

#### 3.1 贝叶斯优化流程的伪代码
