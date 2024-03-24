本文考虑对应常见的序列映射方法进行归纳总结。

# 序列映射的一般定义

考虑序列映射$f:\mathbb R^{n\times d}\to \mathbb R^{n\times d}$，或$\mathbf Y = f(\mathbf  X)$：
$$
\left[\begin{matrix}
\mathbf y_1^\top \\
\vdots \\
\mathbf y_n^\top
\end{matrix}
\right]=\mathbf Y =f(\mathbf X)=
f\left(
\left[\begin{matrix}
\mathbf x_1^\top \\
\vdots \\
\mathbf x_n^\top
\end{matrix}
\right]
\right)\\
\mathbf y_m = f(\mathbf x_1,\ldots,
\mathbf x_n )_m
$$
特别的，我们考虑causal映射，即：
$$
\mathbf y_m= f(\mathbf x_1, \ldots, \mathbf x_m)_m\triangleq f_m(\mathbf x_1, \ldots, \mathbf x_m),
$$
其中
$$
f_m: \mathbb R^{m\times d}\to \mathbb R^d
$$
一个常见的causal映射的例子为language model。

因为一般的映射可以通过两次causal映射得到，例如：
$$
y_m =f_m(x_1, \ldots, x_m) +\bar f_{n-m}(x_{m+1},\ldots, x_n).
$$
所以下文我们只考虑causal映射。



# 基于memory的序列映射

受到RNN的启发，我们通过memory进行序列映射的构造：

(old version, for reference)
- memory $\mathbf m_t \in \mathbb R^{k\times d}$；
- forget gate $\mathbf f_t \in \mathbb R^{k\times ?}$;
- input gate $\mathbf i_t \in \mathbb R^{k}$;
- input state $\mathbf u_t \in \mathbb R^{d}$;
- output gate $\mathbf o_t \in \mathbb R^{d}$;

(new version)

收到之前工作的启发，我们将序列建模定义为三个过程，Expand, Oscillation, Shrink(EOS)，并定义如下状态：

- memory state $\mathbf m_t \in \mathbb R^{k\times d}$；
- oscillation state $\mathbf o_t \in \mathbb R^{k\times ？}$;
- expand state $\mathbf e_t \in \mathbb R^{k}$;
- input state $\mathbf i_t \in \mathbb R^{d}$;
- shrink state $\mathbf s_t \in \mathbb R^{k}$;

在每个时刻$t$：

input state和expand state利用外积到新的memory $\bar {\mathbf m}_t=\mathbf e_t \mathbf i_t^\top$；

然后利用下式进行更新（$\mathbf m_0$初始化为$\mathbf 0\in \mathbb R^{k\times d}$）：
$$
\mathbf m_{t}=f(\mathbf o_t , \mathbf m_{t-1}) + \mathbf e_t \mathbf i_t^\top.
$$
其中$f=\odot$（逐元素乘，此时$?=d$）或$f=.$（矩阵乘法，此时$?=k$）。

最后output state通过dot product从memory中得到最终的输出$\mathbf y_t$:
$$
\mathbf y_t =\mathbf m_t^{\top} \mathbf s_t  \in \mathbb R^d.
$$
forget state, input state, expand state, shrink state都是通过$\mathbf  x_t$计算得到(或者不依赖于$\mathbf x_t$)。

为了方便后续讨论，我们暂且将该方法记为LCSM。我们称该过程为：Expand, Oscillation, Shrink (EOS).



# 例子

上述的定义看起来有点古怪（但思路和普通的RNN也没什么区别？），在这节中，我们将指出上述定义包含了很多被广泛使用的序列建模方式，我们将各个元素的对应关系列在下表中：

对于$f$，我们用1表示$f=\odot$，用2表示$f=.$，$\mathbf 1^{(k)}\in \mathbb R^k, \mathbf 1^{(k)}_j = 1,j=1,\ldots, k, \mathbf J^{(k)}=\mathbf 1^{(k)}{\mathbf 1^{(k)}}^\top$

| method           | shrink state                    | oscillation state                         | expand state                      | input state                     | memory size  | $f$                |
| ---------------- | ------------------------------ | ---------------------------------------------- | ------------------------------- | ------------------------------- | ------------ | ------------ |
| Linear Attention | $\mathbf q_t\in \mathbb R^{k}$ | $\mathbf J^{(k)}\in \mathbb R^{k\times k}$ | $\mathbf k_t \in \mathbb R^{k}$ | $\mathbf v_t \in \mathbb R^{d}$ | $k\times d$  | 1     |
| S4               | $\mathbf C\in \mathbb R^ k $   | $\mathbf A\in \mathbb R^{k\times k}$           | $\mathbf B\in \mathbb R^{k}$    | $\mathbf u_t \in \mathbb R^1$   | $k\times 1$  | 2      |
| S5               | $\mathbf C\in \mathbb R^k $   | $\mathbf A\in \mathbb R^{k\times k}$           | $\mathbf B\in \mathbb R^{k}$    | $\mathbf u_t \in \mathbb R^d$   | $k \times d$ | 2      |
| TNL              | $\mathbf q_t\in \mathbb R^{k}$ | $\lambda \mathbf J^{(k)}\in \mathbb R^{k\times k}$ | $\mathbf k_t \in \mathbb R^{k}$ | $\mathbf v_t \in \mathbb R^{d}$ | $k\times d$  | 1      |
| Mamba            | $\mathbf C_t\in \mathbb R^k $ | $\mathbf A_t\in \mathbb R^{k\times k}$         | $\mathbf B_t\in \mathbb R^{k}$  | $\mathbf u_t \in \mathbb R^d$   | $k\times d$  | 1 |
| RWKV-4 | $\mathbf R_t \in \mathbb R^1$ | $\exp(-w ) \in \mathbb R^{1\times 1}$ | $\exp(\mathbf k_t) \in \mathbb R^{1}$ | $\mathbf v_t \mathbf \in \mathbb R^1$ | $1\times 1$ | 1 |
| Cosformer | $\mathbf q_t\in \mathbb R^{k}$ | $\exp(i\theta)\mathbf J^{(k)}\in \mathbb R^{k\times k}$ | $\mathbf k_t \in \mathbb R^{k}$ | $\mathbf v_t \in \mathbb R^{d}$ | $k\times d$ | 1 |
| Lrpe | $\mathbf q_t\in \mathbb R^{k}$ | $\exp(i\Theta) {\mathbf 1^{(k)}}^{\top}$ | $\mathbf k_t \in \mathbb R^{k}$ | $\mathbf v_t \in \mathbb R^{d}$ | $k\times d$ | 1 |



## Linear Attention[1]

在Linear Attention中，我们通过输入得到$\mathbf x_t \in \mathbb R^{d}$得到query $\mathbf q_t  \in \mathbb R^{k}$，key $\mathbf k_t  \in \mathbb R^{k}$, value $\mathbf v_t  \in \mathbb R^{d}$，并通过下式递推计算：
$$
[\mathbf {kv}]_t =[\mathbf {kv}]_{t-1} + \mathbf k_t \mathbf  v_t^\top . \\
\mathbf y_t =  [\mathbf {kv}]_t^{\top } \mathbf q_t.
$$
可以看到，Linear Attention是MNet的一个特例。



## S4[2]

在S4中，计算式如下：
$$
\mathbf{m}_t=\mathbf{A} \mathbf{m}_{t-1}+\mathbf{B} \mathbf{u}_t^{\top}\\
\mathbf{y}_t=\mathbf{m}_t^{\top} \mathbf{C} .
$$
可以看到，S4同样是MNet的一个特例。

备注：原始的S4定义的是$\mathbb R^{n\times 1}\to \mathbb R^{n\times 1}$的映射$f_i,i=1,\ldots ,d$，通过如下方式定义$\mathbb R^{n\times d} \to \mathbb R^{n\times d}$的映射$f$：
$$
f(\mathbf X)_{mi} =f(\mathbf X_{:, i})_m.
$$

即对每个channel定义一个S4。



## S5[3]

S5的递推式和S4相同，唯一的区别是直接定义映射$\mathbb R^{n\times d} \to \mathbb R^{n\times d}$：
$$
\mathbf{m}_t=\mathbf{A} \mathbf{m}_{t-1}+\mathbf{B} \mathbf{u}_t^{\top}\\
\mathbf{y}_t=\mathbf{m}_t^{\top} \mathbf{C} .
$$
维度区别见表格。



## TNL[4]

TNL(Transnormer LLM)是在Linear Attention基础上加上了指数衰减：
$$
[\mathbf {kv}]_t =\lambda [\mathbf {kv}]_{t-1} + \mathbf k_t \mathbf  v_t ^\top. \\
\mathbf y_t =  [\mathbf {kv}]_t^{\top } \mathbf q_t.
$$



## Mamba[5]

Mamba的递推式同样简单：
$$
\mathbf{m}_t=\mathbf{A}_t \odot \mathbf{m}_{t-1}+\mathbf{B}_t \mathbf{u}_t^{\top}\\
\mathbf{y}_t=\mathbf{m}_t^{\top} \mathbf{C_t} .
$$



## RWKV-4[6]

我们忽略RWKV的分母项，那么RWKV的递推式可以简化为：
$$
\mathbf {m}_t =\exp(-w) \mathbf {m}_{t-1} + \exp( \mathbf k_t) \mathbf  v_t^\top . \\
\mathbf y_t =  \mathbf {m}_t^{\top } \mathbf r_t.
$$
备注：原始的RWKV定义的是$\mathbb R^{n\times 1}\to \mathbb R^{n\times 1}$的映射$f_i,i=1,\ldots ,d$，通过如下方式定义$\mathbb R^{n\times d} \to \mathbb R^{n\times d}$的映射$f$：
$$
f(\mathbf X)_{mi} =f(\mathbf X_{:, i})_m.
$$

即对每个channel定义一个RWKV。



## Cosformer[7]

在Cosformer中，我们通过输入得到$\mathbf x_t \in \mathbb R^{d}$得到query $\mathbf q_t  \in \mathbb R^{k}$，key $\mathbf k_t  \in \mathbb R^{k}$, value $\mathbf v_t  \in \mathbb R^{d}$，并通过下式递推计算：
$$
[\mathbf {kv}]_t =\exp(i\theta)[\mathbf {kv}]_{t-1} + \mathbf k_t \mathbf  v_t^\top . \\
\mathbf y_t =  \mathrm{Rel}\{[\mathbf {kv}]_t \}^{\top } \mathbf q_t.
$$
证明：
$$
\begin{aligned}
{[\mathbf {kv}]_t }&= \sum_{s=1}^t \exp(i (t-s) \theta)\mathbf k_s\mathbf  v_s^\top \\
\mathbf y_t&= \mathrm{Rel}\{[\mathbf {kv}]_t \}^{\top} \mathbf q_t \\
&= \mathrm{Rel}\left\{
\sum_{s=1}^t \exp(i (t-s) \theta)\mathbf v_s \mathbf  k_s ^\top\mathbf q_t
\right\} \\
&=
\sum_{s=1}^t \cos((t-s) \theta)\mathbf v_s \mathbf  k_s ^\top\mathbf q_t
 \\
\end{aligned}
$$



## Lrpe[8]

在Lrpe中，我们通过输入得到$\mathbf x_t \in \mathbb R^{d}$得到query $\mathbf q_t  \in \mathbb R^{k}$，key $\mathbf k_t  \in \mathbb R^{k}$, value $\mathbf v_t  \in \mathbb R^{d}$，并通过下式递推计算：
$$
[\mathbf {kv}]_t =\Lambda [\mathbf {kv}]_{t-1} + \mathbf k_t \mathbf  v_t^\top . \\
\Lambda =\mathrm{diag}\{\exp(i\theta_1),\ldots, \exp(i\theta_k) \}.    \\
\mathbf y_t =  \mathrm{Rel}\{[\mathbf {kv}]_t \}^{\top } \mathbf q_t.
$$
说明：
$$
\begin{aligned}
{[\mathbf {kv}]_t } &= \sum_{s=1}^t \Lambda^{t-s}\mathbf k_s\mathbf  v_s^\top \\
\mathbf y_t&= \mathrm{Rel}\{[\mathbf {kv}_t] \}^{\top} \mathbf q_t \\
&= \mathrm{Rel}\left\{
\sum_{s=1}^t \Lambda^{t-s}\mathbf v_s \mathbf  k_s ^\top\mathbf q_t
\right\} \\
&=
\sum_{s=1}^t \mathbf v_s \mathbf  k_s ^\top \bar \Lambda_{t-s}\mathbf q_t
 \\
 \bar  \Lambda_{t-s}&= \mathrm{diag}\{\cos((t-s)\theta_1),\ldots, \cos((t-s)\theta_k) \}
\end{aligned}
$$



# 简化

为了简化讨论，当$f=.$时候，我们假设$\mathbf o_t$可对角化，这在实际中是一个常见的假设，cite Dss，此时$\mathbf o_t=\text{Diag}\{{\mathbf {\bar o_ t}}\}, \mathbf {\bar {o}_t}\in \mathbb R^{k}$：
$$
\mathbf m_{t}=\mathbf o_t  \mathbf m_{t-1} + \mathbf e_t \mathbf i_t^\top
=\left( \mathbf {\bar {o}_t}{\mathbf 1^{(k)}}^\top \right) \odot \mathbf m_{t-1} + \mathbf e_t \mathbf i_t^\top.
$$
所以不失一般性，我们在正文中只考虑$f=\odot$的情况，并在附录中讨论$\mathbf o_t$不可对角化的几个例子。





# Backward

现在已经定义了LCSM的Forward形式，接下来就是Backward形式，为了方便叙述，我们将$f=\odot$的情形称为Type1，$f=.$的形式称为Type2。

Type1:
$$
\mathbf m_{t}=\mathbf f_t \odot \mathbf m_{t-1} + \mathbf e_t \mathbf i_t^\top,\\
\mathbf y_t =\mathbf m_t^{\top} \mathbf s_t .
$$
Type2:
$$
\mathbf m_{t}=\mathbf f_t\mathbf m_{t-1} + \mathbf e_t \mathbf i_t^\top,\\
\mathbf y_t =\mathbf m_t^{\top} \mathbf s_t
$$



## Type1

$$
\mathbf {ds}_t = \mathbf m_t \mathbf {dy_t} \in \mathbb R^{k},\\
\mathbf {dm}_{t-1}=\mathbf f_t \odot \mathbf {dm}_{t} + \mathbf s_{t-1}  \mathbf {dy}_{t-1} ^{\top}\in \mathbb R^{k\times d}, \mathbf {dm}_n=\mathbf s_{n}  \mathbf {dy}_{n} ^{\top}\in \mathbb R^{k\times d},\\
\mathbf {df}_{t}= \mathbf {dm}_{t}\odot \mathbf m_{t-1} \in \mathbb R^{k\times d}, \\
\mathbf {de}_{t}= \mathbf {dm}_{t} \mathbf i_t \in \mathbb R^{k}, \\

\mathbf {di}_{t}= \mathbf {dm}_{t}^{\top} \mathbf e_t \in \mathbb R^{d}. \\
$$



## Type2

$$
\mathbf {ds}_t = \mathbf m_t \mathbf {dy_t} \in \mathbb R^{k},\\
\mathbf {dm}_{t-1}=\mathbf f_t \mathbf {dm}_{t} + \mathbf s_{t-1}  \mathbf {dy}_{t-1} ^{\top}\in \mathbb R^{k\times d}, \mathbf {dm}_n=\mathbf s_{n}  \mathbf {dy}_{n} ^{\top}\in \mathbb R^{k\times d},\\
\mathbf {df}_{t}=  \mathbf {dm}_{t} \mathbf m_{t-1}^{\top} \in \mathbb R^{k\times k}, \\
\mathbf {de}_{t}= \mathbf {dm}_{t} \mathbf i_t \in \mathbb R^{k}, \\

\mathbf {di}_{t}= \mathbf {dm}_{t}^{\top} \mathbf e_t \in \mathbb R^{d}. \\
$$



# 如何计算state

另一个问题是如何计算shrink state, oscillation state,expand state：

- 通过SSM参数化的形式计算，以及nn.Linear计算；
- 对shrink state, expand state是否使用激活函数；
  - 类似linear attention中的kernel function；
- oscillation state的计算方式：我们利用einsum的形式分别比较了各种形式的构造方式；





# 实验分类

## 是否需要特殊构造

在这个部分，我们比较了SSM参数化和nn.Linear参数化的区别。



## 是否data dependent

通过对表格的总结，我们可以看出来对Lcsm的分类首先可以分为shrink state, oscillation state,expand state是否依赖于输入（即是否含有下标$t$），对于oscillation state，我们还考虑了几个特殊情况，即使用复数，不可学习的data independent，全1的情形共11种情况。对于data dependent的情况，我们假设oscillation state的每个元素属于$[0, 1]$，并且用$\mathrm{sigmoid(x)}^{1/\tau}$计算，关于$\tau$的实验将在后续进行讨论，对于data independent的情况，我们使用alibi的方式进行初始化。



## oscillation state的构造方式

要得到$k\times d$的oscillation state，有多种构造方式，我们通过einsum的形式列举了如下几种可能性。



## activation function test

为了比较激活函数是否有作用，我们测试了一些主流的激活函数。



## tau test

$\tau $可以控制震荡速率，所以在此我们也测试了其性能。





## 实验

我们在wikitext和mqar上进行了实验，结果如下。



- wikitext
- 









# Navie computation

我们使用[9]的方法并行计算出$\mathbf m_t, t=1,\ldots, n$，然后计算出$\mathbf y_t$，这种方法的问题是空间复杂度是$O(nkd)$。



# Fast computation

以block为粒度操作即可。



现在考虑Block粒度进行操作，假设将$\mathbf X\in \mathbb R^{n\times d}$分解为$m$个Blocks $\mathbf X_1, \ldots, \mathbf X_m$，其中Block size为$B$，$m=n/B$，$\mathbf X_t \in \mathbb R^{m\times d}$，记：
$$
\mathbf M_t = \mathbf m_{tB}.
$$
记：
$$
\mathbf {a}_t =\mathbf e_t \mathbf i_t^\top, t=1,\ldots,n. \\
\mathbf {a}_0 = \mathbf m_0, \\
\mathbf {\bar f}_t  =\prod_{j=1}^t \mathbf f_j, \mathbf {\bar f}_0=1.
$$
那么
$$
\mathbf m_{t}=\sum_{j=0}^t \frac{\mathbf {\bar f}_t}{\mathbf {\bar f}_j}\odot \mathbf a_j
=\mathbf {\bar f}_t \odot \sum_{j=0}^t \frac{\mathbf a_j}{\mathbf {\bar f}_j}
$$
考虑$j=tB+k, 1\le k \le B$，那么：
$$
\begin{aligned}
\mathbf m_{tB+k}&= \mathbf {\bar f}_{tB+k} \odot \sum_{j=0}^{tB+k} \frac{\mathbf a_j}{\mathbf {\bar f}_j}  \\
&= \mathbf {\bar f}_{tB+k} \odot \sum_{j=0}^{(t-1)B} \frac{\mathbf a_j}{\mathbf {\bar f}_j} +
\mathbf {\bar f}_{tB+k} \odot \sum_{j=(t-1)B+1}^{tB+k} \frac{\mathbf a_j}{\mathbf {\bar f}_j} \\
&=\frac{\mathbf {\bar f}_{tB+k}}{\mathbf {\bar f}_{(t-1)B} }  \odot \mathbf {\bar f}_{(t-1)B}
\odot \sum_{j=0}^{(t-1)B} \frac{\mathbf a_j}{\mathbf {\bar f}_j} +\mathbf {\bar f}_{tB+k} \odot \sum_{j=(t-1)B+1}^{tB+k} \frac{\mathbf a_j}{\mathbf {\bar f}_j} \\
&= \frac{\mathbf {\bar f}_{tB+k}}{\mathbf {\bar f}_{(t-1)B} } \odot \mathbf M_{t-1}+
\mathbf {\bar f}_{tB+k} \odot \sum_{j=(t-1)B+1}^{tB+k} \frac{\mathbf a_j}{\mathbf {\bar f}_j} \\

\mathbf y_{tB+k}&= \mathbf m_{tB+k}^{\top} \mathbf s_{tB+k} \\
&=\left(

 \frac{\mathbf {\bar f}_{tB+k}}{\mathbf {\bar f}_{(t-1)B} } \odot \mathbf M_{t-1}+
\mathbf {\bar f}_{tB+k} \odot \sum_{j=(t-1)B+1}^{tB+k} \frac{\mathbf a_j}{\mathbf {\bar f}_j}

\right)^{\top} \mathbf s_{tB+k}

\end{aligned}
$$
所以：
$$
\begin{aligned}
\mathbf M_{t}
&=\mathbf m_{tB}\\
&=\mathbf {\bar f}_{tB} \odot \sum_{j=0}^{tB} \frac{\mathbf a_j}{\mathbf {\bar f}_j}  \\
&= \mathbf {\bar f}_{tB} \odot \sum_{j=0}^{(t-1)B} \frac{\mathbf a_j}{\mathbf {\bar f}_j} +
\mathbf {\bar f}_{tB} \odot \sum_{j=(t-1)B+1}^{tB} \frac{\mathbf a_j}{\mathbf {\bar f}_j} \\
&=\frac{\mathbf {\bar f}_{tB}}{\mathbf {\bar f}_{(t-1)B} }  \odot \mathbf {\bar f}_{(t-1)B}
\odot \sum_{j=0}^{(t-1)B} \frac{\mathbf a_j}{\mathbf {\bar f}_j} +\mathbf {\bar f}_{tB} \odot \sum_{j=(t-1)B+1}^{tB} \frac{\mathbf a_j}{\mathbf {\bar f}_j} \\
&= \frac{\mathbf {\bar f}_{tB}}{\mathbf {\bar f}_{(t-1)B} } \odot \mathbf M_{t-1}+
\mathbf {\bar f}_{tB} \odot \sum_{j=(t-1)B+1}^{tB} \frac{\mathbf a_j}{\mathbf {\bar f}_j}
\end{aligned}
$$
注意到$\mathbf {\bar f}_{tB+k} \odot \sum_{j=(t-1)B+1}^{tB+k} \frac{\mathbf a_j}{\mathbf {\bar f}_j}$实际上是递推式：
$$
\mathbf m_{j}=\mathbf f_j \odot \mathbf m_{j-1} + \mathbf e_j \mathbf i_j^\top, j=
$$








以Block形式，我们有：
$$
\mathbf Y_t = \mathbf S_t\mathbf M_{t-1}+ \mathbf S_t .  \\
\mathbf M_{t} = \mathbf M_{t-1}
$$



$$
\mathbf m_{t}=\mathbf f_t \odot \mathbf m_{t-1} + \mathbf e_t \mathbf i_t^\top
$$



## Sequential Reverse

考虑如下递推：
$$
x_{t}=a_{t+1} x_{t+1} + b_{t}, t=1,\ldots, n
$$
展开可得：
$$
\begin{aligned}
x_{n}
&=a_{n+1} x_{n+1} + b_{n}\\
&=a_{n+1}\left(
 x_{n+1} + \frac{ b_{n}}{a_{n+1}}
\right) \\
x_{n-1}&= a_n x_n + b_{n-1}\\
&= a_{n+1} a_n  \left(
 x_{n+1} + \frac{ b_{n}}{a_{n+1}} +
 \frac{b_{n-1}}{a_{n+1}a_n}
\right) \\

x_t&= \left( \prod_{s=t+1}^{n+1} a_s \right) \odot
\left(
x_{n+1} + \sum_{j=t}^n \frac{b_j}{\prod_{s=j+1}^{n+1} a_s  }
\right)


\end{aligned}
$$


反向的时候，维护一个$a_{n+2}$永远为1，然后接受上一步传入的$a_{n+1}$，如果没有，这一项变为1，concat $a_n$，舍弃最后一项（下标最小的一项），最后计算出的$m_0$是$0$。





# Citations

1. Angelos Katharopoulos, Apoorv Vyas, Nikolaos Pappas, and François Fleuret. Transformers are rnns: Fast autoregressive transformers with linear attention. In Proceedings of the 37th International Conference on Machine Learning, ICML 2020, 13-18 July 2020, Virtual Event, volume 119 of Proceedings of Machine Learning Research, pages 5156–5165. PMLR, 2020.
2. Albert Gu, Karan Goel, and Christopher Ré. Efficiently modeling long sequences with structured state spaces. In The Tenth International Conference on Learning Representations, ICLR 2022, Virtual Event, April 25-29, 2022. OpenReview.net, 2022.
3. Jimmy T.H. Smith, Andrew Warrington, & Scott Linderman (2023). Simplified State Space Layers for Sequence Modeling. In *The Eleventh International Conference on Learning Representations* .
4. Zhen Qin, Dong Li, Weigao Sun, Weixuan Sun, Xuyang Shen, Xiaodong Han, Yunshen Wei, Baohong Lv, Xiao Luo, Yu Qiao, & Yiran Zhong. (2023). TransNormerLLM: A Faster and Better Large Language Model with Improved TransNormer.
5. Albert Gu, & Tri Dao. (2023). Mamba: Linear-Time Sequence Modeling with Selective State Spaces.
6. Bo Peng, Eric Alcaide, Quentin Anthony, Alon Albalak, Samuel Arcadinho, Stella Biderman, Huanqi Cao, Xin Cheng, Michael Chung, Matteo Grella, Kranthi Kiran GV, Xuzheng He, Haowen Hou, Jiaju Lin, Przemyslaw Kazienko, Jan Kocon, Jiaming Kong, Bartlomiej Koptyra, Hayden Lau, Krishna Sri Ipsit Mantri, Ferdinand Mom, Atsushi Saito, Guangyu Song, Xiangru Tang, Bolun Wang, Johan S. Wind, Stanislaw Wozniak, Ruichong Zhang, Zhenyuan Zhang, Qihang Zhao, Peng Zhou, Qinghua Zhou, Jian Zhu, & Rui-Jie Zhu. (2023). RWKV: Reinventing RNNs for the Transformer Era.
7. Zhen Qin, Weixuan Sun, Hui Deng, Dongxu Li, Yunshen Wei, Baohong Lv, Junjie Yan, Lingpeng Kong, & Yiran Zhong. (2022). cosFormer: Rethinking Softmax in Attention.
8. Zhen Qin, Weixuan Sun, Kaiyue Lu, Hui Deng, Dongxu Li, Xiaodong Han, Yuchao Dai, Lingpeng Kong, & Yiran Zhong. (2023). Linearized Relative Positional Encoding.



# Log

1. 20240217: 初版。
2. 20240226: 修复笔误。
3. 20240305: 修改名称，添加更多的例子和backwar。
