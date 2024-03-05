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

(old version)
- memory $\mathbf m_t \in \mathbb R^{d\times e}$；
- forget gate $\mathbf f_t \in \mathbb R^{d\times e}$;
- input gate $\mathbf i_t \in \mathbb R^{e}$;
- input state $\mathbf u_t \in \mathbb R^{d}$;
- output gate $\mathbf o_t \in \mathbb R^{e}$;

(new version)
- memory state $\mathbf m_t \in \mathbb R^{k\times d}$；
- forget state $\mathbf f_t \in \mathbb R^{k\times d}$;
- expand state $\mathbf e_t \in \mathbb R^{k}$;
- input state $\mathbf i_t \in \mathbb R^{d}$;
- shrink state $\mathbf s_t \in \mathbb R^{k}$;

在每个时刻$t$：

input state和expand state利用外积映射$f: \mathbb R^{k}\times \mathbb R^d \to \mathbb R^{k\times d}$得到新的memory $\bar {\mathbf m}_t=f_1(\mathbf e_t, \mathbf i_t)=\mathbf e_t^\top \mathbf i_t$；

然后利用下式进行更新（$\mathbf m_0$初始化为$\mathbf 0\in \mathbb R^{k\times d}$）：
$$
\mathbf m_{t}=\mathbf f_t \odot \mathbf m_{t-1} + \mathbf e_t^\top \mathbf i_t.
$$
其中$\odot$是逐元素乘。

最后output state通过dot product从memory中得到最终的输出$\mathbf y_t$:
$$
\mathbf y_t =\mathbf m_t^{\top} \mathbf s_t  \in \mathbb R^d.
$$
forget state, input state, expand state, shrink state都是通过$\mathbf  x_t$计算得到(或者不依赖于$\mathbf x_t$)。

为了方便后续讨论，我们暂且将该方法记为MNet（Memory Network).



# 例子

上述的定义看起来有点古怪（但思路和普通的RNN也没什么区别？），在这节中，我们将指出上述定义包含了很多被广泛使用的序列建模方式，我们将各个元素的对应关系列在下表中：

| method           | shrink state                    | forget state                                    | expand state                      | input state                     | memory size  | $f_1$       | $f_2$                   |
| ---------------- | ------------------------------ | ---------------------------------------------- | ------------------------------- | ------------------------------- | ------------ | ----------- | ----------------------- |
| Linear Attention | $\mathbf q_t\in \mathbb R^{k}$ | $\mathbf I_e\in \mathbb R^{d\times d}$         | $\mathbf k_t \in \mathbb R^{e}$ | $\mathbf v_t \in \mathbb R^{d}$ | $d\times e$  | out product | matrix production       |
| S4               | $\mathbf C\in \mathbb R^ e $   | $\mathbf A\in \mathbb R^{e\times e}$           | $\mathbf B\in \mathbb R^{e}$    | $\mathbb u_t \in \mathbb R^1$   | $1\times e$  | out product | matrix production       |
| S5               | $\mathbf C\in \mathbb R^ e $   | $\mathbf A\in \mathbb R^{e\times e}$           | $\mathbf B\in \mathbb R^{e}$    | $\mathbb u_t \in \mathbb R^d$   | $d \times e$ | out product | matrix production       |
| TNL              | $\mathbf q_t\in \mathbb R^{e}$ | $\mathbf \lambda I_e\in \mathbb R^{d\times d}$ | $\mathbf k_t \in \mathbb R^{e}$ | $\mathbf v_t \in \mathbb R^{d}$ | $d\times e$  | out product | matrix production       |
| Mamba            | $\mathbf C_t\in \mathbb R^ e $ | $\mathbf A_t\in \mathbb R^{d\times e}$         | $\mathbf B_t\in \mathbb R^{e}$  | $\mathbb u_t \in \mathbb R^d$   | $d\times e$  | out product | element wise production |
|                  |                                |                                                |                                 |                                 |              |             |                         |



## Linear Attention[1]

在Linear Attention中，我们通过输入得到$\mathbf x_t \in \mathbb R^{d}$得到query $\mathbf q_t  \in \mathbb R^{e}$，key $\mathbf k_t  \in \mathbb R^{e}$, value $\mathbf v_t  \in \mathbb R^{d}$，并通过下式递推计算：
$$
\mathbf {kv}_t =\mathbf {kv}_{t-1} + \mathbf k_t \mathbf  v_t ^\top. \\
\mathbf y_t =  \mathbf {kv}_t^{\top } \mathbf q_t.
$$
可以看到，Linear Attention是MNet的一个特例。



## S4[2]

在S4中，计算式如下：
$$
\mathbf{m}_t=\mathbf{A} \mathbf{m}_{t-1}+\mathbf{B} \mathbf{u}_t\\
\mathbf{y}_t=\mathbf{C} \mathbf{u}_t.
$$
可以看到，S4同样是MNet的一个特例。

备注：原始的S4定义的是$\mathbb R^{n\times 1}\to \mathbb R^{n\times 1}$的映射$f_i,i=1,\ldots ,d$，通过如下方式定义$\mathbb R^{n\times d} \to \mathbb R^{n\times d}$的映射$f$：
$$
f(\mathbf X)_{mi} =f(\mathbf X_{:, i})_m.
$$



## S5[3]

S5的递推式和S4相同，唯一的区别是直接定义映射$\mathbb R^{n\times d} \to \mathbb R^{n\times d}$，区别见表格。



## TNL[4]

TNL(Transnormer LLM)是在Linear Attention基础上加上了指数衰减：
$$
\mathbf {kv}_t =\lambda \mathbf {kv}_{t-1} + \mathbf k_t \mathbf  v_t ^\top. \\
\mathbf y_t =  \mathbf {kv}_t^{\top } \mathbf q_t.
$$


## Mamba[5]

Mamba的递推式同样简单：
$$
\mathbf{m}_t=\mathbf{A_t} \odot \mathbf{m}_{t-1}+\mathbf{B_t} \mathbf{u}_t\\
\mathbf{y}_t=\mathbf{C_t} \mathbf{u}_t.
$$



# Citations

1. Angelos Katharopoulos, Apoorv Vyas, Nikolaos Pappas, and François Fleuret. Transformers are rnns: Fast autoregressive transformers with linear attention. In Proceedings of the 37th International Conference on Machine Learning, ICML 2020, 13-18 July 2020, Virtual Event, volume 119 of Proceedings of Machine Learning Research, pages 5156–5165. PMLR, 2020.
2. Albert Gu, Karan Goel, and Christopher Ré. Efficiently modeling long sequences with structured state spaces. In The Tenth International Conference on Learning Representations, ICLR 2022, Virtual Event, April 25-29, 2022. OpenReview.net, 2022.
3. Jimmy T.H. Smith, Andrew Warrington, & Scott Linderman (2023). Simplified State Space Layers for Sequence Modeling. In *The Eleventh International Conference on Learning Representations* .
4. Zhen Qin, Dong Li, Weigao Sun, Weixuan Sun, Xuyang Shen, Xiaodong Han, Yunshen Wei, Baohong Lv, Xiao Luo, Yu Qiao, & Yiran Zhong. (2023). TransNormerLLM: A Faster and Better Large Language Model with Improved TransNormer.
5. Albert Gu, & Tri Dao. (2023). Mamba: Linear-Time Sequence Modeling with Selective State Spaces.