The following text summarizes the common sequence mapping methods.

# General Definition of Sequence Mapping

Consider a sequence mapping $f: \mathbb R^{n\times d}\to \mathbb R^{n\times d}$, or $\mathbf Y = f(\mathbf  X)$:
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
In particular, we consider causal mapping, i.e.:
$$
\mathbf y_m= f(\mathbf x_1, \ldots, \mathbf x_m)_m\triangleq f_m(\mathbf x_1, \ldots, \mathbf x_m),
$$
where
$$
f_m: \mathbb R^{m\times d}\to \mathbb R^d.
$$
A common example of a causal mapping is the language model.

Since a general mapping can be obtained through two causal mappings, for example:
$$
y_m =f_m(x_1, \ldots, x_m) +\bar f_{n-m}(x_{m+1},\ldots, x_n).
$$
So, we will only consider causal mappings in the following text.



# Sequence Mapping based on Memory

Inspired by RNN, we construct sequence mapping through memory:

- memory $\mathbf m_t \in \mathbb R^{e\times d}$;
- forget gate $\mathbf f_t \in \mathbb R^{e\times d}$;
- input gate $\mathbf i_t \in \mathbb R^{e}$;
- input state $\mathbf u_t \in \mathbb R^{d}$;
- output gate $\mathbf o_t \in \mathbb R^{e}$;

At each time step $t$:

The input and input gate utilize the mapping $f_1: \mathbb R^{e}\times \mathbb R^d \to \mathbb R^{e\times d}$ to obtain a new memory $\bar {\mathbf m}_t=f_1(\mathbf i_t, \mathbf u_t)$;

Then, the following update is performed ($\mathbf m_0$ is initialized to $\mathbf 0\in \mathbb R^{e\times d}$):
$$
\mathbf m_{t}=f_2(\mathbf f_t, \mathbf m_{t-1}) + \bar {\mathbf m}_t.
$$
where $f_2$ is the mapping $\mathbb R^{e\times d}\times  \mathbb R^{e\times d} \to \mathbb R^{e\times d}$.

Finally, the output gate obtains the final output $\mathbf y_t$ from the memory through dot product:
$$
\mathbf y_t =\mathbf m_t^{\top} \mathbf o_t  \in \mathbb R^d.
$$
The forget gate, input gate, input state, and output gate are all calculated based on $\mathbf  x_t$.

For convenience in the following discussion, we temporarily refer to this method as MNet (Memory Network).



# Examples

The above definition may seem a bit strange (but isn't the idea similar to regular RNN?), in this section, we will point out that the above definition includes many widely used sequence modeling methods. We list the corresponding relationships of each element in the table below:

| method           | output gate                    | forget gate                                    | input gate                      | input state                     | memory size  | $f_1$       | $f_2$                   |
| ---------------- | ------------------------------ | ---------------------------------------------- | ------------------------------- | ------------------------------- | ------------ | ----------- | ----------------------- |
| Linear Attention | $\mathbf q_t\in \mathbb R^{e}$ | $\mathbf I_e\in \mathbb R^{e\times e}$         | $\mathbf k_t \in \mathbb R^{e}$ | $\mathbf v_t \in \mathbb R^{d}$ | $e\times d$  | out product | matrix production       |
| S4               | $\mathbf C\in \mathbb R^ e $   | $\mathbf A\in \mathbb R^{e\times e}$           | $\mathbf B\in \mathbb R^{e}$    | $\mathbb u_t \in \mathbb R^1$   | $e\times 1$  | out product | matrix production       |
| S5               | $\mathbf C\in \mathbb R^ e $   | $\mathbf A\in \mathbb R^{e\times e}$           | $\mathbf B\in \mathbb R^{e}$    | $\mathbb u_t \in \mathbb R^d$   | $e \times d$ | out product | matrix production       |
| TNL              | $\mathbf q_t\in \mathbb R^{e}$ | $\mathbf \lambda I_e\in \mathbb R^{e\times e}$ | $\mathbf k_t \in \mathbb R^{e}$ | $\mathbf v_t \in \mathbb R^{d}$ | $e\times d$  | out product | matrix production       |
| Mamba            | $\mathbf C_t\in \mathbb R^ e $ | $\mathbf A_t\in \mathbb R^{d\times e}$         | $\mathbf B_t\in \mathbb R^{e}$  | $\mathbb u_t \in \mathbb R^d$   | $e\times d$  | out product | element wise production |
|                  |                                |                                                |                                 |                                 |              |             |                         |

## Linear Attention[1]

In Linear Attention, we obtain query $\mathbf q_t  \in \mathbb R^{e}$, key $\mathbf k_t  \in \mathbb R^{e}$, value $\mathbf v_t  \in \mathbb R^{d}$ from the input $\mathbf x_t \in \mathbb R^{d}$, and calculate recursively as follows:
$$
\mathbf {kv}_t =\mathbf {kv}_{t-1} + \mathbf k_t \mathbf  v_t ^\top. \\
\mathbf y_t =  \mathbf {kv}_t^{\top } \mathbf q_t.
$$
It can be seen that Linear Attention is a special case of MNet.



## S4[2]

In S4, the calculation is as follows:
$$
\mathbf{m}_t=\mathbf{A} \mathbf{m}_{t-1}+\mathbf{B} \mathbf{u}_t\\
\mathbf{y}_t= \mathbf{m}_t^{\top} \mathbf{C}.
$$
It can be seen that S4 is also a special case of MNet.

Note: The original definition of S4 was a mapping $f_i,i=1,\ldots ,d$ defined as $\mathbb R^{n\times 1}\to \mathbb R^{n\times 1}$, and defined the mapping $f$ from $\mathbb R^{n\times d} \to \mathbb R^{n\times d}$ as follows:
$$
f(\mathbf X)_{mi} =f(\mathbf X_{:, i})_m.
$$



## S5[3]

The recursive formula for S5 is the same as S4, the only difference is the direct definition of the mapping $\mathbb R^{n\times d} \to \mathbb R^{n\times d}$, as shown in the table.



## TNL[4]

TNL (Transnormer LLM) adds exponential decay to Linear Attention:
$$
\mathbf {kv}_t =\lambda \mathbf {kv}_{t-1} + \mathbf k_t \mathbf  v_t ^\top. \\
\mathbf y_t =  \mathbf {kv}_t^{\top } \mathbf q_t.
$$



## Mamba[5]

The recursive formula for Mamba is also simple:
$$
\mathbf{m}_t=\mathbf{A_t} \odot \mathbf{m}_{t-1}+\mathbf{B_t} \mathbf{u}_t\\
\mathbf{y}_t=\mathbf{m}_t^{\top}\mathbf{C_t} .
$$




# Citations

1. Angelos Katharopoulos, Apoorv Vyas, Nikolaos Pappas, and François Fleuret. Transformers are rnns: Fast autoregressive transformers with linear attention. In Proceedings of the 37th International Conference on Machine Learning, ICML 2020, 13-18 July 2020, Virtual Event, volume 119 of Proceedings of Machine Learning Research, pages 5156–5165. PMLR, 2020.
2.  Albert Gu, Karan Goel, and Christopher Ré. Efficiently modeling long sequences with structured state spaces. In The Tenth International Conference on Learning Representations, ICLR 2022, Virtual Event, April 25-29, 2022. OpenReview.net, 2022.
3. Jimmy T.H. Smith, Andrew Warrington, & Scott Linderman (2023). Simplified State Space Layers for Sequence Modeling. In *The Eleventh International Conference on Learning Representations* .
4. Zhen Qin, Dong Li, Weigao Sun, Weixuan Sun, Xuyang Shen, Xiaodong Han, Yunshen Wei, Baohong Lv, Xiao Luo, Yu Qiao, & Yiran Zhong. (2023). TransNormerLLM: A Faster and Better Large Language Model with Improved TransNormer.
5. Albert Gu, & Tri Dao. (2023). Mamba: Linear-Time Sequence Modeling with Selective State Spaces.



# Log

1. 20240217: First version.
2. 20240226: Fix typo.