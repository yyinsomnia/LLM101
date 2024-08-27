


### Reward modeling (RM)

Starting from the SFT model with the final unembedding layer removed, we trained a model to take in a prompt and response, and output a scalar reward. In this paper, we only use 6B RMs, as this saves a lot of compute, and we found that 175B RM training could be unstable and thus was less suitable to be used as the value function during RL (see Appendix C for more details).

In Stiennon et al. (2020), the RM is trained on a dataset of comparisons between two model outputs on the same input. They use a cross-entropy loss, with the comparisons as labels—the difference in rewards represents the log odds that one response will be preferred to the other by a human labeler.

In order to speed up comparison collection, we present labelers with anywhere between \( K = 4 \) and \( K = 9 \) responses to rank. This produces \( \binom{K}{2} \) comparisons for each prompt shown to a labeler. Since comparisons are very correlated within each labeling task, we found that if we simply shuffle the comparisons into one dataset, a single pass over the dataset caused the reward model to overfit.\(^5\)

Instead, we train on all \( \binom{K}{2} \) comparisons from each prompt as a single batch element. This is much more computationally efficient because it only requires a single forward pass of the RM for each completion (rather than \( \binom{K}{2} \) forward passes for \( K \) completions) and, because it no longer overfits, it achieves much improved validation accuracy and log loss.

Specifically, the loss function for the reward model is:

\[
\text{loss} (\theta) = -\frac{1}{\binom{K}{2}} E_{(x,y_w,y_l) \sim D} \left[ \log \left( \sigma \left( r_\theta (x, y_w) - r_\theta (x, y_l) \right) \right) \right]
\]

where \( r_\theta (x, y) \) is the scalar output of the reward model for prompt \( x \) and completion \( y \) with parameters \( \theta \), \( y_w \) is the preferred completion out of the pair of \( y_w \) and \( y_l \), and \( D \) is the dataset of human comparisons.


\(^5\) That is, if each of the possible \( \binom{K}{2} \) comparisons is treated as a separate data point, then each completion will potentially be used for \( K-1 \) separate gradient updates. The model tends to overfit after a single epoch, so repeating data within an epoch also causes it to overfit.

Finally, since the RM loss is invariant to shifts in reward, we normalize the reward model using a bias
so that the labeler demonstrations achieve a mean score of 0 before doing RL.

### RM 问答

#### 用的是什么loss function？

在这篇论文中，训练奖励模型时使用的是**交叉熵损失函数**（Cross-Entropy Loss）。具体来说，模型被训练来预测人类偏好排序中哪一个响应更好。交叉熵损失函数被用来衡量模型预测与实际人类偏好排序之间的差异。通过最小化这个损失，奖励模型能够更好地学习到人类的偏好，从而在强化学习的过程中为语言模型提供准确的奖励信号。

#### 这里面的 \( \binom{K}{2} \)  是什么意思

在论文中，\( \binom{K}{2} \) 是组合符号，表示从 \(K\) 个样本中选出 2 个样本的所有可能组合。具体来说，如果有 \(K\) 个模型生成的响应，\( \binom{K}{2} \) 表示从中选择任意两个响应进行比较和排序。这个符号在奖励模型训练中用于构造可能的响应对，让模型学习人类的偏好排序，从而生成更符合人类期望的输出。

#### WHY train on all \( \binom{K}{2} \) comparisons from each prompt as a single batch element 


\(^5\) 中提到作者强调了在训练奖励模型时面临的一个挑战：由于每个标注任务中的比较高度相关，如果将这些比较数据简单地混合在一起，模型可能会迅速过拟合。这是因为这些比较虽然来自不同的响应，但它们共享相同的提示（prompt），因此它们之间存在很强的关联性。

当模型在训练数据中看到相同提示下的多次比较时，可能会过度记住这些具体的比较而不是学习更通用的特征或模式。这种情况会导致模型在训练数据上表现良好，但在处理未见过的数据时效果不佳，即表现出过拟合。

为了应对这个问题，论文中提到了一种方法，即将所有与同一个提示相关的比较作为一个批处理元素进行训练。这种方法通过增加批次中的样本多样性，帮助模型学习到更广泛的特征，从而减少过拟合的风险。换句话说，这种方法旨在使模型关注不同提示之间的区别，而不仅仅是特定提示下的细节，从而提高模型的泛化能力。

我自己的想法是，如果放到一个batch，会提升任务的难度，迫使模型不去死记硬背。因为如果全局shuffle，可能已经在某个prompt下学习到了一些completions的比较信号，这样再做一些新的比较的时候，就会降低难度。而一次一个batch，则是同时学习，不存在先学习到了一些的情况。
举个例子，如果我是一个策略的高T，取评审工程的述职。如果我能每讲完一个同学，看到和之前的排序，对比完全看不到，最后一次性给出排序。前者肯定会让我和其他工程评委之间的打分跟接近。

#### loss function 里面的 E～D是什么意思？

在论文中的损失函数表达式里，符号 \(E_{\sim D}\) 表示对数据分布 \(D\) 进行期望（Expectation）。具体来说，\(D\) 是训练数据的分布，包含了不同的提示（prompts）及其对应的响应。这个期望值表示在整个数据集上计算损失函数的平均值，而不是仅针对单一的样本或批次。通过计算期望，模型在整个数据集上进行优化，确保损失函数反映整体数据的趋势，从而训练出更加泛化的模型。

#### 既然是平均值，为什么还要除以 \( \binom{K}{2} \)

在损失函数中，除以  \( \binom{K}{2} \) 的目的是对所有可能的比较对数目进行归一化。这么做是为了确保损失函数的值反映出每个提示下所有可能的比较对的平均效果，而不是让某些提示下生成的比较对因为数量多而在损失函数中占据过大的权重。

尽管期望值 \(E_{\sim D}\) 表示对整个数据集的平均值，但  \( \binom{K}{2} \) 是用来规范化每个提示下的比较对的数量，确保每个提示对最终损失的贡献是公平的，并且不受生成响应数量的影响。

#### 为什么要对两个r的差值，用 \( \sigma \)  然后再取 \( log \)

在论文中，对两个得分 \( r(a) \) 和 \( r(b) \) 的差值使用逻辑函数 \( \sigma \) （sigmoid）然后再取对数（log）是一种标准的处理方式，用于将差值映射到概率空间。具体来说，这种处理有以下几个目的：

1. **概率映射**：逻辑函数 \( \sigma(x) = \frac{1}{1 + e^{-x}} \) 将任意实数 \( x \) 映射到 [0, 1] 之间，解释为概率。这表明响应 \( a \) 相对于响应 \( b \) 的优劣程度。

2. **稳定性**：取对数后，损失函数在优化时更加稳定，特别是在处理非常大的或非常小的差值时。对数函数将乘法转换为加法，简化了模型更新的过程。

3. **梯度优化**：使用对数后，梯度计算更容易且更平滑，提升了模型的训练效果。这样处理后的损失函数对于小的差值有更高的敏感度，而对于大的差值则更平滑。

这个步骤帮助模型更好地学习和区分人类偏好，并确保模型能够更有效地优化生成的输出。
