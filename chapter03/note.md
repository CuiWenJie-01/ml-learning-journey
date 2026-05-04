# Accuracy Using Cross-Validation利用交叉验证测量准确性
运行train_a_binary_classifier.pyd的结果（交叉验证计算 Accuracy）：
```
[0.9578  0.9651  0.97005]
Accuracy: 0.9643166666666666
```
**注意：**
这里面Accuracy: 0.9643166666666666不是表示“模型很强啊！已经96%了！”，是表示“不是5”占96%的概率。是“5”的概率是4.03%。因为是基于MNIST 二分类问题是：“是不是 5？”的

---

# Precision and Recall(Confusion Matrices 混淆矩阵)
运行train_a_binary_classifier.pyd的结果：
```
[[53657   922]
 [ 1324  4097]]
```
![alt text](images/混淆矩阵.png)
```
             预测
           0(False)   1(True)
真实 0        TN         FP
真实 1        FN         TP
```
---

# 📊 Precision & Recall（精确率与召回率）

---

## 📌 1. Precision（精确率）

👉 含义：
“在所有预测为 5 的样本中，有多少是真的 5？”

---

### 📐 公式

$$
Precision = \frac{TP}{TP + FP}
$$

---

### 📊 代入计算

$$
Precision = \frac{4097}{4097 + 922}
$$

$$
Precision \approx 0.816
$$

---

### 🧠 解释

👉 ≈ **81.6%**

模型预测为“5”的样本中，有 **81% 是正确的**

* ✔ 说明：预测质量还可以
* ❌ 仍然存在误报（FP）

---

## 📌 2. Recall（召回率）

👉 含义：
“在所有真实的 5 中，有多少被模型找出来？”

---

### 📐 公式

$$
Recall = \frac{TP}{TP + FN}
$$

---

### 📊 代入计算

$$
Recall = \frac{4097}{4097 + 1324}
$$

$$
Recall \approx 0.756
$$

---

### 🧠 解释

👉 ≈ **75.6%**

所有真实的 5 中，只找出了 **75%**

* ❗说明：漏掉约 **25% 的 5（FN）**

---

## 🚨 3. 关键结论

你的模型表现：

* ✔ Precision ≈ 81.6% → 预测“5”时比较可靠
* ❌ Recall ≈ 75.6% → 漏掉较多真实“5”

---

## 📉 4. 对比 Accuracy（准确率）

$$
Accuracy \approx 0.95
$$

---

### ⚠️ 重要问题

虽然 Accuracy 很高：

* ❗仍然漏掉约 **25% 的 5**

👉 说明：

💥 **Accuracy 无法反映类别不均衡问题**

---

## 🧠 一句话总结

* Precision → “预测对不对”
* Recall → “有没有漏掉”

---

# Precision(精确率) & Recall（召回率）
运行train_a_binary_classifier.pyd的结果：
```
Precision: 0.6461647727272727
Recall: 0.8391440693598967
```
# 📊 Precision & Recall 结果解读

---

## 🧠 1. 一句话总结

👉 你的模型：

> “找得很全，但有点乱说”

---

## 🔵 2. Recall = 0.839（很高）

$$
Recall = \frac{TP}{TP + FN}
$$

### 含义：

$$
Recall \approx 0.839
$$

👉 在所有真实的“5”里，你找到了 **83.9%**

✔ 说明：

* FN（漏掉的5）较少
* 模型“比较积极”

📌 直觉：

> 宁可多抓，也不漏掉

---

## 🔴 3. Precision = 0.646（偏低）

$$
Precision = \frac{TP}{TP + FP}
$$

### 含义：

$$
Precision \approx 0.646
$$

👉 你预测为“5”的里面，只有 **64.6%是真的**

❌ 说明：

* FP（误报）较多
* 有不少“假5”

📌 直觉：

> 经常把不是5的也当成5

---

## ⚖️ 4. 两者关系（核心）

$$
Recall \uparrow \quad \text{Precision} \downarrow
$$

👉 说明：

模型使用了**较低的决策阈值（threshold）**

---

## 🧪 5. 混淆矩阵视角

| 类型 | 含义     |
| -- | ------ |
| TP | 正确预测为5 |
| FP | 错误预测为5 |
| FN | 漏掉的5   |
| TN | 正确预测非5 |

---

## 📉 6. 模型本质

👉 当前模型特点：

* FN ↓（漏得少）
* FP ↑（误报多）

---

## 🎯 7. 业务含义

### ✔ 适合 Recall 重要的场景

* 疾病筛查
* 欺诈检测
* 安全检测

---

### ❌ 不适合 Precision 要求高的场景

* 自动审批
* 垃圾邮件误删敏感场景
* 高精度分类

---

## 🧠 8. 一句话总结

👉 模型：

> “抓得很全，但误判较多”

---

## 🚀 9. 下一步（Hands-On ML关键）

建议继续学：

### ⭐ Precision-Recall 曲线

$$
\text{Precision-Recall Curve}
$$

---

### ⭐ F1-score

$$
F1 = \frac{2 \cdot Precision \cdot Recall}{Precision + Recall}
$$

---

### ⭐ threshold 调整

👉 控制 Precision / Recall trade-off

---

