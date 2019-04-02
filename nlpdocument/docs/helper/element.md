# 可以支持的元素
本文档支持markdown拓展功能，可以使用诸如：数学公式、语法高亮，高亮标注等等原生markdown不支持的功能。

## 代码
### 插入代码
```` markdown
``` python
import numpy as np
import torch.nn as nn
```
````
效果：
``` python
import numpy as np
import torch.nn as nn
```

### 代码特定行高亮
````markdown
``` python hl_lines="3 4"
""" Bubble sort """
def bubble_sort(items):
    for i in range(len(items)):
        for j in range(len(items) - 1 - i):
            if items[j] > items[j + 1]:
                items[j], items[j + 1] = items[j + 1], items[j]
```
````
效果：
``` python hl_lines="3 4"
""" Bubble sort """
def bubble_sort(items):
    for i in range(len(items)):
        for j in range(len(items) - 1 - i):
            if items[j] > items[j + 1]:
                items[j], items[j + 1] = items[j + 1], items[j]
```
更多有关代码的使用细节可以参考[CodeHilite](https://squidfunk.github.io/mkdocs-material/extensions/codehilite/)

## 表格
```markdown
dog | bird | cat
----|------|----
foo | foo  | foo
bar | bar  | bar
baz | baz  | baz
```
效果：

dog | bird | cat
----|------|----
foo | foo  | foo
bar | bar  | bar
baz | baz  | baz

## 数学公式
文档支持使用Latex进行数学公式的编辑

```markdown
$$
\frac{n!}{k!(n-k)!} = \binom{n}{k}
$$
```
效果：

$$
\frac{n!}{k!(n-k)!} = \binom{n}{k}
$$

## Admonition

```markdown
!!! info "Ref"
    [Massive Exploration of Neural Machine Translation Architectures](https://arxiv.org/abs/1703.03906), Denny Britz, Anna Goldie et al.
```
效果：

!!! info "Ref"
    [Massive Exploration of Neural Machine Translation Architectures](https://arxiv.org/abs/1703.03906), Denny Britz, Anna Goldie et al.

```markdown
!!! question "Why"
    From the authors: "*This way, [...] that makes it easy for SGD to “establish communication” between the input and the output. We found this simple data transformation to greatly improve the performance of the LSTM.*"
```
效果：

!!! question "Why"
    From the authors: "*This way, [...] that makes it easy for SGD to “establish communication” between the input and the output. We found this simple data transformation to greatly improve the performance of the LSTM.*"

更多有关Admonition的使用细节可以参考[Admonition](https://squidfunk.github.io/mkdocs-material/extensions/admonition/)

