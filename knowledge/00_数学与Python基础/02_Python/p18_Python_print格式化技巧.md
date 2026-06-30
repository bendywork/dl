# Python print 格式化输出技巧

## 1. `:>N` — 右对齐占位

`>` 表示右对齐，数字为宽度。

```python
for i in [0, 1, 9, 10]:
    print(f"{i:>2}. 条目")

#  0. 条目
#  1. 条目
#  9. 条目
# 10. 条目    ← 小数点对齐
```

| 符号 | 含义 |
|------|------|
| `:>2` | 右对齐占 2 格 |
| `:<2` | 左对齐占 2 格 |
| `:^2` | 居中占 2 格 |
| `:02` | 右侧补零（如 `09`） |

---

## 2. 条件拼接字符串

一行搞定"有内容就显示，没有就跳过"：

```python
tag_str = f"  [{tag}]" if tag else ""
print(f"{title}{tag_str}")
```

等价于啰嗦写法：

```python
if tag:
    print(f"{title}  [{tag}]")
else:
    print(f"{title}")
```

---

## 3. 数字递增显示

`data-index` 从 0 开始，展示给用户时 +1：

```python
print(f"{(rank + 1):>2}. {title}")
# 0 → 显示 " 1."
# 9 → 显示 "10."
```

---

## 4. sorted + lambda — 按字典字段排序

```python
results = [{"rank": 5}, {"rank": 0}, {"rank": 3}]
sorted(results, key=lambda x: x["rank"])
# → [{"rank": 0}, {"rank": 3}, {"rank": 5}]
```

---

## 5. `or 0` — 空值兜底

```python
x = None
print(x or 0)   # → 0

x = 5
print(x or 0)   # → 5
```

`None`、空字符串、0 在布尔上下文中为 `False`，触发 `or` 后面的默认值。

常与排序配合：

```python
sorted(results, key=lambda x: x["rank"] or 0)
# rank 为 None 时当成 0 参与排序
```
