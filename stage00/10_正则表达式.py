# ============================================================
# 章节：正则表达式
# 内容：re 模块 / 字符匹配 / 元字符 / 特殊序列 /
#        字符集 / 分组 / Pattern 方法 / Match 方法 / 编译模式
# ============================================================

import re

# ============================================================
# 简介：正则表达式入门示例 —— 验证 QQ 邮箱
# ============================================================

# ---- 知识点：使用 re.compile 编译正则，fullmatch 进行完整匹配 ----
"""
假设QQ邮箱的规则如下:
- 结尾是 @qq.com
- @前面只能是数字
- 且长度为5-11位
- 且不能以0开头
"""
pattern = r'[1-9]\d{4,10}@qq\.com'
p = re.compile(pattern)

def isValid(email):
    if p.fullmatch(email):
        print("有效的QQ邮箱")
    else:
        print("无效的QQ邮箱")


# ============================================================
# 字符匹配
# ============================================================

# ---- 知识点：普通字符 —— 只与自身匹配 ----
p = re.compile(r"test123")
print(p.search("atest123b"))


# ============================================================
# 特殊字符（元字符）：.  ^  $  *  +  ?  { }  [ ]  \  |  ( )
# ============================================================

# ---- 知识点：. 匹配除换行符以外的任意一个字符；DOTALL 模式下匹配包括换行符 ----
p = re.compile(r".")
print(p.match("abc"))
print(p.match("9bc"))
print(p.match("@bc"))
print(p.match(".bc"))
print(p.match("\tbc"))
print(p.match("\nbc"))    # 不匹配换行符

p = re.compile(r".", flags=re.DOTALL)
print(p.match("\nbc"))    # DOTALL 模式下匹配换行符


# ---- 知识点：^ 匹配字符串开头；MULTILINE 模式下还匹配每行开头 ----
p = re.compile(r"^ab")
print(p.findall("abcd\nabfg"))          # 只匹配第一行开头

p = re.compile(r"^ab", flags=re.MULTILINE)
print(p.findall("abcd\nabfg"))          # 匹配每行开头


# ---- 知识点：$ 匹配字符串末尾或末尾换行符之前；MULTILINE 下还匹配每行末尾 ----
p = re.compile(r"cd$")
print(p.findall("abcd\n"))

p = re.compile(r"cd$", flags=re.MULTILINE)
print(p.findall("abcd\nefcd"))

# $ 找到两个（空的）匹配：一个在换行符之前，一个在字符串末尾
p = re.compile(r"$")
print(p.findall("abcd\n"))


# ---- 知识点：* 匹配前面的正则 0 到任意次，贪婪 ----
p = re.compile(r"ab*")
print(p.search("a"))
print(p.search("ab"))
print(p.search("abb"))
print(p.search("abbbc"))


# ---- 知识点：+ 匹配前面的正则 1 到任意次，贪婪 ----
p = re.compile(r"ab+")
print(p.search("a"))
print(p.search("ab"))
print(p.search("abb"))
print(p.search("abbbc"))


# ---- 知识点：? 匹配前面的正则 0 到 1 次，贪婪 ----
p = re.compile(r"ab?")
print(p.search("a"))
print(p.search("ab"))
print(p.search("abb"))
print(p.search("abbbc"))


# ---- 知识点：*? +? ?? 非贪婪模式（尽量少匹配） ----
p = re.compile(r'<.*>')
print(p.search('<a> b <c>'))    # 贪婪：匹配最长

p = re.compile(r'<.*?>')
print(p.search('<a> b <c>'))    # 非贪婪：匹配最短

p = re.compile(r"ab+?")
print(p.search("abbbc"))        # 只匹配 'ab'

p = re.compile(r"ab??")
print(p.search("abc"))          # 只匹配 'a'


# ---- 知识点：{m} 匹配恰好 m 次重复 ----
p = re.compile(r"ab{2}")
print(p.search("abc"))
print(p.search("abbc"))
print(p.search("abbbc"))


# ---- 知识点：{m,n} 匹配 m 到 n 次（贪婪）；忽略 m 则下限为 0，忽略 n 则上限无限 ----
p = re.compile(r"ab{2,4}")
print(p.search("abc"))
print(p.search("abbc"))
print(p.search("abbbc"))
print(p.search("abbbbc"))
print(p.search("abbbbbc"))

p = re.compile(r"ab{,4}")
print(p.search("ac"))
print(p.search("abc"))

p = re.compile(r"ab{2,}")
print(p.search("abbbbc"))
print(p.search("abbbbbc"))


# ---- 知识点：{m,n}? 非贪婪版本的 {m,n} ----
p = re.compile(r"ab{2,4}?")
print(p.search("abc"))
print(p.search("abbc"))
print(p.search("abbbc"))
print(p.search("abbbbc"))
print(p.search("abbbbbc"))

p = re.compile(r"ab{,4}?")
print(p.search("ac"))
print(p.search("abc"))

p = re.compile(r"ab{2,}?")
print(p.search("abbbbc"))
print(p.search("abbbbbc"))


# ---- 知识点：| 或操作，A|B 匹配 A 或 B，先匹配到的优先 ----
p = re.compile(r"d|e|b")
print(p.search("abc"))
print(p.search("aebcd"))


# ============================================================
# 特殊序列（反斜杠序列）
# ============================================================

# ---- 知识点：\ 转义特殊字符 ----
# 只匹配 * 号
p = re.compile(r"\*")
print(p.fullmatch("*"))

# 只匹配 + 号
p = re.compile(r"\+")
print(p.fullmatch("+"))

# 只匹配 ? 号
p = re.compile(r"\?")
print(p.fullmatch("?"))


# ---- 知识点：\number 引用对应分组匹配的内容 ----
# \1 匹配的内容和第1组一定一样
p = re.compile(r"(.+) \1")
print(p.search("ab abc"))
print(p.search("5 5"))

# 两个组匹配的内容不一定一样
p = re.compile(r"(.+) (.+)")
print(p.search("ab abc"))
print(p.search("5 5"))


# ---- 知识点：\A 匹配字符串开头，MULTILINE 模式下不识别换行（与 ^ 的区别） ----
p = re.compile(r"^ab")
print(p.findall("abcd\nabfg"))

p = re.compile(r"^ab", flags=re.MULTILINE)
print(p.findall("abcd\nabfg"))

p = re.compile(r"\Aab")
print(p.findall("abcd\nabfg"))

p = re.compile(r"\Aab", flags=re.MULTILINE)
print(p.findall("abcd\nabfg"))


# ---- 知识点：\b 匹配单词边界；\B 匹配非单词边界 ----
p = re.compile(r"er\b")
print(p.search("never"))
print(p.search("verb"))

p = re.compile(r"\ba\b")
print(p.search("I have a dog"))

p = re.compile(r"er\B")
print(p.search("never"))
print(p.search("verb"))

p = re.compile(r"\Ba\B")
print(p.search("I have a dog"))


# ---- 知识点：\d 匹配任意数字字符，等价于 [0-9] ----
p = re.compile(r"\d")
print(p.search("a1234b"))

p = re.compile(r"\d+")
print(p.search("a1234b"))


# ---- 知识点：\D 匹配任意非数字字符，等价于 [^0-9] ----
p = re.compile(r"\D")
print(p.search("ab1234c"))

p = re.compile(r"\D+")
print(p.search("ab1234c"))


# ---- 知识点：\s 匹配任意空白符 ----
p = re.compile(r"a\sb")
print(p.search("adb a bc"))


# ---- 知识点：\S 匹配任意非空白符 ----
p = re.compile(r"a\Sb")
print(p.search("adb a bc"))


# ---- 知识点：\w 匹配字母/数字/下划线，等价于 [a-zA-Z0-9_] ----
p = re.compile(r"a\wb")
print(p.findall("adba9ba_ba b"))


# ---- 知识点：\W 匹配非字母非数字非下划线，等价于 [^a-zA-Z0-9_] ----
p = re.compile(r"a\Wb")
print(p.findall("adba9ba_ba b"))


# ---- 知识点：\Z 只匹配字符串末尾，MULTILINE 模式下不识别换行 ----
p = re.compile(r"cd\Z")
print(p.findall("abcd"))

# 结尾是 '\n'，不是 'cd'
print(p.findall("abcd\n"))

# MULTILINE 模式下，\Z 不识别换行
p2 = re.compile(r"cd\Z", flags=re.MULTILINE)
print(p2.findall("abcd\nef"))

# 只会找到一个（空的）匹配
p = re.compile(r"\Z")
print(p.findall("abcd\n"))


# ---- 知识点：\n \t \\ \' \" 标准转义字符在正则中同样支持 ----
p = re.compile(r"\n")
print(p.findall("\n"))

p = re.compile(r"\t")
print(p.findall("\t"))

p = re.compile(r"\\")
print(p.findall("\\"))

p = re.compile(r"\'")
print(p.findall("'"))

p = re.compile(r"\"")
print(p.findall('"'))


# ============================================================
# [] 字符集
# ============================================================

# ---- 知识点：[] 字符集 —— 匹配集合中的任意一个字符 ----
p = re.compile(r"[amk]")
print(p.findall("I have a monkey"))


# ---- 知识点：[] 字符集 —— 用 - 表示字符范围 ----
p = re.compile(r"[a-y]")
print(p.findall("ahzyqAHZYQ"))

p = re.compile(r"[0-5][A-Y]")
print(p.findall("a0hzyq125A6HZYQ"))


# ---- 知识点：[] 字符集 —— 特殊字符在字符集中失去特殊含义，变为普通字符 ----
p = re.compile(r"[.+]")
print(p.findall("abc"))

p = re.compile(r"[.+]")
print(p.findall("a.b+c.d+"))


# ---- 知识点：[] 字符集 —— 特殊序列 \d \s \w 在字符集中可被使用 ----
p = re.compile(r"[\d]")
print(p.search("a1234b"))

p = re.compile(r"[\d+]")
print(p.findall("a1234b+"))

p = re.compile(r"[a\sb]")
print(p.findall("adb a bc"))

p = re.compile(r"[\w]")
print(p.findall("adb_a b!c"))


# ---- 知识点：[] 字符集 —— ^ 放在首位表示取反，匹配不在集合内的字符 ----
p = re.compile(r"[^5]")
print(p.findall("5a b512!5"))

p = re.compile(r"[^^]")
print(p.findall("5a^b512!5"))


# ---- 知识点：[] 字符集 —— 匹配 [ 和 ]，用反斜杠转义 ----
p = re.compile(r"[\[\]]")
print(p.findall("[]"))


# ============================================================
# 分组
# ============================================================

# ---- 知识点：(…) 捕获分组 —— 匹配括号内的正则，并记录子组内容 ----
# 组 0 表示整个匹配；子组从 1 开始向右编号
p = re.compile(r"b(.+)a(.+)e")

m = p.match("babacdefg")
print(m)
print(m.group(1), m.group(2))
print(m.span(1), m.span(2))
print(m.start(1), m.end(1))
print(m.start(2), m.end(2))

# 多个分组，findall 返回元组列表
print(p.findall("babacdefg"))

# 引用第 1 组匹配的内容（\1）
p = re.compile(r"b(.+)a(\1)e")
print(p.findall("babaabefg"))


# ---- 知识点：(?:…) 非捕获分组 —— 不创建新组，匹配内容不可被引用 ----
p = re.compile(r"b(?:.+)a(?:.+)e")
m = p.match("babacdefg")
print(m)


# ============================================================
# 编译正则表达式
# ============================================================

# ---- 知识点：re.compile(pattern, flags=0) 编译正则，返回 Pattern 对象 ----
p = re.compile('ab*', flags=0)
print(p)


# ============================================================
# 执行匹配 —— Pattern 实例对象方法
# ============================================================

# ---- 知识点：Pattern.search() 扫描整个字符串，返回第一个匹配的 Match 对象 ----
p = re.compile('og')
m = p.search("dog")
print(m)

print(p.search("dog", 2))          # pos=2，从第 2 个字符开始
print(p.search("dog", endpos=2))   # endpos=2，只搜索前 2 个字符


# ---- 知识点：Pattern.match() 从字符串起始位置匹配，起始不匹配则返回 None ----
p = re.compile('og')

print(p.match("dog"))              # 从起始位置匹配，'dog' 的起始不是 'og'
print(p.search("dog", 1))         # search 带 pos=1，从 'o' 开始


# ---- 知识点：Pattern.fullmatch() 整个字符串都匹配才返回 Match 对象 ----
p = re.compile('o[gh]')

print(p.fullmatch("ogh"))          # 长度3，不完全匹配
print(p.fullmatch("og"))           # 完全匹配
print(p.fullmatch("oh"))           # 完全匹配
print(p.fullmatch("dog"))          # 不匹配
print(p.fullmatch("dog", 1))       # pos=1，从 'o' 开始，完全匹配 'og'


# ---- 知识点：Pattern.findall() 找到所有不重复匹配，以列表返回 ----
p = re.compile(r'\d')
print(p.findall("Ten years ago, Three dogs"))
print(p.findall("10 years ago, 3 dogs"))

# 多个分组，返回元组列表
p = re.compile(r'(\d+)-(\D)')
print(p.findall("Ten-years ago, Three-dogs"))
print(p.findall("101-years ago, 3-dogs"))

# 当捕获分组被重复时，组号也重复，后面组的结果会把前面组的结果覆盖
p = re.compile(r"(\d)(\d){2}")   # 等价于 (\d)(\d)(\d)，后面两个组号都为 2
print(p.findall("1234567890"))


# ---- 知识点：Pattern.split() 按匹配子串分割字符串，以列表返回 ----
# 若有捕获分组，分组匹配的内容也会包含在结果中
p = re.compile(r"\W+")
print(p.split('Words, words, words.'))

p = re.compile(r"(\W+)")
print(p.split('Words, words, words.'))

print(p.split('...Words, words, words...'))


# ---- 知识点：Pattern.sub() 替换所有匹配项，repl 可以是字符串或函数 ----
p = re.compile(r'blue|white|red')
# 把每一个从左开始非重叠匹配的字符串用其他字符串替换
print(p.sub('colour', 'blue socks and red shoes'))
print(p.sub('colour', 'blue socks and red shoes', count=1))


def func(matchobj):
    if matchobj.group() == '-':
        return ' '
    return '-'

# 把每一个从左开始非重叠匹配的对象作为参数传入函数调用
# 这个函数只能有一个匹配对象参数，并返回一个替换字符串
p = re.compile(r'-{1,2}')
print(p.sub(func, 'pro----gram-files'))


# ============================================================
# Match 实例对象方法
# ============================================================

# ---- 知识点：Match.group(*groupN) 返回一个或多个子组的匹配结果 ----
p = re.compile(r"b(.+)a(.+)e")
m = p.match("babacdefg")
print(m)
print(m.group())          # 等同于 group(0)，整个匹配结果
print(m.group(0))
print(m.group(1))
print(m.group(2))
print(m.group(2, 1, 0))   # 多个参数，返回元组


# ---- 知识点：Match.start() / Match.end() 返回匹配的起始/结束位置 ----
p = re.compile(r"b(.+)a(.+)e")
m = p.match("babacdefg")
print(m)
print(m.start(), m.end())
print(m.start(1), m.end(1))
print(m.start(2), m.end(2))


# ---- 知识点：Match.span() 返回 (start, end) 元组 ----
p = re.compile(r"b(.+)a(.+)e")
m = p.match("babacdefg")
print(m)
print(m.span())
print(m.span(0))
print(m.span(1))
print(m.span(2))


# ============================================================
# 模块级别函数（不需要先创建 Pattern 对象）
# re.search / re.match / re.fullmatch / re.findall / re.split / re.sub
# ============================================================

# ---- 知识点：re 模块级别函数 —— 与 Pattern 方法等价，直接传入 pattern 字符串 ----
# 示例：re.search
print(re.search(r'\d+', "abc 123 def"))

# 示例：re.findall
print(re.findall(r'\d+', "abc 123 def 456"))

# 示例：re.sub
print(re.sub(r'\d+', 'NUM', "abc 123 def 456"))

# 示例：re.split
print(re.split(r'\W+', 'Words, words, words.'))


# ============================================================
# 编译模式（flags）
# ============================================================

# ---- 知识点：re.I / re.IGNORECASE 忽略大小写匹配 ----
p = re.compile(r"[a-z]+", flags=re.IGNORECASE)
print(p.match("aAbBcC"))


# ---- 知识点：re.M / re.MULTILINE 多行匹配，影响 ^ 和 $ ----
p = re.compile(r"^ab", flags=re.MULTILINE)
print(p.findall("abcd\nabfg"))

p = re.compile(r"cd$", flags=re.MULTILINE)
print(p.findall("abcd\nefcd"))


# ---- 知识点：re.S / re.DOTALL 使 . 匹配包括换行符在内的所有字符 ----
p1 = re.compile(r".")
p2 = re.compile(r".", flags=re.DOTALL)
print(p1.search("\nbc"))    # 不匹配换行符
print(p2.search("\nbc"))    # 匹配换行符


# === END p12 ===
