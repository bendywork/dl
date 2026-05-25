# ============================================================
# 章节：面向对象编程（p08）
# ============================================================


# === 面向过程编程示例 ===

def get_up(name):
    print(f'{name}睁开眼睛')
    print(f'{name}起身')
    print(f'{name}穿好衣服')


def wash(name):
    print(f'{name}刷牙')
    print(f'{name}洗脸')


def eat(name):
    print(f'{name}吃菜')
    print(f'{name}扒饭')


def login_id(name):
    print(f'{name}输入账号密码')
    print(f'{name}登陆账号成功')


def study(name):
    print(f'{name}看视频')
    print(f'{name}查资料')
    print(f'{name}写代码')


def sleep(name):
    print(f'{name}脱掉外套')
    print(f'{name}躺下')
    print(f'{name}闭上眼睛')


# 面向过程：模拟学生的一天
count_s = 0
stu1 = '张三'
age1 = 18
grade1 = '高三'
print(f'大家好! 我是{stu1}, 今年{age1}岁, 目前正在读{grade1}!')
count_s += 1
get_up(stu1)
wash(stu1)
eat(stu1)
login_id(stu1)
study(stu1)
eat(stu1)
study(stu1)
eat(stu1)
wash(stu1)
sleep(stu1)
print(f'当前统计的学生人数为: {count_s}')


# 面向过程：模拟老师的一天（老师有额外的 clock_in / work 方法）
def clock_in(name):
    print(f'{name}录入指纹')
    print(f'{name}打卡成功')


def work(name):
    print(f'{name}授课')
    print(f'{name}答疑')
    print(f'{name}写代码')


count_t = 0
t1 = '老赵'
age_t1 = 39
department = '教学部'
print(f'大家好! 我是{t1}, 今年{age_t1}岁, 在{department}任职!')
count_t += 1
get_up(t1)
wash(t1)
eat(t1)
clock_in(t1)
work(t1)
eat(t1)
work(t1)
eat(t1)
wash(t1)
sleep(t1)
print(f'当前统计的老师人数为: {count_t}')


# === 面向对象编程示例：类的定义与继承（模拟学生和老师的一天） ===

class Person:

    def __init__(self, name, age):
        self.name = name
        self.age = age
        self.show_time()

    def get_up(self):
        print(f'{self.name}睁开眼睛')
        print(f'{self.name}起身')
        print(f'{self.name}穿好衣服')

    def wash(self):
        print(f'{self.name}刷牙')
        print(f'{self.name}洗脸')

    def eat(self):
        print(f'{self.name}吃菜')
        print(f'{self.name}扒饭')

    def sleep(self):
        print(f'{self.name}脱掉外套')
        print(f'{self.name}躺下')
        print(f'{self.name}闭上眼睛')

    def show_time(self):
        pass


class Student(Person):

    count_s = 0

    def __init__(self, name, age, grade):
        self.grade = grade
        super(Student, self).__init__(name, age)
        Student.count_s += 1

    def show_time(self):
        print(f'大家好! 我是{self.name}, 今年{self.age}岁, 目前正在读{self.grade}!')

    def login_id(self):
        print(f'{self.name}输入账号密码')
        print(f'{self.name}登陆账号成功')

    def study(self):
        print(f'{self.name}看视频')
        print(f'{self.name}查资料')
        print(f'{self.name}写代码')

    @classmethod
    def publish(cls):
        print(f'当前统计的学生人数为: {cls.count_s}')


class Teacher(Person):

    count_t = 0

    def __init__(self, name, age, department):
        self.department = department
        super(Teacher, self).__init__(name, age)
        Teacher.count_t += 1

    def show_time(self):
        print(f'大家好! 我是{self.name}, 今年{self.age}岁, 在{self.department}任职!')

    def clock_in(self):
        print(f'{self.name}录入指纹')
        print(f'{self.name}打卡成功')

    def work(self):
        print(f'{self.name}授课')
        print(f'{self.name}答疑')
        print(f'{self.name}写代码')

    @classmethod
    def publish(cls):
        print(f'当前统计的老师人数为: {cls.count_t}')


stu1_obj = Student('张三', 18, '高三')
stu2_obj = Student('李四', 16, '高一')
stu3_obj = Student('王五', 17, '高二')
Student.publish()

t1_obj = Teacher('老赵', 39, '教学部')
t2_obj = Teacher('老孙', 45, '后勤部')
Teacher.publish()


# === 类对象、实例对象、类属性、实例属性 ===

"""
类, 类对象
object是所有类的父类, 通常省略不写
"""
class Student_Attr(object):

    school = '深兰教育'  # 类属性(类变量)

    def __init__(self, name, age):
        self.name = name  # 实例属性(实例变量)
        self.age = age


"""
魔术方法(特殊方法): 官方定义好的, 以两个下划线开头并且以
两个下划线结尾来命名的方法
魔术方法特点: 一般不需要主动调用, 在满足特定条件时, 会被自动调用

__new__称为构造方法, 用来创建实例对象, 并返回该实例对象
__init__称为初始化方法, 可以对实例对象进行属性定制, 没有返回值

每当实例化时, 先自动调用魔术方法__new__(cls, *args, **kwargs),
把要实例化的类对象(即: Student)作为实参传递给形参cls,
并把实例化时传入的其他实参(即: '张三', 28)传递给形参*args, **kwargs,
然后__new__方法根据cls创建出一个对应的实例对象, 并返回该实例对象(即: stu1=该实例对象)

再自动调用魔术方法__init__(self, name, age), 把__new__方法创建的实例对象(即: stu1)
作为实参传递给形参self, 实例化时传入的其他实参(即: '张三', 28)分别传给形参name, age,
然后__init__方法再对self进行属性定制(inplace操作)
"""
stu1_a = Student_Attr('张三', 28)
stu2_a = Student_Attr('李四', age=32)

""" 调用实例属性: 只能用实例对象调用, 不能用类对象调用 """
print(stu1_a.name)
print(stu2_a.name)

print(getattr(stu1_a, 'age'))
print(getattr(stu1_a, 'adres', '该实例属性不存在'))

"""
调用类属性: 既可以用类对象调用(推荐), 也可以用实例对象调用
注意: 当实例属性和类属性同名时, 实例对象优先调用实例属性
"""
print(Student_Attr.school)
print(stu1_a.school)
print(stu2_a.school)

print(getattr(Student_Attr, 'school'))
print(getattr(Student_Attr, 'adres', '该类属性不存在'))

""" 修改实例属性: 只能用实例对象修改 """
stu1_a.age = 29
print(stu1_a.age)

setattr(stu1_a, 'age', 27)
print(stu1_a.age)

""" 修改类属性: 只能用类对象修改 """
Student_Attr.school = '深兰大学'
print(Student_Attr.school)

setattr(Student_Attr, 'school', '深兰教育')
print(Student_Attr.school)

"""
动态定义实例属性: 当实例对象修改的属性不存在时, 则新增该实例属性
"""
stu1_a.school = 'ShenLanEdu'
print(stu1_a.school)   # 给stu1_a新增一个实例属性
print(Student_Attr.school)  # 类属性不变

setattr(stu2_a, 'adres', '威宁路')
print(stu2_a.adres)   # 给stu2_a新增一个实例属性

""" 动态定义类属性: 当类对象修改的属性不存在时, 则新增该类属性 """
Student_Attr.subject = 'AI'
print(Student_Attr.subject)   # 新增一个类属性
print(stu1_a.subject)
print(stu2_a.subject)

setattr(Student_Attr, 'course', '人工智能')
print(Student_Attr.course)   # 新增一个类属性
print(stu1_a.course)
print(stu2_a.course)

""" 删除属性: 可以用del语句 """
del stu1_a.age
delattr(stu1_a, 'name')

del Student_Attr.school
delattr(Student_Attr, 'subject')

""" 判定属性是否存在 """
print(hasattr(Student_Attr, 'school'))
print(hasattr(stu1_a, 'name'))
print(hasattr(stu2_a, 'age'))


# === 与属性操作相关的内置函数：delattr ===

class Person_Del:

    eat = "rice"

    def __init__(self, age):
        self.age = age


p_del = Person_Del(18)
print(Person_Del.eat)
""" 等价于 del Person_Del.eat """
delattr(Person_Del, "eat")   # 删除类属性eat
# print(Person_Del.eat)  # AttributeError: 已被删除

print(p_del.age)
""" 等价于 del p_del.age """
delattr(p_del, "age")   # 删除实例属性age
# print(p_del.age)  # AttributeError: 已被删除


# === 与属性操作相关的内置函数：getattr ===

class Person_Get:

    eat = "rice"

    def __init__(self, age):
        self.age = age


p_get = Person_Get(18)

""" 等价于 Person_Get.eat """
print(getattr(Person_Get, "eat"))

""" 等价于 p_get.age """
print(getattr(p_get, "age"))

print(getattr(p_get, "height", 178))
# print(getattr(p_get, "height"))  # AttributeError: 无默认值时属性不存在会报错


# === 与属性操作相关的内置函数：hasattr ===

class Person_Has:

    eat = "rice"

    def __init__(self, age):
        self.age = age


p_has = Person_Has(18)
print(hasattr(Person_Has, "eat"))    # True
print(hasattr(p_has, "eat"))         # True
print(hasattr(p_has, "age"))         # True
print(hasattr(p_has, "height"))      # False


# === 与属性操作相关的内置函数：setattr ===

class Person_Set:

    eat = "rice"

    def __init__(self, age):
        self.age = age


p_set = Person_Set(18)
setattr(Person_Set, "eat", "noodles")
print(Person_Set.eat)

setattr(Person_Set, "drink", "water")
print(Person_Set.drink)

setattr(p_set, "age", 29)
print(p_set.age)

setattr(p_set, "height", 178)
print(p_set.height)


# === 类方法、对象方法、静态方法 ===

"""
通常把定义在类中的函数叫方法（method）
对象方法的第一个参数位隐式的接收了实例对象
类方法的第一个参数位隐式的接收了类对象
"""

class Student_Methods:

    school = '深兰教育'

    def __init__(self, name):
        self.name = name

    def study1(self, course):       # 对象方法
        print(f'{self.name}在学习{course}课!')

    @classmethod                    # 类方法装饰器
    def study2(cls, course):
        print(f'{cls.school}的学生在学习{course}课!')
        print(f'{Student_Methods.school}的学生在学习{course}课!')

    @staticmethod                   # 静态方法装饰器
    def study3(course):
        print(f'{Student_Methods.school}的学生在学习{course}课!')

    @property                       # 只读属性装饰器
    def study4(self):
        return f'{self.name}在学习Python课'


stu1_m = Student_Methods('张三')
stu2_m = Student_Methods('李四')

"""
调用对象方法: 通常用实例对象去调用, 用类对象调用时需要主动给self传实参
"""
stu1_m.study1('Python')
stu2_m.study1('机器学习')
Student_Methods.study1(stu1_m, 'Python')
Student_Methods.study1(stu2_m, '机器学习')

"""
调用类方法: 既可以用类对象调用(推荐), 也可以用实例对象调用
"""
Student_Methods.study2('Python')
stu1_m.study2('Python')
stu2_m.study2('Python')

"""
调用静态方法: 既可以用类对象调用(推荐), 也可以用实例对象调用
"""
Student_Methods.study3('Python')
stu1_m.study3('Python')
stu2_m.study3('Python')

"""
调用只读属性: 用实例对象调用
"""
print(stu1_m.study4)
print(stu2_m.study4)


# === 封装：私有属性与私有方法 ===

"""
在属性名或方法名前面加两个下划线开头, 声明为私有属性或私有方法
私有属性或私有方法只能在该类的内部调用, 不能在该类的外部直接调用
"""

class Person_Encap:

    school = '深兰教育'
    __eat = 'rice'        # 私有类属性

    def __init__(self, name, age):
        self.name = name
        self.__age = age  # 私有实例属性

    def get_up(self):
        print(f'{self.name}起床了!')

    def __sleep(self):    # 私有方法
        print(f'{self.name}睡觉了!')

    @classmethod
    def get_eat(cls):
        return cls.__eat

    def get_age(self):
        return self.__age

    def call_sleep(self):
        self.__sleep()


print(Person_Encap.school)
# Person_Encap.__eat  # Error：私有类属性不能在外部访问
print(Person_Encap.get_eat())

p1 = Person_Encap('张三', 19)
print(p1.name)
# p1.__age  # Error：私有实例属性不能在外部访问
print(p1.get_age())

p1.get_up()
# p1.__sleep()  # Error：私有方法不能在外部调用
p1.call_sleep()


# === 继承：单继承 ===

"""
所有的类都默认继承内置的 object 类，通常不用显式的写出来
子类继承父类后，就可以调用父类中的属性和方法
继承顺序：先找当前类，再找父类，再找父类的父类，依此类推
"""

class A_Single:   # 父类
    pass


class B_Single(A_Single):   # 子类
    pass


class Person_Inherit:

    state = "China"

    @staticmethod
    def eat():
        print('吃饭')

    @staticmethod
    def speak():
        print('说话')


class Student_Inherit(Person_Inherit):

    @staticmethod
    def study():
        print('读书')


class Worker_Inherit(Person_Inherit):

    @staticmethod
    def work():
        print('搬砖')


Student_Inherit.study()
Student_Inherit.eat()
Student_Inherit.speak()
print(Student_Inherit.state)

Worker_Inherit.work()
Worker_Inherit.eat()
Worker_Inherit.speak()
print(Worker_Inherit.state)


# === 继承：多级继承（单继承链） ===

class Animal_Single:

    @staticmethod
    def eat():
        print('吃东西')


class Cat_Single(Animal_Single):

    @staticmethod
    def catch_mouse():
        print('抓老鼠')


class Ragdoll_Single(Cat_Single):

    @staticmethod
    def cute():
        print('卖萌')


Ragdoll_Single.cute()
Ragdoll_Single.catch_mouse()
Ragdoll_Single.eat()


# === 继承：多重继承 ===

"""
继承顺序：先找当前类，再按照从左往右的顺序依次找对应的父类
"""

class Animal_Multi:

    @staticmethod
    def eat():
        print('吃东西')


class Cat_Multi:

    @staticmethod
    def catch_mouse():
        print('抓老鼠')


class Ragdoll_Multi(Cat_Multi, Animal_Multi):

    @staticmethod
    def cute():
        print('卖萌')


Ragdoll_Multi.cute()
Ragdoll_Multi.catch_mouse()
Ragdoll_Multi.eat()


# === 方法重写 ===

"""
在继承关系中，当父类的方法不能满足子类的需求时，可以在子类重写父类的该方法
"""

class Animal_Override:

    def __init__(self, food):
        self.food = food

    def eat(self):
        print(f"动物吃{self.food}")


class Cat_Override(Animal_Override):

    # 为了实现'猫吃鱼'的功能, 而不是父类的'动物吃鱼'
    # 子类对eat方法重写
    def eat(self):
        print(f"猫吃{self.food}")


c_override = Cat_Override("鱼")
c_override.eat()


# === super()：调用父类方法 ===

"""
super是内置的类, 可以表示指定类的父类（超类）
适用场景：在子类重写父类方法后，想再调用父类的该方法
"""

class Animal_Super:

    def eat(self):
        print("吃东西")


class Cat_Super(Animal_Super):

    def eat(self):
        print("吃鱼")


class Ragdoll_Super(Cat_Super):

    def eat(self):
        print("喝咖啡")


rd = Ragdoll_Super()
rd.eat()
super(Ragdoll_Super, rd).eat()   # rd调用Ragdoll_Super父类(Cat_Super)的eat方法
super(Cat_Super, rd).eat()       # rd调用Cat_Super父类(Animal_Super)的eat方法

c_super = Cat_Super()
c_super.eat()                    # c_super调用Cat_Super类中的eat方法
super(Cat_Super, c_super).eat()  # c_super调用Cat_Super父类(Animal_Super)的eat方法


# === 继承中的 __init__ 方法 ===

class A_Init:

    def __init__(self, name):
        self.name = name
        self.Q()

    def Q(self):
        print(self.name, 'Q方法被调用')


class B_Init(A_Init):
    pass


b_init = B_Init('张三')
b_init.Q()


class C_Init(A_Init):

    def __init__(self, name):
        self.name = name   # 未调用super，不会执行父类__init__中的self.Q()


c_init = C_Init('赵六')
c_init.Q()


class D_Init(A_Init):

    def __init__(self, name):
        super(D_Init, self).__init__('李四')   # 先用'李四'调用父类__init__
        self.name = name                       # 再覆盖为'王五'


d_init = D_Init('王五')
d_init.Q()


# === 与继承相关的两个内置函数：isinstance ===

"""
isinstance(object, classinfo)
object：实例对象
classinfo：类对象或者由多个类对象构成的元组
判定 object 是否为 classinfo 的实例对象或者其子类的实例对象
"""

class A_Isinstance:
    pass


class B_Isinstance(A_Isinstance):
    pass


class C_Isinstance(A_Isinstance):
    pass


a_i = A_Isinstance()
b_i = B_Isinstance()
c_i = C_Isinstance()
print(isinstance(a_i, A_Isinstance))          # True
print(type(a_i) == A_Isinstance)              # True

print(isinstance(b_i, A_Isinstance))          # True，考虑继承
print(type(b_i) == A_Isinstance)              # False，type不考虑继承
print(isinstance(c_i, A_Isinstance))          # True，考虑继承
print(type(c_i) == A_Isinstance)              # False，type不考虑继承
print(isinstance(c_i, (B_Isinstance, A_Isinstance)))  # True，c_i是A子类的实例


# === 与继承相关的两个内置函数：issubclass ===

"""
issubclass(class, classinfo)
class：类对象
classinfo：类对象或者由多个类对象构成的元组
判定 class 是否为 classinfo 的子类
该函数会把自己视作为自己的子类
"""

class A_Issubclass:
    pass


class B_Issubclass(A_Issubclass):
    pass


class C_Issubclass(A_Issubclass):
    pass


print(issubclass(B_Issubclass, A_Issubclass))               # True
print(issubclass(C_Issubclass, A_Issubclass))               # True
print(issubclass(A_Issubclass, A_Issubclass))               # True，类会被视作其自身的子类
print(issubclass(C_Issubclass, (B_Issubclass, A_Issubclass)))  # True


# === 多态性 ===

"""
多态性是指具有不同内容的方法可以使用相同的方法名，
则可以用一个方法名调用不同内容的方法
"""

class Apple:

    @staticmethod
    def change():
        return '啊~ 我变成了苹果汁!'


class Banana:

    @staticmethod
    def change():
        return '啊~ 我变成了香蕉汁!'


class Mango:

    @staticmethod
    def change():
        return '啊~ 我变成了芒果汁!'


class Juicer:

    @staticmethod
    def work(fruit):
        print(fruit.change())


"""
三个内容不同的change方法使用相同的名字命名,
只要改变change的调用对象, 就可以调用不同内容的方法
"""
Juicer.work(Apple)
Juicer.work(Banana)
Juicer.work(Mango)

# === END p08 ===
