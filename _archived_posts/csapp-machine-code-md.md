---
title: machine_code（1）
author: haomingx
avatar: /images/favicon.png
authorDesc: 不断折腾
comments: true
date: 2022-12-27 20:37:07
authorLink:
authorAbout:
categories:
series: csapp
tags:
---

#### 处理器模型发展

https://hansimov.gitbook.io/csapp/part1/ch03-machine-level-representing-of-programs/3.1-a-historial-perspective

#### 摩尔定律

#### 芯片构造

![image-20221227205856257](https://haoming2003.oss-cn-hangzhou.aliyuncs.com/image-20221227205856257.png)

当时标准桌面型号有四个核心，服务器级别的机器有八个核心（上图）

芯片周围连接外围设备的接口：

- DDR是连接到主存的方式，即所谓的DRAM（Dynamic动态 RAM随机访问机）
- PCI是一种同步的独立于处理器的32位或64位局部总线，主要用于连接显示卡、网卡、声卡
- SATA是与不同类型盘连接
- USB接口与USB设备连接
- ethernet网络连接

集成到芯片上的不止是处理器还有很多逻辑单元

#### 处理器架构

| **架构**   | **特点**                     | **代表性的厂商**           | **运营机构**     |
| ---------- | ---------------------------- | -------------------------- | ---------------- |
| **X86**    | **性能高，速度快，兼容性好** | **英特尔，AMD**            | **英特尔**       |
| **ARM**    | **成本低，低功耗**           | **苹果，谷歌，IBM，华为**  | **英国ARM公司**  |
| **RISC-V** | **模块化，极简，可拓展**     | **三星，英伟达，西部数据** | **RISC-V基金会** |
| **MIPS**   | **简洁，优化方便，高拓展性** | **龙芯**                   | **MIPS科技公司** |

**X86在PC上占据大部分份额，ARM在手机处理器上占据绝对份额**

#### c代码运行的过程

```c
gcc -Og -S sum.c
// 将c代码转换为assembly代码
```

-s为-stop停在把c转化为汇编的时刻

-Og是我希望编译器做什么样的优化的规范，这样才能读懂

具体过程：https://www.cnblogs.com/carpenterlee/p/5994681.html

```C
objdump -d sum > sum.d
```

反汇编，sum.d的内容

```c
a.out:     file format elf64-x86-64
     
Disassembly of section .init:

0000000000001000 <_init>:
    1000:   f3 0f 1e fa             endbr64
    1004:   48 83 ec 08             sub    $0x8,%rsp
    1008:   48 8b 05 d9 2f 00 00    mov    0x2fd9(%rip),%rax        # 3fe8 <__gmon_start__>
    100f:   48 85 c0                test   %rax,%rax
    1012:   74 02                   je     1016 <_init+0x16>
    1014:   ff d0                   callq  *%rax
    1016:   48 83 c4 08             add    $0x8,%rsp
    101a:   c3                      retq
        ...
```

**反汇编程序无法访问源代码，甚至无法访问汇编代码，它只是通过实际目标代码文件中的字节来辨别出来的**

或者使用GDB

```c
>> gdb
...
(gdb)
(gdb) disassemble bitXor
Dump of assembler code for function bitXor:
   0x0000000000001149 <+0>:     endbr64
   0x000000000000114d <+4>:     push   %rbp
   0x000000000000114e <+5>:     mov    %rsp,%rbp
   0x0000000000001151 <+8>:     mov    %edi,-0x4(%rbp)
   0x0000000000001154 <+11>:    mov    %esi,-0x8(%rbp)
   0x0000000000001157 <+14>:    mov    -0x4(%rbp),%eax
   0x000000000000115a <+17>:    xor    -0x8(%rbp),%eax
   0x000000000000115d <+20>:    pop    %rbp
   0x000000000000115e <+21>:    retq
End of assembler dump.
```

**此方法前面显示的是16进制地址而非像objdum那样的字节级编码**

#### Assembly Characteristics: Data Types 汇编语言特性

- "Integer" data type of 1,2,4,or 8 bytes，不区分unsigned和signed
- floating point 有4,8,10,bytes
- 没有数组以及一些数据结构，只是内存中连续存储的单元

#### x86-64 Integer Registers

有16个寄存器

![img](https://img-blog.csdnimg.cn/20190723112517340.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3NoZW5kZXpodXRp,size_16,color_FFFFFF,t_70)

注意到 %r代表62位操作 %e代表了32位的操作，%e版本只是%r实体的低32位。

**%rsp寄存器存的是栈指针，它能告诉你程序执行到哪儿了**

#### 移动数据

##### 格式

```assembly
moveq Source Dest
```

##### 操作数类型

![img](https://img-blog.csdnimg.cn/20190723113325695.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3NoZW5kZXpodXRp,size_16,color_FFFFFF,t_70)

##### 操作数组合

![img](https://img-blog.csdnimg.cn/20190723113456800.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3NoZW5kZXpodXRp,size_16,color_FFFFFF,t_70)

#### 理解Swap()函数

![img](https://img-blog.csdnimg.cn/20190723125635539.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3NoZW5kZXpodXRp,size_16,color_FFFFFF,t_70)

使用x86-64的时候，函数参数总是出现在某些特定的寄存器中，**%rdi将是第一个参数寄存器，%rsi将是第二个参数寄存器**。最多可以有6个

#### 完整的内存地址模式

![img](https://img-blog.csdnimg.cn/20190723125841927.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3NoZW5kZXpodXRp,size_16,color_FFFFFF,t_70)

下面是内存完整模式的一个例子

![img](https://img-blog.csdnimg.cn/20190723130008460.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3NoZW5kZXpodXRp,size_16,color_FFFFFF,t_70)

#### 地址计算

![img](https://img-blog.csdnimg.cn/20190723131714576.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3NoZW5kZXpodXRp,size_16,color_FFFFFF,t_70)

![image-20221229121353910](C:\Users\xhm\AppData\Roaming\Typora\typora-user-images\image-20221229121353910.png)

![image-20221229121547636](C:\Users\xhm\AppData\Roaming\Typora\typora-user-images\image-20221229121547636.png)

![img](https://img-blog.csdnimg.cn/20190723131745599.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3NoZW5kZXpodXRp,size_16,color_FFFFFF,t_70)

### control

#### 处理器状态 (x86-64, Partial)

![img](https://img-blog.csdnimg.cn/20190724111040785.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3NoZW5kZXpodXRp,size_16,color_FFFFFF,t_70)

图上的CF,ZF,SF,OF就是微机学过的状态位，其中各自代表的意思如下

![img](https://img-blog.csdnimg.cn/20190724111626651.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3NoZW5kZXpodXRp,size_16,color_FFFFFF,t_70)

![img](https://img-blog.csdnimg.cn/20190724112028471.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3NoZW5kZXpodXRp,size_16,color_FFFFFF,t_70)

 

![img](https://img-blog.csdnimg.cn/20190724112037452.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3NoZW5kZXpodXRp,size_16,color_FFFFFF,t_70)

![img](https://img-blog.csdnimg.cn/2019072411205365.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3NoZW5kZXpodXRp,size_16,color_FFFFFF,t_70)

![img](https://img-blog.csdnimg.cn/20190724112113371.png)