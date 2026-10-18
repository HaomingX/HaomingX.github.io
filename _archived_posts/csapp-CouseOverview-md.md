---
title: csapp_CouseOverview
author: haomingx
avatar: /images/favicon.png
authorDesc: 不断折腾
categories: 
comments: true
date: 2022-12-23 14:03:50
authorLink:
authorAbout:
series: csapp
tags: csapp
---

---

---

写在前面; 今天是12.23，经历了新冠的抗争后开始学习csapp，教材课程及练习都使用CMU 15-213。希望能在这个寒假学习完毕

---

### Ints are not Interagers, Float are not Reals

Example 1:

- Float's : Yes!
- Int's:
  - 40000 * 40000 -> 1600000000
  - 50000 * 50000 -> ??

Example 2:  Is (x + y) + z = x + (y  + z )   ?

- Int's:: Yes!
- Float's : Not Sure!!

Float will throw the more number!

### Memory Referencing Bug Example

![image-20221223150035984](https://haoming2003.oss-cn-hangzhou.aliyuncs.com/202212231500118.png)

![image-20221223151356105](https://haoming2003.oss-cn-hangzhou.aliyuncs.com/202212231513136.png)

**For sure, this is influenced by your gcc version and IDE**.

