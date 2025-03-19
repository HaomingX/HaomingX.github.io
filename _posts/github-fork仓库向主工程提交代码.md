---
title: github fork仓库向主工程提交代码
date: 2022-09-22 02:01:42
tags: github代码提交
cover: https://haoming2003.oss-cn-hangzhou.aliyuncs.com/202209220205897.jpg
---



## 1. fork并关联本地

进入我的主页，找到这个仓库

![image-20220922020848487](https://haoming2003.oss-cn-hangzhou.aliyuncs.com/202209220208551.png)

点击右上角的fork，然后你的主页里就多了一个同样的仓库了，相当于做了一个镜像开了个分支

然后在本地合适位置（最好别带中文）建立一个同名文件夹（名字不影响，但是为了一致嘛），然后在文件夹中打开git bash(path配置好了的话，powershell也可以)，然后按照如下流程输入（有梯子的话最好打开梯子）

```bash
# 克隆fork后仓库到本地,yourname为你的github名
git clone （fork后的url）
```

然后你的文件夹下就会出现本项目已有所有文件，然后你就可以在本地仓库的对应文件夹（你的名字）添加你的学习文件了

```bash
# add到本地暂存区, .是add所有新文件的意思
git add .

# commit到本地仓库
git commit -m "first_commit"

# 关联到你的远程仓库
git remote add origin your_url

# push到你的远程仓库
git push -u origin main
```

![img](https://haoming2003.oss-cn-hangzhou.aliyuncs.com/202209220214651.webp)

然后你fork的仓库会出现你的新增文件



## 2.关联主工程

关联主工程：

```bash
git remote add okex(自定义分支名) (主工程的git url)
# 查看关联情况
git remote -v
```

![在这里插入图片描述](https://haoming2003.oss-cn-hangzhou.aliyuncs.com/202209220224884.png)

拉取主工程各分支信息到本地：

```bash
git fetch okex(自定义分支名)
```

![在这里插入图片描述](https://haoming2003.oss-cn-hangzhou.aliyuncs.com/202209220224876.png)

在本地切换到主分支的某分支（比如develop）：

```bash
git checkout develop
```

在此分支的基础上创建一个自己的分支：

```bash
git checkout -b michael.w
```

开始做代码修改。

代码commit后向自己的repo push代码：

```bash
git push
```

这里可能报错，请根据报错内容自行纠正

![在这里插入图片描述](https://haoming2003.oss-cn-hangzhou.aliyuncs.com/202209220224883.png)

1. 从自己的repo中向主工程发起request pull：
   ![在这里插入图片描述](https://haoming2003.oss-cn-hangzhou.aliyuncs.com/202209220224879.png)
   选择要提交的目标分支：
   ![在这里插入图片描述](https://haoming2003.oss-cn-hangzhou.aliyuncs.com/202209220224980.png)

### 如何将主分支的更新进度同步到我的repo中

假设主工程的开发分支时main

```shell
# 切到本地的main分支
git checkout main
# 将okex的的main分支拉取下来并与本地现在所处分支合并
git pull okex main
# 推到我的repo
git push
```



---

> **本文参考了[wgy的博客](https://blog.csdn.net/michael_wgy_/article/details/104589800)，侵删**

> **由于github默认分支改变，以上master记得改为main**

---

