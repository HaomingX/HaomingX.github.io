---

title: github+hexo+butterfly搭建博客
date: 2022-09-21 01:06:52
tags: 博客搭建
top_img: https://haoming2003.oss-cn-hangzhou.aliyuncs.com/202209210056379.jpg
cover: https://haoming2003.oss-cn-hangzhou.aliyuncs.com/202209210056384.jpg
---

---

突然想到搭建一个博客玩，其实之前也在csdn上发过一点，但是没坚持下来，太失败了

希望这次可以坚持下来，下面记录一下搭建过程

---



### 环境准备

1. github账号

2. nodejs, npm（版本别太低）

上网搜具体的安装教程，肯定比我写得好

### 步骤

#### 创建**username.github.io**的项目

（记住**username**跟你**github**名称同名）



在合适的地方新建一个文件夹，用来存放自己的博客文件，我的放在`D:\blog`下

**在该目录下**

#### 安装**Hexo**

```bash
npm i hexo-cli -g
```

可能会有几个报错，忽略

安装完后用 **hexo -v** 验证是否安装成功

#### **初始化**并生成网页

```bash
hexo init

npm install # 安装必备组件

hexo g # 生成静态网页

hexo s # 打开本地服务器,打开http://localhost:4000/,就有效果了
```

**ctrl** + **c**关闭本地服务器



#### 连接github和本地

在根目录下

```bash
git config --global user.name "HaomingX"
git config --global user.email "978545377@qq.com"
# 根据你注册github的信息替换成你自己的
```

生成密钥SSH key

```bash
ssh-keygen -t rsa -C "978545377@qq.com"
```

打开[github](http://github.com/)，点击`settings`，再点击`SSH and GPG keys`，新建一个SSH，名字任意

```bash
cat ~/.ssh/id_rsa.pub
```

复制到ssh密匙框中，保存

输入`ssh -T git@github.com`，如果说了Hi 用户名!,你就成功了



打开博客根目录下的`_config.yml`文件，这是博客的配置文件

修改最后一行的配置：

```bash
deploy:
  type: git
  repository: https://github.com/HaomingX/HaomingX.github.io
  branch: main
```

#### 写文章

根目录下安装扩展

```bash
npm i hexo-deployer-git
```

```bash
# 创建文章
hexo new post "文章名"
```

打开`D:\blog\source\_posts`的目录，可以发现下面多了一个`.md`文件

编写完后

```bash
hexo g
hexo s

hexo d # 上传到github
```

打开你的[github.io](https://github.io/)主页就能看到发布的文章



### butterfly美化

[可以跟这个博主的教程走，写得很好](https://tzy1997.com/articles/hexo1603/)