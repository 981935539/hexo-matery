---
title: Hexo+Github博客搭建
top: false
cover: false
toc: true
mathjax: true
date: 2019-11-15 03:02:58
password:
summary:
tags: 
- Github
- Hexo
- node.js
categories: 随笔
---

<iframe frameborder="no" border="0" marginwidth="0" marginheight="0" width=330 height=86 src="//music.163.com/outchain/player?type=2&id=4913023&auto=1&height=66"></iframe>

# 前言

​	**准备工作**

+ Github账号
+ node.js、hexo、npm安装



# 一、安装node.js

1. 下载windows版node.js

   下载地址: https://nodejs.org/en/download/

   选择Windows Installer(.msi) 64-bit

2. 双击node-v12.13.0-x64.msi, 一直next安装完成

3. 测试是否安装成功

   win+R键，输入cmd,然后回车，打开cmd窗口

   输入node -v 	显示node.js版本

   输入npm -v 	显示npm版本

   安装完成

   

# 二、安装hexo

1. 先创建hexo的安装目录, 例如:  F:\LearnSpace\Blog

2. cd Blob  进入Blob目录

3. npm install hexo-cli -g    安装hexo

4. hexo -v  验证是否安装成功

5. npm init blog    初始化blog文件夹，存放博客

6. npm install 安装必备组件

7. cd blog

8. hexo g    生成静态网页

9. hexo s     打开本地服务器

10. http://localhost:4000/    打开网页

11. ctrl + c   关闭本地服务器

    

# 三、连接Github与本地

1. 新建一个名为`你的github用户名.github.io`的仓库，比如说，如果你的`Github`用户名是test，那么你就新建`test.github.io`的仓库（必须是你的用户名，其它名称无效），将来你的网站访问地址就是` http://test.github.io` 了。

   点击`Settings`，向下拉到最后有个`GitHub Pages`，点击`Choose a theme`选择一个主题。然后等一会儿，再回到`GitHub Pages`, 就会像下面一样

   ![](2.png)

2. 修改配置文件

   编辑blog根目录下的`_config.yml`, 修改最后一行的配置

```
deploy:
  type: git
  repository: https://github.com/981935539/981935539.github.io.git
  branch: master
```

3. 安装Git部署插件: `npm install hexo-deployer-git --save`

# 四、编辑第一篇博客

``` 
hexo new post "first-article"  # 创建第一篇博客
hexo g  # 生成静态网页
hexo s  # 本地预览效果
hexo d  # 上传github
```

此时可以在github.io主页就能看到发布的文章啦。

# 五、绑定域名

1. 以阿里云为例，如下图所示，添加两条解析记录:

​	![](1.png)

2. 然后打开你的Github博客项目，点击`settings`，拉到下面`Custom domain`处，填上你自己的域名，保存

3. 这时候你的`F:\LearnSpace\Blog\blob\source` 会出现一个CNAME的文件

4. 如果没有CNAME文件

   打开你本地博客`/source`目录，我的是`F:\LearnSpace\Blog\blob\source`，新建`CNAME`文件，注意没有后缀。然后在里面写上你的域名，保存。最后运行`hexo g`、`hexo d`上传到Github。

   

# 六、hexo常用命令

```
npm install hexo-cli -g  	# 安装hexo
npm uninstall hexo-cli -g  	# 卸载hexo

hexo generate #生成静态页面至public目录
hexo server #开启预览访问端口（默认端口4000，'ctrl + c'关闭server）
hexo deploy #部署到GitHub
hexo help  # 查看帮助
hexo version  #查看Hexo的版本

# 缩写
hexo n == hexo new
hexo g == hexo generate
hexo s == hexo server
hexo d == hexo deploy

# 组合
hexo s -g #生成并本地预览
hexo d -g #生成并上传
```



# 七、写博客的规范

1. _config.yml

   冒号后面必须有一个空格，否则会出问题

2. 图片

   引用图片需要把图片放在对应的文件夹中，只需要写文件名就可以了

3. 文章头设置

   模板在/scaffolds/post.md

   ```
   --- 
   title: {{ title }} # 文章名称
   date: {{ date }} # 文章生成时间
   top: false 
   cover: false 
   password: 
   toc: true 
   mathjax: true 
   summary: 
   tags:
   -- [tag1]
   -- [tag2]
   -- [tag3]
   categories: 
   -- [cat1]
   ---
   ```

   

# 八、备份博客源文件

​	博客已经搭建完成，但是博客仓库只是保存生成的静态网页文件，是没有博客源文件的，如果电脑出现了问题，那就麻烦了，所以源文件也需要备份一下。

1. 在`Github`上创建一个与本地仓库同名的仓库, 我的是`hexo-matery`

2. 初始化本地仓库

   ```shell
   git init       
   添加.gitignore文件
   .gitignore
       .DS_Store
       Thumbs.db
       *.log
       public/
       .deploy*/
       .vscode/
   ```

   

3. 连接到远程`Github`,

   ```shell
   git remote add github git@github.com:981935539/hexo-matery.git
   git fetch
   git merge --allow-unrelated-histories github/master
   ```

4. 推送本地源文件到`Github`

   ```shell
   git add .
   git commit -m "第一次备份本地仓库"
   git push --set-upstream github master
   ```

   

5. 现在在任何一台电脑上, 执行`git clonegit@github.com:981935539/hexo-matery.git`

   就可以把博客源文件复制到本地。

   

# 九、Ubuntu安装node.js和hexo

```shell
tar -xvf node-v12.13.0-linux-x64.tar.xz
sudo mv node-v12.13.0-linux-x64 /usr/local
sudo ln -s /usr/local/node-v12.13.0-linux-x64/bin/node /usr/local/bin/node
sudo ln -s /usr/local/node-v12.13.0-linux-x64/bin/npm /usr/local/bin/npm

sudo npm install -g hexo
sudo ln -s /usr/local/node-v12.13.0-linux-x64/bin/hexo /usr/local/bin/hexo
```



# 十、参考

​	https://godweiyang.com/2018/04/13/hexo-blog/#toc-heading-9

​	https://www.cnblogs.com/liuxianan/p/build-blog-website-by-hexo-github.html