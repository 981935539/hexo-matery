---
title: CMake用法总结
top: false
cover: false
toc: true
mathjax: true
date: 2019-11-18 16:18:09
password:
summary:
tags:
- CMake
- Make
- C++
categories: C++
---

# 前言



# 一、`CMake`的作用

大家都知道, 源文件的编译步骤为:

+ 预处理: 宏定义展开, 头文件展开, 条件编译
+ 编译: 检查语法, 生成编译文件
+ 汇编: 将汇编文件生成目标文件(二进制文件)
+ 链接: 将目标文件链接成目标程序

但如果源文件太多，一个一个编译就会特别麻烦，为什么不批处理编译源文件呢，于是就有了make工具，它是一个自动化编译工具，你可以使用一条命令实现完全编译。还可以指定文件编译的顺序。但是使用make编译源码，需要编写一个规则文件，make依据它来批处理编译，这个文件就是makefile，所以编写makefile文件也是一个程序员所必备的技能。
 对于一个大工程，编写makefile实在是件复杂的事，于是人们又想，为什么不设计一个工具，读入所有源文件之后，自动生成makefile呢，于是就出现了`cmake`工具，它能够输出各种各样的makefile或者project文件,从而帮助程序员减轻负担。但是随之而来也就是编写cmakelist文件，它是cmake所依据的规则。所以在编程的世界里没有捷径可走，还是要脚踏实地的。

 原文件－－camkelist ---cmake ---makefile ---make ---生成可执行文件

# 二、`CMake基本语法规则`

1. 变量使用${}方式取值，但是在 IF 控制语句中是直接使用变量名

2. 指令(参数1  参数2  ...)

   参数使用括弧括起，参数之间使用空格或分号分开

3. 指令是大小写无关的，参数和变量是大小写相关的。推荐全部使用大写指令

4. 关于双引号的疑惑

   ```shell
   SET(SRC_LIST main.c)也可以写成 SET(SRC_LIST “main.c”)
   是没有区别的，但是假设一个源文件的文件名是 fu nc.c(文件名中间包含了空格)。这时候就必须使用双引号，如果写成了 SET(SRC_LIST fu nc.c)，就会出现错误，提示你找不到 fu 文件和 nc.c 文件。这种情况，就必须写成:SET(SRC_LIST “fu nc.c”)
   ```

   

# 三、内部构建与外部构建

内部构建就是在项目跟目录直接编译

引出了我们对外部编译的探讨，外部编译的过程如下：

1. 首先，请清除 t1 目录中除 main.c CmakeLists.txt 之外的所有中间文件，最关键的是 CMakeCache.txt。
2. 在 t1 目录中建立 build 目录，当然你也可以在任何地方建立 build 目录，不一定必须在工程目录中。
3. 进入 build 目录，运行 cmake ..(注意,..代表父目录，因为父目录存在我们需要的CMakeLists.txt，如果你在其他地方建立了 build 目录，需要运行 cmake <工程的全路径>)，查看一下 build 目录，就会发现了生成了编译需要的 Makefile 以及其他的中间文件.
4. 运行 make 构建工程，就会在当前目录(build 目录)中获得目标文件 hello。
5. 上述过程就是所谓的 out-of-source 外部编译，一个最大的好处是，对于原有的工程没有任何影响，所有动作全部发生在编译目录。通过这一点，也足以说服我们全部采用外部编译方式构建工程。
6. 这里需要特别注意的是：
   通过外部编译进行工程构建，HELLO_SOURCE_DIR 仍然指代工程路径，即/backup/cmake/t1, 而 HELLO_BINARY_DIR 则指代编译路径，即/backup/cmake/t1/build

#　四、安装库和INSTALL指令

有两种安装方式，一种是从代码编译后直接 make install 安装，一种是cmake的install 指令安装。

## 1、`make install`

```shell
DESTDIR=
install:
	mkdir -p $(DESTDIR)/usr/bin
	install -m 755 hello $(DESTDIR)/usr/bin
你可以通过:
	make install
将 hello 直接安装到/usr/bin 目录，也可以通过 make install
DESTDIR=/tmp/test 将他安装在/tmp/test/usr/bin 目录，打包时这个方式经常被使用。稍微复杂一点的是还需要定义 PREFIX，一般 autotools 工程，会运行这样的指令:
./configure –prefix=/usr 
或者./configure --prefix=/usr/local 
来指定PREFIX
比如上面的 Makefile 就可以改写成:
DESTDIR=
PREFIX=/usr
install:
	mkdir -p $(DESTDIR)/$(PREFIX)/bin
	install -m 755 hello $(DESTDIR)/$(PREFIX)/bin
```



## 2、`cmake INSTALL`指令安装

这里需要引入一个新的 cmake 指令 INSTALL 和一个非常有用的变量
CMAKE_INSTALL_PREFIX。CMAKE_INSTALL_PREFIX 变量类似于 configure 脚本的 –prefix，常见的使用方法看起来是这个样子：
	`cmake -DCMAKE_INSTALL_PREFIX=/usr ..`
INSTALL 指令用于定义安装规则，安装的内容可以包括目标二进制、动态库、静态库以及文件、目录、脚本等。

INSTALL 指令包含了各种安装类型，我们需要一个个分开解释：
目标文件的安装：

```
INSTALL(TARGETS targets...
	[[ARCHIVE|LIBRARY|RUNTIME]
	[DESTINATION <dir>]
	[PERMISSIONS permissions...]
	[CONFIGURATIONS [Debug|Release|...]]
	[COMPONENT <component>]
	[OPTIONAL]
] [...])
```



参数中的 TARGETS 后面跟的就是我们通过 ADD_EXECUTABLE 或者 ADD_LIBRARY 定义的
目标文件，可能是可执行二进制、动态库、静态库。
目标类型也就相对应的有三种，ARCHIVE 特指静态库，LIBRARY 特指动态库，RUNTIME
特指可执行目标二进制。
DESTINATION 定义了安装的路径，如果路径以/开头，那么指的是绝对路径，这时候
CMAKE_INSTALL_PREFIX 其实就无效了。如果你希望使用 CMAKE_INSTALL_PREFIX 来
定义安装路径，就要写成相对路径，即不要以/开头，那么安装后的路径就是
${CMAKE_INSTALL_PREFIX}/<DESTINATION 定义的路径>
举个简单的例子：

```shell
INSTALL(TARGETS myrun mylib mystaticlib
	RUNTIME DESTINATION bin
	LIBRARY DESTINATION lib
	ARCHIVE DESTINATION libstatic
)
```

上面的例子会将：
可执行二进制 myrun 安装到${CMAKE_INSTALL_PREFIX}/bin 目录
动态库 libmylib 安装到${CMAKE_INSTALL_PREFIX}/lib 目录
静态库 libmystaticlib 安装到${CMAKE_INSTALL_PREFIX}/libstatic 目录
特别注意的是你不需要关心 TARGETS 具体生成的路径，只需要写上 TARGETS 名称就可以
了。  

普通文件的安装：

```shell
INSTALL(FILES files... DESTINATION <dir>
	[PERMISSIONS permissions...]
	[CONFIGURATIONS [Debug|Release|...]]
	[COMPONENT <component>]
	[RENAME <name>] [OPTIONAL])
```



可用于安装一般文件，并可以指定访问权限，文件名是此指令所在路径下的相对路径。如果
默认不定义权限 PERMISSIONS，安装后的权限为：
OWNER_WRITE, OWNER_READ, GROUP_READ,和 WORLD_READ，即 644 权限。
非目标文件的可执行程序安装(比如脚本之类)：

```
INSTALL(PROGRAMS files... DESTINATION <dir>
	[PERMISSIONS permissions...]
	[CONFIGURATIONS [Debug|Release|...]]
	[COMPONENT <component>]
	[RENAME <name>] [OPTIONAL])
```

跟上面的 FILES 指令使用方法一样，唯一的不同是安装后权限为:
OWNER_EXECUTE, GROUP_EXECUTE, 和 WORLD_EXECUTE，即 755 权限
目录的安装：

```shell
INSTALL(DIRECTORY dirs... DESTINATION <dir>
	[FILE_PERMISSIONS permissions...]
	[DIRECTORY_PERMISSIONS permissions...]
	[USE_SOURCE_PERMISSIONS]
	[CONFIGURATIONS [Debug|Release|...]]
	[COMPONENT <component>]
	[[PATTERN <pattern> | REGEX <regex>]
	[EXCLUDE] [PERMISSIONS permissions...]] [...])
```


这里主要介绍其中的 DIRECTORY、PATTERN 以及 PERMISSIONS 参数。

DIRECTORY 后面连接的是所在 Source 目录的相对路径，但务必注意：abc 和 abc/有很大的区别。
如果目录名不以/结尾，那么这个目录将被安装为目标路径下的 abc，如果目录名以/结尾，代表将这个目录中的内容安装到目标路径，但不包括这个目录本身。
PATTERN 用于使用正则表达式进行过滤，PERMISSIONS 用于指定 PATTERN 过滤后的文件权限。
我们来看一个例子:

```shell
INSTALL(DIRECTORY icons scripts/ DESTINATION 	share/myproj
PATTERN "CVS" EXCLUDE
PATTERN "scripts/*"
PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ
GROUP_EXECUTE GROUP_READ)

```

这条指令的执行结果是：
将 icons 目录安装到 <prefix>/share/myproj，将 scripts/中的内容安装到<prefix>/share/myproj不包含目录名为 CVS 的目录，对于 scripts/*  文件指定权限为 OWNER_EXECUTE   OWNER_WRITE OWNER_READ GROUP_EXECUTE GROUP_READ.

安装时 CMAKE 脚本的执行：

```
INSTALL([[SCRIPT <file>] [CODE <code>]] [...])
SCRIPT 参数用于在安装时调用 cmake 脚本文件（也就是<abc>.cmake 文件）
CODE 参数用于执行 CMAKE 指令，必须以双引号括起来。比如：
INSTALL(CODE "MESSAGE(\"Sample install message.\")")
```



# 五、静态库和动态库构建

## 1、ADD_LIBRARY指令

```shell
ADD_LIBRARY(libname [SHARED|STATIC|MODULE]
	[EXCLUDE_FROM_ALL]
	source1 source2 ... sourceN)
# 不需要写全lib<libname>.so, 只需要填写<libname>,cmake系统会自动为你生成，lib<libname>.X

# 类型有三种:
	SHARED，动态库	.so
	STATIC，静态库	.a
	MODULE，在使用 dyld 的系统有效，如果不支持 dyld，则被当作 SHARED 对待。
	
#EXCLUDE_FROM_ALL 参数的意思是这个库不会被默认构建，除非有其他的组件依赖或者手工构建。
```


## 2、指定库的生成路径

​	两种方法

1. ADD_SUBDIRECTORY指令来指定一个编译输出位置
2. 在CMakeLists.txt中添加　SET(LIBRARY_OUTPUT_PATH <路径>)来指定一个新的位置


## 3、同时生成动态库和静态库

因为ADD_SUBDIRECTORY的TARGET(libname)是唯一的，所以生成动态库和静态库不能指定相同的名称，想要有相同的名称需要用到SET_TARGET_PROPERTIES指令。

SET_TARGET_PROPERTIES，其基本语法是：

```shell
SET_TARGET_PROPERTIES(target1 target2 ...
	PROPERTIES prop1 value1
	prop2 value2 ...)
# 举例
ADD_LIBRARY(hello SHARED ${LIBHELLO_SRC})　# 动态库
ADD_LIBRARY(hello_static STATIC ${LIBHELLO_SRC}) # 静态库
SET_TARGET_PROPERTIES(hello_static PROPERTIES OUTPUT_NAME "hello")
```

这条指令可以用来设置输出的名称，对于动态库，还可以用来指定动态库版本和 API 版本。

与他对应的指令是：
	GET_TARGET_PROPERTY(VAR target property)

举例

```shell
GET_TARGET_PROPERTY(OUTPUT_VALUE hello_static OUTPUT_NAME)
MESSAGE(STATUS “This is the hello_static
OUTPUT_NAME:”${OUTPUT_VALUE})
# 如果没有这个属性定义，则返回 NOTFOUND.
```

## 4、动态库版本号

```shell
SET_TARGET_PROPERTIES(hello PROPERTIES VERSION 1.2 SOVERSION 1)
# VERSION 指代动态库版本，SOVERSION 指代 API 版本。
# 在 build/lib 目录会生成：
    libhello.so.1.2
    libhello.so.1->libhello.so.1.2
    libhello.so -> libhello.so.1
```



# 六、使用共享库和头文件

## 1.`INCLUDE_DIRECTORIES`指令

`INCLUDE_DIRECTORIES([AFTER|BEFORE] [SYSTEM] dir1 dir2 ...)`
这条指令可以用来向工程添加多个特定的头文件搜索路径，路径之间用空格分割，如果路径中包含了空格，可以使用双引号将它括起来，默认的行为是追加到当前的头文件搜索路径的
后面，你可以通过两种方式来进行控制搜索路径添加的方式：
１. CMAKE_INCLUDE_DIRECTORIES_BEFORE，通过 SET 这个 cmake 变量为 on，可以将添加的头文件搜索路径放在已有路径的前面。
２. 通过 AFTER 或者 BEFORE 参数，也可以控制是追加还是置前。

## 2. `LINK_DIRECTORIES`和 `TARGET_LINK_LIBRARIES`

```shell
LINK_DIRECTORIES(directory1 directory2 ...)
# 这个指令非常简单，添加非标准的共享库搜索路径，比如，在工程内部同时存在共享库和可执行二进制，在编译时就需要指定一下这些共享库的路径。
# TARGET_LINK_LIBRARIES 的全部语法是:
TARGET_LINK_LIBRARIES(target library1
	<debug | optimized> library2
...)
# 这个指令可以用来为 target 添加需要链接的共享库
```

## 3. `FIND`系列指令

1. 特殊的环境变量` CMAKE_INCLUDE_PATH` 和`CMAKE_LIBRARY_PATH`

   务必注意，这两个是环境变量而不是 cmake 变量

2. `CMAKE_INCLUDE_PATH`和`CMAKE_LIBRARY_PATH`是配合`FIND_PATH`和`FIND_LIBRARY`指令使用的

3. find_path指令

   ```shell
   find_path (<VAR> NAMES name)
   # <VAR>查找的库文件路径报存在变量VAR中
   # 默认搜索路径为`CMAKE_INCLUDE_PATH`
   
   find_path (<VAR> NAMES name PATHS paths... [NO_DEFAULT_PATH])
   #　指定搜索路径
   # NO_DEFAULT_PATH　不使用默认搜索路径　
   # 举例
   为了将程序更智能一点，我们可以使用 CMAKE_INCLUDE_PATH 来进行，使用 bash 的方法
   如下：export CMAKE_INCLUDE_PATH=/usr/include/hello
   然后在头文件中将 INCLUDE_DIRECTORIES(/usr/include/hello)替换为：
   FIND_PATH(myHeader hello.h)
   IF(myHeader)
   	INCLUDE_DIRECTORIES(${myHeader})
   ENDIF(myHeader)
   ```

## 4. 共享库和头文件指令总结

1. **FIND_PATH** 查找头文件所在目录
2. **INCLUDE_DIRECTORIES**　添加头文件目录
3. **FIND_LIBRARY** 查找库文件所在目录
4. **LINK_DIRECTORIES**   添加库文件目录
5. **LINK_LIBRARIES**　添加需要链接的库文件路径，注意这里是全路径
6. **TARGET_LINK_LIBRARIES **　给TARGET链接库



# 七、Find模块

## 1.Find模块使用

```shell
FIND_PACKAGE(XXX)
IF(XXX_FOUND)
	INCLUDE_DIRECTORIES(${XXX_INCLUDE_DIR})
	TARGET_LINK_LIBRARIES(xxxtest ${XXX_LIBRARY})
ELSE(XXX_FOUND)
	MESSAGE(FATAL_ERROR ”XXX library not found”)
ENDIF(XXX_FOUND)
```

对于系统预定义的 Find<name>.cmake 模块，使用方法一般如上例所示：
每一个模块都会定义以下几个变量
	• <name>_FOUND
	• <name>_INCLUDE_DIR or <name>_INCLUDES
	• <name>_LIBRARY or <name>_LIBRARIES
你可以通过<name>_FOUND 来判断模块是否被找到，如果没有找到，按照工程的需要关闭某些特性、给出提醒或者中止编译





## 2.find_package指令

```shell
find_package(<PackageName> [QUIET] [REQUIRED] [[COMPONENTS] [components...]]
             [OPTIONAL_COMPONENTS components...]
             [NO_POLICY_SCOPE])
             
# 查找并从外部项目加载设置，
# <PackageName>_FOUND 将设置为指示是否找到该软件包, 如果查找到，该变量为true
# [QUIET], 设置该变量，不会打印任何消息，且		   <PackageName>_FIND_QUIETLY为true
# [REQUIRED] 设置该变量，如果找不到软件包，该选项将停止处理并显示一条错误消息，且设置<PackageName>_FIND_REQUIRED为true,不过不指定该参数，即使没有找到，也能编译通过
```

find_package采用两种模式搜索库：

-  **Module模式**：搜索**CMAKE_MODULE_PATH**指定路径下的**FindXXX.cmake**文件，执行该文件从而找到XXX库。其中，具体查找库并给**XXX_INCLUDE_DIRS**和**XXX_LIBRARIES**两个变量赋值的操作由FindXXX.cmake模块完成。
-  **Config模式**：搜索**XXX_DIR**指定路径下的**XXXConfig.cmake**文件，执行该文件从而找到XXX库。其中具体查找库并给**XXX_INCLUDE_DIRS**和**XXX_LIBRARIES**两个变量赋值的操作由XXXConfig.cmake模块完成。

两种模式看起来似乎差不多，不过cmake默认采取**Module**模式，如果Module模式未找到库，才会采取Config模式。如果**XXX_DIR**路径下找不到XXXConfig.cmake或`<lower-case-package-name>`config.cmake文件，则会找/usr/local/lib/cmake/XXX/中的XXXConfig.cmake文件。总之，Config模式是一个备选策略。通常，库安装时会拷贝一份XXXConfig.cmake到系统目录中，因此在没有显式指定搜索路径时也可以顺利找到。

总结：CMake搜索的顺序为: 首先在`CMAKE_MODULE_PATH`中搜索名为`Find<PackageName>.cmake`的文件，然后在`<PackageName>_DIR`名为`PackageName>Config.cmake`或`<lower-case-package-name>-config.cmake`的文件，如果还是找不到，则会去`/usr/local/lib/cmake`中查找`Find<PackageName>.cmake`文件。

所以我们可以通过`CMAKE_MODULE_PATH`或`<PackageName>_DIR`变量指定cmake文件路径。

## 3.自定义Find模块

```shell
# 查找HELLO的头文件目录
FIND_PATH(HELLO_INCLUDE_DIR hello.h /usr/include/hello
/usr/local/include/hello)
# 查找HELLO的动态库
FIND_LIBRARY(HELLO_LIBRARY NAMES hello PATH /usr/lib
/usr/local/lib)
IF (HELLO_INCLUDE_DIR AND HELLO_LIBRARY)
	SET(HELLO_FOUND TRUE)
ENDIF (HELLO_INCLUDE_DIR AND HELLO_LIBRARY)
IF (HELLO_FOUND)
	# 如果不指定QUIET参数，就打印信息
	IF (NOT HELLO_FIND_QUIETLY)
		MESSAGE(STATUS "Found Hello: ${HELLO_LIBRARY}")
	ENDIF (NOT HELLO_FIND_QUIETLY)
ELSE (HELLO_FOUND)
	# 如果设置了REQUIRED参数就报错
	IF (HELLO_FIND_REQUIRED)
		MESSAGE(FATAL_ERROR "Could not find hello library")
	ENDIF (HELLO_FIND_REQUIRED)
ENDIF (HELLO_FOUND)
```

# 八、`CMake`常用变量

## 1.`cmake` 变量引用的方式：

使用${}进行变量的引用。在 IF 等语句中，是直接使用变量名而不通过${}取值

## 2.`cmake` 自定义变量的方式：

主要有隐式定义和显式定义两种，前面举了一个隐式定义的例子，就是 PROJECT 指令，他会隐式的定义<projectname>_BINARY_DIR 和<projectname>_SOURCE_DIR 两个变量。
显式定义的例子我们前面也提到了，使用 SET 指令，就可以构建一个自定义变量了。比如:

SET(HELLO_SRC main.SOURCE_PATHc)，就PROJECT_BINARY_DIR 可以通过${HELLO_SRC}来引用这个自定义变量了.

## 3.`cmake` 常用变量

### 1. CMAKE_BINARY_DIR/PROJECT_BINARY_DIR/<projectname>_BINARY_DIR_

这三个变量指代的内容是一致的，如果是 in source 编译，指得就是工程顶层目录，如果是 out-of-source 编译，指的是工程编译发生的目录。PROJECT_BINARY_DIR 跟其他指令稍有区别，现在，你可以理解为他们是一致的。

### 2. CMAKE_SOURCE_DIR/PROJECT_SOURCE_DIR/<projectname>_SOURCE_DIR

这三个变量指代的内容是一致的，不论采用何种编译方式，都是工程顶层目录。

### 3. CMAKE_CURRENT_SOURCE_DIR

指的是**当前处理的** CMakeLists.txt 所在的路径

### 4. CMAKE_CURRRENT_BINARY_DIR

如果是 in-source 编译，它跟 CMAKE_CURRENT_SOURCE_DIR 一致，如果是 out-ofsource 编译，他指的是 target 编译目录。
使用我们上面提到的 ADD_SUBDIRECTORY(src bin)可以更改这个变量的值。
使用 SET(EXECUTABLE_OUTPUT_PATH <新路径>)并不会对这个变量造成影响，它仅仅修改了最终目标文件存放的路径。

### ５. CMAKE_CURRENT_LIST_FILE

​	输出调用这个变量的 CMakeLists.txt 的完整路径

### 6. CMAKE_CURRENT_LIST_LINE

​	输出这个变量所在的行

### 7. CMAKE_MODULE_PATH

这个变量用来定义自己的 cmake 模块所在的路径。如果你的工程比较复杂，有可能会自己编写一些 cmake 模块，这些 cmake 模块是随你的工程发布的，为了让 cmake 在处理CMakeLists.txt 时找到这些模块，你需要通过 SET 指令，将自己的 cmake 模块路径设
置一下。比如
SET(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
这时候你就可以通过 INCLUDE 指令来调用自己的模块了。

### 8. EXECUTABLE_OUTPUT_PATH 和 LIBRARY_OUTPUT_PATH

分别用来重新定义最终结果的存放目录，前面我们已经提到了这两个变量。

### 9. PROJECT_NAME

返回通过 PROJECT 指令定义的项目名称。

## 4. cmake 调用环境变量的方式

使用$ENV{NAME}指令就可以调用系统的环境变量了。
比如MESSAGE(STATUS “HOME dir: $ENV{HOME}”)
设置环境变量的方式是：SET(ENV{变量名} 值)

### 1. CMAKE_INCLUDE_CURRENT_DIR

自动添加 CMAKE_CURRENT_BINARY_DIR 和 CMAKE_CURRENT_SOURCE_DIR 到当前处理
的 CMakeLists.txt。相当于在每个 CMakeLists.txt 加入：
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_BINARY_DIR}
${CMAKE_CURRENT_SOURCE_DIR})

### 2. CMAKE_INCLUDE_DIRECTORIES_PROJECT_BEFORE

将工程提供的头文件目录始终至于系统头文件目录的前面，当你定义的头文件确实跟系统发生冲突时可以提供一些帮助。

### 3. CMAKE_INCLUDE_PATH 和 CMAKE_LIBRARY_PATH 我们在上一节已经提及。

## 5. 系统信息

1. CMAKE_MAJOR_VERSION，CMAKE 主版本号，比如 2.4.6 中的 2

2. CMAKE_MINOR_VERSION，CMAKE 次版本号，比如 2.4.6 中的 4

3. CMAKE_PATCH_VERSION，CMAKE 补丁等级，比如 2.4.6 中的 6

4. CMAKE_SYSTEM，系统名称，比如 Linux-2.6.22

5. CMAKE_SYSTEM_NAME，不包含版本的系统名，比如 Linux

6. CMAKE_SYSTEM_VERSION，系统版本，比如 2.6.22

7. CMAKE_SYSTEM_PROCESSOR，处理器名称，比如 i686.

8. UNIX，在所有的类 UNIX 平台为 TRUE，包括 OS X 和 cygwin

9. WIN32，在所有的 win32 平台为 TRUE，包括 cygwin

   

## 6.主要的开关选项：

1. CMAKE_ALLOW_LOOSE_LOOP_CONSTRUCTS，用来控制 IF ELSE 语句的书写方式，在
   下一节语法部分会讲到。

2. BUILD_SHARED_LIBS
   这个开关用来控制默认的库编译方式，如果不进行设置，使用 ADD_LIBRARY 并没有指定库类型的情况下，默认编译生成的库都是静态库。
   如果 SET(BUILD_SHARED_LIBS ON)后，默认生成的为动态

3. CMAKE_C_FLAGS
   设置 C 编译选项，也可以通过指令 ADD_DEFINITIONS()添加。

4. CMAKE_CXX_FLAGS
   设置 C++编译选项，也可以通过指令 ADD_DEFINITIONS()添加。

   

# 九、`CMake`常用指令

## 1. 基本指令

### MESSAGE

```shell
message([<mode>] "message to display" ...)
可选<mode>关键字确定消息的类型:
FATAL_ERROR	立即终止所有 cmake 过程
SEND_ERROR 产生错误，生成过程被跳过
WARNING
AUTHOR_WARNING
NOTICE
STATUS	输出前缀为—的信息
VERBOSE
DEBUG
TRACE
```

### PROJECT

```shell
project(<PROJECT-NAME> [<language-name>...])
project(<PROJECT-NAME>
        [VERSION <major>[.<minor>[.<patch>[.<tweak>]]]]
        [LANGUAGES <language-name>...])
        
设置项目的名称，并将其存储在变量中 PROJECT_NAME。从顶层调用时， CMakeLists.txt还将项目名称存储在变量CMAKE_PROJECT_NAME中。

同时设置变量

PROJECT_SOURCE_DIR， <PROJECT-NAME>_SOURCE_DIR
PROJECT_BINARY_DIR， <PROJECT-NAME>_BINARY_DIR

https://cmake.org/cmake/help/v3.15/command/project.html
```



### SET

```shell
将普通变量，缓存变量或环境变量设置为给定值。
指定<value>...占位符的此命令的签名期望零个或多个参数。多个参数将以分号分隔的列表形式加入，以形成要设置的实际变量值。零参数将导致未设置普通变量。unset() 命令显式取消设置变量。
1、设置正常变量
set(<variable> <value>... [PARENT_SCOPE])
<variable>在当前函数或目录范围内设置给定值。
如果PARENT_SCOPE给出了该选项，则将在当前作用域上方的作用域中设置变量。
2、设置缓存变量
set(<variable> <value>... CACHE <type> <docstring> [FORCE])
3、设置环境变量
set(ENV{<variable>} [<value>])

```



### add_executable

```shell
使用指定的源文件生成可执行文件
add_executable(<name> [WIN32] [MACOSX_BUNDLE]
               [EXCLUDE_FROM_ALL]
               [source1] [source2 ...])
<name>可执行文件名, <name>与逻辑目标名称相对应，并且在项目中必须是全局唯一的。构建的可执行文件的实际文件名是基于本机平台（例如<name>.exe或<name>）的约定构造的 。
默认情况下，将在与调用命令的源树目录相对应的构建树目录中创建可执行文件。
               
```

### add_subdirectory

```shell
在构建中添加一个子目录。
add_subdirectory(source_dir [binary_dir] [EXCLUDE_FROM_ALL])
将一个子目录添加到构建中。source_dir指定源CMakeLists.txt和代码文件所在的目录。binary_dir指定了输出文件放置的目录以及编译输出的路径。EXCLUDE_FROM_ALL 参数的含义是将这个目录从编译过程中排除，比如，工程的 example，可能就需要工程构建完成后，再进入 example 目录单独进行构建(当然，你也可以通过定义依赖来解决此类问题)。
如果没有指定binary_dir,那么编译结果(包括中间结果)都将存放在
build/source_dir 目录(这个目录跟原有的 source_dir 目录对应)，指定binary_dir 目录后，相当于在编译时将 source_dir 重命名为binary_dir，所有的中间结果和目标二进制都将存放在binary_dir 目录。
```

### subdirs

```shell
构建多个子目录
subdirs(dir1 dir2 ...[EXCLUDE_FROM_ALL exclude_dir1 exclude_dir2 ...]
        [PREORDER] )
        
            
不论是 SUBDIRS 还是 ADD_SUBDIRECTORY 指令(不论是否指定编译输出目录)，我们都可以通过 SET 指令重新定义EXECUTABLE_OUTPUT_PATH 和 LIBRARY_OUTPUT_PATH 变量
来指定最终的目标二进制的位置(指最终生成的 hello 或者最终的共享库，不包含编译生成的中间文件)
SET(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)
SET(LIBRARY_OUTPUT_PATH ${PROJECT_BINARY_DIR}/lib)
在第一节我们提到了<projectname>_BINARY_DIR 和 PROJECT_BINARY_DIR 变量，他们指的编译发生的当前目录，如果是内部编译，就相当于 PROJECT_SOURCE_DIR 也就是工程代码所在目录，如果是外部编译，指的是外部编译所在目录，也就是本例中的两个指令分别定义了：可执行二进制的输出路径为 build/bin 和库的输出路径为 build/lib.
```



### add_library

```shell
ADD_LIBRARY(libname [SHARED|STATIC|MODULE]
[EXCLUDE_FROM_ALL]
source1 source2 ... sourceN)
你不需要写全 libhello.so，只需要填写 hello 即可，cmake 系统会自动为你生成
libhello.X
类型有三种:
SHARED，动态库
STATIC，静态库
MODULE，在使用 dyld 的系统有效，如果不支持 dyld，则被当作 SHARED 对待。
EXCLUDE_FROM_ALL 参数的意思是这个库不会被默认构建，除非有其他的组件依赖或者手
工构建。
```



### include_directories

```shell
将include目录添加到构建中
include_directories([AFTER|BEFORE] [SYSTEM] dir1 [dir2 ...])
将给定目录添加到编译器用于搜索头文件的路径中。
这条指令可以用来向工程添加多个特定的头文件搜索路径，路径之间用空格分割，如果路径
中包含了空格，可以使用双引号将它括起来，默认的行为是追加到当前的头文件搜索路径的
后面，你可以通过两种方式来进行控制搜索路径添加的方式：
１，CMAKE_INCLUDE_DIRECTORIES_BEFORE，通过 SET 这个 cmake 变量为 on，可以
将添加的头文件搜索路径放在已有路径的前面。
２，通过 AFTER 或者 BEFORE 参数，也可以控制是追加还是置前。
```

### target_link_libraries & link_directories

```shell
TARGET_LINK_LIBRARIES(target library1
<debug | optimized> library2
...)
这个指令可以用来为 target 添加需要链接的共享库，本例中是一个可执行文件，但是同样
可以用于为自己编写的共享库添加共享库链接。
为了解决我们前面遇到的 HelloFunc 未定义错误，我们需要作的是向
src/CMakeLists.txt 中添加如下指令：
TARGET_LINK_LIBRARIES(main hello)
也可以写成
TARGET_LINK_LIBRARIES(main libhello.so)
```

### ADD_DEFINITIONS

```shell
向 C/C++编译器添加-D 定义，比如:
ADD_DEFINITIONS(-DENABLE_DEBUG -DABC)，参数之间用空格分割。
如果你的代码中定义了#ifdef ENABLE_DEBUG #endif，这个代码块就会生效。如果要添加其他的编译器开关，可以通过 CMAKE_C_FLAGS 变量和 CMAKE_CXX_FLAGS 变量设置。
```



### ADD_DEPENDENCIES

```shell
定义 target 依赖的其他 target，确保在编译本 target 之前，其他的 target 已经被构建。
ADD_DEPENDENCIES(target-name depend-target1
depend-target2 ...)
```



### ADD_TEST 与 ENABLE_TESTING 指令。

```shell
ENABLE_TESTING 指令用来控制 Makefile 是否构建 test 目标，涉及工程所有目录。语法很简单，没有任何参数，ENABLE_TESTING()，一般情况这个指令放在工程的主CMakeLists.txt 中.
ADD_TEST 指令的语法是:
	`ADD_TEST(testname Exename arg1 arg2 ...)`
testname 是自定义的 test 名称，Exename 可以是构建的目标文件也可以是外部脚本等等。后面连接传递给可执行文件的参数。如果没有在同一个 CMakeLists.txt 中打开
	ENABLE_TESTING()指令，任何 ADD_TEST 都是无效的。
比如我们前面的 Helloworld 例子，可以在工程主 CMakeLists.txt 中添加

ADD_TEST(mytest ${PROJECT_BINARY_DIR}/bin/main)
ENABLE_TESTING()
生成 Makefile 后，就可以运行 make test 来执行测试了。
```



### AUX_SOURCE_DIRECTORY

```shell
基本语法是：
AUX_SOURCE_DIRECTORY(dir VARIABLE)
作用是发现一个目录下所有的源代码文件并将列表存储在一个变量中，这个指令临时被用来
自动构建源文件列表。因为目前 cmake 还不能自动发现新添加的源文件。
比如
AUX_SOURCE_DIRECTORY(. SRC_LIST)
ADD_EXECUTABLE(main ${SRC_LIST})
你也可以通过后面提到的 FOREACH 指令来处理这个 LIST
```



###　CMAKE_MINIMUM_REQUIRED

```sehll
其语法为 CMAKE_MINIMUM_REQUIRED(VERSION versionNumber [FATAL_ERROR])
比如 CMAKE_MINIMUM_REQUIRED(VERSION 2.5 FATAL_ERROR)
如果 cmake 版本小与 2.5，则出现严重错误，整个过程中止。
```



### EXEC_PROGRAM

在 CMakeLists.txt 处理过程中执行命令，并不会在生成的 Makefile 中执行。具体语法为：

```shell
EXEC_PROGRAM(Executable [directory in which to run]
[ARGS <arguments to executable>]
[OUTPUT_VARIABLE <var>]
[RETURN_VALUE <var>])
```

用于在指定的目录运行某个程序，通过 ARGS 添加参数，如果要获取输出和返回值，可通过OUTPUT_VARIABLE 和 RETURN_VALUE 分别定义两个变量.
这个指令可以帮助你在 CMakeLists.txt 处理过程中支持任何命令，比如根据系统情况去修改代码文件等等。
举个简单的例子，我们要在 src 目录执行 ls 命令，并把结果和返回值存下来。
可以直接在 src/CMakeLists.txt 中添加：
EXEC_PROGRAM(ls ARGS "*.c" OUTPUT_VARIABLE LS_OUTPUT RETURN_VALUE LS_RVALUE)
IF(not LS_RVALUE)
	MESSAGE(STATUS "ls result: " ${LS_OUTPUT})
ENDIF(not LS_RVALUE)
在 cmake 生成 Makefile 的过程中，就会执行 ls 命令，如果返回 0，则说明成功执行，
那么就输出 ls *.c 的结果。关于 IF 语句，后面的控制指令会提到。

### FILE 指令

文件操作指令，基本语法为:

```shell
FILE(WRITE filename "message to write"... )
FILE(APPEND filename "message to write"... )
FILE(READ filename variable)
FILE(GLOB variable [RELATIVE path] [globbing
expressions]...)
FILE(GLOB_RECURSE variable [RELATIVE path]
[globbing expressions]...)
FILE(REMOVE [directory]...)
FILE(REMOVE_RECURSE [directory]...)
FILE(MAKE_DIRECTORY [directory]...)
FILE(RELATIVE_PATH variable directory file)
FILE(TO_CMAKE_PATH path result)
FILE(TO_NATIVE_PATH path result)
```



这里的语法都比较简单，不在展开介绍了。

### INCLUDE 指令

```shell
用来载入 CMakeLists.txt 文件，也用于载入预定义的 cmake 模块.
	INCLUDE(file1 [OPTIONAL])
	INCLUDE(module [OPTIONAL])
OPTIONAL 参数的作用是文件不存在也不会产生错误。
你可以指定载入一个文件，如果定义的是一个模块，那么将在 CMAKE_MODULE_PATH 中搜索这个模块并载入。
载入的内容将在处理到 INCLUDE 语句是直接执行。
```



## 2. 控制指令：

### 1. IF 指令

基本语法为：

```shell
IF(expression)

# THEN section.

COMMAND1(ARGS ...)
COMMAND2(ARGS ...)
...
ELSE(expression)

# ELSE section.

COMMAND1(ARGS ...)
COMMAND2(ARGS ...)
...
ENDIF(expression)
```



另外一个指令是 ELSEIF，总体把握一个原则，凡是出现 IF 的地方一定要有对应的
ENDIF.出现 ELSEIF 的地方，ENDIF 是可选的。
表达式的使用方法如下:
IF(var)，如果变量不是：空，0，N, NO, OFF, FALSE, NOTFOUND 或
<var>_NOTFOUND 时，表达式为真。
IF(NOT var )，与上述条件相反。
IF(var1 AND var2)，当两个变量都为真是为真。
IF(var1 OR var2)，当两个变量其中一个为真时为真。
IF(COMMAND cmd)，当给定的 cmd 确实是命令并可以调用是为真。
IF(EXISTS dir)或者 IF(EXISTS file)，当目录名或者文件名存在时为真。
IF(file1 IS_NEWER_THAN file2)，当 file1 比 file2 新，或者 file1/file2 其中有一个不存在时为真，文件名请使用完整路径。
IF(IS_DIRECTORY dirname)，当 dirname 是目录时，为真。
IF(variable MATCHES regex)
IF(string MATCHES regex)
当给定的变量或者字符串能够匹配正则表达式 regex 时为真。比如：
IF("hello" MATCHES "ell")
MESSAGE("true")
ENDIF("hello" MATCHES "ell")
IF(variable LESS number)
IF(string LESS number)
IF(variable GREATER number)
IF(string GREATER number)
IF(variable EQUAL number)
IF(string EQUAL number)
数字比较表达式
IF(variable STRLESS string)
IF(string STRLESS string)
IF(variable STRGREATER string)
IF(string STRGREATER string)
IF(variable STREQUAL string)
IF(string STREQUAL string)
按照字母序的排列进行比较.
IF(DEFINED variable)，如果变量被定义，为真。
一个小例子，用来判断平台差异：
IF(WIN32)
MESSAGE(STATUS “This is windows.”)
#作一些 Windows 相关的操作
ELSE(WIN32)
MESSAGE(STATUS “This is not windows”)
#作一些非 Windows 相关的操作
ENDIF(WIN32)
上述代码用来控制在不同的平台进行不同的控制，但是，阅读起来却并不是那么舒服，
ELSE(WIN32)之类的语句很容易引起歧义。
这就用到了我们在“常用变量”一节提到的 CMAKE_ALLOW_LOOSE_LOOP_CONSTRUCTS 开
关。
可以 SET(CMAKE_ALLOW_LOOSE_LOOP_CONSTRUCTS ON)
这时候就可以写成:
IF(WIN32)
ELSE()
ENDIF()
如果配合 ELSEIF 使用，可能的写法是这样:
IF(WIN32)
#do something related to WIN32
ELSEIF(UNIX)
#do something related to UNIX
ELSEIF(APPLE)
#do something related to APPLE
ENDIF(WIN32)

### 2. WHILE

WHILE 指令的语法是：

```shell
WHILE(condition)
COMMAND1(ARGS ...)
COMMAND2(ARGS ...)
...
ENDWHILE(condition)
```



其真假判断条件可以参考 IF 指令。

### 3. FOREACH

FOREACH 指令的使用方法有三种形式：

```shell
1，列表
FOREACH(loop_var arg1 arg2 ...)
COMMAND1(ARGS ...)
COMMAND2(ARGS ...)
...
ENDFOREACH(loop_var)
像我们前面使用的 AUX_SOURCE_DIRECTORY 的例子
AUX_SOURCE_DIRECTORY(. SRC_LIST)
FOREACH(F ${SRC_LIST})
MESSAGE(${F})
ENDFOREACH(F)
2，范围
FOREACH(loop_var RANGE total)
ENDFOREACH(loop_var)
从 0 到 total 以１为步进
举例如下：
FOREACH(VAR RANGE 10)
MESSAGE(${VAR})
ENDFOREACH(VAR)
最终得到的输出是：
0 1 2 3 4 5 6 7 8 9
10
３，范围和步进
FOREACH(loop_var RANGE start stop [step])
ENDFOREACH(loop_var)
从 start 开始到 stop 结束，以 step 为步进，
举例如下
FOREACH(A RANGE 5 15 3)
MESSAGE(${A})
ENDFOREACH(A)
最终得到的结果是：
5 8
11
14
这个指令需要注意的是，知道遇到 ENDFOREACH 指令，整个语句块才会得到真正的执行。
```



# 十、`CMakeLists`配置模板

## １.基本配置

```shell
cmake_minimum_required(VERSION 3.14)
project(XXX_Project)

# 设置CMAKE版本
set(CMAKE_CXX_STANDARD 14)

# 设置输出目录为 build/Debug/bin build/Debug/lib
# 并缓存路径
set(OUTPUT_DIRECTORY_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/build/${CMAKE_BUILD_TYPE})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${OUTPUT_DIRECTORY_ROOT}/bin" CACHE PATH "Runtime directory" FORCE)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${OUTPUT_DIRECTORY_ROOT}/lib" CACHE PATH "Library directory" FORCE)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${OUTPUT_DIRECTORY_ROOT}/lib" CACHE PATH "Archive directory" FORCE)

# 添加src子目录
add_subdirectory(src)
```

## ２.依赖库相关配置

**`OPenCV`依赖库**

将`OpenCV`依赖库下的`share/OpenCV`中，`OpenCVConfig.cmake`复制一份叫`FindOpenCV.cmake`，然后在根目录的CMakeLists.txt添加如下配置

```shell
#　添加make文件搜索路径
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ~/3rdparty/OpenCV-3.4.7/share/OpenCV)

# 查找cmake文件，并初始化变量
find_package(OpenCV REQUIRED)
# 添加头文件搜索路径
include_directories(${OpenCV_INCLUDE_DIRS})

# 给执行程序添加链接库
add_executable(XXXXMain main.cpp)
target_link_libraries(XXXXMain ${OpenCV_LIBS})
```



# 十一、参考

1. [http://file.ncnynl.com/ros/CMake%20Practice.pdf](http://file.ncnynl.com/ros/CMake Practice.pdf)
2. https://cmake.org/cmake/help/latest/guide/tutorial/index.html