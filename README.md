# VeloVGI WorkStation

## 流程图

![](img/pipeline.png)

## 文件说明

1. velovgi: 模型文件
2. docs: 操作文档
3. data：数据
4. notebook: 示例notebook
    - local_pc: 本地个人计算机的notebook（WSL，Ubuntu环境）
    - lab_server：实验室服务器的notebook（Ubuntu环境）
    - cluster_server：集群服务器的notebook

## 23_04_24_1 结构初始化，项目的模型velovgi代码结构调整

1. 参考模型结构
    - ScVelo: https://github.com/theislab/scvelo/tree/master/scvelo
    - Scanpy: https://github.com/scverse/scanpy/tree/master/scanpy
    - CellRank: https://github.com/theislab/cellrank
    - Dynamo: https://github.com/aristoteleo/dynamo-release/tree/master/dynamo
    - LatentVelo: https://github.com/Spencerfar/LatentVelo/tree/main/latentvelo
    - scvi-tool: https://github.com/scverse/scvi-tools/tree/main/scvi
2. 思考：
    - ScVelo的结构最清晰合理，主要参考这里的结构
    - scvi-tool都是用深度学习的，参考其中模型结构
    - LatentVelo基于深度学习，与本项目应该最类似

3. 本项目模型设计：
    - preprocessing(pp): 预处理部分
    - tools(tl): 这里存放之前写好的velovgi模型代码
    - plotting(pl): 绘图部分
    - utils：比较杂的工具部分
    - \ __init__: 管理整体包

4. 把之前的项目的模型velovgi代码结构调整，使用了`notebook/local_pc/23_04_24_02_项目结构导入_调整文件结构.ipynb`进行测试

## 23_04_24_2 文档构建与API添加

1. docs文档构建（在相同环境下构建文档，相关包从https://github.com/readthedocs-examples/example-jupyter-book/提示安装）
    - 创建GitHub项目
    - .gitignore：文件忽略构建build文件
    - 本地构建并上传
        ```{bash}
        jupyter-book build docs/
        git add .
        git commit -m "init docs"
        git push
        ```
2. APi添加前，需要手动把velvgi包添加到python解释器的搜索路径下，参考：https://blog.csdn.net/mifangdebaise/article/details/124804735。
    ```
    cd /usr/local/conda/envs/velovi-env/lib/python3.8/site-packages
    vim velovgi.pth
    /mnt/h/F_bak/Python进阶/scRNA/Other/velovgi_workstation
    ```
3. 配置文件docs/_config.yml下添加插件
    ```
    - sphinx.ext.autosummary
    - sphinx.ext.autodoc
    - sphinx.ext.napoleon # 文档格式解析
    - sphinx.ext.viewcode # 参数
    ```
4. 在md中使用rst代码写文档，参考：https://jupyterbook.org/en/stable/advanced/developers.html
    ```{eval-rst}
    .. currentmodule:: velovgi
    ```
    - .. currentmodule: 指定当前所在的模块
    ```{eval-rst}
    .. autosummary::
    :nosignatures:
    :toctree: generated/
    :recursive:
    
    pl
    
    ```
    - .. autosummary: 自动展示
    - :toctree: generated/ : 指定生成rst文件的位置
    - recursive: 递归文件夹内所有文档
    - 之后展示API列表，会根据docstring注释结果构建文档，

5. 分别在docs/api_complex.md, **docs/api_simplified.md（这里只有包的全名才能成功，别称不行）**中引入尝试
6. 提交
    ```{bash}
    git add .
    git commit -m "docs add API"
    git push
    ```




​    
