x_max_task_num 问题在这个修改了内容导致问题



服务器用的
处理器: Intel(R) Xeon(R) Gold 6336Y CPU @ 2.40GHz （24个核心cpu）
内存: 48GB RAM DDR4 
    程序过程 并行用到了9个进程 
    cpu占用共占用9个核心，每个核心80%
    内存每个进程占用0.3%内存，共占用1.3G内存

时间:
    分别跑不同的 T Z
    1W 迭代  
        T=7 12h
        T=14 29h
        T=21 51h

    10W 迭代 
        T=7 145h/24=6天
        T=14 330h/24=13.75天
        T=21 580h/24=24.17天

使用方法
python task_allocation_problem.py 
程序log 查看 /home/junchengshen/code/MDP/log/task_allocation_problem/process_0.log


linux 系统启动
nohup python task_allocation_problem.py > task_allocation_problem.log 2>&1 &