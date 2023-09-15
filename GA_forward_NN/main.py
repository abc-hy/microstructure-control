import geatpy as ea  # import geatpy
from ga_optimization_geatpy_SEGA_NN1_25in_1296out import MyProblem
import numpy as np
## Written by Haiying Yang
## Date: August,2021

if __name__ == '__main__':
    """===============================实例化问题对象==========================="""
    problem = MyProblem()  # 生成问题对象
    """=================================种群设置=============================="""
    Encoding = 'BG'  # 编码方式
    NIND =  15 # 种群规模
    Field = ea.crtfld(Encoding, problem.varTypes, problem.ranges, problem.borders)  # 创建区域描述器
    population = ea.Population(Encoding, Field, NIND)  # 实例化种群对象（此时种群还没被初始化，仅仅是完成种群对象的实例化）
    """===============================算法参数设置============================="""
    myAlgorithm = ea.soea_SEGA_templet(problem, population)  # 实例化一个算法模板对象
    myAlgorithm.MAXGEN = 40  # 最大进化代数
    myAlgorithm.logTras = 0  # 设置每隔多少代记录日志，若设置成0则表示不记录日志
    myAlgorithm.verbose = False  # 设置是否打印输出日志信息
    myAlgorithm.drawing = 0  # 设置绘图方式（0：不绘图；1：绘制结果图；2：绘制目标空间过程动画；3：绘制决策空间过程动画）
    myAlgorithm.recOper.XOVR = 0.6 # 设置交叉概率
    myAlgorithm.mutOper.pm = 0.1 # 设置变异概率
    
    
    """==========================调用算法模板进行种群进化========================"""
    [BestIndi, population] = myAlgorithm.run()  # 执行算法模板，得到最优个体以及最后一代种群
    
    BestIndi.save()  # 把最优个体的信息保存到文件中

    f_best=myAlgorithm.trace["f_best"]
    print(f_best)
    
    

    np.savetxt('f_best.txt',f_best,fmt='%f',delimiter=' ')
    
    
    """=================================输出结果=============================="""
    print('评价次数：%s' % myAlgorithm.evalsNum)
    print('时间已过 %s 秒' % myAlgorithm.passTime)
    if BestIndi.sizes != 0:
        print('最优的目标函数值为：%s' % BestIndi.ObjV[0][0])
        print('最优的控制变量值为：')
        for i in range(BestIndi.Phen.shape[1]):
            print(BestIndi.Phen[0, i])
    else:
        print('没找到可行解。')
    # print(type(BestIndi.Phen[0, :]))
    # print(BestIndi.Phen[0, :])
    # print(type(BestIndi.ObjV[0][0]))
    # print(BestIndi.ObjV[0][0])
    
    best_nn_parameter=np.array([int(BestIndi.Phen[0, 0]),int(BestIndi.Phen[0, 1]),int(BestIndi.Phen[0, 2])])
    best_nn_parameter=[int(BestIndi.Phen[0, 0]),int(BestIndi.Phen[0, 1]),int(BestIndi.Phen[0, 2])]

    # print(type(best_nn_parameter))
    # print(best_nn_parameter)
    np.savetxt('best_nn_parameter.txt',best_nn_parameter,fmt='%f',header='Hidden_layer_number Hidden_layer_nodes Regularization_parameter epochs')
    optimized_objective_value = np.array([float(BestIndi.ObjV[0][0])])
    print(type(optimized_objective_value))
    np.savetxt('optimized_objective_value.txt',optimized_objective_value,fmt='%.6f')

    # best_gen = np.argmax(obj_trace[:, 1]) # 记录最优种群是在哪一代
    
    # best_ObjV = obj_trace[best_gen, 1]
    # print('最优的目标函数值为：%s'%(best_ObjV))
    # print('最优的决策变量值为：')
    # for i in range(var_trace.shape[1]):
    #     print(var_trace[best_gen, i])
    # print('有效进化代数：%s'%(obj_trace.shape[0]))
    # print('最优的一代是第%s 代'%(best_gen + 1))
    # print('评价次数：%s'%(myAlgorithm.evalsNum))
    # print('时间已过%s 秒'%(myAlgorithm.passTime))
