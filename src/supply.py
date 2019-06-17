import random
import numpy as np
from gurobipy import *
import matplotlib.pyplot as plt


J_NUM = 3  # 拟计划建配送中心的个数
S_NUM = 5  # 供应商的个数
R_NUM = 10  # 零售商个数
L_NUM = 2  # 回收中心的个数

np.random.seed(1)
random.seed(32)  # 随机数种子

h = 0.01 * round(random.random() * 10)  # 单位产品单位运输距离的运输成本                           #单位产品重量(t)
m_tax = 0.04
K = 0.12  # 碳税率
gamma = 0.1  # 新产品生产过程中的碳排放因子
delta = 0.001  # 回收品再造过程中的碳排放因子
alpha = 0.0785  # 产品运输过程中的碳排放因子

W_sr = 2500 + 28 * np.random.rand(S_NUM, R_NUM) * 10
GO_rs = 250 + 3 * np.random.rand(R_NUM, S_NUM) * 10
P_r = np.array([3599, 3499, 3399, 3299, 3699, 3659, 3559, 3359, 3459, 3529])
Po_s = 410 + 3 * np.random.rand(S_NUM, 1) * 10
e_r = 150 + 3 * np.random.rand(1, R_NUM) * 10
Cn_s = 1500 + 25 * np.random.rand(S_NUM, 1) * 10
Co_s = 24 + 8 * np.random.rand(S_NUM, 1) * 10
C_j = 15000 + 678 * np.random.rand(1, J_NUM)
I_J = 25 + 2 * np.random.rand(1, J_NUM)

do_r = 300 + 8 * np.random.rand(1, R_NUM) * 80
d_r = 1000 + 7 * np.random.rand(1, R_NUM) * 100
do_r = do_r.astype(np.int32)
d_r = d_r.astype(np.int32)
eta_j = 8000 + 28 * np.random.rand(1, J_NUM)


beta_sjr = np.ones((R_NUM, S_NUM, J_NUM), dtype=np.float32)  # 生成距离张量
beta_sjr[0] = [820 + 16 * np.random.rand(1, J_NUM) * 10, 431 + 16 * np.random.rand(1, J_NUM) * 10, \
               726.2 + 16 * np.random.rand(1, J_NUM) * 10, 1170.8 + 16 * np.random.rand(1, J_NUM),
               931.8 + 16 * np.random.rand(1, J_NUM) * 10]

beta_sjr[1] = [810 + 15 * np.random.rand(1, J_NUM) * 10, 421 + 15 * np.random.rand(1, J_NUM) * 10, \
               716.2 + 15 * np.random.rand(1, J_NUM) * 10, 1160.8 + 15 * np.random.rand(1, J_NUM),
               951.8 + 15 * np.random.rand(1, J_NUM) * 10]

beta_sjr[2] = [810 + 15 * np.random.rand(1, J_NUM) * 10, 421 + 15 * np.random.rand(1, J_NUM) * 10, \
               716.2 + 15 * np.random.rand(1, J_NUM) * 10, 1160.8 + 15 * np.random.rand(1, J_NUM),
               951.8 + 15 * np.random.rand(1, J_NUM) * 10]

beta_sjr[3] = [810 + 15 * np.random.rand(1, J_NUM) * 10, 421 + 15 * np.random.rand(1, J_NUM) * 10, \
               716.2 + 15 * np.random.rand(1, J_NUM) * 10, 1160.8 + 15 * np.random.rand(1, J_NUM),
               951.8 + 15 * np.random.rand(1, J_NUM) * 10]

beta_sjr[4] = [810 + 15 * np.random.rand(1, J_NUM) * 10, 421 + 15 * np.random.rand(1, J_NUM) * 10, \
               716.2 + 15 * np.random.rand(1, J_NUM) * 10, 1160.8 + 15 * np.random.rand(1, J_NUM),
               951.8 + 15 * np.random.rand(1, J_NUM) * 10]

beta_sjr[5] = [810 + 15 * np.random.rand(1, J_NUM) * 10, 421 + 15 * np.random.rand(1, J_NUM) * 10, \
               716.2 + 15 * np.random.rand(1, J_NUM) * 10, 1160.8 + 15 * np.random.rand(1, J_NUM),
               951.8 + 15 * np.random.rand(1, J_NUM) * 10]

beta_sjr[6] = [810 + 15 * np.random.rand(1, J_NUM) * 10, 421 + 15 * np.random.rand(1, J_NUM) * 10, \
               716.2 + 15 * np.random.rand(1, J_NUM) * 10, 1160.8 + 15 * np.random.rand(1, J_NUM),
               951.8 + 15 * np.random.rand(1, J_NUM) * 10]

beta_sjr[7] = [810 + 15 * np.random.rand(1, J_NUM) * 10, 421 + 15 * np.random.rand(1, J_NUM) * 10, \
               716.2 + 15 * np.random.rand(1, J_NUM) * 10, 1160.8 + 15 * np.random.rand(1, J_NUM),
               951.8 + 15 * np.random.rand(1, J_NUM) * 10]

beta_sjr[8] = [810 + 15 * np.random.rand(1, J_NUM) * 10, 421 + 15 * np.random.rand(1, J_NUM) * 10, \
               716.2 + 15 * np.random.rand(1, J_NUM) * 10, 1160.8 + 15 * np.random.rand(1, J_NUM),
               951.8 + 15 * np.random.rand(1, J_NUM) * 10]

beta_sjr[9] = [810 + 15 * np.random.rand(1, J_NUM) * 10, 421 + 15 * np.random.rand(1, J_NUM) * 10, \
               716.2 + 15 * np.random.rand(1, J_NUM) * 10, 1160.8 + 15 * np.random.rand(1, J_NUM),
               951.8 + 15 * np.random.rand(1, J_NUM) * 10]

beta_rls = np.ones((S_NUM, R_NUM, L_NUM), dtype=np.float32)  # 生成距离张量
beta_rls[0] = [820 + 16 * np.random.rand(1, L_NUM) * 10, 431 + 16 * np.random.rand(1, L_NUM) * 10, \
               726.2 + 16 * np.random.rand(1, L_NUM) * 10, 1170.8 + 16 * np.random.rand(1, L_NUM),
               931.8 + 16 * np.random.rand(1, L_NUM) * 10, 820 + 16 * np.random.rand(1, L_NUM) * 10,
               431 + 16 * np.random.rand(1, L_NUM) * 10, \
               726.2 + 16 * np.random.rand(1, L_NUM) * 10, 1170.8 + 16 * np.random.rand(1, L_NUM),
               931.8 + 16 * np.random.rand(1, L_NUM) * 10]

beta_rls[1] = [810 + 15 * np.random.rand(1, L_NUM) * 10, 421 + 15 * np.random.rand(1, L_NUM) * 10, \
               716.2 + 15 * np.random.rand(1, L_NUM) * 10, 1160.8 + 15 * np.random.rand(1, L_NUM),
               951.8 + 15 * np.random.rand(1, L_NUM) * 10, 820 + 16 * np.random.rand(1, L_NUM) * 10,
               431 + 16 * np.random.rand(1, L_NUM) * 10, \
               726.2 + 16 * np.random.rand(1, L_NUM) * 10, 1170.8 + 16 * np.random.rand(1, L_NUM),
               931.8 + 16 * np.random.rand(1, L_NUM) * 10]

beta_rls[2] = [810 + 15 * np.random.rand(1, L_NUM) * 10, 421 + 15 * np.random.rand(1, L_NUM) * 10, \
               716.2 + 15 * np.random.rand(1, L_NUM) * 10, 1160.8 + 15 * np.random.rand(1, L_NUM),
               951.8 + 15 * np.random.rand(1, L_NUM) * 10, 820 + 16 * np.random.rand(1, L_NUM) * 10,
               431 + 16 * np.random.rand(1, L_NUM) * 10, \
               726.2 + 16 * np.random.rand(1, L_NUM) * 10, 1170.8 + 16 * np.random.rand(1, L_NUM),
               931.8 + 16 * np.random.rand(1, L_NUM) * 10]

beta_rls[3] = [810 + 15 * np.random.rand(1, L_NUM) * 10, 421 + 15 * np.random.rand(1, L_NUM) * 10, \
               716.2 + 15 * np.random.rand(1, L_NUM) * 10, 1160.8 + 15 * np.random.rand(1, L_NUM),
               951.8 + 15 * np.random.rand(1, L_NUM) * 10, 820 + 16 * np.random.rand(1, L_NUM) * 10,
               431 + 16 * np.random.rand(1, L_NUM) * 10, \
               726.2 + 16 * np.random.rand(1, L_NUM) * 10, 1170.8 + 16 * np.random.rand(1, L_NUM),
               931.8 + 16 * np.random.rand(1, L_NUM) * 10]

beta_rls[4] = [810 + 15 * np.random.rand(1, L_NUM) * 10, 421 + 15 * np.random.rand(1, L_NUM) * 10, \
               716.2 + 15 * np.random.rand(1, L_NUM) * 10, 1160.8 + 15 * np.random.rand(1, L_NUM),
               951.8 + 15 * np.random.rand(1, L_NUM) * 10, 820 + 16 * np.random.rand(1, L_NUM) * 10,
               431 + 16 * np.random.rand(1, L_NUM) * 10, \
               726.2 + 16 * np.random.rand(1, L_NUM) * 10, 1170.8 + 16 * np.random.rand(1, L_NUM),
               931.8 + 16 * np.random.rand(1, L_NUM) * 10]


dist_sjr = {}
for i in range(R_NUM):
    for j in range(S_NUM):
        for k in range(J_NUM):
            arc = ('forward_{}'.format(i), 'forward_{}'.format(j), 'forward_{}'.format(k))
            dist_sjr[arc] = beta_sjr[i, j, k]


dist_rls = {}
for i in range(S_NUM):
    for j in range(R_NUM):
        for k in range(L_NUM):
            arc = ('back_{}'.format(i), 'back_{}'.format(j), 'back_{}'.format(k))
            dist_rls[arc] = beta_rls[i, j, k]


back_arc, back_dis = multidict(dist_rls)
forward_arc, forward_dis = multidict(dist_sjr)


def set_model1(Multi = False,P_epson=12222):

   model1 = Model('supply2')
   model1.Params.LogToConsole = 0
   forward = model1.addVars(forward_arc, vtype=GRB.INTEGER, name="forward")
   back = model1.addVars(back_arc, vtype=GRB.INTEGER, name="back")

   '''
   forward = m.addVars(forward_arc,vtype=GRB.CONTINUOUS,lb=0,name="forward")
   back = m.addVars(back_arc,vtype=GRB.CONTINUOUS,lb=0,name="back")
   '''
   Y_number = [i for i in range(J_NUM)]
   Y_j = model1.addVars(Y_number, vtype=GRB.BINARY)

   # 目标 1： 成本最小
   SC = sum(forward.sum('*','forward_{}'.format(i),'*') * Cn_s[i,0] for i in range(S_NUM)) + \
    K * gamma * forward.sum('*','*','*') + \
    sum(back.sum('back_{}'.format(i),'back_{}'.format(j),'*') * GO_rs[j,i] for j in range(R_NUM) for i  in range(S_NUM)) + \
    sum(back.sum('back_{}'.format(i),'*','*') * Co_s[i,0] for i in range(S_NUM))

   JC = (h + K *m_tax*alpha) * (quicksum(forward[arc] * forward_dis[arc] for arc in forward_arc) + 
        quicksum(back[arc] * back_dis[arc] for arc in back_arc))+ \
       quicksum(Y_j[i] * C_j[0,i] for i in range(J_NUM)) +\
       quicksum(K * Y_j[i] * I_J[0,i] for i in range(J_NUM))
   RC = quicksum(forward.sum('forward_{}'.format(i),'forward_{}'.format(j),'*') * W_sr[j,i] for i in range(R_NUM) for j in range(S_NUM)) + \
       quicksum(back.sum('*','back_{}'.format(j),'*') * e_r [0,j] for j in range(R_NUM))
   model1.setObjective(SC + JC + RC, GRB.MINIMIZE)


   Cons1 = model1.addConstrs(forward.sum('forward_{}'.format(i), '*', '*') == d_r[0, i] for i in range(R_NUM))

   ## 约束2 零售商经回收中心到供应商的商品总和应该等于零售商回收到的商品之和
   Cons2 = model1.addConstrs(back.sum('*', 'back_{}'.format(i), '*') == do_r[0, i] for i in range(R_NUM))

   ## 约束3 中转站产品数量约束
   Cons3 = model1.addConstrs(forward.sum('*', '*', 'forward_{}'.format(i)) <= eta_j[0, i] * Y_j[i] for i in range(J_NUM))

   ## 约束4 零售商收到的旧产品理论上要小于实际卖出总产品
   Cons4 = model1.addConstr(forward.sum('*', '*', '*') >= sum(d_r[0]))


   if Multi:
      SP = quicksum(
       forward.sum('forward_{}'.format(i), 'forward_{}'.format(j), '*') * W_sr[j, i] for i in range(R_NUM) for j in
       range(S_NUM)) + quicksum(back.sum('back_{}'.format(i),'*', '*') * Po_s[i, 0] for i in range(S_NUM))

      RP = quicksum(
       forward.sum('forward_{}'.format(j), '*', '*') * P_r[j] for j in range(R_NUM)) + \
        quicksum(back.sum('back_{}'.format(i),'back_{}'.format(j), '*') * GO_rs[j, i] for j in range(R_NUM) for i in range(S_NUM))

      Cons5 = model1.addConstr(SP+RP >= P_epson)



   model1.optimize()
   Current_Cost = model1.ObjVal
   Current_Profit = get_objprofit(model1,forward,back,Y_j)
   
   '''
   Current_Profit = get_objcost(m)
   print(Current_Cost,Current_Profit)
   assert(abs(Current_Profit - Current_Cost) <= 1)
   '''
   return Current_Cost , Current_Profit


def set_model2(Multi = False,C_epson=12222):

   model2 = Model('supply2')
   model2.Params.LogToConsole = 0
   forward = model2.addVars(forward_arc, vtype=GRB.INTEGER, name="forward")
   back = model2.addVars(back_arc, vtype=GRB.INTEGER, name="back")

   '''
   forward = m.addVars(forward_arc,vtype=GRB.CONTINUOUS,lb=0,name="forward")
   back = m.addVars(back_arc,vtype=GRB.CONTINUOUS,lb=0,name="back")
   '''
   Y_number = [i for i in range(J_NUM)]
   Y_j = model2.addVars(Y_number, vtype=GRB.BINARY)

   SP = quicksum(
       forward.sum('forward_{}'.format(i), 'forward_{}'.format(j), '*') * W_sr[j, i] for i in range(R_NUM) for j in
       range(S_NUM)) + quicksum(back.sum('back_{}'.format(i),'*', '*') * Po_s[i, 0] for i in range(S_NUM))

   RP = quicksum(
       forward.sum('forward_{}'.format(j), '*', '*') * P_r[j] for j in range(R_NUM)) + \
        quicksum(back.sum('back_{}'.format(i),'back_{}'.format(j), '*') * GO_rs[j, i] for j in range(R_NUM) for i in range(S_NUM))

   model2.setObjective(RP + SP, GRB.MAXIMIZE)
   ## 约束1 产品数量与库存数量之和要满足消费者需求

   Cons1 = model2.addConstrs(forward.sum('forward_{}'.format(i), '*', '*') == d_r[0, i] for i in range(R_NUM))

   ## 约束2 零售商经回收中心到供应商的商品总和应该等于零售商回收到的商品之和
   Cons2 = model2.addConstrs(back.sum('*', 'back_{}'.format(i), '*') == do_r[0, i] for i in range(R_NUM))

   ## 约束3 中转站产品数量约束
   Cons3 = model2.addConstrs(forward.sum('*', '*', 'forward_{}'.format(i)) <= eta_j[0, i] * Y_j[i] for i in range(J_NUM))

   ## 约束4 零售商收到的旧产品理论上要小于实际卖出总产品
   Cons4 = model2.addConstr(forward.sum('*', '*', '*') >= sum(d_r[0]))

   if Multi:
      SC = sum(forward.sum('*','forward_{}'.format(i),'*') * Cn_s[i,0] for i in range(S_NUM)) + \
       K * gamma * forward.sum('*','*','*') + \
       sum(back.sum('back_{}'.format(i),'back_{}'.format(j),'*') * GO_rs[j,i] for j in range(R_NUM) for i  in range(S_NUM)) + \
       sum(back.sum('back_{}'.format(i),'*','*') * Co_s[i,0] for i in range(S_NUM))

      JC = (h + K *m_tax*alpha) * (quicksum(forward[arc] * forward_dis[arc] for arc in forward_arc) + 
           quicksum(back[arc] * back_dis[arc] for arc in back_arc))+ \
          quicksum(Y_j[i] * C_j[0,i] for i in range(J_NUM)) +\
          quicksum(K * Y_j[i] * I_J[0,i] for i in range(J_NUM))
      RC = quicksum(forward.sum('forward_{}'.format(i),'forward_{}'.format(j),'*') * W_sr[j,i] for i in range(R_NUM) for j in range(S_NUM)) + \
          quicksum(back.sum('*','back_{}'.format(j),'*') * e_r [0,j] for j in range(R_NUM))

      Cons5 = model2.addConstr(SC + JC + RC <= C_epson)


   model2.optimize()
   Current_Profit = model2.ObjVal

   Current_Cost = get_objcost(model2,forward,back,Y_j)

   '''
   Current_Cost = get_objprofit(m)
   assert (abs(Current_Profit - Current_Cost) < 1)

   '''

   return Current_Profit , Current_Cost




def get_objcost(model,forward,back,Y_j):

   solution_forward = model.getAttr('x', forward)
   solution_back =  model.getAttr('x', back)
   solution_yj = model.getAttr('x', Y_j)

   SC1_value = []
   for j in range(S_NUM):
      value = []
      for i in range(R_NUM):
         for k in range(J_NUM):
            value_ij = solution_forward['forward_{}'.format(i), 'forward_{}'.format(j),'forward_{}'.format(k)]
            value.append(value_ij) 
      total_value = sum(value) * Cn_s[j,0] 
      SC1_value.append(total_value)
   SC1_total = sum(SC1_value)

   SC2_value = []
   value = [] 
   for i in range(R_NUM):
      for j in range(S_NUM):
         for k in range(J_NUM):
            value_ij = solution_forward['forward_{}'.format(i), 'forward_{}'.format(j),'forward_{}'.format(k)]
            value.append(value_ij)
   total_value = sum(value) * K * gamma 
   SC2_value.append(total_value)
   SC2_total = sum(SC2_value) 

   SC3_value = []
   for i in range(S_NUM):
      for j in range(R_NUM):
         value = [] 
         for k in range(L_NUM):
            value_ij = solution_back['back_{}'.format(i), 'back_{}'.format(j),'back_{}'.format(k)]
            value.append(value_ij)
         total_value = sum(value) * GO_rs[j, i]
         SC3_value.append(total_value)
   SC3_total = sum(SC3_value)


   SC4_value = []
   for i in range(S_NUM):
      value = []
      for j in range(R_NUM):
         for k in range(L_NUM):
            value_ij = solution_back['back_{}'.format(i), 'back_{}'.format(j),'back_{}'.format(k)]
            value.append(value_ij) 
      total_value = sum(value) * Co_s[i,0]
      SC4_value.append(total_value)
   SC4_total = sum(SC4_value)


   JC1_total = (h + K *m_tax*alpha) * (sum([solution_forward[arc] * forward_dis[arc] for arc in forward_arc]) + 
        sum([solution_back[arc] * back_dis[arc] for arc in back_arc]))
   JC2_total = sum([solution_yj[i] * C_j[0,i] for i in range(J_NUM)])+\
       sum(K * solution_yj[i] * I_J[0,i] for i in range(J_NUM))

   RC1_value = []
   for i in range(R_NUM):
      for j in range(S_NUM):
         value = [] 
         for k in range(J_NUM):
            value_ij = solution_forward['forward_{}'.format(i), 'forward_{}'.format(j),'forward_{}'.format(k)]
            value.append(value_ij)
         total_value = sum(value) * W_sr[j, i]
         RC1_value.append(total_value)
   RC1_total = sum(RC1_value)

   RC2_value = []
   for j in range(R_NUM):
      value = []
      for i in range(S_NUM):
         for k in range(L_NUM):
            value_ij = solution_back['back_{}'.format(i), 'back_{}'.format(j),'back_{}'.format(k)]
            value.append(value_ij) 
      total_value = sum(value) * e_r [0,j]
      RC2_value.append(total_value)
   RC2_total = sum(RC2_value)

   Total_cost = SC1_total + SC2_total + SC3_total + SC4_total + JC1_total + JC2_total + RC1_total + RC2_total

   return Total_cost


def get_objprofit(model,forward,back,Y_j):

   solution_forward = model.getAttr('x', forward)
   solution_back =  model.getAttr('x', back)
   SP1_value = []
   for i in range(R_NUM):
      for j in range(S_NUM):
         value = [] 
         for k in range(J_NUM):
            value_ij = solution_forward['forward_{}'.format(i), 'forward_{}'.format(j),'forward_{}'.format(k)]
            value.append(value_ij)
         total_value = sum(value) * W_sr[j, i]
         SP1_value.append(total_value)
   SP1_total = sum(SP1_value)

   SP2_value = []
   for i in range(S_NUM):
      value = []
      for j in range(R_NUM):
         for k in range(L_NUM):
            value_ij = solution_back['back_{}'.format(i), 'back_{}'.format(j),'back_{}'.format(k)]
            value.append(value_ij) 
      total_value = sum(value) * Po_s[i, 0]
      SP2_value.append(total_value)
   SP2_total = sum(SP2_value)

   RP1_value = []
   for i in range(R_NUM):
      value = []
      for j in range(S_NUM):
         for k in range(J_NUM):
            value_ij = solution_forward['forward_{}'.format(i), 'forward_{}'.format(j),'forward_{}'.format(k)]
            value.append(value_ij) 
      total_value = sum(value) * P_r[i]
      RP1_value.append(total_value)
   RP1_total = sum(RP1_value)

   RP2_value = []
   for i in range(S_NUM):
      for j in range(R_NUM):
         value = [] 
         for k in range(L_NUM):
            value_ij = solution_back['back_{}'.format(i), 'back_{}'.format(j),'back_{}'.format(k)]
            value.append(value_ij)
         total_value = sum(value) * GO_rs[j, i]
         RP2_value.append(total_value)
   RP2_total = sum(RP2_value)

   Total_profit = SP1_total + SP2_total + RP1_total + RP2_total 
   return Total_profit



def multi_iteration(cost1, profit1,profit2,cost2):

   ##设定C_epson 和 P_epson 的范围
   C_min = cost1
   C_p = cost2
   P_max = profit2
   P_c = profit1

   ##设定最大迭代次数
   iterations  = 200

   ##确定步长
   C_Steplenth = (C_p - C_min) / iterations
   P_Steplenth = (P_max - P_c) / iterations 

   step_plot = []
   ## 记录上一步的值，以便判断终止条件
   LAST_cost1 = cost1 
   LAST_profit1 = profit1
   LAST_profit2 = profit2 
   LAST_cost2 = cost2
   #终止条件表达式
   jude = lambda x,y,z,w: (x-y) / (z-w)
   #绘图中间值
   cost1_plot = []
   profit1_plot = []
   cost2_plot = []
   profit2_plot = []

   ##设定两个画布
   ax1 = plt.subplot(1,2,1)
   ax2 = plt.subplot(1,2,2)

   #交互式绘图
   plt.ion()
   ##迭代开始
   for step in range(1,iterations + 1):
      C_epson = C_min + step * C_Steplenth
      P_epson = P_max - step * P_Steplenth

      cost1, profit1 = set_model1(Multi=True,P_epson=P_epson) # Mc model
      profit2,cost2 = set_model2(Multi=True,C_epson=C_epson)  # Mp model

      judgeMp =  jude(cost2,LAST_cost2,profit2,LAST_profit2)
      judgeMc =  jude(LAST_profit1,profit1,LAST_cost1,cost1)

      print("********step%d**********"%step)
      print(LAST_profit2,'***MP***OPT_Value***',profit2,'Cepson',P_epson) 
      print(LAST_cost2,'****MP***PROFIT***',cost2,'Pepson',C_epson) 
      print(LAST_cost1,'***MC***OPT_Value***',cost1) 
      print(LAST_profit1,'****MP***COST***',profit1) 
      print(judgeMp,'***MC***judge***') 
      print(judgeMc,'****MP***judge***\n')  

      LAST_cost1 = cost1 
      LAST_profit1 = profit1
      LAST_profit2 = profit2 
      LAST_cost2 = cost2 

      cost1_plot.append(cost1)
      profit1_plot.append(profit1)
      cost2_plot.append(cost2)
      profit2_plot.append(profit2)
      step_plot.append(step)

      ax1.plot(step_plot,cost2_plot,color='r')
      ax1.plot(step_plot,profit2_plot,color='g')
      ax2.plot(step_plot,cost1_plot,color='r')
      ax2.plot(step_plot,profit1_plot,color='g')  
      plt.show()   
      plt.pause(0.1)
   plt.ioff()


if __name__ == '__main__':

   cost1, profit1 = set_model1()
   profit2,cost2 = set_model2()
   ##得到初始解
   print(cost1,cost2)
   print(profit1,profit2)
   multi_iteration(cost1, profit1, profit2, cost2)
















