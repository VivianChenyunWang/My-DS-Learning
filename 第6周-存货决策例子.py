#!/usr/bin/env python
# coding: utf-8

# In[25]:


################################################################################################
#                                   最优订货量决策例子
#                                    （2021-02-27）
################################################################################################

import pandas as pd
import numpy as np

import matplotlib
import matplotlib.pyplot as plt

import math
import itertools

from datetime import datetime

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


# # 任务一：定义并检查总成本计算函数

# ## 1. 定义成本计算函数 inventory_cost_df

# In[26]:


#-----------------------------------------------------------------------------------------------#
# inventory_cost_df：给定订货策略 Q_list，成本结果函数
#-----------------------------------------------------------------------------------------------#
def inventory_cost_df(days, P_list, D_list, Q_list, S=250, C=3.2, IC=12, fast=False):
    
    ### params ###
    # 单位订货固定成本（shipment cost）：S = 240
    # 单位产品日均存货成本：C = 3
    # 单位产品日均缺货成本：IC = 10

    # 准备空列表
    day_i_list = [i for i in range(days)] # 总天数
    Stock_list = [None for i in range(days)]
    Ins_list = [None for i in range(days)]
    Cost_pq_list = [None for i in range(days)]
    Cost_ship_list = [None for i in range(days)]
    Cost_stock_list = [None for i in range(days)]
    Cost_ins_list = [None for i in range(days)]
    Cost_list = [None for i in range(days)]
    
    ### 按日处理 ###
    for day_i in range(days):
        
        # 昨日库存：
        if day_i == 0:
            l_stock = 0
        else:
            l_stock = Stock_list[day_i-1]
        # 今日价格：
        p = P_list[day_i]
        # 今日订货量：
        q = Q_list[day_i]
        # 今日需求量：
        d = D_list[day_i]
        # 更新今日库存 = 昨日库存 + 今日购进 - 今日使用量
        stock = l_stock + q - d # 今日结束时的库存（负数表示缺货）
        Stock_list[day_i] = stock # 记录
        # 今日缺货量：
        if Stock_list[day_i] < 0:
            ins = abs(Stock_list[day_i]) # 缺货量（正数）
        else:
            ins = 0
        Ins_list[day_i] = ins # 记录
        
        ### 今日成本计算 ###
        # 购进产品总价值
        cost_pq = q * p 
        # 运输成本 ，如果进货（q > 0），则付出运输成本 S，否则为 0
        if q > 0:
            cost_ship = S 
        else:
            cost_ship = 0
        # 存储成本，假设日内需求均匀发生，带来半数的等效库存成本
        cost_stock = C * (max(stock, 0) + min(d, max(q + l_stock, 0))/2) 
        # 缺货成本 = IC * 缺货量（正数）
        cost_ins = IC * ins 
        # 总成本
        cost = cost_pq + cost_ship + cost_stock + cost_ins 
        
        # 修改相应 list 中该日的数值
        Cost_pq_list[day_i] = cost_pq
        Cost_ship_list[day_i] = cost_ship
        Cost_stock_list[day_i] = cost_stock
        Cost_ins_list[day_i] = cost_ins
        Cost_list[day_i] = cost
        
    ### 存入 pandas df ###
    if fast == True:
        data = Cost_list
        df = pd.DataFrame(data=data, columns=['总成本'])
        df['累计总成本'] = df['总成本'].cumsum()
    else:
        data = zip(day_i_list, P_list, D_list, Q_list, Stock_list, Ins_list, Cost_list, Cost_pq_list, Cost_ship_list, Cost_stock_list, Cost_ins_list)
        df = pd.DataFrame(data=data, columns=['日期', '价格', '需求量', '订货量', '库存量', '缺货量', '总成本', '购进货物价值', '运输成本', '库存成本', '缺货成本'])
        df['累计需求量'] = df['需求量'].cumsum()
        df['累计订货量'] = df['订货量'].cumsum()
        df['累计总成本'] = df['总成本'].cumsum()
        df['累计运输成本'] = df['运输成本'].cumsum()
        df['累计库存成本'] = df['库存成本'].cumsum()
        df['累计缺货成本'] = df['缺货成本'].cumsum()
    
    #### return ###
    return df
days = 60
D1 = [10, 8, 6, 9, 4, 0, 3, 0, 0, 0, 5, 6, 14, 0, 1, 2, 3, 3, 6, 1, 4, 8, 15, 7, 9, 6, 9, 4, 8, 11, 3, 19, 18, 23, 21, 20, 15, 15, 17, 21, 10, 10, 18, 17, 15, 17, 12, 15, 12, 16, 18, 15, 12, 12, 10, 13, 13, 12, 10, 9]
P1 = [20, 19, 21, 20, 19, 21, 21, 19, 20, 18, 21, 21, 21, 18, 20, 20, 20, 20, 21, 17, 20, 19, 20, 22, 22, 20, 21, 18, 20, 21, 20, 20, 21, 21, 22, 20, 20, 19, 21, 22, 20, 19, 20, 18, 20, 19, 20, 20, 21, 20, 21, 19, 17, 20, 19, 20, 20, 20, 21, 22]
S = 250
C = 3.2
IC = 12


# ## 2. 用 EOQ 公式验证成本函数的正确性

# ### 2.1 根据 EOQ 公式，计算 Economic Order Quantity 的 Q_list

# In[27]:


#-----------------------------------------------------------------------------------------------#
# 用 EOQ 公式，计算 Economic Order Quantity (EOQ) 的 Q_list
#-----------------------------------------------------------------------------------------------#
### params ###
# 总天数：
days = 60
day_i_list = [i for i in range(days)]
# 日均订货量（demand）：
D = 10
# 产品单位价格：
P = 20
# 单位订货固定成本（shipment cost）：
S = 240.0
# 单位产品日均存货成本：
C = 3.0
# 单位产品日均缺货成本：
IC = 10.0
    
### Q_list_EOQ ###
# 最优每次订货量（quantity）：Q = sqrt(2 * D * S / C)
Q = (2 * D * S / C) ** 0.5
print(f'EOQ公式 最优每次订货量：{Q}')
# 最优订货次数：N = D * days / Q
N = D * days / Q
print(f'EOQ公式 最优订货次数：{N}')
# 订货周期天数：T = 1 / N * days
T = 1 / N * days
print(f'EOQ公式 订货周期天数：{T}')
# 订货计划 Q_list_EOQ
Q_list_EOQ = []
for day_i in range(days):
    # 如果到了订货周期
    if day_i % T == 0:
        Q_list_EOQ.append(Q)
    else:
        Q_list_EOQ.append(0)

print(f'EOQ公式 订货计划：{Q_list_EOQ}')   


# In[ ]:





# ### 2.2 根据 EOQ 公式，计算 Economic Order Quantity 的总成本

# In[28]:


### 用 EOQ 公式，计算 Economic Order Quantity (EOQ) 的成本

# EOQ 模型总固定订购成本（运输成本）：S * N = S * D * days / Q
print(f'EOQ公式 总运输成本：{S * D * days / Q}')
# EOQ 模型总库存成本（储藏/存放成本）：(Q / 2) * C * days
print(f'EOQ公式 总库存成本：{(Q / 2) * C * days}')
# EOQ 模型总成本：
print(f'EOQ公式 总成本：{S * D * days / Q + (Q / 2) * C * days}')


# ### 2.3 用自定义函数计算 EOQ 模型的成本，检查是否与公式结果一致

# In[29]:


### 用自定义函数计算 EOQ 模型的成本 ###
days = 60
P_list = [P for i in range(days)]
D_list = [D for i in range(days)]
# inventory_cost_df(days, P_list, D_list, Q_list_EOQ, fast=True)
df = inventory_cost_df(days, P_list, D_list, Q_list_EOQ)
# df[['累计需求量', '累计订货量', '累计总成本', '累计运输成本', '累计库存成本', '累计缺货成本']].iloc[-1]
df


# In[24]:


### 用自定义函数计算另一个 Q_list 的成本 ###

Q_list_2 = [30.0, 0, 0, 0, 30.0, 0, 0, 0, 30.0, 0, 0, 0,30.0, 0, 0, 0, 30.0, 0, 0, 0, 30.0, 0, 0, 0, 30.0, 0, 0, 0, 30.0, 0, 0, 0, 30.0, 0, 0, 0, 30.0, 0, 0, 0, 50.0, 0, 0, 0, 50.0, 0, 0, 0, 50.0, 0, 0, 0, 50.0, 0, 0, 0, 50.0, 0, 0, 0]
df = inventory_cost_df(days, P_list, D_list, Q_list_2)
df[['累计需求量', '累计订货量', '累计总成本', '累计运输成本', '累计库存成本', '累计缺货成本']].iloc[-1]


# In[ ]:





# # 任务二：优化求解总成本最低策略

# ## 2.1 用编程枚举方法，寻找 EOQ

# ### 2.1.1  n_k_pairs

# In[30]:


### n_k_pairs：分 n 批，每批订购 k 个，共订购 n * k >= quantity, 可能的整数对集合 ###
import math

def n_k_pairs(quantity, days):
    n_k_pairs = []
    for n in range(1, days+1):
        # n 批需要订购的个数 k
        k = math.ceil(quantity / n) 
        n_k_pairs.append((n, k))
    return n_k_pairs

print(n_k_pairs(600, 60))


# ### 2.1.2  nk_Q_lists

# In[31]:


### nk_Q_lists：给定天数、总量，等量、整批、等间隔订购的可选策略集合 ###


# 给定 quantity，求 (n, k), 并得到所有可行的 Q_list
def nk_Q_lists(quantity, days):
    
    nk_Q_lists = []
    
    # 遍历 n, k pairs
    for n, k in n_k_pairs(quantity, days):
        
        Q_list = []
        
        # 遍历每日
        for day_i in range(days):
            # 已达到总数则后续全为 0
            if sum(Q_list) >= quantity:
                Q_list.append(0)
                continue 
            # 要求首日订购
            if day_i == 0:
                Q_list.append(k)
                continue  
            # 如果到了订货周期，则订购（如果今天再不订购，则截至今日结束进度落后）
            if sum(Q_list)/quantity < (day_i+1)/days:
                # 本次订购 k，但不超过 quantity
                if sum(Q_list) + k > quantity:
                    Q_list.append(quantity - sum(Q_list))
                else:
                    Q_list.append(k)
            else:
                Q_list.append(0)  
                
        # add to nk_Q_lists
        if Q_list not in nk_Q_lists:
            nk_Q_lists.append(Q_list)
        
    # return
    return nk_Q_lists


# In[32]:


# 本例中 n, k 法可选 Q_list
for i in nk_Q_lists(600, 60):
    print(i)


# ### 2.1.3  solution_nk

# In[33]:


### 遍历 n, k Q_list，寻找成本最低的策略 ###

def solution_nk(days, P_list, D_list):

    # lowest_record
    lowest_record = 9999999999
    quantity = sum(D_list)
    
    #print(n_k_pairs(quantity, days))
    
    # all possible stategies
    for Q_list in nk_Q_lists(quantity, days):
        # total cost for Q_list.
        cost = inventory_cost_df(days, P_list, D_list, Q_list, fast=True)['累计总成本'].iloc[days-1]
        # update lowest_record
        if cost <= lowest_record:
            lowest_record = cost
            best_Q_list = Q_list
            best_df = inventory_cost_df(days, P_list, D_list, best_Q_list)
    # return
    return best_df


# In[34]:


### 验证EOQ ###

days = 60
P0 = [20 for i in range(days)]
D0 = [10 for i in range(days)]

solution_nk(days, P0, D0)


# ## 2.2 使用循环自定义一个策略

# ### 2.2.1 新场景：P 和 D 为变量

# In[35]:


### 生成新的 Pi 和 Di：变量 ###
def generate_P(days=60, set_seed=True):
    if set_seed == True:
        np.random.seed(1234)
    P_list = [max(round(i), 1) for i in np.random.normal(20, 1, 60)]
    return P_list

def generate_D(days=60, set_seed=True):
    if set_seed == True:
        np.random.seed(1234)
    D_list = [i for i in np.random.normal(10, 2, 60)]
    for i in range(len(D_list)):
        D_list[i] = round(max(D_list[i] + i/2 - 15.8, 0))
    D_list[0] = 8
    D_list[1] = 3
    D_list[6] = 2
    return D_list

D1 = generate_D(days=60)
P1 = generate_P(days=60)


# In[36]:


### matplotlib plot ###

plt.style.use('default')
fig, ax = plt.subplots(dpi=300)
plt.rcParams['figure.dpi'] = 300
plt.rcParams['figure.figsize'] = [10, 6]
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False  

#
ax.plot([i for i in range(days)], D1, marker='.', alpha=0.9, linewidth=0.8, color='green')
ax.set_yticks(range(0, 31, 5))
ax.set(xlabel='date', ylabel='D', title='New Parameters')
ax.grid()
ax.legend(loc='upper left', frameon=True, labels=['D'])
#
ax2 = ax.twinx()
ax2.plot([i for i in range(days)], P1, marker='.', alpha=0.9, linewidth=0.8, color='red')
ax2.set_yticks(range(15, 26, 1))
ax2.set(ylabel='P')
ax2.legend(loc='upper right', frameon=True, labels=['P'])
#
plt.xticks(range(0,61,5), fontproperties='Times New Roman', size=8)
plt.show()


# In[37]:


### 此时 n, k 法的结果（不理想）###

solution_nk(days, P1, D1)


# ### 2.2.2 自定义策略：solution_min_batchcost

# In[38]:


### solution_min_batchcost: 使某一次订货量达到单次单位成本最低  ###

def solution_min_batchcost(days, P_list, D_list):

    # 空列表
    Q_list_selected = [0 for i in range(days)]
    Stock_list = [None for i in range(days)]
    
    # 逐日处理
    for day_i in range(days):

        # 如果昨日已经购进过，不再购进
        if Q_list_selected[day_i-1] != 0:
            best_Q = 0
        else:
            # reset record
            lowest_record = 9999999999

            ### 进货目标为：使本次进货的单位成本最低 ###

            # 仍需购进的总数：remain_num, 期初库存：begin_stock
            if day_i == 0:
                remain_num = sum(D_list)
                begin_stock = 0
            else:
                remain_num = sum(D_list[day_i:])
                begin_stock = Stock_list[day_i-1]

            ### 假设当日订购 try_num 个，后续都是 0，判断单位成本（try_num: 1～200）###
            for try_num in range(1, min(remain_num + 1, 201)):
                # Planned_stock_list
                Planned_stock_list = [None for i in range(days)]
                if day_i == 0:
                    Planned_Q_list = [0 for i in range(0, days)]
                    Planned_Q_list[day_i] = try_num
                else:
                    Planned_Q_list = Q_list_selected[0:day_i] + [0 for i in range(day_i, days)]
                    Planned_Q_list[day_i] = try_num
                # cycle_end
                for cycle_end in range(day_i, days):
                    # cycle_end 这一天的日末库存
                    if day_i == 0 or cycle_end == day_i:
                        Planned_stock_list[cycle_end] = try_num - D_list[cycle_end]
                    else:
                        Planned_stock_list[cycle_end] = Planned_stock_list[cycle_end-1] - D_list[cycle_end]
                    # 如果库存 <= 0，则该周期结束，得到 cycle_end 的值
                    if Planned_stock_list[cycle_end] <= 0:
                        break

                ### 本周期的总成本 ###
                # 起始数：day_i-1 的总成本
                if day_i == 0:
                    cost_begin = 0
                else:
                    cost_begin = inventory_cost_df(days, P_list, D_list, Planned_Q_list, fast=True)['累计总成本'].iloc[day_i-1]
                # 截止数：cycle_end 的总成本
                cost_end = inventory_cost_df(days, P_list, D_list, Planned_Q_list, fast=True)['累计总成本'].iloc[cycle_end]
                # 订购 try_num 个货物的 unit_cost
                unit_cost_this_cycle = (cost_end - cost_begin) / try_num  
                # 寻找新的 lowest_record
                if unit_cost_this_cycle < lowest_record:
                    lowest_record = unit_cost_this_cycle
                    best_Q = try_num

            # 如果不进货，则需要考虑由于今日不进货所导致的缺货成本 (D_list[day_i] - begin_stock) * IC
            target_unit_cost = 19000 / sum(D_list)
            inaction_unit_cost = (D_list[day_i] - begin_stock) * IC / 1
            # unit cost 过高，则选择不进货：既高于设定成本上限，又高于本日不进货的假设成本
            if lowest_record > target_unit_cost and lowest_record > inaction_unit_cost:
                best_Q = 0

        # 本日 Q 确定完毕，记录到 Q_list_selected
        Q_list_selected[day_i] = best_Q
        # 更新实际历史库存信息
        if day_i == 0:
            Stock_list[day_i] = Q_list_selected[0] - D_list[day_i]
        else:
            Stock_list[day_i] = Stock_list[day_i-1] + Q_list_selected[day_i] - D_list[day_i]

    # cost of Q_list_selected
    best_df = inventory_cost_df(days, P_list, D_list, Q_list_selected)
    return best_df
    


# In[39]:


# solution_min_batchcost result
our_Q_list = solution_min_batchcost(days, P1, D1)['订货量']
inventory_cost_df(days, P1, D1, our_Q_list)


# In[40]:


# 刚刚的解
print([i for i in our_Q_list])


# In[ ]:





# ## 2.3 使用优化算法

# ### 2.3.1 局部优化

# In[41]:


### 使用优化算法 ###

from scipy import optimize


### 为了加速运算，重新定义总成本目标函数：inventory cost ###
def inventory_cost(days, P_list, D_list, Q_list, S=240, C=3, IC=10):
    # 准备空列表
    Stock_list = [None for i in range(days)]
    Cost_list = [None for i in range(days)]
    ### 按日处理 ###
    for day_i in range(days):
        # 昨日库存：
        if day_i == 0:
            l_stock = 0
        else:
            l_stock = Stock_list[day_i-1]
        # 今日价格：
        p = P_list[day_i]
        # 今日订货量：
        q = Q_list[day_i]
        # 今日需求量：
        d = D_list[day_i]
        # 更新今日库存 = 昨日库存 + 今日购进 - 今日使用量
        stock = l_stock + q - d # 今日结束时的库存（负数表示缺货）
        Stock_list[day_i] = stock # 记录
        # 今日缺货量：
        if Stock_list[day_i] < 0:
            ins = abs(Stock_list[day_i]) # 缺货量（正数）
        else:
            ins = 0
        ### 今日成本计算 ###
        # 购进产品总价值
        cost_pq = q * p 
        # 运输成本 ，如果进货（q > 0），则付出运输成本 S，否则为 0
        if q > 0:
            cost_ship = S 
        else:
            cost_ship = 0
        # 存储成本，假设日内需求均匀发生，带来半数的等效库存成本
        cost_stock = C * (max(stock, 0) + min(d, max(q + l_stock, 0))/2) 
        # 缺货成本 = IC * 缺货量（正数）
        cost_ins = IC * ins 
        # 总成本
        cost = cost_pq + cost_ship + cost_stock + cost_ins 
        # 修改相应 list 中该日的数值
        Cost_list[day_i] = cost
        
    return sum(Cost_list)


# In[42]:


### 原方案结果+局部优化 ###

def min_batchcost_optimized(days, P_list, D_list):

    # start with solution_min_batchcost solution
    our_Q_list = solution_min_batchcost(days, P_list, D_list)['订货量']
        
    # optimize.minimize
    func = lambda x: inventory_cost(days, P_list, D_list, x)
    # x0 = [100] * 60 # 起始点
    x0 = our_Q_list # 起始点
    lw = [0] * days
    up = [200] * days
    our_Q_list_new = [int(i) for i in optimize.minimize(func, x0, bounds=list(zip(lw, up)))['x']]

    result_df = inventory_cost_df(days, P_list, D_list, our_Q_list_new)
    
    return result_df

min_batchcost_optimized(days, P1, D1)


# In[ ]:





# ### 2.3.2 模拟退火 (Dual Annealing optimization)

# In[44]:


### 模拟退火全局优化 ###

def min_batchcost_dual_annealing(days, P_list, D_list, startwith='min_batchcost'):

    if startwith=='min_batchcost':
        # start with solution_min_batchcost solution
        our_Q_list = solution_min_batchcost(days, P_list, D_list)['订货量']
        # optimize.dual_annealing
        func = lambda x: inventory_cost(days, P_list, D_list, x)
        x0 = our_Q_list # 起始点
        lw = [0] * days
        up = [200] * days
    else:
        # optimize.dual_annealing
        func = lambda x: inventory_cost(days, P_list, D_list, x)
        x0 = [100] * days # 起始点
        lw = [0] * days
        up = [200] * days

    # 退火 1 次
    x_solution = optimize.dual_annealing(func, list(zip(lw, up)), maxiter=1000, x0=x0)['x']
    solution_Q_list = [int(i) for i in x_solution]
    best_solution = inventory_cost_df(days, P_list, D_list, solution_Q_list)

    # 退火 3 次
    best_cost = 99999999
    for i in range(3):
        print(f'now round {i}:')    
        # 优化求解
        x_solution = optimize.dual_annealing(func, list(zip(lw, up)), maxiter=1000, x0=x0)['x']
        solution_Q_list = [int(i) for i in x_solution]
         # 记录更优结果
        if inventory_cost(days, P1, D1, solution_Q_list) < best_cost:
            best_cost = inventory_cost(days, P_list, D_list, solution_Q_list)
            best_solution = inventory_cost_df(days, P_list, D_list, solution_Q_list)
            x0 = solution_Q_list
            print(f'new record: {best_cost}')
    
    return best_solution

min_batchcost_dual_annealing(days, P1, D1)


# In[ ]:





# ## 2.4 四种策略对比：哪种效果最好？

# In[21]:


#-----------------------------------------------------------------------------------------------#
# 四种策略对比：哪种效果最好？
#-----------------------------------------------------------------------------------------------#

days = 60
quantity = 600

# 随机生成 3 次 P 和 D 的分布（进行 3 次比赛）
for i in range(3):
    
    # round
    print(f'\n第 {i+1} 次比赛：')
    # rand P_list, D_list
    P_list = generate_P(days=60, set_seed=False)
    D_list = generate_D(days=60, set_seed=False)
    
    # EOQ
    start_time = datetime.now()
    EOQ_cost = inventory_cost_df(days, P_list, D_list, Q_list_EOQ)['累计总成本'].iloc[days-1]
    end_time = datetime.now()
    ms = (end_time - start_time).microseconds
    sec = (end_time - start_time).seconds
    delta_time = sec + ms / 1000000
    print(f'[0] EOQ: cost = {EOQ_cost}, 计算耗时 {delta_time:.3f} seconds.')
    
    # solution_nk
    start_time = datetime.now()
    EOQ_cost = solution_nk(days, P_list, D_list)['累计总成本'].iloc[days-1]
    end_time = datetime.now()
    ms = (end_time - start_time).microseconds
    sec = (end_time - start_time).seconds
    delta_time = sec + ms / 1000000
    print(f'[1] solution_nk: cost = {EOQ_cost}, 计算耗时 {delta_time:.3f} seconds.')
    
    # solution_min_batchcost
    start_time = datetime.now()
    solution1_cost = solution_min_batchcost(days, P_list, D_list)['累计总成本'].iloc[days-1]
    end_time = datetime.now()
    ms = (end_time - start_time).microseconds
    sec = (end_time - start_time).seconds
    delta_time = sec + ms / 1000000
    print(f'[2] solution_min_batchcost: cost = {solution1_cost}, 计算耗时 {delta_time:.3f} seconds.')
    
    # min_batchcost_optimized
    start_time = datetime.now()
    solution2_cost = min_batchcost_optimized(days, P_list, D_list)['累计总成本'].iloc[days-1]
    end_time = datetime.now()
    ms = (end_time - start_time).microseconds
    sec = (end_time - start_time).seconds
    delta_time = sec + ms / 1000000
    print(f'[3] min_batchcost_optimized: cost = {solution2_cost}, 计算耗时 {delta_time:.3f} seconds.')
    
    # min_batchcost_dual_annealing
    start_time = datetime.now()
    solution3_cost = min_batchcost_dual_annealing(days, P_list, D_list)['累计总成本'].iloc[days-1]
    end_time = datetime.now()
    ms = (end_time - start_time).microseconds
    sec = (end_time - start_time).seconds
    delta_time = sec + ms / 1000000
    print(f'[4] min_batchcost_dual_annealing = {solution3_cost}, 计算耗时 {delta_time:.3f} seconds.')



# In[ ]:




