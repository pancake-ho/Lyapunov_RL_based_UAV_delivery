


'''
setting
1. field node 생성
 - 각 quality level(1,2,3)마다 lamda의 PPP로 생성임 -> total 비율이 전체 면적의 0.4인거
 - field의 면적은 세로 2, 가로 20
 
 


step
1. MK를 받고 Bk랑 비교해서 진행




'''


import numpy as np
import matplotlib.pyplot as plt
import cmath
import math
import random
np.random.seed(6)


#directory = 'python3.8.5/DTRL/result_plot3/'
directory = './DTRL_downgrade/result_plot/'




# environment로서 동작하는 CLASS
class Env():
    
    
  def __init__(self):
        
        self.max_episode = 1000
        self.intensity_lambda=3 #최초 0.4로 썼음
        self.V =40   # 최초 0.015로 썼음
        self.agent_T = 5 #  agent의 large time scale T
        #self.user_mobility = 2/self.agent_T #기존 mobility 2였음 high mobility일때는 2, low mobility일 때는 0.1사용했음
        self.user_mobility = 0.04
        # setting에 agent_T * mobility를 추가해서 기존 mobility=2처럼 그림 그리려면 mobility=2/agent_T로 해야함
        
        self.user_queue = 0
        self.queue_tilde = 100   #최초 100으로 썼음
        self.P_bar = 3
        self.node_position_x = []
        self.node_position_y = []
        self.node_quality = []
        #self.node_list = []
        self.quality_levels=3
        self.Nq_values=[1, 2, 3]  # file size
        self.Pq_values=[4/7, 2/7, 1/7] # probability of each quality level level. 낮은 quality caching 확률이 4/7, 중간 quality caching 확률이 2/7, 높은 quality caching 확률이 1/7인 경우.
        #self.Pq_values=[1/3, 1/3, 1/3] # probability of each quality level level
        #self.Pq_values=[1/7, 2/7, 4/7] # probability of each quality level level
        #self.Pq_values = [1/7, 1/7, 1/7]
        self.field_x = 20 # 0 ~ 20
        self.field_y = 2 # 0 ~ 2
        self.field_area = self.field_x * self.field_y# * 50 * 50
        self.num_node = 0
        self.alpha_list_x = []
        self.alpha_list_y = []
        self.user_position_x = []
        self.user_position_y = []
        user_x = []
        for i in range(int(self.field_x/ (self.user_mobility * self.agent_T) )):  #/5는 기존 mobility=2, agent_t=5일 때 cell안겹치고 인접하던 scale맞추려고 넣어준 계수임
            user_x.append(self.field_x - i * (self.user_mobility*self.agent_T) )
        for element in reversed(user_x):
            self.user_position_x.append(element)
        for _ in range(len(self.user_position_x)):
            self.user_position_y.append(self.field_y/2)
            
        self.Mq_list= [(0,0)
                        ,(1,1),(1,2),(1,3)
                        ,(2,1),(2,2),(2,3)
                        ,(3,1),(3,2),(3,3)
                        ,(4,1),(4,2),(4,3)
                        ,(5,1),(5,2),(5,3)
                        ,(6,1),(6,2),(6,3)
                        ,(7,1),(7,2),(7,3)
                        ,(8,1),(8,2),(8,3)
                        ,(9,1),(9,2),(9,3)
                        ]
        
        
        # Rayleigh fading channel model
        self.INR = 5 #dB단위
        self.Bandwidth = 1 # MHz단위 1MHz
        #self.transmit_SNR = 10**2.5 #dB단위가 아니라 MB단위
        #self.transmit_SNR = 10**1.5
        #self.transmit_SNR = 10**2.0
        self.transmit_SNR = 10**3.0
        
        
        self.distances = [] #self.make_distance_list()
        self.path_loss_exponent = 2
        self.Queue_departure = 1
        self.user_radius = 1 # 1 = 50m임. 실제로 계산할 때는 cell안에서 0~1인 distance값에 50을 곱해줘야 함
        self.coherence_time = 1
        self.fast_fading_gain = abs(np.random.normal(0,1))
        self.shadowing_variance = 4 #dB
        self.shadowing_effect = abs(np.random.normal(0,self.shadowing_variance))
        self.channelGain = 0 #cmath.sqrt(  self.shadowing_effect / ((self.DistanceTtoR)**self.path_loss_exponent) ) * self.fast_fading_gain
        self.ChannelCapacity = self.Bandwidth * math.log2( (1 + self.transmit_SNR * abs(self.channelGain)*abs(self.channelGain)) / (1+self.INR) )
        self.ChannelCapacity_list = []
        
        self.small_time = 0 # Mq time
        self.large_time = 0 # alpha scan time
        self.episode = 0
        self.max_time = len(self.user_position_x)
        self.q_t = 0
        
        self.count_alpha = []
        
        self.accepted_chunk = 0
        self.accumulated_file_size = 0
        self.buffering_count = 0        
      
      
      #매 episode마다 Environment 초기화하는 함수
  def setting(self):
        self.node_position_x = []
        self.node_position_y = []
        self.node_quality = []
        
        '''
        # node분포 확률이 최소 1/7이면서 1/7단위로 랜덤하게 배정되도록 하는 부분
        self.Pq_values = [1/7, 1/7, 1/7]
        for i in range(4):
            a = random.randint(1,3)
            if (a == 1):
                self.Pq_values[0] = self.Pq_values[0] + 1/7
            elif (a == 2):
                self.Pq_values[0] = self.Pq_values[1] + 1/7
            elif (a == 3):
                self.Pq_values[0] = self.Pq_values[2] + 1/7
        ''' 

        #self.num_node = np.random.poisson(self.intensity_lambda * self.field_area)
        for level in range(self.quality_levels):
            num_node = np.random.poisson(self.intensity_lambda * self.field_area * self.Pq_values[level])
            #print("num node each quality level : ", num_node)
            for i in range(num_node):
                    self.node_position_x.append(self.field_x * np.random.rand())
                    self.node_position_y.append(self.field_y * np.random.rand())
                    self.node_quality.append(self.Nq_values[level])
        self.num_node = len(self.node_position_x)
        self.make_distance_list()
        
        # 디버깅 목적으로 시각화를 통해 코드가 정상적으로 동작하는지 확인하기 위해 graph를 그리는 함수
  def plot_graph(self):
        '''
        plt.subplots(figsize=(10, 2))
        plt.xlim(0,self.field_x)
        plt.ylim(0,self.field_y)
        for i in range(self.num_node):
            if(self.node_quality[i]==1):
                plt.plot(self.node_position_x[i],self.node_position_y[i], 'ro')
            elif(self.node_quality[i]==2):
                plt.plot(self.node_position_x[i],self.node_position_y[i], 'go')
            else:
                plt.plot(self.node_position_x[i],self.node_position_y[i], 'bo')
        for i in range(len(self.user_position_x)):
            plt.plot(self.user_position_x[i], self.user_position_y[i], 'k^')                              
        # quality level = 1이면 red, 2이면 green, 3이면 blue로 표현
        # user node는 검은색(plt에서 k)으로 표현
        #plt.plot(self.node_position_x,self.node_position_y, 'bo') 
        #print("user position x : ", self.user_position_x)
        #print("user position y : ", self.user_position_y)       
        #print("node x : ", self.node_position_x)
        #print("node y : ", self.node_position_y)
        #print("node quality : ", self.node_quality)
        plt.show()
        '''
                
        
        #f, ax = plt.subplots(figsize=(self.field_x+2, self.field_y+1))
        #figure, ax = plt.subplots(2,1, figsize=(self.field_x+2, 2*(self.field_y+1)))
        figure, ax = plt.subplots(2,1, figsize=(10, 5))
        plt.subplot(2,1,1)
        plt.xlim(0,self.field_x+1)
        plt.ylim(-1,self.field_y+1)
        circles = []
        #ax[0] = plt.axes(xlim=(-1,self.field_x+1),ylim=(-1,self.field_y+1))


        count1 = 0
        count2 = 0
        count3 = 0
        count4 = 0
        count5 = 0
        
        for i in range(self.num_node):
            if(self.node_quality[i]==1):
                if(count1==0):
                    ax[0].plot(self.node_position_x[i],self.node_position_y[i], 'ro', label="quality level 1")
                    count1 = 1
                else:
                    ax[0].plot(self.node_position_x[i],self.node_position_y[i], 'ro')
            elif(self.node_quality[i]==2):
                if(count2==0):
                    ax[0].plot(self.node_position_x[i],self.node_position_y[i], 'go', label="quality level 2")
                    count2 = 1
                else:
                    ax[0].plot(self.node_position_x[i],self.node_position_y[i], 'go')                    
            else:
                if(count3==0):
                    ax[0].plot(self.node_position_x[i],self.node_position_y[i], 'bo', label="quality level 3")
                    count3 = 1
                else:
                    ax[0].plot(self.node_position_x[i],self.node_position_y[i], 'bo')    
                                    
        for i in range(len(self.user_position_x)):
            if(count4==0):
                ax[0].plot(self.user_position_x[i], self.user_position_y[i], 'k^', label="user")
                count4 = 1
            else:
                ax[0].plot(self.user_position_x[i], self.user_position_y[i], 'k^')
                

        #for i in range(len(self.alpha_list_x)):
        #ax[0].plot(self.alpha_list_x, self.alpha_list_y, color = 'violet', marker='o', linestyle='-', label="linked node")
        for i in range(len(self.alpha_list_x)):
            if(count5==0):
                ax[0].plot([self.alpha_list_x[i],self.user_position_x[self.count_alpha[i]]], [self.alpha_list_y[i],self.user_position_y[self.count_alpha[i]]], color = 'violet', marker='o', linestyle='-', label="linked node")
                count5 = 1
            else:
                ax[0].plot([self.alpha_list_x[i],self.user_position_x[self.count_alpha[i]]], [self.alpha_list_y[i],self.user_position_y[self.count_alpha[i]]], color = 'violet', marker='o', linestyle='-')
                

        for i in range(len(self.user_position_x)):
            circles.append(plt.Circle((self.user_position_x[i],self.user_position_y[i]),1, fc= 'yellow', ec='orange', alpha=0.3))
        
        #print(self.alpha_list_x)
        for i in range(len(self.user_position_x)):
            ax[0].add_patch(circles[i])
        
        plt.legend(ncols=5)
            
            
            
            
            #############     
        #plt.subplots(2)
        #plt.xlim(-1,self.field_x+1)
        #plt.ylim(-1,self.field_y+1)
        circles = []        
        plt.subplot(2,1,2)
        plt.xlim(0,self.field_x+1)
        plt.ylim(-1,self.field_y+1)
        #ax[1] = plt.axes(xlim=(-1,self.field_x+1),ylim=(-1,self.field_y+1))
        

        count1 = 0
        count2 = 0
        count3 = 0
        count4 = 0
        
        
        for i in range(self.num_node):
            if(self.node_quality[i]==1):
                if(count1==0):
                    ax[1].plot(self.node_position_x[i],self.node_position_y[i], 'ro', label="quality level 1")
                    count1 = 1
                else:
                    ax[1].plot(self.node_position_x[i],self.node_position_y[i], 'ro')
            elif(self.node_quality[i]==2):
                if(count2==0):
                    ax[1].plot(self.node_position_x[i],self.node_position_y[i], 'go', label="quality level 2")
                    count2 = 1
                else:
                    ax[1].plot(self.node_position_x[i],self.node_position_y[i], 'go')                    
            else:
                if(count3==0):
                    ax[1].plot(self.node_position_x[i],self.node_position_y[i], 'bo', label="quality level 3")
                    count3 = 1
                else:
                    ax[1].plot(self.node_position_x[i],self.node_position_y[i], 'bo')    
                    
        for i in range(len(self.user_position_x)):
            if(count4==0):
                ax[1].plot(self.user_position_x[i], self.user_position_y[i], 'k^', label="user")
                count4 = 1
            else:
                ax[1].plot(self.user_position_x[i], self.user_position_y[i], 'k^')
                
        
        for i in range(len(self.user_position_x)):
            circles.append(plt.Circle((self.user_position_x[i],self.user_position_y[i]),1, fc= 'yellow', ec='orange', alpha=0.3))
        
        for i in range(len(self.user_position_x)):
            ax[1].add_patch(circles[i])
        
            
        plt.legend(ncols=4)
            
        
        
        plt.tight_layout()
        #plt.show()
        #plt.pause(0)
        plt.savefig('{}{}{}.png'.format(directory,self.transmit_SNR, self.episode))
        #plt.close()
        plt.cla()
        plt.clf()
        plt.close('all')
        #https://hyeonchan523.github.io/python/matplotlib-memory-leak/
        #plt 사용시 memory leaking 해결
        
        # 시간에 따라 변화하는 vehicle-RSU 사이 distance를 update 해주는 함수
  def make_distance_list(self): # user radius = 1 이라고 했을 때의 distance임. 실제 distance값은 user radius값을 곱해줘야함
      for i in range(len(self.user_position_x)):
        distances = []
        for j in range(len(self.node_position_x)):
            distances.append(math.sqrt((self.user_position_x[i] - self.node_position_x[j]) * (self.user_position_x[i] - self.node_position_x[j]) +
                                        (self.user_position_y[i] - self.node_position_y[j]) * (self.user_position_y[i] - self.node_position_y[j])))
        self.distances.append(distances)
        # self.distances는 user의 [ [time=0일 때 모든 node에 대한 거리들(list)] [time=1일 때 모든 node에 대한 거리들] [time=2일 때 모든 node에 대한 거리들] ]형식임
      #print("distances : ", self.distances)
        
        
     # 시간에 따라 변화하는 channel capacity를 sampling하는 함수   
  def calculate_capacity(self,large_time): # large time은 현재 user의 위치가 어디인지 알기 위해 사용하는 것
    self.ChannelCapacity_list = []
    for i in range(len(self.node_position_x)):
        self.fast_fading_gain = abs(np.random.normal(0,1))
        self.shadowing_effect = abs(np.random.normal(0,self.shadowing_variance))
        self.channelGain = cmath.sqrt(  self.shadowing_effect / ((self.user_radius*self.distances[large_time][i])**self.path_loss_exponent) ) * self.fast_fading_gain
        self.ChannelCapacity = self.Bandwidth * math.log2( (1 + (self.transmit_SNR * abs(self.channelGain)*abs(self.channelGain)) / (1+self.INR)) )
        
        #print("component in channel gain : ",  self.shadowing_effect / ((self.user_radius*self.distances[large_time][i])**self.path_loss_exponent))
        #print("abs channel gain : ", abs(self.channelGain))
        #print("distances[large_time][i] : ", self.distances[large_time][i])        
        #print("channel gain : ", self.channelGain)
        #print("channel gain square : ", self.channelGain * self.channelGain)
        
        
        self.ChannelCapacity_list.append(self.ChannelCapacity)        
    
  # episode가 끝나고 environment의 변수값을 초기화하는 함수
  def reset(self):
      #self.setting()
      self.small_time = 0
      self.large_time = 0
      self.episode = self.episode+1
      self.alpha_list_x = []
      self.alpha_list_y = []
      self.distances = []   
      self.q_t = 0
      self.user_queue = 0
      self.count_alpha = []
      self.accepted_chunk = 0
      self.accumulated_file_size = 0
      self.buffering_count = 0
      
      
      
      
      # small time step마다 environment에서 next state와 reward를 도출해주는 함수.
  def step(self, node, action): # action은 Mq_list에 대한 index값 0~27로 받을거임, action을 M,q로 받음
      delta_q = 0
      if (node==None):
          M = 0
          q = 0
          self.user_queue = max(self.user_queue - 1, 0) + M
          next_state = [self.user_queue, self.q_t]
          if(self.user_position_x[self.large_time] >= self.field_x):
              done = True
          else:
              done = False
      elif(action[0]==0):
          M=0
          q=0
          self.user_queue = max(self.user_queue - 1, 0) + M
          next_state = [self.user_queue, self.q_t]
          print("node's caching quality : ", self.node_quality[node])
          if(self.user_position_x[self.large_time] >= self.field_x):
              done = True
          else:
              done = False
          
      
      else:
          M = action[0]
          q = action[1]
          self.calculate_capacity(self.large_time)
          #print("CC list : ", self.ChannelCapacity_list)
          print("Bk : ", self.ChannelCapacity_list[node])
          print("node's caching quality : ", self.node_quality[node])
          if M * q > self.ChannelCapacity_list[node]:
              M = 0
              q = 0
          else:
              print("transmission success")
              delta_q = abs(self.q_t - q)
              self.q_t = q
              self.accepted_chunk = self.accepted_chunk + M
              self.accumulated_file_size = self.accumulated_file_size + M*q
          self.user_queue = max(self.user_queue - 1, 0) + M
          next_state = [self.user_queue, self.q_t]
          if(self.user_position_x[self.large_time] >= self.field_x):
              done = True
          else:
              done = False
              
      self.small_time = self.small_time + 1
      if(self.small_time%self.agent_T==0):
          self.large_time = self.large_time + 1
      
      buffering_panalty = 0
      if(self.user_queue==0):
          self.buffering_count = self.buffering_count+1
          buffering_panalty = 10
      
      #reward = math.sqrt(M)*q*q - buffering_panalty - delta_q
      #reward = math.sqrt(M) + 20*q - buffering_panalty - delta_q
      #reward = 10*q - buffering_panalty - delta_q
      #reward = 30*q - buffering_panalty
      reward = math.sqrt(M) + 30*q - buffering_panalty - delta_q
      #reward = -self.user_queue + 30*q - buffering_panalty - delta_q
      
      return next_state, reward, done
      
  # get_reward라고 되어있지만, 정확히는 reward가 아니라 performance evaluation(성능 평가)를 계산하는 함수    
  def get_rewards(self):
      if(self.accepted_chunk==0):
          return self.buffering_count, 0
      meanbitrate = self.accumulated_file_size/self.accepted_chunk
      return self.buffering_count, meanbitrate
      
      # plot graph에서 디버깅하기 위해 넣어둔 함수로 추정됨
  def save_alpha(self,alpha):
      if(alpha != None):
          self.alpha_list_x.append(self.node_position_x[alpha])
          self.alpha_list_y.append(self.node_position_y[alpha])
          self.count_alpha.append(self.large_time)
      
    # vehicle의 communication range안에 있는 RSU를 확인하는 함수       
  def check_in_cell(self,large_time):
      check_in_cell = []
      distances_in_cell = []
      quality_in_cell = []
      for i in range(len(self.node_position_x)):
            #print("large time : ", env.large_time)
            #print("i : ", i)
            #print("distances' size : ", len(env.distances), '\t', len(env.distances[env.large_time]))
                
        if self.distances[large_time][i] <= 1.0:
            check_in_cell.append(1)
            distances_in_cell.append(self.distances[large_time][i])
            quality_in_cell.append(self.node_quality[i])
        else:
            check_in_cell.append(0)

      if(1 not in check_in_cell):
        return None
      else:
          sorted_distance_in_cell = sorted(distances_in_cell)
          count = 0
          node_state = [[],[]] #[[distances],[qualities]]형태로 만들거임
          for i in range(len(sorted_distance_in_cell)):
              node_state[0].append(sorted_distance_in_cell[i])
              node_state[1].append(quality_in_cell[distances_in_cell.index(sorted_distance_in_cell[i])])
              count=count+1
              if(count==5):
                  break
          if(len(node_state[0]) < 5):
              while(len(node_state[0])<5):
                  node_state[0].append(1)
                  node_state[1].append(1)
      return node_state
  
  
  
      #환경이 정상적으로 구현되는지 테스트용
if __name__ == "__main__":
    env = Env()
    env.setting()
    env.plot_graph()

      
      