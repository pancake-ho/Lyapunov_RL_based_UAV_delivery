
import numpy as np
import random
from collections import defaultdict
from collections import deque
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import RandomUniform
import copy
import math
import matplotlib.pyplot as plt
#import logging # 파이썬에서 log를 저장하기 위한 모듈이라는데, print로 출력된 결과를 저장하는 모듈이 아님
import sys #print로 출력한 결과를 txt로 저장하기 위한 모듈

from environment import Env

np.random.seed(7)
tf.random.set_seed(1)


directory = './DTRL_downgrade/result_plot/'
log_file_name = 'result_log.txt'
#SNR_text = ['15dB', '20dB', '30dB']
#SNR_list = [10**1.5, 10**2.0, 10**3.0]
SNR_list = [ [10**1.5, '15dB'],    [10**2.0, '20dB'],  [10**2.5, '25dB'] , [10**3.0, '30dB']  ]
#sys.stdout = open('{}{}'.format(directory,log_file_name), 'w')
#sys.stdout = open('./DTRL/result_plot3/result_log.txt', 'w')


# chunk delivery DRL Class. M : chunk amounts, Q: bitrate.
class DQN_MQ(tf.keras.Model): #딥러닝 모델 제작
    def __init__(self, action_size):
        super(DQN_MQ, self).__init__()
        
        #self.input_layer = tf.keras.layers.Input(input_shape=(1, 1))
        self.fc1 = Dense(24, activation='relu', name = 'fc1', input_shape=(1,2)) # input size 고려해주기
        self.fc2 = Dense(24,activation='relu', name = 'fc2')
        self.fc_out = Dense(action_size, name = 'fc_out', kernel_initializer=RandomUniform(-1e-3, 1e-3))
        #https://keras.io/ko/initializers/
        # kernel_initializer, bias_initiallizer가 있는데 각각 weight, bias의 초기값을 설정하는 flag임
        # -0.001~0.001로 설정해준 이유는 reduce variance를 위해서임. 
        # weight가 update되더라도, 저범위에서 시작하면 weight가 작은 값이 나올거고
        # 그러면 Q의 범위가 작아지니까.
        # M = 1~9, K=1~3으로 3*9=27 + (0,0)으로 action_size=28이다.
        self.build((1,2))
        #https://github.com/keras-team/keras/issues/4753
        # set_weights()함수를 쓸 때 model 또는layer를 먼저 build로 구축해두고 사용하는 것이 안전함. 그래서 build사용으로 에러 해결


    def call(self, x):
        #DL에 입력 넣어주려면 최소 [[sample1],[sample2], ---] 형태로 2차원이여야함.
        #그래서 원래 [M,K]로 했던 state를 [[M,K],[M,K],---]형태로 바꿔줘야함. 
        #x = self.input_layer(x)
        x = self.fc1(x)
        x = self.fc2(x)
        q=self.fc_out(x)
        return q
    
#link scheduling DRL Class. S: scheduling    
class DQN_scheduling(tf.keras.Model):
    def __init__(self, action_size):
        super(DQN_scheduling, self).__init__()
        #self.input_layer = tf.keras.layers.Input(input_shape=(1, 1))
        self.fc1 = Dense(24, activation='relu', name = 'fc1', input_shape=(1,10)) # input size 고려해주기
        self.fc2 = Dense(24,activation='relu', name = 'fc2')
        self.fc_out = Dense(action_size, name = 'fc_out', kernel_initializer=RandomUniform(-1e-3, 1e-3))
        #https://keras.io/ko/initializers/
        # kernel_initializer, bias_initiallizer가 있는데 각각 weight, bias의 초기값을 설정하는 flag임
        # -0.001~0.001로 설정해준 이유는 reduce variance를 위해서임. 
        # weight가 update되더라도, 저범위에서 시작하면 weight가 작은 값이 나올거고
        # 그러면 Q의 범위가 작아지니까.
        # M = 1~9, K=1~3으로 3*9=27 + (0,0)으로 action_size=28이다.
        self.build((1,10)) # input shape에 맞춰서 build해줘야 함
        #https://github.com/keras-team/keras/issues/4753
        # set_weights()함수를 쓸 때 model 또는layer를 먼저 build로 구축해두고 사용하는 것이 안전함. 그래서 build사용으로 에러 해결


    def call(self, x):
        #DL에 입력 넣어주려면 최소 [[sample1],[sample2], ---] 형태로 2차원이여야함.
        #그래서 원래 [M,K]로 했던 state를 [[M,K],[M,K],---]형태로 바꿔줘야함. 
        #x = self.input_layer(x)
        x = self.fc1(x)
        x = self.fc2(x)
        q=self.fc_out(x)
        return q

# video delivery DRL과 link scheduling DRL을 모두 포함하는 하나의 DQN Agent Class.
class DQNAgent:

    def __init__(self, state_size, action_size_MQ, action_size_scheduling):
        #상태와 행동의 크기 정의
        self.state_size = state_size
        self.action_size_MQ = action_size_MQ
        self.action_size_scheduling = action_size_scheduling
        
        #action은 M,K형태고 1~9,1~3 + (0,0)으로 28개
        self.action_list= [(0,0)
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

        #하이퍼 파라미터 설정
        self.learning_rate=0.001
        self.discount_factor = 0.5 #임의로 설정한 discount_factor
        self.epsilon = 1.0 #초기 epsilon 값
        self.epsilon_decay = 0.99999#epsilon 값 감쇄량 # 최초0.999였음
        self.epsilon_min = 0.01#epsilon 최소 값
        self.batch_size_MQ = 64
        self.batch_size_S = 64
        self.train_start = 500
        self.q_table = defaultdict(float)

        #리플레이 메모리, 최대 크기 2000
        self.memory_MQ = deque(maxlen=2000)
        self.memory_scheduling = deque(maxlen=2000)

        #모델, 타깃 모델 생성
        self.behavior_MQ_model = DQN_MQ(self.action_size_MQ)#replay 메모리 저장용
        self.target_MQ_model = DQN_MQ(self.action_size_MQ)#정책 학습용
        self.behavior_S_model = DQN_scheduling(self.action_size_scheduling)
        self.target_S_model = DQN_scheduling(self.action_size_scheduling)
        self.optimizer = Adam(lr=self.learning_rate)
        #타깃 모델 초기화
        self.update_target_MQ_model()
        self.update_target_S_model()
        
        
        self.T = 5 # large time scale # 최초 T는 5였음
        
        
        
        
        
        
        
        
        

    # delivery 타깃 모델을 모델의 가중치로 업데이트
    def update_target_MQ_model(self):
        #self.target_model.set_weights(self.behavior_model.get_weights())
        
        '''
        print("behavior model : " )
        #self.behavior_model.summary()
        print("behavior model layer 0 wegith : ", self.behavior_model.layers[0].name, '\t', self.behavior_model.layers[0].get_weights())
        print("behavior model layer 1 wegith : ", self.behavior_model.layers[1].name, '\t', self.behavior_model.layers[1].get_weights())
        print("behavior model layer 2 wegith : ", self.behavior_model.layers[2].name, '\t', self.behavior_model.layers[2].get_weights())
        
        print("target model : ")
        #self.target_model.summary()
        print("target model layer 0 wegith : ", self.target_model.layers[0].name, '\t', self.target_model.layers[0].get_weights())
        print("target model layer 1 wegith : ", self.target_model.layers[1].name, '\t', self.target_model.layers[1].get_weights())
        print("target model layer 2 wegith : ", self.target_model.layers[2].name, '\t', self.target_model.layers[2].get_weights())
        '''
        
        weights = self.behavior_MQ_model.get_weights()
        #print("behavior model's weights : ", weights)
        #print("target model's weights : ", self.target_model.get_weights())
        
        #print("target model's weights : ", self.target_model.get_weights())
        self.target_MQ_model.set_weights(weights)
        
        '''
        T를 늘리기 ex)20 or mobility 줄이기
        추후) lambda 조금만 키우기
        결과 plot scheduling 잘 됐는지 확인용
        caching된 quality 내에서만 q(t) 제대로 하는지 확인        
        '''
        
    # scheduling target model의 weight를 behavior model의 weight로 update. 
    def update_target_S_model(self):
        
        weights = self.behavior_S_model.get_weights()
        #print("behavior model's weights : ", weights)
        #print("target model's weights : ", self.target_model.get_weights())
        
        #print("target model's weights : ", self.target_model.get_weights())
        self.target_S_model.set_weights(weights)
        

    # <s, a, r, s'> 리플레이 메모리에 저장
    def append_MQ_sample(self, state, action, reward, next_state, done):
        self.memory_MQ.append((state, action, reward, next_state, done))
    def append_S_sample(self, state, action, reward, next_state, done):
        self.memory_scheduling.append((state, action, reward, next_state, done))
        
    # (M,K)형태인 action을 입력으로 받아서 action_list에 있는 index로 mapping해주는 함수
    def action_transform(self,action):
        for i in range(len(self.action_list)):
            if(action==self.action_list[i]):
                return i

    #리플레이 메모리에서 무작위로 추출한 배치로 delivery DRL 모델 학습
    def train_MQ_model(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        #메모리에서 배치 크기만큼 무작위로 샘플 추출
        mini_batch = random.sample(self.memory_MQ, self.batch_size_MQ)
        #print("mini_batch : ", mini_batch)
        states = np.array([sample[0] for sample in mini_batch])
        actions = np.array([sample[1] for sample in mini_batch]) #아직 (M,K)상태
        rewards = np.array([sample[2] for sample in mini_batch])
        next_states = np.array([sample[3][0] for sample in mini_batch])
        dones = np.array([sample[4] for sample in mini_batch])
        #print("states : ", states)
        # 학습 파라미터
        model_params = self.behavior_MQ_model.trainable_variables
        with tf.GradientTape() as tape:

            #현재 상태에 대한 모델의 큐함수
            predicts = self.behavior_MQ_model(states)
            one_hot_action = tf.one_hot(actions, self.action_size_MQ)
            #print("prediction : ", predicts)
            #print("prediction : ", tf.shape(predicts))
            # print(tf.shape(one_hot_action))
            predicts = tf.reduce_sum(one_hot_action * predicts, axis=1)
            #여기가 off policy인 이유 중 behaior가 epsilon으로 구한 action에 대한 Q값을 가져오는 거라서.
            
            #print("action곱 :", actions)
            #print("reduce_sum(action*predicts,axis=-1) : ", predicts)
            #다음 상태에 대한 타깃 모델의 큐함수
            
            target_predicts = self.target_MQ_model(next_states)
            target_predicts = tf.stop_gradient(target_predicts)

            #벨만 최적 방정식을 이용한 업데이트 타깃
            
            max_q = np.amax(target_predicts, axis=-1)
            #여기가 Greedy policy로 off policy인 부분임
            targets = rewards + (1-dones) * self.discount_factor * max_q
            loss = tf.reduce_mean(tf.square(targets - predicts)) # loss 함수로 mse 사용

            # 오류 함수를 줄이는 방향으로 모델 업데이트
            grads = tape.gradient(loss, model_params)
            self.optimizer.apply_gradients(zip(grads, model_params))


    #리플레이 메모리에서 무작위로 추출한 배치로 scheduling DRL 모델 학습
    def train_S_model(self):

        #메모리에서 배치 크기만큼 무작위로 샘플 추출
        #print("scheduling memory : ", self.memory_scheduling)
        mini_batch = random.sample(self.memory_scheduling, self.batch_size_S)
        #print("mini_batch : ", mini_batch)
        states = np.array([sample[0][0] for sample in mini_batch])
        actions = np.array([sample[1] for sample in mini_batch]) #아직 (M,K)상태
        rewards = np.array([sample[2] for sample in mini_batch])
        next_states = np.array([sample[3][0] for sample in mini_batch])
        dones = np.array([sample[4] for sample in mini_batch])
        #print("states : ", states)
        # 학습 파라미터
        model_params = self.behavior_S_model.trainable_variables
        with tf.GradientTape() as tape:

            #현재 상태에 대한 모델의 큐함수
            predicts = self.behavior_S_model(states)
            one_hot_action = tf.one_hot(actions, self.action_size_scheduling)
            #print("prediction : ", predicts)
            #print("prediction : ", tf.shape(predicts))
            # print(tf.shape(one_hot_action))
            predicts = tf.reduce_sum(one_hot_action * predicts, axis=1)
            #여기가 off policy인 이유 중 behaior가 epsilon으로 구한 action에 대한 Q값을 가져오는 거라서.
            
            #print("action곱 :", actions)
            #print("reduce_sum(action*predicts,axis=-1) : ", predicts)
            #다음 상태에 대한 타깃 모델의 큐함수
            
            target_predicts = self.target_S_model(next_states)
            target_predicts = tf.stop_gradient(target_predicts)

            #벨만 최적 방정식을 이용한 업데이트 타깃
            
            max_q = np.amax(target_predicts, axis=-1)
            #여기가 Greedy policy로 off policy인 부분임
            targets = rewards + (1-dones) * self.discount_factor * max_q
            loss = tf.reduce_mean(tf.square(targets - predicts)) # loss 함수로 mse 사용

            # 오류 함수를 줄이는 방향으로 모델 업데이트
            grads = tape.gradient(loss, model_params)
            self.optimizer.apply_gradients(zip(grads, model_params))

    # 모든 에피소드에서 에이전트가 방문한 상태의 큐 함수를 업데이트
   

    # 입실론 탐욕 정책에 따라서 delivery 행동을 반환 Behavior policy : epsilon greedy
    def get_Mq(self, state, alpha, env):
        print("current state : ", state[0])
        if(alpha == None):
            print("node not in cell : (0,0)")
            return (0,0)
        if(state[0][1]==0):
            state[0][1]=2  # 우리 q_t가 1~3이고 차이가 1 이하여야 하니까 최초 q_t==0일 때 q_t=2로 만들어서 아무 q나 선택 가능하도록 한 것
            
            
        node_level = env.node_quality[alpha]
        if np.random.rand() <= self.epsilon:            
            possible_action_list = []
            for i in range(len(self.action_list)):
                action = self.action_list[i]
                if( (action[1] <= node_level) ):
                    possible_action_list.append(i)
            if len(possible_action_list)==0 :
                action = (0,0)
            else:
                action = self.action_list[possible_action_list[np.random.randint(0,len(possible_action_list))]]
            
            print('get_action의 epsilon greedy policy에 따른 random action : ', action)
            return action # action은 (M,K) format
        else:
            
            q_value= self.behavior_MQ_model(state) 
            action = self.action_list[np.argmax(q_value)]
            copy_action_list = copy.deepcopy(self.action_list)
            copy_q_value = copy.deepcopy(q_value)
            copy_q_value = copy_q_value.numpy()
            copy_q_value = copy_q_value.tolist()[0]
            #print(type(copy_q_value))
            #print(copy_q_value)
            while(True):                
                if( (action[1] > node_level) ):
                    #print('copy action list : ', copy_action_list)
                    #print('copy q value : ', copy_q_value)
                    impossible_q_index = np.argmax(copy_q_value)
                    copy_q_value.pop(impossible_q_index)
                    copy_action_list.pop(impossible_q_index) 
                    #print("copy action list : ", copy_action_list)      
                    if(copy_action_list==[]):
                        action = (0,0)
                        print("No possible action")
                        return action             
                    action = copy_action_list[np.argmax(copy_q_value)]                                    
                else:
                    break
            
            
            
            
            print("get_action의 epsilon greedy policy에 따른 action : ", action)
            return action # action은 (M,K) format
        
        
    # scheduling DRL로 scheduling을 수행하는 함수. behavior policy : epsilon greedy policy 
    def get_alpha(self,env):
        
        env.make_distance_list()
        node_state = env.check_in_cell(env.large_time)
        print("node state : ",node_state)
        if(node_state==None):
            return None, None, None
        input_node_state = np.reshape(node_state, [1,10])
        #print("input node state : ", input_node_state)
        q_value = self.behavior_S_model(input_node_state)
        #print("q value : ", q_value)
        if np.random.rand() <= self.epsilon:
            link_node_index = random.randint(0,4)
            while(node_state[0][link_node_index]==1):
                link_node_index = random.randint(0,4)
            print("random link node index : ", link_node_index)
        else:
            link_node_index = np.argmax(q_value)
            copy_q_value = copy.deepcopy(q_value)
            copy_q_value = copy_q_value.numpy()
            copy_q_value = copy_q_value.tolist()[0]
            
            original_q_value = copy.deepcopy(q_value)
            original_q_value = original_q_value.numpy()
            original_q_value = original_q_value.tolist()[0]
            
            while(node_state[0][link_node_index]==1):
                copy_q_value.remove(original_q_value[link_node_index])
                link_node_index = original_q_value.index(max(copy_q_value))
            print("greedy link node index : ",link_node_index)
        env_link_node_index = env.distances[env.large_time].index(node_state[0][link_node_index])
        
        
        
        return env_link_node_index, node_state, link_node_index
        #env_link_node_index는 현재 user위치에서 모든 node중에서 link node의 index
        #link_node_index는 현재 user위치에서 cell안의 거리순 5개 이내 node들 중에서의 index
        #그래프 그릴땐 env_link_node_index사용하고, agent학습할 때는 link_node_index 사용
        
        
        
        
        
    # target scheduling model을 사용해 scheduling하는 함수. target policy : greedy policy        
    def test_get_alpha(self,env):
        
        env.make_distance_list()
        node_state = env.check_in_cell(env.large_time)
        print("node state : ",node_state)
        if(node_state==None):
            return None, None, None
        input_node_state = np.reshape(node_state, [1,10])
        #print("input node state : ", input_node_state)
        q_value = self.target_S_model(input_node_state)
        #print("q value : ", q_value)
        link_node_index = np.argmax(q_value)
        copy_q_value = copy.deepcopy(q_value)
        copy_q_value = copy_q_value.numpy()
        copy_q_value = copy_q_value.tolist()[0]
        
        original_q_value = copy.deepcopy(q_value)
        original_q_value = original_q_value.numpy()
        original_q_value = original_q_value.tolist()[0]
        
        while(node_state[0][link_node_index]==1):
            copy_q_value.remove(original_q_value[link_node_index])
            link_node_index = original_q_value.index(max(copy_q_value))
        print("link node index : ",link_node_index)
        env_link_node_index = env.distances[env.large_time].index(node_state[0][link_node_index])      
        
        return env_link_node_index, node_state, link_node_index
        #env_link_node_index는 현재 user위치에서 모든 node중에서 link node의 index
        #link_node_index는 현재 user위치에서 cell안의 거리순 5개 이내 node들 중에서의 index
        #그래프 그릴땐 env_link_node_index사용하고, agent학습할 때는 link_node_index 사용
        
        
        
        
        
        
    # target delivery model을 사용해 delivery하는 함수. target policy : greedy policy   
    def test_get_Mq(self, state, alpha, env):
       
        print("current state : ", state[0])
        if(alpha == None):
            print("node not in cell : (0,0)")
            return (0,0)
        if(state[0][1]==0):
            state[0][1]=2  # 우리 q_t가 1~3이고 차이가 1 이하여야 하니까 최초 q_t==0일 때 q_t=2로 만들어서 아무 q나 선택 가능하도록 한 것
            
            
        node_level = env.node_quality[alpha]
    
        q_value= self.target_MQ_model(state) 
        action = self.action_list[np.argmax(q_value)]
        copy_action_list = copy.deepcopy(self.action_list)
        copy_q_value = copy.deepcopy(q_value)
        copy_q_value = copy_q_value.numpy()
        copy_q_value = copy_q_value.tolist()[0]
        #print(type(copy_q_value))
        #print(copy_q_value)
        while(True):                
            if( (action[1] > node_level) ):
                #print('copy action list : ', copy_action_list)
                #print('copy q value : ', copy_q_value)
                impossible_q_index = np.argmax(copy_q_value)
                copy_q_value.pop(impossible_q_index)
                copy_action_list.pop(impossible_q_index) 
                #print("copy action list : ", copy_action_list)      
                if(copy_action_list==[]):
                    action = (0,0)
                    print("No possible action")
                    return action             
                action = copy_action_list[np.argmax(copy_q_value)]                                    
            else:
                break
        print("get_action의 epsilon greedy policy에 따른 greedy action : ", action)
        return action # action은 (M,K) format
    
        
        
        
        
        
        
    # 메인 함수

if __name__ == "__main__":  # 해당 .py파일이 module로 import되면 작동하지 않고, main으로 사용될 때만 동작하는 코드라는 뜻. __name__이 file이름이고 이게, main으로 되느냐임. agent에서 compile을 시작할 떄



    env = Env() #환경 객체 만들기(정보를 담고 있음) __init__ 받아옴

    env.transmit_SNR = SNR_list[3][0]
    # 매번 SNR 바꿔서 실험할 때마다 index값 바꿔줘야함

    #state_table = [[] for i in range(env.max_episode)]   # 디버깅용 밟은 state저장하는 table
    #action_table = [[] for i in range(env.max_episode)]  # 디버깅용 취한 action저장하는 table
    #total_reward= [[0.] for i in range(env.max_episode)] # 디버깅용 받은 reward저장하는 table
    #plot_q = [] # 디버깅용 q값을 plot하려고 만든 table

    action_MQ_size = 28 # m(t), k(t)
    action_S_size = 5
    state_size = 2 # Z(t), q(t)

    agent = DQNAgent(state_size, action_MQ_size,action_S_size)  # agent 객체 만들기 __init__ 받아옴

    #agent.update_target_model() 
    #https://keras.io/ko/models/about-keras-models/
    #keras set_weights 함수 docs
    
    total_reward = [0 for i  in range(env.max_episode)]
    total_mean_bitrate = [0 for i  in range(env.max_episode)]
    total_buffering = [0 for i  in range(env.max_episode)]
    
    
    

    for episode in range(env.max_episode): # episode를 진행하는 반복문, max_episode 일단 1000으로 설정했음
        #break
        env.setting()
        prev_state = [env.user_queue, 0] #state 리셋 list 형태[Q_t,n_t]를 state로 쓰겠다.
        prev_state = np.reshape(prev_state, [1, state_size])
        #prev_state = np.reshape(prev_state, [1, None])
        #print("state :", prev_state)
        #print("[[0,0]] : ",[[0,0]] )
        large_reward = 0

        while True: # time step을 진행하는 반복문
            if (env.small_time % agent.T == 0):                
                env_alpha, node_state, action_alpha = agent.get_alpha(env)
                env.save_alpha(env_alpha)
                done = False
                if(large_reward!=0):
                    #if( prev_node_state == None):
                    #https://ddka.tistory.com/entry/python%EC%97%90%EC%84%9C-list-%EB%98%90%EB%8A%94-numpyarray-%EB%B3%80%EC%88%98%EC%9D%98-%EA%B0%92%EC%9D%B4-%EA%B0%99%EC%9D%80%EC%A7%80-%EB%B9%84%EA%B5%90%ED%95%98%EB%8A%94-%EB%B0%A9%EB%B2%95
                    # numpy array랑 None이랑 ==연산자로 비교가 불가능해서 if문 안돌아가고 error발생함. 아래 np.array_equal 사용해야함
                    if( np.array_equal(prev_node_state,None)):
                        input_prev_node_state = np.reshape([0,0,0,0,0,0,0,0,0,0], [1,10])
                    else:
                        input_prev_node_state = np.reshape(prev_node_state,[1,10])
                    #if( node_state == None ):
                    if( np.array_equal(node_state,None)):
                        input_node_state = np.reshape([0,0,0,0,0,0,0,0,0,0], [1,10])
                    else:
                        input_node_state = np.reshape(node_state,[1,10])
                    if(env.large_time==len(env.user_position_x)):
                        done = True
                    if(action_alpha!=None):
                        agent.append_S_sample(input_prev_node_state, action_alpha, large_reward, input_node_state, done)
                prev_node_state = node_state
                large_reward=0
                    

            print("episode, small_time, large_time : ", episode, env.small_time, env.large_time)
            action = agent.get_Mq(prev_state, env_alpha, env)
            next_state, reward, done = env.step(env_alpha, action) # time+1 되는 시점
            next_state = np.reshape(next_state, [1, state_size])#reshape 해주는 이유
            
            large_reward = large_reward + reward
            #if (env.small_time % agent.T == 0):
                #env.save_alpha(alpha)
            
            total_reward[episode] = total_reward[episode] + reward
            #total_buffering[episode][0] = total_reward[episode][0] + buffering
            #total_mean_bitrate[episode][0] = total_mean_bitrate[episode][0] + bitrate
            
            #total_reward[episode][0] = total_reward[episode][0]+reward # 디버깅용 reward 더하는 코드
            #plot_q.append(agent.behavior_model(np.reshape([0, 0],[1,2]))[0])

            action = agent.action_transform(action)
            agent.append_MQ_sample(prev_state[0], action, reward, next_state, done)
            
            #print("agent.memory : ", len(agent.memory))
            print("\n\n\n")

            #타임스텝마다 학습
            if len(agent.memory_MQ) >= agent.train_start:
                #print("behavior model summary : ", agent.behavior_model.summary())
                #print("target model summary : ", agent.target_model.summary())
                
                #weights= agent.behavior_model.get_weights() 
                #print(weights[0].shape)
                #print(weights[1].shape)
                #print(weights[2].shape)
                #print(weights[3].shape)               
                #print("agent.behavior_model.weights : ", weights)
                #print("agent.behavior_model.bias : ", bias)
                #weights = agent.target_model.get_weights()   
                #print(weights[0].shape)
                #print(weights[1].shape)
                #print(weights[2].shape)
                #print(weights[3].shape)                                       
                #print("agent.target_model.weights : ", weights)
                #print("agent.target_model.bias : ", bias)
                agent.train_MQ_model()
                agent.train_S_model()
                

            prev_state = next_state
            

            # 디버깅용 원하는 [[state],(action)]을 넣으면 step단위로 q_function의 변화를 볼 수 있다.

            #state_table[episode].append(next_state)  # 디버깅용 state table만드는 코드
            #action_table[episode].append(prev_action) # 디버깅용 action table만드는 코드
            
            if done: #마지막 update까지 하고 디버깅용 저장까지 하고 break해야함
                #각 에피소드마다 타깃 모델을 모델의 가중치로 업데이트
                agent.update_target_MQ_model()
                agent.update_target_S_model()
                #에피소드마다 학습 결과 출력
                buffering, meanbitrate = env.get_rewards()
                total_buffering[episode] =  buffering
                total_mean_bitrate[episode] = meanbitrate                

                print("\n끝\n")
                break
        #if(env.max_episode - episode < 10):
            #env.plot_graph()

        total_buffering[episode] = total_buffering[episode]/env.small_time
        if(env.episode == env.max_episode):
            break 
        #env.plot_graph()
        env.reset()  # 환경 리셋(q,n을 가지고 있고, 이를 초기화 함)
    
    
    
    
    
    
    
    
    
    
    ##################################################
    # target model로 test 돌리는 부분
    
    
    sys.stdout = open('{}{}{}'.format(directory,SNR_list[3][1],log_file_name), 'w')
    print("\n\n\n\ntarget model test\n\n\n\n")
    print("test in {}\n\n".format(SNR_list[3][1]))
    
    env2 = Env()
    env2.max_episode = 10
    env2.transmit_SNR = SNR_list[3][0]
    
    #total_reward2 = [0 for i  in range(env2.max_episode)]
    total_mean_bitrate2 = [0 for i  in range(env2.max_episode)]
    total_buffering2 = [0 for i  in range(env2.max_episode)]
    
    
    for episode2 in range(env2.max_episode):
        
        env2.setting()
        prev_state = [env2.user_queue, 0] #state 리셋 list 형태[Q_t,n_t]를 state로 쓰겠다.
        prev_state = np.reshape(prev_state, [1, state_size])
    
        large_reward = 0
        while True: # time step을 진행하는 반복문
            if (env2.small_time % agent.T == 0):
                            
                env_alpha, node_state, action_alpha = agent.test_get_alpha(env2)
                env2.save_alpha(env_alpha)
                done = False
                if(large_reward!=0):
                    #if( prev_node_state == None):
                    #https://ddka.tistory.com/entry/python%EC%97%90%EC%84%9C-list-%EB%98%90%EB%8A%94-numpyarray-%EB%B3%80%EC%88%98%EC%9D%98-%EA%B0%92%EC%9D%B4-%EA%B0%99%EC%9D%80%EC%A7%80-%EB%B9%84%EA%B5%90%ED%95%98%EB%8A%94-%EB%B0%A9%EB%B2%95
                    # numpy array랑 None이랑 ==연산자로 비교가 불가능해서 if문 안돌아가고 error발생함. 아래 np.array_equal 사용해야함
                    if( np.array_equal(prev_node_state,None)):
                        input_prev_node_state = np.reshape([0,0,0,0,0,0,0,0,0,0], [1,10])
                    else:
                        input_prev_node_state = np.reshape(prev_node_state,[1,10])
                    #if( node_state == None ):
                    if( np.array_equal(node_state,None)):
                        input_node_state = np.reshape([0,0,0,0,0,0,0,0,0,0], [1,10])
                    else:
                        input_node_state = np.reshape(node_state,[1,10])
                    if(env2.large_time==len(env.user_position_x)):
                        done = True
                    if(action_alpha!=None):
                        agent.append_S_sample(input_prev_node_state, action_alpha, large_reward, input_node_state, done)
                prev_node_state = node_state
                large_reward=0

            print("episode, small_time, large_time : ", episode2, env2.small_time, env2.large_time)
            action = agent.test_get_Mq(prev_state, env_alpha, env2)
            next_state, reward, done = env2.step(env_alpha, action) # time+1 되는 시점
            next_state = np.reshape(next_state, [1, state_size])#reshape 해주는 이유
            
            large_reward = large_reward + reward
            
            #if (env2.small_time % agent.T == 0):
                #env2.save_alpha(alpha)
            
            #total_reward2[episode2] = total_reward2[episode2] + reward

            action = agent.action_transform(action)
            agent.append_MQ_sample(prev_state[0], action, reward, next_state, done)
            
            #print("agent.memory : ", len(agent.memory))
            print("\n\n\n")
    
            prev_state = next_state
            

            
            if done: 
                buffering, meanbitrate = env2.get_rewards()
                total_buffering2[episode2] =  buffering
                total_mean_bitrate2[episode2] = meanbitrate                

                print("\n끝\n")
                break

        total_buffering2[episode2] = total_buffering2[episode2]/env2.small_time
        env2.plot_graph()
        env2.reset() 
        if(episode2 == env2.max_episode-1):
            break 
    print("{} mean of buffering in Test : ".format(SNR_list[3][1]), tf.reduce_sum(total_buffering2)/len(total_buffering2))
    print("{} mean of bitrate in Test : ".format(SNR_list[3][1]), tf.reduce_sum(total_mean_bitrate2)/len(total_mean_bitrate2))
    #여기서 mean은 episode당 bitrate, buffering을 의미함

    agent.target_S_model.save("{}{}target_S_model".format(directory,SNR_list[3][1]))
    agent.target_MQ_model.save("{}{}target_MQ_model".format(directory,SNR_list[3][1]))
    
    ##################################################
    #plot code
    plt.figure('train_reward')
    plt.xlabel('episode')
    plt.ylabel('total_reward')
    #plt.ylim(-100,60000)
    plt.plot(total_reward)
    plt.title("reward at each episode in Train")

    plt.savefig('{}{}{}.png'.format(directory, SNR_list[3][1],'train_reward'))
    plt.cla()
    plt.clf()
    plt.close('all')
    
    plt.figure('train_bitrate')
    plt.xlabel('episode')
    plt.ylabel('total_mean_bitrate')
    plt.plot(total_mean_bitrate)
    plt.title("bitrate at each episode in Train")

    plt.savefig('{}{}{}.png'.format(directory, SNR_list[3][1],'train_bitrate'))
    plt.cla()
    plt.clf()
    plt.close('all')


    
    plt.figure('train_buffering')
    plt.xlabel('episode')
    plt.ylabel('total_buffering')
    plt.plot(total_buffering)
    plt.title("buffering at each episode in Train")
    #plt.show()

    plt.savefig('{}{}{}.png'.format(directory, SNR_list[3][1],'train_buffering'))
    plt.cla()
    plt.clf()
    plt.close('all')

    
    '''
    plt.figure()
    plt.xlabel('episode')
    plt.ylabel('test_reward')
    plt.plot(total_reward2)
    plt.title("reward at each episode in Test")
    '''
    
    plt.figure('test_bitrate')
    plt.xlabel('episode')
    plt.ylabel('test_mean_bitrate')
    plt.plot(total_mean_bitrate2)
    plt.title("bitrate at each episode in Test")
    
    
    plt.savefig('{}{}{}.png'.format(directory, SNR_list[3][1],'test_bitrate'))
    plt.cla()
    plt.clf()
    plt.close('all')

    
    
    
    plt.figure('test_buffering')
    plt.xlabel('episode')
    plt.ylabel('test_buffering')
    plt.plot(total_buffering2)
    plt.title("buffering at each episode in Test")
    #plt.show()
    
    plt.savefig('{}{}{}.png'.format(directory, SNR_list[3][1],'test_buffering'))
    plt.cla()
    plt.clf()
    plt.close('all')

    
    
    
    sys.stdout.close()
    
        
    
    
    
    
    '''
    logger = logging.getLogger('simple_example')
    logger.setLevel(logging.DEBUG)
    
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    
    fh = logging.FileHandler(filename="python3.8.5/DTRL/result_plot/result_log.log")
    fh.setLevel(logging.DEBUG)
    
    logger.addHandler(ch)
    logger.addHandler(ch)
    '''
    #print("state_table :  ", *state_table[:],sep='\n')     #state table 기록 보여줄 때 주석 해제
    #print("action_table : ", *action_table[:],sep='\n')    #action table 기록 보여줄 때 주석 해제
    #print("q_table : ", *q_list,sep='\n')                  #최종 q_table 보여줄 때 주석 해제
    
    #print("plot", plot_q) # step단위 q값 변화 텍스트로 보여줄 때 주석 해제
    #plt.xlabel('episode')
    #plt.ylabel('total_reward')
    #plt.plot(total_reward)
    #plt.plot(total_reward,'o')  # 선 말고 dot으로 plot 해줄 때 주석 해제
    #보통 결과 확인할 때, 매 step마다 state랑 action이 어떻게 됐는지랑 한 episode의 total reward합이 어떻게 변하는지를 보여주면 성능을 알 수 있다.
    #plt.xlabel('step')         #step단위 q값 보여줄 때 주석 해제
    #plt.ylabel('q_funtion')    #step단위 q값 보여줄 때 주석 해제
    #plt.plot(plot_q)           #step단위 q값 보여줄 때 주석 해제
    #plt.show()
    





#https://needjarvis.tistory.com/230
# 현재 error 상황은 weight구조와 bias구조가 안맞다는 것.






