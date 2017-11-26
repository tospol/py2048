# 2048.py
# Written in python / pygame by DavidSousaRJ - david.sousarj@gmail.com
# License: Creative Commons
# Sorry about some comments in portuguese!
import os
import sys
import pygame
from pygame.locals import *
from random import randint
import tensorflow as tf
import copy
import random
import numpy as np
from collections import deque
from msvcrt import kbhit,getch
import time

TABLE = np.zeros([4,4]) #[[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev = 0.01)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.01, shape = shape)
    return tf.Variable(initial)

def conv2d(x, W, stride):
    return tf.nn.conv2d(x, W, strides = [1, stride, stride, 1], padding = "SAME")

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = "SAME")

def isgameover(TABLE):
    status = 0
    zerocount = 0
    for LINE in TABLE:
        if 2048 in LINE:
            status = 1
            return status
        elif 0 not in LINE:
            zerocount += 1
    if zerocount == 4:
        # condicoes de gameover: nao ter zero e nao ter consecutivo igual
        # procura consecutivos horizontal
        for i in range(4):
            for j in range(3):
                if TABLE[i][j] == TABLE[i][j + 1]:
                    return status
        # procura consecutivos na vertical
        for j in range(4):
            for i in range(3):
                if TABLE[i][j] == TABLE[i + 1][j]:
                    return status
        status = 2
    return status

def onehot(k):
    a = np.zeros(4)
    a[k] = 1
    return a
# regras do 2048
# define a direcaoo jogada, p.ex. : cima
# para cada coluna, de cima pra baixo
# move o numero para o zero-consecutivo-mais-longe
# se o nao-zero-mais-perto e igual ao numero, combina


def moveup(pi, pj, T):
    justcomb = False
    score = 0
    while pi > 0 and (T[pi - 1][pj] == 0 or
                      (T[pi - 1][pj] == T[pi][pj] and not justcomb)):
        if T[pi - 1][pj] == 0:
            T[pi - 1][pj] = T[pi][pj]
            T[pi][pj] = 0
            pi -= 1
        elif T[pi - 1][pj] == T[pi][pj]:
            if score < T[pi][pj]:
                score = T[pi][pj]
            T[pi - 1][pj] += T[pi][pj]
            T[pi][pj] = 0
            pi -= 1
            justcomb = True
    return T,score


def movedown(pi, pj, T):
    justcomb = False
    score = 0
    while pi < 3 and (T[pi + 1][pj] == 0 or
                      (T[pi + 1][pj] == T[pi][pj] and not justcomb)):
        if T[pi + 1][pj] == 0:
            T[pi + 1][pj] = T[pi][pj]
            T[pi][pj] = 0
            pi += 1
        elif T[pi + 1][pj] == T[pi][pj]:
            if score < T[pi][pj]:
                score = T[pi][pj]
            T[pi + 1][pj] += T[pi][pj]
            T[pi][pj] = 0
            pi += 1
            justcomb = True
    return T,score


def moveleft(pi, pj, T):
    justcomb = False
    score = 0
    while pj > 0 and (T[pi][pj - 1] == 0 or
                      (T[pi][pj - 1] == T[pi][pj] and not justcomb)):
        if T[pi][pj - 1] == 0:
            T[pi][pj - 1] = T[pi][pj]
            T[pi][pj] = 0
            pj -= 1
        elif T[pi][pj - 1] == T[pi][pj]:
            if score < T[pi][pj]:
                score = T[pi][pj]
            T[pi][pj - 1] += T[pi][pj]
            T[pi][pj] = 0
            pj -= 1
            justcomb = True
    return T,score


def moveright(pi, pj, T):
    justcomb = False
    score = 0
    while pj < 3 and (T[pi][pj + 1] == 0 or
                      (T[pi][pj + 1] == T[pi][pj] and not justcomb)):
        if T[pi][pj + 1] == 0:
            T[pi][pj + 1] = T[pi][pj]
            T[pi][pj] = 0
            pj += 1
        elif T[pi][pj + 1] == T[pi][pj]:
            if score < T[pi][pj]:
                score = T[pi][pj]
            T[pi][pj + 1] += T[pi][pj]
            T[pi][pj] = 0
            pj += 1
            justcomb = True
    return T,score


def randomfill(TABLE):
    # search for zero in the game table
    flatTABLE = np.count_nonzero(TABLE)
    if flatTABLE == 16:
        return TABLE
    empty = False
    w = 0
    while not empty:
        w = randint(0, 15)
        if TABLE[w // 4][w % 4] == 0:
            empty = True
    z = randint(1, 5)
    if z == 5:
        TABLE[w // 4][w % 4] = 4
    else:
        TABLE[w // 4][w % 4] = 2
    return TABLE

def key(DIRECTION, TABLE):
    score = 0
    if DIRECTION == 0: #w
        for pi in range(1, 4):
            for pj in range(4):
                if TABLE[pi][pj] != 0:
                    TABLE,s = moveup(pi, pj, TABLE)
                    if score < s:
                        score = s
    elif DIRECTION == 1: #s
        for pi in range(2, -1, -1):
            for pj in range(4):
                if TABLE[pi][pj] != 0:
                    TABLE,s = movedown(pi, pj, TABLE)
                    if score < s:
                        score = s
    elif DIRECTION == 2: #a
        for pj in range(1, 4):
            for pi in range(4):
                if TABLE[pi][pj] != 0:
                    TABLE,s = moveleft(pi, pj, TABLE)
                    if score < s:
                        score = s
    elif DIRECTION == 3: #d
        for pj in range(2, -1, -1):
            for pi in range(4):
                if TABLE[pi][pj] != 0:
                    TABLE,s = moveright(pi, pj, TABLE)
                    if score < s:
                        score = s

    return TABLE,score


def showtext(TABLE):
    os.system('clear')
    for LINE in TABLE:
        for N in LINE:
            print("%4s" % N, end=' ')
        print("")
    print(score)
    print(reward)
    print(TABLE)

########################################################################
# Parte Grafica
width = 400
height = 400
boxsize = min(width, height) // 4
margin = 5
thickness = 0
STATUS = 0

color1 = (100, 100, 100)
color2 = (200, 200, 200)

dictcolor = {
    0: color2,
    2: (150, 150, 150),
    4: (180, 180, 180),
    8: (255, 200, 200),
    16: (255, 100, 100),
    32: (255, 50, 50),
    64: (255, 0, 0),
    128: (255, 200, 50),
    256: (200, 180, 100),
    512: (150, 150, 150),
    1024: (100, 100, 200),
    2048: (50, 50, 255)}

# Init screen
pygame.init()
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption('Python 2048 by DavidSousaRJ')
myfont = pygame.font.SysFont("Impact", 30)


def gameover(STATUS):
    if STATUS == 1:
        label = myfont.render("You win! :)", 1, (255, 255, 255))
        screen.blit(label, (100, 100))
    elif STATUS == 2:
        label = myfont.render("Game over! :(", 1, (255, 255, 255))
        screen.blit(label, (100, 100))
    pygame.display.update()


def show(TABLE):
    screen.fill(color1)
    for i in range(4):
        for j in range(4):
            pygame.draw.rect(screen, dictcolor[TABLE[i][j]],
                             (j * boxsize + margin,
                              i * boxsize +
                              margin,
                              boxsize - 2 *
                              margin,
                              boxsize - 2 * margin),
                             thickness)
            if TABLE[i][j] != 0:
                label = myfont.render(
                    "%4s" % (int(TABLE[i][j])), 1, (255, 255, 255))
                screen.blit(
                    label, (j * boxsize + 4 * margin,
                            i * boxsize + 5 * margin))
    pygame.display.update()

def createNetwork():
    # network weights
    W_conv1 = weight_variable([4, 4, 4, 32])
    b_conv1 = bias_variable([32])

    W_conv2 = weight_variable([3, 3, 32, 64])
    b_conv2 = bias_variable([64])

    #W_conv3 = weight_variable([3, 3, 64, 64])
    #b_conv3 = bias_variable([64])

    W_fc1 = weight_variable([1024, 256])
    b_fc1 = bias_variable([256])

    W_fc2 = weight_variable([256, 4])
    b_fc2 = bias_variable([4])

    # input layer
    s = tf.placeholder("float", [None,4,4,4])

    # hidden layers
    h_conv1 = tf.nn.relu(conv2d(s, W_conv1,1) + b_conv1) # 4x4x32
    #h_pool1 = max_pool_2x2(h_conv1)

    h_conv2 = tf.nn.relu(conv2d(h_conv1, W_conv2,1) + b_conv2) #4x4x64
    #h_pool2 = max_pool_2x2(h_conv2)

    #h_conv3 = tf.nn.relu(conv2d(h_conv2, W_conv3, 1) + b_conv3)
    #h_pool3 = max_pool_2x2(h_conv3)

    #h_pool3_flat = tf.reshape(h_pool3, [-1, 256])
    h_conv3_flat = tf.reshape(h_conv2, [-1, 1024])

    h_fc1 = tf.nn.relu(tf.matmul(h_conv3_flat, W_fc1) + b_fc1)

    # readout layer
    readout = tf.matmul(h_fc1, W_fc2) + b_fc2

    return s, readout, h_fc1

GAMMA = 0.99 # decay rate of past observations
OBSERVE = 20000. # timesteps to observe before training
EXPLORE = 30000. # frames over which to anneal epsilon
FINAL_EPSILON = 0.0001 # final value of epsilon
INITIAL_EPSILON = 0.9 # starting value of epsilon
REPLAY_MEMORY = 50000 # number of previous transitions to remember
BATCH = 32 # size of minibatch
ACTIONS = 4

stepDelay = 0
log_level = 1
lastscore = 0
score = 0
reward = 0
# paintCanvas
TABLE = randomfill(TABLE)
TABLE = randomfill(TABLE)
show(TABLE)
showtext(TABLE)
running = True
g_terminal = False
win = 2048
lose = -1


sess = tf.InteractiveSession()
s, readout, h_fc1 = createNetwork()

# define the cost function
a = tf.placeholder("float", [None, ACTIONS])
y = tf.placeholder("float", [None])
readout_action = tf.reduce_sum(tf.multiply(readout, a), reduction_indices=1)
cost = tf.reduce_mean(tf.square(y - readout_action))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cost)

# open up a game state to communicate with emulator
# game_state = game.GameState()

# store the previous observations in replay memory
D = deque()

# saving and loading networks
saver = tf.train.Saver()
sess.run(tf.initialize_all_variables())
checkpoint = tf.train.get_checkpoint_state("saved_networkscon")
if checkpoint and checkpoint.model_checkpoint_path:
    saver.restore(sess, checkpoint.model_checkpoint_path)
    print("Successfully loaded:", checkpoint.model_checkpoint_path)
else:
    print("Could not find old network weights")

epsilon = INITIAL_EPSILON
t = 0
s_t = np.stack((TABLE,TABLE,TABLE,TABLE),axis=2)
s_tr = np.stack((TABLE,TABLE,TABLE,TABLE),axis=2)
maxScore = 0

while True:
    #for event in pygame.event.get():
    while True:
        if kbhit():
            akey = ord(getch())
            if akey==49:
                stepDelay = 0
            elif akey==50:
                stepDelay = 0.1
            elif akey==51:
                stepDelay = 0.5
            elif akey==52:
                
                print("quit")
                pygame.quit()
                sys.exit()
        #if event.type == QUIT:
        #    print("quit")
        #    pygame.quit()
        #    sys.exit()
        if 1==1:
            if running:
                pygame.event.pump()

                readout_t = readout.eval(feed_dict={s : [s_t]})[0]
                #a_t = readout_t
                if random.random() <= epsilon:
                    d = randint(0,3)
                    print("---------------random action----------------",d)

                else:
                    d = np.argmax(readout_t)
                a_t = onehot(d)
                #d = randint(0,3)
                #if d == 0:
                if epsilon > FINAL_EPSILON and t > OBSERVE:
                    epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE

                #new_table = key(d, copy.deepcopy(TABLE))


                new_table,reward = key(d, copy.deepcopy(TABLE))
                if not np.array_equal(new_table , TABLE):
                    g_terminal = False
                    TABLE = randomfill(new_table)
                    lastscore = score
                    score = score + reward
                    if reward == 0:
                        reward = 0.1
                    #else:
                    #    reward = score
                    show(TABLE)
                    #showtext(TABLE)
                    STATUS = isgameover(TABLE)
                    if STATUS == 1:
                        reward = win*10
                    elif STATUS == 2:
                        reward = np.max(TABLE)
                        if reward > maxScore:
                            maxScore = reward
                    if STATUS > 0:
                        g_terminal = True
                        running = False
                        gameover(STATUS)
                else:
                    #score = score - 0.1
                    reward = -1
                    g_terminal = True
                    running = False
                    
                s_tr1 = np.stack((new_table, s_tr[:,:,0], s_tr[:,:,1], s_tr[:,:,2]), axis=2)
                s_t1 = np.stack((TABLE, s_t[:,:,0], s_t[:,:,1], s_t[:,:,2]), axis=2)
                D.append((s_t, a_t, reward, s_tr1, g_terminal))
                if len(D) > REPLAY_MEMORY:
                    D.popleft()
                # only train if done observing
                if t > OBSERVE:
                    # sample a minibatch to train on
                    minibatch = random.sample(D, BATCH)

                    # get the batch variables
                    s_j_batch = [d[0] for d in minibatch]
                    a_batch = [d[1] for d in minibatch]
                    r_batch = [d[2] for d in minibatch]
                    s_j1_batch = [d[3] for d in minibatch]

                    y_batch = []
                    readout_j1_batch = readout.eval(feed_dict = {s : s_j1_batch})
                    for i in range(0, len(minibatch)):
                        terminal = minibatch[i][4]
                        # if terminal, only equals reward
                        if terminal:
                            y_batch.append(r_batch[i])
                        else:
                            y_batch.append(r_batch[i] + GAMMA * np.max(readout_j1_batch[i]))

                    # perform gradient step

                    train_step.run(feed_dict = {
                        y : y_batch,
                        a : a_batch,
                        s : s_j_batch}
                    )

                # update the old values
                s_tr = s_tr1
                s_t = s_t1
                t += 1

                # save progress every 10000 iterations
                if t % 10000 == 0:
                    saver.save(sess, 'saved_networkscon/py2048', global_step = t)

                if stepDelay > 0:
                    time.sleep(stepDelay)
                # print info
                if t % log_level == 0:
                    state = ""
                    if t <= OBSERVE:
                        state = "observe"
                    elif t > OBSERVE and t <= OBSERVE + EXPLORE:
                        state = "explore"
                    else:
                        state = "train"

                    print("TIMESTEP", t, "/ STATE", state, \
                        "/ EPSILON", epsilon, "/ ACTION", d, "/ HI-score", maxScore, "/ Q_MAX %e" % np.max(readout_t), "/ REWARD", reward)

            else:
                TABLE = np.zeros([4,4])
                lastscore = 0
                score = 0
                reward = 0
                # paintCanvas
                TABLE = randomfill(TABLE)
                TABLE = randomfill(TABLE)
                show(TABLE)
                #showtext(TABLE)
                running = True
                g_terminal = False
                s_t = np.stack((TABLE,TABLE,TABLE,TABLE),axis=2)
                s_tr = np.stack((TABLE,TABLE,TABLE,TABLE),axis=2)
                # if d == 1:
                #     TABLE,reward = key('s', TABLE)
                #     TABLE = randomfill(TABLE)
                #     lastscore = score
                #     score = score + reward
                #     if reward == 0:
                #         reward = 0.1
                #     show(TABLE)
                #     showtext(TABLE)
                #     STATUS = isgameover(TABLE)
                #     if STATUS > 0:
                #         running = False
                #         gameover(STATUS)
                # if d == 2:
                #     TABLE,reward = key('a', TABLE)
                #     TABLE = randomfill(TABLE)
                #     lastscore = score
                #     score = score + reward
                #     if reward == 0:
                #         reward = 0.1
                #     show(TABLE)
                #     showtext(TABLE)
                #     STATUS = isgameover(TABLE)
                #     if STATUS > 0:
                #         running = False
                #         gameover(STATUS)
                # if d == 3:
                #     TABLE,reward = key('d', TABLE)
                #     TABLE = randomfill(TABLE)
                #     lastscore = score
                #     score = score + reward
                #     if reward == 0:
                #         reward = 0.1
                #     show(TABLE)
                #     showtext(TABLE)
                #     STATUS = isgameover(TABLE)
                #     if STATUS > 0:
                #         running = False
                #         gameover(STATUS)

# end
