import os, sys, pygame
from random import randint
import random, copy
from collections import defaultdict
from tensorforce.agents import DQNAgent
import tensorflow as tf
import warnings
warnings.filterwarnings('ignore')

class StackList(object):
    def __init__(self):
        self._data = []

    def add(self, level):
        if len(level):
            self._data.insert(0, level)

    def pop(self):
        if not self._data:
            return None
        item = self._data[0].pop()
        if len(self._data[0]) == 0:
            self._data.pop(0)
        return item

class Rectangulo(pygame.sprite.Sprite):
    def __init__(self,pos,color):
        pygame.sprite.Sprite.__init__(self)
        self.image = pygame.Surface((16, 16)).convert()
        self.image.fill(color)
        self.rect = self.image.get_rect(center=pos)

class Maze(pygame.sprite.Sprite):
    def __init__(self, screen, width=21, height=21, exit_cell=(1, 1)):
        pygame.sprite.Sprite.__init__(self)
        if width % 2 == 0:
            width += 1
        if height % 2 == 0:
            height += 1
        self.width = width
        self.height = height
        self.start_cell = None
        self.exit_cell = exit_cell
        self.steps = 0
        self.maze = None
        self.screen = screen
        self.create()

    def create(self):
        self.maze = [[1, ] * self.width  for _ in range(self.height)]
        visited = []
        stack = StackList()
        item = (None, self.exit_cell)
        while item is not None:
            prev, cell = item
            if cell not in visited:
                x, y = cell
                self.maze[y][x] = 0
                if prev and cell:
                    self._remove_wall(prev, cell)
                neighbors = [x for x in self._get_neighbors(cell) if x not in visited]
                random.shuffle(neighbors)
                stack.add([(cell, x) for x in neighbors])
                visited.append(cell)
            item = stack.pop()

    def __str__(self):
        res = ''
        for row in self.maze:
            for col in row:
                res += col and '##' or '  '
            res += '\n'
        return res

    def _get_neighbors(self, cell):
        x, y = cell
        neighbors = []

        # Left
        if x - 2 > 0:
            neighbors.append((x-2, y))
        # Right
        if x + 2 < self.width:
            neighbors.append((x+2, y))
        # Up
        if y - 2 > 0:
            neighbors.append((x, y-2))
        # Down
        if y + 2 < self.height:
            neighbors.append((x, y+2))

        return neighbors

    def _remove_wall(self, cell, neighbor):
        """
        Remove the wall between two cells
        Example:
          Given the cells a and b
          The wall between them is w
          # # # # #
          # # # # #
          # a w b #
          # # # # #
          # # # # #
        """
        x0, y0 = cell
        x1, y1 = neighbor
        # Vertical
        if x0 == x1:
            x = x0
            y = int((y0 + y1) / 2)
        # Horizontal
        if y0 == y1:
            x = int((x0 + x1) / 2)
            y = y0

        # remove walls
        self.maze[y][x] = 0

        self.start_cell = neighbor
        self.steps += 2

    def _update_start_cell(self, cell, depth):
        if depth > self.recursion_depth:
            self.start_cell = cell
            self.steps = depth * 2 # wall + cell

    def crearRectangulos(self):
        self.lista_rect=list()
        for x in range(len(self.maze[0])):
            for y in range(len(self.maze)):
                if(self.maze[y][x]==1):
                    self.lista_rect.append(Rectangulo((16*x+1,16*y+1),(255,255,255)))
                elif(self.maze[y][x]==2):
                    self.lista_rect.append(Rectangulo((16*x+1,16*y+1),(0,255,0)))

    def buscarColisiones(self, x_ball, y_ball):
        if(self.maze[y_ball][x_ball]==1):
            return 1
        elif(self.maze[y_ball][x_ball]==2):
            return 2
        return 0

    def show(self, verbose=False):
        sprites=pygame.sprite.Group()
        for rectangulo in self.lista_rect:
            sprites.add(rectangulo)
            sprites.draw(self.screen)

    def showTerminal(self):
        for x in range(len(self.maze[0])):
            print(self.maze[x])

    def ponerMetaEn(self, x, y):
        self.maze[y][x]=2

class MyDQNAgent(object):
    def __init__(self,states,actions):
        self.q_agent = DQNAgent(states, actions,
            network = [dict(type='conv2d', size=3),
                dict(type='flatten'),
                dict(type='dense', size=32),
                dict(type='dense', size=32)],
            update_mode=dict(
                unit='timesteps',
                batch_size=8,
                frequency=8
            ),
            memory=dict(
                type='replay',
                include_next_states=True,
                capacity=100
            ),
            optimizer=dict(
                type='adam',
                learning_rate=1e-2
            ),
            states_preprocessing=[
                dict(type='normalize'),
                #dict(type='running_standardize'),
                #dict(type='sequence')
            ],
            target_sync_frequency=10,
            actions_exploration=dict(
                 type="epsilon_decay",
                 initial_epsilon=1.0,
                 final_epsilon=0.1,
                 timesteps=10
            )
        )


class Ball(pygame.sprite.Sprite):
    def __init__(self, mapa, pos_inicial, pos=(0, 0)):
        pygame.sprite.Sprite.__init__(self)
        self.pos = pos
        self.image = pygame.Surface((15, 15)).convert()
        self.image.fill((255, 0, 0))
        self.rect = self.image.get_rect(center=self.pos)
        self.speed_x = 0
        self.speed_y = 0
        self.last_move = 4
        self.map_x = pos_inicial[0]
        self.map_y = pos_inicial[1]
        self.prev_x = 1
        self.prev_y = 1
        tamaño = (len(mapa[0]),len(mapa))

        states = dict(type='float',shape=(2,tamaño[0],tamaño[1]))
        #states = dict( type='float',shape=(tamaño[0]*tamaño[1]+1,) )
        actions = dict( type='int',shape=(1,),num_actions=5 )
        myAgent = MyDQNAgent(states,actions)
        self.q_agent =  myAgent.q_agent

        self.mapa = copy.deepcopy(mapa)
        self.mapa[self.map_y][self.map_x] = 3

    def actualizarMapa(self,valido):
        if (self.last_move==0):
            self.mapa[self.map_y+1][self.map_x]=valido
        elif (self.last_move==1):
            self.mapa[self.map_y][self.map_x+1]=valido
        elif (self.last_move==2):
            self.mapa[self.map_y-1][self.map_x]=valido
        elif (self.last_move==3):
            self.mapa[self.map_y][self.map_x-1]=valido
        self.mapa[self.map_y][self.map_x]=3

    def reset(self, map_x, map_y, mapa):
        self.pos = (map_x*16,map_y*16)
        self.rect = self.image.get_rect(center=self.pos)
        self.last_move = 4
        self.prev_x = 1
        self.prev_y = 1
        self.speed_x = 0
        self.speed_y = 0
        self.mapa = copy.deepcopy(mapa)
        self.map_x = map_x
        self.map_y = map_y
        self.mapa[self.map_y][self.map_x]=3

    def update(self):
        self.rect.move_ip(self.speed_x, self.speed_y)
        self.speed_x = 0
        self.speed_y = 0

    def neg_y(self,valido):
        #print("Arriba")
        self.speed_y += -16
        self.last_move = 0
        self.prev_y = self.map_y
        self.map_y += -1
        self.actualizarMapa(valido)
    def neg_x(self,valido):
        #print("Izquierda")
        self.speed_x += -16
        self.last_move = 1
        self.prev_x = self.map_x
        self.map_x += -1
        self.actualizarMapa(valido)
    def pos_y(self,valido):
        #print("Abajo")
        self.speed_y += 16
        self.last_move = 2
        self.prev_y = self.map_y
        self.map_y += 1
        self.actualizarMapa(valido)
    def pos_x(self,valido):
        #print("Derecha")
        self.speed_x += 16
        self.last_move = 3
        self.prev_x = self.map_x
        self.map_x += 1
        self.actualizarMapa(valido)

    def move_back(self):
        if(self.last_move==0):
            self.pos_y(1)
        elif(self.last_move==1):
            self.pos_x(1)
        elif(self.last_move==2):
            self.neg_y(1)
        elif(self.last_move==3):
            self.neg_x(1)

    def stop(self):
        self.speed_x = 0
        self.speed_y = 0

    def moverse(self, actions):
        if (actions==0):
            self.pos_x(0)
        elif (actions==1):
            self.pos_y(0)
        elif (actions==2):
            self.neg_x(0)
        elif (actions==3):
            self.neg_y(0)
        elif (actions==4):
            self.last_move=4
            self.prev_x = self.map_x
            self.prev_y = self.map_y



def main():
    size = width, height = 300, 300
    screen = pygame.display.set_mode(size)
    pygame.display.set_caption('Maze')
    meta = (3,3)
    tamaño = (7,7)
    pos_inicial = (1,1)
    try:
        filename = os.path.join(
	    os.path.dirname(__file__),
	    'assets',
	    'graphics',
	    'fondo.jpeg')
        background = pygame.image.load(filename)
        background = background.convert()
    except pygame.error as e:
        print ('Cannot load image: ', filename)
        raise SystemExit(str(e))

    maze = Maze(screen, tamaño[0], tamaño[1])
    maze.ponerMetaEn(meta[0],meta[1])
    maze.crearRectangulos()

    # Insertamos esta linea aqui
    ball = Ball(maze.maze,pos_inicial,(16,16))

    sprites = pygame.sprite.Group(ball)

    clock = pygame.time.Clock()
    fps = 60

    pygame.key.set_repeat(1, int(1000/fps))
    maze.showTerminal()

    stay = 1
    num_pasos = 0
#    flat_mapa = []
    num_choques = 0
    mapa_anterior = copy.deepcopy(ball.mapa)
    ball.q_agent.restore_model("./model")

    while stay:
        clock.tick(fps)
        ball.stop()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                 pygame.quit()
                 return
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                 stay=0

        num_pasos += 1
        #for i in range(len(ball.mapa)):
        #    for j in range(len(ball.mapa[0])):
        #        flat_mapa.append(ball.mapa[i][j])
        #flat_mapa.append(num_pasos)

        #actions = ball.q_agent.act(flat_mapa, deterministic=False)
        actions = ball.q_agent.act([mapa_anterior, ball.mapa], deterministic=False)
        #flat_mapa = []
        mapa_anterior = copy.deepcopy(ball.mapa)
        ball.moverse(actions)
        print(actions[0])

        if (maze.buscarColisiones(ball.map_x, ball.map_y)==1):
            ball.move_back()
            num_choques += 1
        elif(maze.buscarColisiones(ball.map_x, ball.map_y)==2):
            stay=0

        screen.blit(background,(0, 0))
        sprites.update()
        sprites.draw(screen)
        maze.show()
        pygame.display.flip()

        #input()

    print("Numero de pasos: %d" %num_pasos)
    print("Numero de choques: %d" %num_choques)
    pygame.quit()

if __name__ == '__main__':
    main()
