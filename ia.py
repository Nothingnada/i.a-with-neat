import pygame
from pygame.locals import *
import random
import neat
import os

pygame.init()

tela = pygame.display.set_mode((600,400))
pygame.display.set_caption('ia')

class per:
    pers=None
    x=None
    y=None
    c=None
    def __init__(self,x,y,c):
        self.x=x
        self.y=y
        self.c=c

    def desenhar(self):
        self.pers=pygame.draw.rect(tela,self.c,(self.x,self.y,40,40))
    def colidir(self,col):
        if self.pers.colliderect(col):
            return True


font = pygame.font.SysFont('arial',40,True,True)

perso = per(100, 100,(255,255,255))

redes = []
lista_genomas = []
inimy = []
geracao=0
tempo=999

clock = pygame.time.Clock()
def main(genomas,config):
    global x,y,xi,yi,per,geracao,tempo
    game=True
    geracao+=1
    tempo=999
    x=100
    y=100
    for _, genoma in genomas:
        rede = neat.nn.FeedForwardNetwork.create(genoma,config)
        redes.append(rede)
        genoma.fitness = 0
        lista_genomas.append(genoma)
        inimy.append(per(100,100,(255,0,0)))

    while game:
        clock.tick(50)
        tela.fill((0,0,0))
        msg=f'Geracao: {geracao}'
        

        for ev in pygame.event.get():
            if ev.type == QUIT:
                pygame.quit()
                quit()
            if pygame.key.get_pressed()[K_a]:
                perso.x-=5
            if pygame.key.get_pressed()[K_d]:
                perso.x+=5
            if pygame.key.get_pressed()[K_w]:
                perso.y-=5
            if pygame.key.get_pressed()[K_s]:
                perso.y+=5
            if pygame.key.get_pressed()[K_q]:
                for i,ini in enumerate(inimy):
                    inimy.pop(i)
                    lista_genomas[i].fitness -= 1000
                    lista_genomas.pop(i)
                    redes.pop(i)
                
            
            
        for i,ini in enumerate(inimy):
            output = redes[i].activate((inimy[i].x-perso.x,inimy[i].y-perso.y))
            #print("1:",output[0])
            #print("2:",output[1])
            if output[0] > 0.5:
                ini.x+=1
                lista_genomas[i].fitness += 0.1
            if output[1] > 0.5:
                ini.y+=1
                lista_genomas[i].fitness += 0.1
            if output[0] < 0.5:
                ini.x-=1
                lista_genomas[i].fitness += 0.1
            if output[1] < 0.5:
                ini.y-=1
                lista_genomas[i].fitness += 0.1

        
        for i,ini in enumerate(inimy):
            if ini.x >= 600:
                inimy.pop(i)
                lista_genomas[i].fitness -= 5
                lista_genomas.pop(i)
                redes.pop(i)
                    
            elif ini.x <= 0:
                inimy.pop(i)
                lista_genomas[i].fitness -= 5
                lista_genomas.pop(i)
                redes.pop(i)
                    
            elif ini.y <= 0:
                inimy.pop(i)
                lista_genomas[i].fitness -= 5
                lista_genomas.pop(i)
                redes.pop(i)
                    
            elif ini.y >= 400:
                inimy.pop(i)
                lista_genomas[i].fitness -= 5
                lista_genomas.pop(i)
                redes.pop(i)
                    
       
        if len(inimy) <= 0:
            game=False

        txt=font.render(msg, True,(255,255,255))

        tela.blit(txt,(0,0))

        perso.desenhar()

        

        for i,ini in enumerate(inimy):
            ini.desenhar()

        for i,ini in enumerate(inimy):
            if perso.colidir(ini.pers):
                perso.x = random.randint(100,300)
                perso.y = random.randint(100,300)
                lista_genomas[i].fitness+=100
                tempo+=1000
        if tempo<=0:
            for i,ini in enumerate(inimy):
                    inimy.pop(i)
                    lista_genomas[i].fitness -= 200
                    lista_genomas.pop(i)
                    redes.pop(i)

        tempo-=5
        pygame.display.flip()

def rodar(caminho_config):
    config = neat.config.Config(neat.DefaultGenome,
                                neat.DefaultReproduction,
                                neat.DefaultSpeciesSet,
                                neat.DefaultStagnation,
                                caminho_config)
    populacao = neat.Population(config)
    populacao.add_reporter(neat.StdOutReporter(True))
    populacao.add_reporter(neat.StatisticsReporter())

    
    populacao.run(main)
    


if __name__ == '__main__':
    caminho_config = 'config.txt'
    rodar(caminho_config)
