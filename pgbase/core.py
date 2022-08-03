import pygame
import moderngl
import sys

class Window():
    screen = None
    ctx = None
    
    #should call this once before running anything
    #should not be called while running
    #so ctx and screen should be fixed in practice
    @classmethod
    def setup(cls, size = [2000, 1600]):
        if size is None:
            screen = pygame.display.set_mode(flags = pygame.OPENGL | pygame.DOUBLEBUF | pygame.FULLSCREEN)
        else:
            screen = pygame.display.set_mode(size, flags = pygame.OPENGL | pygame.DOUBLEBUF)
        pygame.init()
        ctx = moderngl.create_context()

        cls.screen = screen
        cls.ctx = ctx

    @classmethod
    def quit(cls):
        cls.ctx.release()
        pygame.quit()
        sys.exit()
    
    def __init__(self, rect = None):
        assert not type(self).screen is None
        assert not type(self).ctx is None
        if rect is None:
            rect = [0, 0, self.screen.get_width(), self.screen.get_height()]
        self.set_rect(rect)

    def set_rect(self, rect):
        self.rect = tuple(rect)

    @property
    def width(self):
        return self.rect[2]
    @property
    def height(self):
        return self.rect[3]

    def tick(self, dt):
        pass
    def draw(self):
        pass
    def event(self, event):
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                raise ExitException()



class ExitException(Exception):
        pass

    

def run(window):
    w, h = window.screen.get_size()
    window.set_rect([0, 0, w, h])
    clock = pygame.time.Clock()
    try:
        while True:
            window.tick(1 / 60)
            
            for event in pygame.event.get():
                window.event(event)
                if event.type == pygame.QUIT:
                    raise ExitException()
                
            clock.tick(60)
            window.draw()
            pygame.display.flip()
    except ExitException as e:
        pass
    except Exception as e: 
        Window.quit()
        raise e




