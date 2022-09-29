import pygame
import moderngl
import sys
import warnings

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
        self.is_active = True

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

    def end(self):
        assert self.is_active
        self.is_active = False
        
    def __del__(self):
        assert not self.is_active


class ExitException(Exception):
        pass


RUN_ROOT_COUNT = 0
def run(window):
    if RUN_ROOT_COUNT == 0:
        warnings.warn("run has been called before run_root. It should only be called after run_root for starting sub-applications.", RuntimeWarning)

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
    window.end()


def run_root(window):
    global RUN_ROOT_COUNT
    if RUN_ROOT_COUNT > 0:
        warnings.warn("run_root has been called already. It should only be called once when the application is being started.", RuntimeWarning)
    RUN_ROOT_COUNT += 1
    run(window)
    Window.quit()
    




