##import mandel
##mandel.run()

##import ftree
##ftree.run()


##import polytopes
##polytopes.example()


##import rsphere
#cc#rsphere.run()




if __name__ == "__main__":
    import chessai
    chessai.standard.run()



##import multiprocessing
##import os
##
##
##class Process(multiprocessing.Process):
##    def __init__(self, error_queue, *args, **kwargs):
##        multiprocessing.Process.__init__(self, *args, **kwargs)
##        self.error_queue = error_queue
##    def run(self):
##        import traceback
##        try:
##            super().run()
##        except Exception as e:
##            self.error_queue.put(f"error in process pid={os.getpid()} with parent={os.getppid()}\n" + "".join(traceback.format_exception(None, e, e.__traceback__)))
##            raise e
##        return
##
##
##def info(title):
##    print(title)
##    print('module name:', __name__)
##    print('parent process:', os.getppid())
##    print('process id:', os.getpid())
##
##def f(name):
##    info('function f')
##    print('hello', name)
##    0/0
##
##if __name__ == '__main__':
##    info('main line')
##    error_queue = multiprocessing.Queue()
##    p = Process(error_queue, target=f, args=('bob',))
##    p.start()
##    p.join()
##    print("bloo", p)
##
##    while not error_queue.empty():
##        e = error_queue.get()
##        raise Exception(e)
