

import time


class runtime:

    def __init__(self):

        self.start = time.time()
        pass

    def touch(self):

        self.stop = time.time()
        self.delta = round(self.stop - self.start, 1)
        print('Use {} second.'.format(self.delta))
        pass
    
    pass

# start_time = time.time()
# main()
# print("--- %s seconds ---" % (time.time() - start_time))