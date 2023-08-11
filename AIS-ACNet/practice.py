import sys
import time
from termcolor import colored


for i in range(5):

    #덮어쓰기를 위해 '\r'이 필요하다.

    sys.stdout.write(colored("\r {} sec".format(str(i)), 'blue'))

#     sys.stdout.flush()
    time.sleep(1)