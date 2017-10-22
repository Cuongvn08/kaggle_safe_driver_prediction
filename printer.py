from datetime import datetime
import pytz
tz = tz=pytz.timezone('Asia/Seoul')


def print_log(message):    
    date_time = datetime.now(tz).strftime('%Y-%m-%d %H:%M:%S')

    # print to console
    print(date_time, message)
    
    # log to file
    log_file = 'log.txt'
    logger = open(log_file, 'a')    
    logger.write(date_time + ': ' + message + '\n')
    logger.flush()
    logger.close()
    

class Printer():
    def __init__(self, title):
        time = datetime.now(tz).strftime('%Y%m%d_%H%M%S')
        logger_path = 'Logger_' + str(time) + '.txt'
        self.logger = open(logger_path, 'w')
        self.logger.write('*** ' + title.upper() + ' ***' + '\n\n')
        self.logger.flush()        
        
    def print(self, message):
        date_time = datetime.now(tz).strftime('%Y-%m-%d %H:%M:%S')
        print(date_time, message)
        self.logger.write(date_time + ' ' + message + '\n')
        self.logger.flush()        
        

        