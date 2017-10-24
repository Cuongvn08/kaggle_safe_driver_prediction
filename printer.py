from datetime import datetime
import pytz
tz = tz=pytz.timezone('Asia/Seoul')


def print_log(message, print_enable = True):
    date_time = datetime.now(tz).strftime('%Y-%m-%d %H:%M:%S')

    # print to console
    if print_enable is True:
        print(date_time, message)
    
    # log to file
    log_file = 'log.txt'
    logger = open(log_file, 'a')    
    logger.write(date_time + ': ' + message + '\n')
    logger.flush()
    logger.close()
    