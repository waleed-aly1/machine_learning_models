import datetime
import os
from pytz import timezone


class LogFileHandler():

    def __init__(self, type):
        cwd = os.getcwd()
        self.timeZone = timezone('UTC')
        dt = datetime.datetime.now(self.timeZone)
        now = datetime.datetime(year=dt.year, month=dt.month, day=dt.day, hour=dt.hour, minute=dt.minute,
                                second=dt.second)
        path = cwd + '/logs/' + type + ' ' + now.date().__str__().replace('-', '')
        try:
            os.makedirs(path)
        except:
            pass

        self.logFile = open(path + '/Log.txt', 'w', encoding="utf-8")
        self.logFile.write(
            '******************************************************run ' + now.__str__() + ' started******************************************************' + '\n')
        self.logFile.flush()
        os.fsync(self.logFile.fileno())

    def write(self, message):
        dt = datetime.datetime.now(self.timeZone)
        now = datetime.datetime(year=dt.year, month=dt.month, day=dt.day, hour=dt.hour, minute=dt.minute,
                                second=dt.second)

        self.logFile.write('@' + now.time().__str__() + ' ' + message.__str__() + '\n')
        self.logFile.flush()
        os.fsync(self.logFile.fileno())
