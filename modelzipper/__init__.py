from termcolor import colored  
from datetime import datetime
from .tutils import *
import pytz

utc_now = datetime.utcnow()
aoe_tz = pytz.timezone('Pacific/Kwajalein')
aoe_now = utc_now.replace(tzinfo=pytz.utc).astimezone(aoe_tz)
aoe_time_str = aoe_now.strftime('%Y-%m-%d %H:%M:%S')

print(colored('ModelZipper package is already loaded, status: >>> ready <<<' + ' (AOE time: ' + aoe_time_str + ')', 'cyan', attrs=['blink']))