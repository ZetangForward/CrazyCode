from termcolor import colored  
from datetime import datetime
from .tutils import *
import pytz

__version__ = '0.2.6'

utc_now = datetime.utcnow()
aoe_tz = pytz.timezone('Pacific/Kwajalein')
aoe_now = utc_now.replace(tzinfo=pytz.utc).astimezone(aoe_tz)
aoe_time_str = aoe_now.strftime('%Y-%m-%d %H:%M:%S')

print(colored(f'ModelZipper is ready for launch🚀 | Current Version🦄 >>> {__version__} <<< | AOE Time🕒 {aoe_time_str}', 'cyan', attrs=['blink']))