import os, sys
try:                                            # if running in CLI
    cur_path = os.path.abspath(__file__)
except NameError:                               # if running in IDE
    cur_path = os.getcwd()

while cur_path.split('/')[-1] != 'ufc':
    cur_path = os.path.abspath(os.path.join(cur_path, os.pardir))    
sys.path.insert(1, os.path.join(cur_path, 'lib', 'python3.7', 'site-packages'))

from lxml import html
import _config
import _connections
import unicodedata
import json
from random import randint
from time import sleep
#from selenium.common.exceptions import ElementNotVisibleException, StaleElementReferenceException



browser = _connections.sel_scraper(headless = False)


url = 'https://www.oddsshark.com/ufc/events'
browser.get(url) 
article_tree = html.fromstring(browser.page_source)

article_tree.xpath('//*[@id="oslive-scoreboard"]/div/div/div/div[2]/div[1]/div/a/text()')
article_tree.xpath('//*[@id="oslive-scoreboard"]/div/div/div/div[5]/div[2]/div[1]/text()')


article_tree.xpath('//*[@id="oslive-scoreboard"]/div/div[1]/div/div[2]/div[2]/div/a/text()')
