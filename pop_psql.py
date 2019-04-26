import os, sys
try:                                            # if running in CLI
    cur_path = os.path.abspath(__file__)
except NameError:                               # if running in IDE
    cur_path = os.getcwd()

while cur_path.split('/')[-1] != 'ufc':
    cur_path = os.path.abspath(os.path.join(cur_path, os.pardir))    
sys.path.insert(1, os.path.join(cur_path, 'lib', 'python3.7', 'site-packages'))


import requests
from lxml import html
from _connections import db_connection
from pg_tables import create_tables
import pandas as pd
from datetime import datetime

def pg_create_table(cur, table_name):  
#    cur, table_name = _psql, 
    try:
        # Truncate the table first
        for script in create_tables[table_name]:
            cur.execute(script)
            cur.execute("commit;")
        print("Created {}".format(table_name))
        
    except Exception as e:
        print("Error: {}".format(str(e)))
        
        
def pg_query(cur, query):
    cur.execute(query)
    data = pd.DataFrame(cur.fetchall())
    return(data) 
    
def pg_insert(cur, script):
    try:
        cur.execute(script)
        cur.execute("commit;")
        
    except Exception as e:
        print("Error: {}".format(str(e)))
        raise(Exception)
        
    

PSQL = db_connection('psql')

def pop_corners():
    all_fights = pg_query(PSQL.client, 'select fight_id, name, date from ufc.fights')
    all_fights.columns = ['fight_id', 'fight_name', 'fight_date']
    all_fighters = pg_query(PSQL.client, "select fighter_id, name from ufc.fighters;")
    all_fighters = {j:i for i,j in all_fighters.values}
    all_corners = pg_query(PSQL.client, "select bout_id, red_corner, blue_corner from ufc.bout_corners")
    all_corners = {i:{'red_corner': j, 'blue_corner': k} for i,j,k in all_corners.values}
    all_bouts = pg_query(PSQL.client, "select bout_id, fight_id from ufc.bouts")
    all_bouts.set_index(0, inplace = True)
    for bout in all_corners.keys():
        all_bouts.drop(bout, axis = 0, inplace = True)
    missing_fights = all_bouts[1].unique()
    total_fights = all_fights['fight_id'].values
    all_fights.set_index('fight_id', inplace = True)
    for fight in all_fights.index:
        if fight not in missing_fights:
            all_fights.drop(fight, axis = 0, inplace = True)
    all_fights.reset_index(inplace = True)
    
    cities = pg_query(PSQL.client, 'select fight_id, city_name from ufc.fights f join ufc.cities c on f.city_id = c.city_id')
    cities = {i:j for i,j in cities.values}
    
    countries = pg_query(PSQL.client, 'select fight_id, country_name from ufc.fights f join ufc.countries c on f.country_id = c.country_id')
    countries = {i:j for i,j in countries.values}
    for fight_id, fight_name, fight_date in all_fights.values[2:]:
        if 'Fight Night' in fight_name:
            if fight_id == '6546af7ab545b90c':
                url = 'https://www.ufc.com/event/ufc-fight-night-czech-republic-2019'
                page = requests.get(url)
            elif fight_id == '84283233ec42be5f':
                url = 'https://www.ufc.com/event/ufc-fight-night-brazil-2019'
                page = requests.get(url)
            elif fight_id == 'de25520d54eab12d':
                url = 'https://www.ufc.com/event/ufc-china-2018'
                page = requests.get(url)
            elif fight_id == 'aa3153a9941b4d44':
                url = 'https://www.ufc.com/event/ufc-south-america-2018'
                page = requests.get(url)
            else:
                try:
                    url = 'https://www.ufc.com/event/ufc-fight-night-%s-%i-%i' % (fight_date.strftime("%B"), int(fight_date.strftime("%d")), int(fight_date.strftime("%Y")))
                    page = requests.get(url)
                    if page.status_code == 404:
                        asdfasdf
    
    #            if fight_id == '80eacd4da0617c57':
    #                url = 'https://www.ufc.com/event/ufc-fight-night-london-2019'
    #            if fight_id == '6546af7ab545b90c':
    #                url = 'https://www.ufc.com/event/ufc-fight-night-czech-republic-2019'
    #            if fight_id == 'a7a79b8efbceaaac':
    #                url = 'https://www.ufc.com/event/ufc-fight-night-phoenix-2019'
                except:
                    try:
                        url = 'https://www.ufc.com/event/ufc-fight-night-%s-%i' % (cities[fight_id], int(fight_date.strftime("%Y")))
                        page = requests.get(url)
                        if page.status_code == 404:
                            asdfasdf
                    except:
                        try:
                            url = 'https://www.ufc.com/event/ufc-fight-night-%s-%s-%i-%i' % (cities[fight_id], fight_date.strftime("%b"), int(fight_date.strftime("%d")), int(fight_date.strftime("%Y")))
                            page = requests.get(url)
                            if page.status_code == 404:
                                asdfasdf    
                        except:
                            try:
                                url = 'https://www.ufc.com/event/ufc-%s-%i' % (cities[fight_id], int(fight_date.strftime("%Y")))
                                page = requests.get(url)
                                if page.status_code == 404:
                                    asdfasdf
                            except:
                                url = 'https://www.ufc.com/event/ufc-%s-%i' % (countries[fight_id], int(fight_date.strftime("%Y")))
                                page = requests.get(url)
                                if page.status_code == 404:
                                    print(fight_name)
                                    asdfasdf
        else:
            url = 'https://www.ufc.com/event/%s' % (fight_name.split(':')[0].replace(' ', '-'))
            page = requests.get(url)
            if page.status_code == 404:
                print(fight_name)
                continue
                
        tree = html.fromstring(page.content)
        reds = tree.xpath('//*[@id="edit-group-main-card"]/div/div/section/ul/li/div/div/div/div[2]/div[4]/text()')
        reds = [i.strip() for i in reds]
        blues = tree.xpath('//*[@id="edit-group-main-card"]/div/div/section/ul/li/div/div/div/div[2]/div[6]/text()')
        blues = [i.strip() for i in blues]
        
        for red, blue in zip(reds, blues):
            if red not in all_fighters.keys():
                print(red)
                continue
            if blue not in all_fighters.keys():
                print(blue)
                continue
            red_corner = all_fighters[red]
            blue_corner = all_fighters[blue]
            
            bout_id = pg_query(PSQL.client, "select bx.bout_id from ufc.bout_fighter_xref bx join ufc.bouts b on b.bout_id = bx.bout_id where fight_id = '%s' and fighter_id = '%s' and opponent_id = '%s'" % (fight_id, red_corner, blue_corner))
            try:
                bout_id = bout_id[0].values[0]
            except:
                continue
            
            if bout_id in all_corners.keys():
                continue
            script = "INSERT INTO ufc.bout_corners(\
                        bout_id, red_corner, blue_corner)\
                        VALUES ('%s', '%s', '%s');" % (bout_id, red_corner, blue_corner)
            pg_insert(PSQL.client, script) 
            all_corners[bout_id] = {'red_corner': red_corner, 'blue_corner': blue_corner}
        
        
        
#def pop_bout_fighter_xref():
#    bout_data = pg_query(PSQL.client, 'select bout_id, winner, loser from ufc.bout_results;')
#    bout_data.rename(columns = {0: 'bout_id', 1: 'winner', 2: 'loser'}, inplace = True)
#    for bout, winner, loser in bout_data.values:
#        script = "INSERT INTO ufc.bout_fighter_xref(\
#                    bout_id, fighter_id, opponent_id)\
#                    	VALUES ('%s', '%s', '%s');" % (bout, winner, loser)
#        pg_insert(PSQL.client, script) 
#        script = "INSERT INTO ufc.bout_fighter_xref(\
#                    bout_id, fighter_id, opponent_id)\
#                    	VALUES ('%s', '%s', '%s');" % (bout, loser, winner)
#        pg_insert(PSQL.client, script) 


def pop_bout_stats():
    all_bouts = pg_query(PSQL.client, 'select b.bout_id from ufc.bouts b full join ufc.bout_stats bs on bs.bout_id = b.bout_id where bs.bout_id is NULL')
    all_fighters = pg_query(PSQL.client, "select fighter_id from ufc.fighters;")
    all_fighters = set([i[0] for i in all_fighters.values]) 
    for bout in all_bouts.values:
        url = 'http://www.ufcstats.com/fight-details/%s' % (bout[0])
        page = requests.get(url)
        tree = html.fromstring(page.content)   
        
        kd = tree.xpath('/html/body/section/div/div/section[2]/table/tbody/tr/td[2]/p/text()')
        kd = [int(i.strip()) for i in kd]
        
        ss = tree.xpath('/html/body/section/div/div/section[2]/table/tbody/tr/td[3]/p/text()')
        ss = [i.strip() for i in ss]  
        ssa = [int(i.split(' of ')[1]) for i in ss]
        sss = [int(i.split(' of ')[0]) for i in ss]
    
        ts = tree.xpath('/html/body/section/div/div/section[2]/table/tbody/tr/td[5]/p/text()')
        ts = [i.strip() for i in ts]  
        tsa = [int(i.split(' of ')[1]) for i in ts]
        tss = [int(i.split(' of ')[0]) for i in ts]
        
        td = tree.xpath('/html/body/section/div/div/section[2]/table/tbody/tr/td[6]/p/text()')
        td = [i.strip() for i in td]  
        tda = [int(i.split(' of ')[1]) for i in td]
        tds = [int(i.split(' of ')[0]) for i in td]   
        
        sub = tree.xpath('/html/body/section/div/div/section[2]/table/tbody/tr/td[8]/p/text()')
        sub = [i.strip() for i in sub]  
    
        pas = tree.xpath('/html/body/section/div/div/section[2]/table/tbody/tr/td[9]/p/text()')
        pas = [i.strip() for i in pas] 
        
        rev = tree.xpath('/html/body/section/div/div/section[2]/table/tbody/tr/td[9]/p/text()')
        rev = [i.strip() for i in rev] 
        
        headss = tree.xpath('/html/body/section/div/div/table/tbody/tr/td[4]/p/text()')
        headss = [i.strip() for i in headss]  
        headssa = [int(i.split(' of ')[1]) for i in headss]
        headsss = [int(i.split(' of ')[0]) for i in headss]
        
        bodyss = tree.xpath('/html/body/section/div/div/table/tbody/tr/td[5]/p/text()')
        bodyss = [i.strip() for i in bodyss]  
        bodyssa = [int(i.split(' of ')[1]) for i in bodyss]
        bodysss = [int(i.split(' of ')[0]) for i in bodyss]
    
        legss = tree.xpath('/html/body/section/div/div/table/tbody/tr/td[6]/p/text()')
        legss = [i.strip() for i in legss]  
        legssa = [int(i.split(' of ')[1]) for i in legss]
        legsss = [int(i.split(' of ')[0]) for i in legss]    
        
        distss = tree.xpath('/html/body/section/div/div/table/tbody/tr/td[7]/p/text()')
        distss = [i.strip() for i in distss]  
        distssa = [int(i.split(' of ')[1]) for i in distss]
        distsss = [int(i.split(' of ')[0]) for i in distss]   
        
        clinss = tree.xpath('/html/body/section/div/div/table/tbody/tr/td[8]/p/text()')
        clinss = [i.strip() for i in clinss]  
        clinssa = [int(i.split(' of ')[1]) for i in clinss]
        clinsss = [int(i.split(' of ')[0]) for i in clinss] 
    
        gndss = tree.xpath('/html/body/section/div/div/table/tbody/tr/td[9]/p/text()')
        clinss = [i.strip() for i in gndss]  
        gndssa = [int(i.split(' of ')[1]) for i in gndss]
        gndsss = [int(i.split(' of ')[0]) for i in gndss] 
        
        fighter_profs = tree.xpath('/html/body/section/div/div/section[2]/table/tbody/tr/td[1]/p/a/@href')
        fighter_profs = [i.replace('http://www.ufcstats.com/fighter-details/', '') for i in fighter_profs]
        
        for _kd, _fighter_profs, _ssa, _sss, _tsa, _tss, _sub, _pas, _rev, _headssa, _headsss, _bodyssa, _bodysss, _legssa, _legsss, _distssa, _distsss, _clinssa, _clinsss, _gndssa, _gndsss, _tda, _tds in zip(kd, fighter_profs, ssa, sss, tsa, tss, sub, pas, rev, headssa, headsss, bodyssa, bodysss, legssa,legsss, distssa, distsss, clinssa, clinsss, gndssa, gndsss, tda, tds):
            if _fighter_profs not in all_fighters:
                continue
            script = "INSERT INTO ufc.bout_stats(\
                    bout_id, fighter_id, kd, ssa, sss, tsa, tss, sub, pas, rev, headssa, headsss, bodyssa, bodysss, legssa, legsss, disssa, dissss, clinssa, clinsss, gndssa, gndsss, tda, tds) \
                    	VALUES ('%s', '%s', %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s);" % (bout[0], _fighter_profs, _kd, _ssa, _sss, _tsa, _tss, _sub, _pas, _rev, _headssa, _headsss, _bodyssa, _bodysss, _legssa, _legsss, _distssa, _distsss, _clinssa, _clinsss, _gndssa, _gndsss, _tda, _tds)
            pg_insert(PSQL.client, script) 
    

def pop_bout_res():
#    modern_fights = pg_query(PSQL.client, "select f.fight_id from ufc.fights f full join ufc.bouts b on b.fight_id = f.fight_id full join ufc.bout_results br on br.bout_id = b.bout_id where date > '1-1-2002' and br.bout_id is NULL;")
    
    modern_fights = pg_query(PSQL.client, "select bs.bout_id from ufc.bout_results br full join ufc.bout_stats bs on bs.bout_id = br.bout_id where br.bout_id is Null")
    
    method_dict = pg_query(PSQL.client, "select * from ufc.methods;")
    method_dict = {v:k for k,v in method_dict.values}
    all_fighters = pg_query(PSQL.client, "select fighter_id from ufc.fighters;")
    all_fighters = set([i[0] for i in all_fighters.values])    
    
    all_bouts = pg_query(PSQL.client, "select bout_id from ufc.bout_results")
    all_bouts = set([i[0] for i in all_bouts.values])
    for fight_id in modern_fights[0].unique():
        url = 'http://www.ufcstats.com/event-details/%s' % (fight_id)
        page = requests.get(url)
        tree = html.fromstring(page.content)
    
        fighter_profs = tree.xpath('/html/body/section/div/div/table/tbody/tr/td[2]/p/a/@href')
        fighter_profs = [i.replace('http://www.ufcstats.com/fighter-details/', '') for i in fighter_profs]
            
        methods = tree.xpath('/html/body/section/div/div/table/tbody/tr/td[8]/p[1]/text()')
        methods = [i.strip() for i in methods]
        
        rounds = tree.xpath('/html/body/section/div/div/table/tbody/tr/td[9]/p/text()')
        try:
            rounds = [int(i.strip()) for i in rounds]
        except:
            continue
        times = tree.xpath('/html/body/section/div/div/table/tbody/tr/td[10]/p/text()')
        times = [i.strip() for i in times]
        minutes = [int(i.split(':')[0]) for i in times]
        seconds = [int(i.split(':')[1]) for i in times]
        
        times = [i*60 + j for i,j in zip(minutes, seconds)]
        lengths =  [(i-1)*300 + j for i,j in zip(rounds, times)]
        
        bouts = tree.xpath('/html/body/section/div/div/table/tbody/tr/@data-link')
        bouts = [i.replace('http://www.ufcstats.com/fight-details/', '') for i in bouts]   
        
        fighters = [fighter_profs[i: i+2] for i in range(0, len(fighter_profs), 2)]
        
        for fighter, bout, method, rnd, time, length in zip(fighters, bouts, methods, rounds, times, lengths):        
            if fighter[0] not in all_fighters or fighter[1] not in all_fighters:
                continue
            
            if bout in all_bouts:
                continue
            script = "INSERT INTO ufc.bout_results(\
                        	bout_id, winner, loser, method_id, rounds, time, length)\
                        	VALUES ('%s', '%s', '%s', %i, %i, %i, %i);" % (bout, fighter[0], fighter[1], method_dict[method], rnd, time, length)
            pg_insert(PSQL.client, script) 
                
            
def pop_bouts():
    modern_fights = pg_query(PSQL.client, "select f.fight_id from ufc.fights f full join ufc.bouts b on b.fight_id = f.fight_id where date > '1-1-2002' and b.fight_id is NULL;")
    #method_dict = pg_query(PSQL.client, "select * from ufc.methods;")
    #method_dict = {v:k for k,v in method_dict.values}
    wc_dict = pg_query(PSQL.client, "select * from ufc.weights;")
    wc_dict = {v:k for k,v in wc_dict.values}
    
    champ_icon = 'http://1e49bc5171d173577ecd-1323f4090557a33db01577564f60846c.r80.cf1.rackcdn.com/belt.png'
    
    for fight_id in modern_fights[0].values:
        url = 'http://www.ufcstats.com/event-details/%s' % (fight_id)
        page = requests.get(url)
        tree = html.fromstring(page.content)
        weight_classes = tree.xpath('/html/body/section/div/div/table/tbody/tr/td[7]/p/text()')
        weight_classes = [i.strip() for i in weight_classes if i.strip() != '']
            
        champ_fights = tree.xpath('/html/body/section/div/div/table/tbody/tr/td[7]/p/img/@src')
        champ_fights = len([i for i in champ_fights if i ==  champ_icon])
        
        bouts = tree.xpath('/html/body/section/div/div/table/tbody/tr/@data-link')
        bouts = [i.replace('http://www.ufcstats.com/fight-details/', '') for i in bouts]   
        
        for i, (bout, weight) in enumerate(zip(bouts, weight_classes)):
            if i <= champ_fights:
                champ = 'TRUE'
            else:
                champ = 'FALSE'
            
        
            script = "INSERT INTO ufc.bouts(\
                        	bout_id, fight_id, weight_id, champ)\
                        	VALUES ('%s', '%s', %i, %s);" % (bout, fight_id, wc_dict[weight.replace("'", '')], champ)
            pg_insert(PSQL.client, script)    


def pop_wc_meth():
    all_weight_class = []
    all_methods = []
    modern_fights = pg_query(PSQL.client, "select f.fight_id from ufc.fights f full join ufc.bouts b on b.fight_id = f.fight_id where date > '1-1-2002' and b.fight_id is NULL;")
    all_fighters = pg_query(PSQL.client, "select fighter_id from ufc.fighters;")
    all_fighters = set([i[0] for i in all_fighters.values])
    for fight_id in modern_fights[0].values[1:]:
        url = 'http://www.ufcstats.com/event-details/%s' % (fight_id)
        page = requests.get(url)
        tree = html.fromstring(page.content)
        
        weight_classes = tree.xpath('/html/body/section/div/div/table/tbody/tr/td[7]/p/text()')
        weight_classes = [i.strip() for i in weight_classes if i.strip() != '']
            
        methods = tree.xpath('/html/body/section/div/div/table/tbody/tr/td[8]/p[1]/text()')
        methods = [i.strip() for i in methods]
        
        for method in methods:
            all_methods.append(method)
        for w_class in weight_classes:
            all_weight_class.append(w_class)
            
    wc_set = set(all_weight_class)
    meth_set = set(all_methods)
    
    wc_id = 0
    
    cur_wc = pg_query(PSQL.client, 'Select weight_id, weight_desc from ufc.weights')
    wc_dict = {k:j for j,k in cur_wc.values}
    wc_id = max(wc_dict.values()) + 1
    for wc in wc_set:
        if wc in wc_dict.keys():
            continue
        script = "INSERT INTO ufc.weights(\
                    	weight_id, weight_desc)\
                    	VALUES (%i, '%s');" % (wc_id, wc.replace("'",''))
        pg_insert(PSQL.client, script)        
        wc_id += 1
        
    cur_meth = pg_query(PSQL.client, 'Select method_id, method_desc from ufc.methods')
    meth_dict = {k:j for j,k in cur_meth.values}
    meth_id = max(meth_dict.values()) + 1
    for meth in meth_set:
        if meth in meth_dict.keys() or meth == '':
            continue
        script = "INSERT INTO ufc.methods(\
                    	method_id, method_desc)\
                    	VALUES (%i, '%s');" % (meth_id, meth)
        pg_insert(PSQL.client, script)        
        meth_id += 1

    
def pop_fighters():
    modern_fights = pg_query(PSQL.client, "select f.fight_id from ufc.fights f full join ufc.bouts b on b.fight_id = f.fight_id where date > '1-1-2002' and b.fight_id is NULL;")
    all_fighters = pg_query(PSQL.client, "select fighter_id from ufc.fighters;")
    all_fighters = [i for i in all_fighters[0].values]
    for fight_id in modern_fights[0].values:
        url = 'http://www.ufcstats.com/event-details/%s' % (fight_id)
        page = requests.get(url)
        tree = html.fromstring(page.content)
    
        fighter_profs = tree.xpath('/html/body/section/div/div/table/tbody/tr/td[2]/p/a/@href')
        fighter_profs = [i.replace('http://www.ufcstats.com/fighter-details/', '') for i in fighter_profs]
        
        for fighter in fighter_profs:
            if fighter in all_fighters:
                continue
            
            f_url = 'http://www.ufcstats.com/fighter-details/%s' % (fighter)
            f_page = requests.get(f_url)
            f_tree = html.fromstring(f_page.content)
            if f_page.status_code == 404:
                print('404')
                continue
            f_name = f_tree.xpath('/html/body/section/div/h2/span[1]/text()')
#            f_tree.xpath('/html/body/section/div/h2/span[1]/text()')
            f_name = f_name[0].strip()
            f_height = f_tree.xpath('/html/body/section/div/div/div[1]/ul/li[1]/text()')
            try:
                f_height_inches = int(f_height[1].strip().split("'")[1].strip().replace('"', ''))
            except:
                all_fighters.append(fighter)
                continue
            f_height_feet = int(f_height[1].strip().split("'")[0].strip())
            f_height = f_height_feet * 12 + f_height_inches
            f_reach = f_tree.xpath('/html/body/section/div/div/div[1]/ul/li[3]/text()')
            try:
                f_reach = int(f_reach[1].strip().replace('"',''))
            except:
                all_fighters.append(fighter)
                continue
            f_stance = f_tree.xpath('/html/body/section/div/div/div[1]/ul/li[4]/text()')
            f_stance = f_stance[1].strip()
            
            f_dob = f_tree.xpath('/html/body/section/div/div/div[1]/ul/li[5]/text()')
            try:
                f_dob = datetime.strptime(f_dob[1].strip(), '%b %d, %Y')
            except:
                all_fighters.append(fighter)
                continue            
            
            script = "INSERT INTO ufc.fighters(\
                        fighter_id, name, height, reach, stance, dob)\
                        VALUES ('%s', '%s', %i, %i, '%s', '%s');" % (fighter, f_name.replace("'",''), f_height, f_reach, f_stance, f_dob)
            pg_insert(PSQL.client, script)
            all_fighters.append(fighter)
        
        
def pop_fights():
    #   pg_create_table(PSQL.client, 'bout_corners')
    #http://www.ufcstats.com/event-details/
    url = 'http://www.ufcstats.com/statistics/events/completed?page=all'
    page = requests.get(url)
    tree = html.fromstring(page.content)
    
    ids = tree.xpath('/html/body/section/div/div/div/div[2]/div/table/tbody/tr/td[1]/i/a/@href')
    ids = [i.replace('http://www.ufcstats.com/event-details/', '') for i in ids]
    names = tree.xpath('/html/body/section/div/div/div/div[2]/div/table/tbody/tr/td[1]/i/a/text()')
    names = [i.strip() for i in names]
    locs = tree.xpath('/html/body/section/div/div/div/div[2]/div/table/tbody/tr/td[2]/text()')
    locs = [i.strip() for i in locs]
    countries = [i.split(',')[-1].strip() for i in locs]
    states = [i.split(',')[1].strip() if len(i.split(',')) == 3 else '' for i in locs]
    cities = [i.split(',')[0].strip() for i in locs]
    dates = tree.xpath('/html/body/section/div/div/div/div[2]/div/table/tbody/tr/td[1]/i/span/text()')
    dates = [i.strip() for i in dates]
    dates = [datetime.strptime(i, '%B %d, %Y') for i in dates]
    
    cur_countries = pg_query(PSQL.client, 'Select * from ufc.countries')
    country_dict = {k:j for j,k in cur_countries.values}
    country_id = max(country_dict.values()) + 1
    for country in countries:
        if country in country_dict.keys():
            continue
        script = "INSERT INTO ufc.countries(\
        	country_id, country_name)\
        	VALUES (%i, '%s');" % (country_id, country)
        pg_insert(PSQL.client, script)
        country_dict[country] = country_id
        country_id += 1
        
        
    cur_states = pg_query(PSQL.client, 'Select state_id, state_name from ufc.states')
    state_dict = {k:j for j,k in cur_states.values}
    state_id = max(state_dict.values()) + 1
    for state, country in zip(states, countries):
        if state == '':
            continue
        if state in state_dict.keys():
            continue
        script = "INSERT INTO ufc.states(\
        	state_id, state_name, country_id)\
        	VALUES (%i, '%s', %i);" % (state_id, state, country_dict[country])
        pg_insert(PSQL.client, script)
        state_dict[state] = state_id
        state_id += 1  
        
    
    cur_cities = pg_query(PSQL.client, 'Select city_id, city_name from ufc.cities')
    city_dict = {k:j for j,k in cur_cities.values}
    city_id = max(city_dict.values()) + 1
    for city, state, country in zip(cities, states, countries):
        if city in city_dict.keys():
            continue
        
        if state == '':
            use_state = 'Null'
        else:
            use_state = state_dict[state]
        script = "INSERT INTO ufc.cities(\
        	city_id, city_name, country_id, state_id)\
        	VALUES (%i, '%s', %i, %s);" % (city_id, city, country_dict[country], use_state)
        pg_insert(PSQL.client, script)
        city_dict[city] = city_id
        city_id += 1  
    
    
    cur_fights = pg_query(PSQL.client, 'Select fight_id from ufc.fights')
    cur_fights = set([i[0] for i in cur_fights.values])
    for fight_id, name, country, state, city, date in zip(ids, names, countries, states, cities, dates):
        if fight_id in cur_fights:
            continue

        if state == '':
            use_state = 'Null'
        else:
            use_state = state_dict[state]    
        
        script = "INSERT INTO ufc.fights(\
                	fight_id, name, country_id, state_id, city_id, date)\
                	VALUES ('%s', '%s', %i, %s, %s, '%s');" % (fight_id, name.replace("'",''), country_dict[country], use_state, city_dict[city], date)
        pg_insert(PSQL.client, script)


