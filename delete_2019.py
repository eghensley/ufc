import os, sys
try:                                            # if running in CLI
    cur_path = os.path.abspath(__file__)
except NameError:                               # if running in IDE
    cur_path = os.getcwd()

while cur_path.split('/')[-1] != 'ufc':
    cur_path = os.path.abspath(os.path.join(cur_path, os.pardir))    
sys.path.insert(1, os.path.join(cur_path, 'lib', 'python3.7', 'site-packages'))

from _connections import db_connection
from db.pop_psql import pg_drop, pg_query


PSQL = db_connection('psql')
bouts = pg_query(PSQL.client, "select b.bout_id from ufc.bouts b join ufc.fights f on f.fight_id = b.fight_id where date > '1-1-2019';")

'1218cba28d4dce94' in [i[0] for i in bouts.values]
for bout in bouts.values:
    pg_drop(PSQL.client, "DELETE from ufc.avg_stats avgs where avgs.bout_id = '%s';" % (bout[0]))
    pg_drop(PSQL.client, "DELETE from ufc.adj_stats avgs where avgs.bout_id = '%s';" % (bout[0]))
    pg_drop(PSQL.client, "DELETE from ufc.adj_avg_stats avgs where avgs.bout_id = '%s';" % (bout[0]))
    pg_drop(PSQL.client, "DELETE from ufc.bout_fighter_xref avgs where avgs.bout_id = '%s';" % (bout[0]))


#    try:
#        pg_query(PSQL.client, "DELETE from ufc.adj_stats;")
#    except:
#        continue
