#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import datetime
import sys
import os
sys.path.append(os.getenv('HIVE_TASK'))
from HiveTask import HiveTask

def get_format_yesterday(num=1,format = '%Y-%m-%d'):
    end_dt = (datetime.date.today() - datetime.timedelta(num)).strftime(format)
    start_dt = (datetime.date.today() - datetime.timedelta(num+90)).strftime(format)
    return start_dt,end_dt
    
def main():
    if(len(sys.argv)>1): # 传了系统参数
        end_dt = sys.argv[1]
        start_dt = (datetime.datetime.strptime(sys.argv[1],'%Y-%m-%d') - datetime.timedelta(90)).strftime( '%Y-%m-%d')
    else:
        start_dt, end_dt = get_format_yesterday()
        
        
    insert_table_sql = """
    ALTER TABLE ***.********** DROP IF EXISTS PARTITION (dt = '{0}');
    insert overwrite table ***.********** partition (dt = '{0}')
    SELECT 
        band ,
        b.*
    from  
        fdm.fdm_pbs_replenishmentall_no_book lateral view json_tuple(processed_info, 'model','days','percent') b as model,days,percent
    where
        dt = '{0}'
    ;
    """
    
    print(insert_table_sql.format(end_dt))
    ht = HiveTask()
    ht.exec_sql(schema_name='app', sql=insert_table_sql.format(end_dt))
    
if __name__ == "__main__":
    main()
