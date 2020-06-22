import psycopg2
import json

with open('./credential.json') as f:
  cre = json.load(f)


connection = psycopg2.connect(user = cre["user"],
                              password = cre["password"],
                              host = "awesome-hw.sdsc.edu",
                              port = "5432",
                              database = "postgres")


cursor = connection.cursor()


def excute_sql(sql_text):
    """ excute sql script and return the ouput
        
        sql_text: a string
        
        return: sql excution result formatted in list
    """
    rows = ""
    try:
        cursor.execute(sql_text)
        connection.commit()
        rows = cursor.fetchall()

    except (Exception, psycopg2.Error) as error :
        print ("Error while connecting to PostgreSQL", error)

    return rows


def excute_sql_file(filename):
    """ 
        excute sql file 
        filename: file path
    
    """
    f = open(filename, "r")
    sql_text = f.read()
    return excute_sql(sql_text)

def keywords_filtering_sql(key_words_list, date_after):
    """ 
        excute sql file based on key words/date filtering
    
    """
    if len(key_words_list) == 0 or date_after ==u'':
        print ("Error from function keywords_filtering_sql. key word list cannot be empty and date_after cannot be empty")
        return []
    
    sql_template = """select distinct id, publishdate, news, language, title, keywords \nfrom usnewspaper \nwhere """
    sql_template += "publishdate >= '{}' and \n".format(date_after)  

    #print(("'" + "', '".join(key_words_list) + "do_not_delete_this_one'"))
    sql_template += "keywords && array[{}]".format( ("'" + "', '".join(key_words_list) + "'") ) 
    
    #condition_template = '''keywords ilike '%{}%' or\n'''   
    #sql_template += "\n".join([condition_template.format(i) for i in key_words_list])
    sql_template += ";"
    

    print("SQL EXECUTING: \n" +sql_template)
    return excute_sql(sql_template)
