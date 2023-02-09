import pymysql

class database:
    def __init__(self):
        self.db = pymysql.connect(host='localhost', user='root', password='brighten0701', db='action', charset='utf8')
        self.cursor = self.db.cursor(pymysql.cursors.DictCursor)
    
    def getFace(self):
        try:
            sql = "select distinct URL from face where video_id = (select distinct URL from video order by url desc limit 1,1)"
            self.cursor.execute(sql)
            datas = self.cursor.fetchall()
            face = []
            for data in datas:
                face.append(data['URL'])
            self.db.commit()
        except Exception as ex:
            print(ex)
        finally:
            self.db.close()
            return face

    
    def getVideo(self):
        try: 
            sql = "select distinct URL from video order by url desc limit 1,1"
            self.cursor.execute(sql)
            data = self.cursor.fetchone()
            video = data['URL']
            self.db.commit()
        except Exception as ex:
            print(ex)
        finally:
            self.db.close()
            return video